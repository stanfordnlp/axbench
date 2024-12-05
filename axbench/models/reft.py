from .model import Model
import torch, einops
from tqdm.auto import tqdm
import os
import pandas as pd
from pyvene import (
    IntervenableConfig,
    IntervenableModel
)
from .interventions import (
    ConceptReFTIntervention,
)
from ..utils.constants import EXAMPLE_TAG
from torch.utils.data import DataLoader
from ..utils.model_utils import (
    remove_gradient_parallel_to_decoder_directions,
    gather_residual_activations, 
    get_lr,
    calculate_l1_losses
)
from ..utils.data_utils import (
    parse_positions, 
    get_intervention_locations,
    InterventionDataCollator
)
from dataclasses import dataclass
from transformers import set_seed, get_scheduler, DataCollatorForSeq2Seq, DataCollator
import transformers, datasets
from typing import Dict, Optional, Sequence, Union, List, Any

# using pyreft out-of-the-box
import pyreft


@dataclass
class InterventionEvalDataCollator(object):
    """Collate examples for Intervention."""
    
    tokenizer: transformers.AutoTokenizer
    data_collator: transformers.DataCollator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        intervention_locations will be something like [1,10,0,0,0] where all 0s are padding intervention locations.
        """
        max_intervention_len = max([len(inst["intervention_locations"][0]) for inst in instances])
        max_seq_len = max([len(inst["input_ids"]) for inst in instances])
        
        for inst in instances:
            non_pad_len = len(inst["input_ids"])
            _intervention_location_paddings = torch.tensor(
                [[-1 for _ in range(max_intervention_len - len(inst["intervention_locations"][0]))] for _ in range(inst["intervention_locations"].shape[0])]) # pointing to the first padding token
            inst["intervention_locations"] = torch.cat([inst["intervention_locations"], _intervention_location_paddings], dim=-1).int()
            inst["intervention_locations"] = inst["intervention_locations"] + 1 # shift by 1 to point to the first non-padding token, and all paddings will be 0.

            _input_id_paddings = torch.tensor(
                [self.tokenizer.pad_token_id for _ in range(max_seq_len - non_pad_len)])
            offset = max_seq_len - non_pad_len
            inst["intervention_locations"] = inst["intervention_locations"] + offset
            inst["input_ids"] = torch.cat((_input_id_paddings, torch.tensor([self.tokenizer.pad_token_id]), inst["input_ids"])).int()
            
            inst["attention_mask"] = (inst["input_ids"] != self.tokenizer.pad_token_id).int()

        batch_inputs = self.data_collator(instances)
        return batch_inputs


def make_eval_data_module(
    tokenizer: transformers.PreTrainedTokenizer, model, df, 
    positions="all", # "all_prompt" or "all" or "f1+l1" (pyreft formatting)
    num_interventions=1, 
    nonstop=True, share_weights=True
):
    """Make dataset and collator for supervised fine-tuning with kl div loss."""
    
    all_base_input_ids, all_intervention_locations = [], []
    for _, row in df.iterrows():
        base_prompt = row["input"]
        base_prompt_ids = tokenizer(
            base_prompt, max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]
        base_prompt_length = len(base_prompt_ids)

        if positions == "all_prompt":
            intervention_locations = torch.tensor([[i for i in range(base_prompt_length)]])
        else:
            first_n, last_n = parse_positions(positions)
            intervention_locations = get_intervention_locations(
                last_position=base_prompt_length, 
                first_n=first_n, 
                last_n=last_n,
                pad_mode="first",
                num_interventions=num_interventions,
                share_weights=share_weights,
            )
        all_base_input_ids.append(base_prompt_ids)
        all_intervention_locations.append(intervention_locations)
        
    eval_dataset = datasets.Dataset.from_dict({
        "input_ids": all_base_input_ids,
        "intervention_locations": all_intervention_locations,
    })
    eval_dataset.set_format(
        type='torch', columns=[
            'input_ids', 'intervention_locations',])

    data_collator_fn = transformers.DefaultDataCollator(
        return_tensors="pt"
    )
    data_collator = InterventionEvalDataCollator(tokenizer=tokenizer, data_collator=data_collator_fn)
    return dict(train_dataset=None, eval_dataset=eval_dataset, data_collator=data_collator)


def apply_chat_template(tokenizer, prompt):
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], 
        tokenize=True, add_generation_prompt=True)[1:]
    return tokenizer.decode(formatted_prompt)


class LoReFT(Model):
    def __str__(self):
        return 'LoReFT'

    def make_model(self, **kwargs):
        mode = kwargs.get("mode", "latent")
        # there is one type of intervention throughout
        if mode == "train":
            if self.training_args.reft_type == "Loreft":
                intervention_cls = pyreft.LoreftIntervention
            else:
                intervention_cls = pyreft.NodireftIntervention
            reft_config = pyreft.ReftConfig(representations=[{
                "layer": l, "component": "block_output",
                "low_rank_dimension": kwargs.get("low_rank_dimension", 1),
                "intervention": intervention_cls(embed_dim=self.model.config.hidden_size,
                low_rank_dimension=kwargs.get("low_rank_dimension", 1))} for l in self.training_args.reft_layers])
            ax_model = pyreft.get_reft_model(self.model, reft_config)
            ax_model.set_device(self.device)
            ax_model.print_trainable_parameters()
            self.ax_model = ax_model
        else:
            reft_config = pyreft.ReftConfig(representations=[{
                "layer": l, "component": "block_output",
                "low_rank_dimension": kwargs.get("low_rank_dimension", 1),
                "intervention": ConceptReFTIntervention(
                    n_concepts=kwargs.get("n_concepts", 1),
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )} for l in self.training_args.reft_layers])
            ax_model = pyreft.get_reft_model(self.model, reft_config)
            ax_model.set_device(self.device)
            self.ax_model = ax_model
        self.number_of_interventions = len(self.training_args.reft_layers)
        self.intervention_positions = self.training_args.reft_positions

    def save(self, dump_dir, **kwargs): 
        # 3D matrices saving is needed for ReFT
        # [n_concept, embed_dim, low_rank_dimension]
        proj_weights = []
        source_weights = []
        source_biases = []
        intervention_names = []
        for intervention_name, intervention in self.ax_model.interventions.items():
            intervention_names.append(intervention_name)
            intervention_state_dict = intervention[0].state_dict()
            proj_weight = intervention_state_dict["rotate_layer"] # [embed_dim, low_rank_dimension]
            source_weight = intervention_state_dict["weight"].T # [embed_dim, low_rank_dimension]
            source_bias = intervention_state_dict["bias"] # [low_rank_dimension]
            proj_weights.append(proj_weight)
            source_weights.append(source_weight)
            source_biases.append(source_bias)

        model_name = kwargs.get("model_name", self.__str__())
        weight_file = dump_dir / f"{model_name}_weight.pt"
        if weight_file.exists():
            existing_weight = torch.load(weight_file)
            for i, intervention_name in enumerate(intervention_names):
                existing_weight[f"{intervention_name}.proj_weight"] = torch.cat(
                    [existing_weight[f"{intervention_name}.proj_weight"], proj_weights[i].cpu().unsqueeze(dim=0)], dim=0)
                existing_weight[f"{intervention_name}.source_weight"] = torch.cat(
                    [existing_weight[f"{intervention_name}.source_weight"], source_weights[i].cpu().unsqueeze(dim=0)], dim=0)
        else:
            existing_weight = {}
            for i, intervention_name in enumerate(intervention_names):
                existing_weight[f"{intervention_name}.proj_weight"] = proj_weights[i].cpu().unsqueeze(dim=0)
                existing_weight[f"{intervention_name}.source_weight"] = source_weights[i].cpu().unsqueeze(dim=0)
        torch.save(existing_weight, weight_file)

        bias_file = dump_dir / f"{model_name}_bias.pt"
        if bias_file.exists():
            existing_bias = torch.load(bias_file)
            for i, intervention_name in enumerate(intervention_names):
                existing_bias[f"{intervention_name}.bias"] = torch.cat(
                    [existing_bias[f"{intervention_name}.bias"], source_biases[i].cpu().unsqueeze(dim=0)], dim=0)
        else:
            existing_bias = {}
            for i, intervention_name in enumerate(intervention_names):
                existing_bias[f"{intervention_name}.bias"] = source_biases[i].cpu().unsqueeze(dim=0)
        torch.save(existing_bias, bias_file)

    def load(self, dump_dir=None, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        weight = torch.load(
            f"{dump_dir}/{model_name}_weight.pt"
        )
        bias = torch.load(
            f"{dump_dir}/{model_name}_bias.pt"
        )
        weight_keys = list(weight.keys())
        n_concepts = weight[weight_keys[0]].shape[0]
        low_rank_dimension = weight[weight_keys[0]].shape[-1]
        self.make_model(n_concepts=n_concepts, low_rank_dimension=low_rank_dimension, **kwargs)
        for intervention_name, intervention in self.ax_model.interventions.items():
            intervention[0].W_proj.data = weight[f"{intervention_name}.proj_weight"]
            intervention[0].W_source.data = weight[f"{intervention_name}.source_weight"]
            intervention[0].b_source.data = bias[f"{intervention_name}.bias"]
        self.ax_model.set_device(self.device)
        # ensure everything is in eval mode
        self.model.eval()
        for k, v in self.ax_model.interventions.items():
            _ = v[0].eval()

    def train(self, examples, **kwargs):
        data_module = pyreft.make_multiple_position_supervised_data_module(
            self.tokenizer, self.model, 
            [e for e in examples.input], 
            [e for e in examples.output], 
            positions=self.intervention_positions, 
            num_interventions=self.number_of_interventions, 
            nonstop=True, share_weights=True
        )
        train_dataloader = DataLoader(
            data_module["train_dataset"], shuffle=True, # we shuffle for examples.
            batch_size=self.training_args.batch_size, collate_fn=data_module["data_collator"])
        optimizer = torch.optim.AdamW(
            self.ax_model.parameters(), lr=self.training_args.lr, weight_decay=self.training_args.weight_decay,
            betas=(0.9, 0.999), eps=1e-8)
        num_training_steps = self.training_args.n_epochs * (len(train_dataloader) // self.training_args.gradient_accumulation_steps)
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer,
            num_warmup_steps=0, num_training_steps=num_training_steps)
        # Main training loop.
        rank = torch.distributed.get_rank()
        progress_bar, curr_step = tqdm(range(num_training_steps), position=rank, leave=True), 0
        
        for epoch in range(self.training_args.n_epochs):
            for step, batch in enumerate(train_dataloader):
                # prepare input
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                unit_locations={"sources->base": (
                    None,
                    inputs["intervention_locations"].permute(1, 0, 2).tolist()
                )}
                # forward
                _, cf_outputs = self.ax_model.forward(
                    base={
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"]
                    }, unit_locations=unit_locations, labels=inputs["labels"],
                    use_cache=False)
                # loss
                loss = cf_outputs.loss.mean()
                loss /= self.training_args.gradient_accumulation_steps
                # grads
                loss.backward()

                # Perform optimization step every gradient_accumulation_steps
                if (step + 1) % self.training_args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    torch.nn.utils.clip_grad_norm_(self.ax_model.parameters(), 1.0)
                    curr_step += 1
                    curr_lr = get_lr(optimizer)
                    # optim
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    progress_bar.set_description(
                        "lr %.6f || loss %.6f " % (curr_lr, loss))
        progress_bar.close()

    @torch.no_grad()
    def predict_steer(self, examples, **kwargs):
        # set tokenizer padding to left
        self.tokenizer.padding_side = "left"
        # depending on the model, we use different concept id columns
        concept_id_col = "concept_id"
        # iterate rows in batch
        batch_size = kwargs.get("batch_size", 64)
        eval_output_length = kwargs.get("eval_output_length", 128)
        temperature = kwargs.get("temperature", 1.0)
        all_generations = []
        all_perplexities = []
        # Main training loop.
        rank = torch.distributed.get_rank()

        data_module = make_eval_data_module(
            self.tokenizer, self.model, examples, 
            positions=self.intervention_positions,
            num_interventions=self.number_of_interventions, 
            nonstop=True, share_weights=True
        )
        eval_dataloader = DataLoader(
            data_module["eval_dataset"], shuffle=False,
            batch_size=kwargs.get("batch_size"), 
            collate_fn=data_module["data_collator"])
        
        torch.cuda.empty_cache()
        all_batch_examples = [examples.iloc[i:i+batch_size] for i in range(0, len(examples), batch_size)]
        progress_bar = tqdm(all_batch_examples, position=rank, leave=True)
        for i, batch in enumerate(eval_dataloader):
            # prepare input
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            unit_locations={"sources->base": (
                None,
                inputs["intervention_locations"].permute(1, 0, 2).tolist()
            )}
            batch_examples = all_batch_examples[i]
            idx = torch.tensor(batch_examples["concept_id"].tolist()).to(self.device)
            _, generations = self.ax_model.generate(
                {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}, 
                unit_locations=unit_locations, intervene_on_prompt=True, 
                subspaces=[{"idx": idx}]*self.number_of_interventions,
                max_new_tokens=eval_output_length, do_sample=True, 
                temperature=temperature,
            )
            # Decode and print only the generated text without prompt tokens
            input_lengths = [len(input_ids) for input_ids in inputs["input_ids"]]
            generated_texts = [
                self.tokenizer.decode(generation[input_length:], skip_special_tokens=True)
                for generation, input_length in zip(generations, input_lengths)
            ]
            all_generations += generated_texts

            # Calculate perplexity for each sequence
            unpruned_generated_texts = [
                self.tokenizer.decode(generation, skip_special_tokens=True)
                for generation in generations
            ]
            batch_input_ids = self.tokenizer(
                unpruned_generated_texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device)
            batch_attention_mask = (batch_input_ids != self.tokenizer.pad_token_id).float()
            
            # Forward pass without labels to get logits
            outputs = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            
            logits = outputs.logits[:, :-1, :].contiguous()  # Remove last token prediction
            target_ids = batch_input_ids[:, 1:].contiguous()  # Shift right by 1
            
            # Calculate loss for each token
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            token_losses = loss_fct(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            # Reshape losses and mask
            token_losses = token_losses.view(batch_input_ids.size(0), -1)
            mask = batch_attention_mask[:, 1:].contiguous()
            
            # Calculate perplexity for each sequence
            seq_lengths = mask.sum(dim=1)
            seq_losses = (token_losses * mask).sum(dim=1) / seq_lengths
            seq_perplexities = torch.exp(seq_losses).tolist()
            all_perplexities.extend(seq_perplexities)
            progress_bar.update(1)

        return {
            "steered_generation": all_generations,
            "perplexity": all_perplexities,
        }