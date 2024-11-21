from .model import Model
import torch
from tqdm.auto import tqdm
import os
import pandas as pd
from pyvene import (
    IntervenableConfig,
    IntervenableModel
)
from .interventions import (
    LoreftIntervention,
    ConceptLoreftIntervention,
)
from ..utils.constants import EXAMPLE_TAG
from torch.utils.data import DataLoader
from ..utils.model_utils import (
    set_decoder_norm_to_unit_norm, 
    remove_gradient_parallel_to_decoder_directions,
    gather_residual_activations, 
    get_lr,
    calculate_l1_losses
)
from transformers import get_scheduler
from transformers import set_seed


class ReFT(Model):
    def __str__(self):
        return 'ReFT'

    def make_model(self, **kwargs):
        mode = kwargs.get("mode", "latent")
        if mode == "train":
            ax = LoreftIntervention(
                embed_dim=self.model.config.hidden_size, 
                low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                dtype=kwargs.get("dtype", torch.bfloat16)
            )
        else:
            ax = ConceptLoreftIntervention(
                n_concepts=kwargs.get("n_concepts", 1),
                embed_dim=self.model.config.hidden_size, 
                low_rank_dimension=kwargs.get("low_rank_dimension", 1),
            )
        layers = self.steering_layers if self.steering_layers else [self.layer]
        self.ax = ax.to(self.device)
        self.ax.train()
        ax_config = IntervenableConfig(representations=[{
            "layer": l,
            "component": f"model.layers[{l}].output",
            "low_rank_dimension": kwargs.get("low_rank_dimension", 1),
            "intervention": self.ax} for l in layers])
        ax_model = IntervenableModel(ax_config, self.model)
        ax_model.disable_model_gradients()    
        ax_model.set_device(self.device)
        self.ax_model = ax_model

    def save(self, dump_dir, **kwargs): 
        # 3D matrices saving is needed for ReFT
        # [n_concept, embed_dim, low_rank_dimension]
        proj_weight = self.ax.proj.weight.data # [embed_dim, low_rank_dimension]
        source_weight = self.ax.learned_source.weight.data # [embed_dim, low_rank_dimension]
        source_bias = self.ax.learned_source.bias.data # [low_rank_dimension]

        model_name = kwargs.get("model_name", self.__str__())
        weight_file = dump_dir / f"{model_name}_weight.pt"
        proj_weight = proj_weight.cpu().unsqueeze(dim=0)
        source_weight = source_weight.cpu().unsqueeze(dim=0)
        if weight_file.exists():
            existing_weight = torch.load(weight_file)
            existing_weight["proj_weight"] = torch.cat([existing_weight["proj_weight"], proj_weight], dim=0)
            existing_weight["source_weight"] = torch.cat([existing_weight["source_weight"], source_weight], dim=0)
        else:
            existing_weight = {
                "proj_weight": proj_weight,
                "source_weight": source_weight
            }
        torch.save(existing_weight, weight_file)

        bias_file = dump_dir / f"{model_name}_bias.pt"
        source_bias = source_bias.cpu().unsqueeze(dim=0)
        if bias_file.exists():
            source_bias = torch.cat([torch.load(bias_file), source_bias], dim=0)
        torch.save(source_bias, bias_file)

    def load(self, dump_dir=None, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        weight = torch.load(
            f"{dump_dir}/{model_name}_weight.pt"
        )
        bias = torch.load(
            f"{dump_dir}/{model_name}_bias.pt"
        )
        n_concepts = weight["proj_weight"].shape[0]
        low_rank_dimension = weight["proj_weight"].shape[-1]
        self.make_model(n_concepts=n_concepts, low_rank_dimension=low_rank_dimension, **kwargs)
        self.ax.W_proj.data = weight["proj_weight"].to(self.device)
        self.ax.W_source.data = weight["source_weight"].to(self.device)
        self.ax.b_source.data = bias.to(self.device)

    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples)
        torch.cuda.empty_cache()

        # Optimizer and lr
        optimizer = torch.optim.AdamW(self.ax_model.parameters(), lr=self.training_args.lr)
        num_training_steps = self.training_args.n_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer,
            num_warmup_steps=0, num_training_steps=num_training_steps)
        # Main training loop.
        rank = torch.distributed.get_rank()
        progress_bar, curr_step = tqdm(range(num_training_steps), position=rank, leave=True), 0
        
        for epoch in range(self.training_args.n_epochs):
            for batch in train_dataloader:
                # prepare input
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                unit_locations={"sources->base": (
                    None,
                    inputs["intervention_locations"].permute(1, 0, 2).tolist()
                )}
        
                # forward
                _, cf_outputs = self.ax_model(
                    base={
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"]
                    }, unit_locations=unit_locations, labels=inputs["labels"],
                    use_cache=False)
                
                # loss
                loss = cf_outputs.loss
                
                # grads
                loss.backward()
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
    def predict_latent(self, examples, **kwargs):
        self.ax.eval()
        batch_size = kwargs.get('batch_size', 32)
        
        all_acts = []
        all_max_act = []
        all_max_act_idx = []
        all_max_token = []
        all_tokens = []
        # Process in batches
        for i in range(0, len(examples), batch_size):
            batch = examples.iloc[i:i + batch_size]
            # Batch encode all inputs
            inputs = self.tokenizer(
                batch["input"].tolist(), return_tensors="pt", 
                add_special_tokens=True, padding=True, truncation=True).to(self.device)
            
            gather_acts = gather_residual_activations(
                self.model, self.layer, inputs)
            _, _, ax_acts = self.ax.encode(
                gather_acts[:, 1:],  # no bos token
                subspaces={
                    "input_subspaces": torch.tensor(batch["concept_id"].tolist()).to(self.device)
                }, k=1)
            seq_lens = inputs["attention_mask"].sum(dim=1) - 1 # no bos token
            # Process each sequence in the batch
            for seq_idx, ax_seq in enumerate(ax_acts):
                acts = ax_seq[:seq_lens[seq_idx]].flatten().data.float().cpu().numpy().tolist()
                acts = [round(x, 3) for x in acts]
                max_act = max(acts)
                max_act_indices = [i for i, x in enumerate(acts) if x == max_act]
                max_act_idx = max_act_indices[0]
                # Get tokens for this specific sequence
                tokens = self.tokenizer.tokenize(batch.iloc[seq_idx]["input"])
                max_token = tokens[max_act_idx]
                all_acts.append(acts)
                all_max_act.append(max_act)
                all_max_act_idx.append(max_act_idx)
                all_max_token.append(max_token)
                all_tokens.append(tokens)
        return {
            "acts": all_acts,
            "max_act": all_max_act,
            "max_act_idx": all_max_act_idx,
            "max_token": all_max_token,
            "tokens": all_tokens
        }
