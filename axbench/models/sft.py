from pathlib import Path
from .model import Model
import torch, einops
from tqdm.auto import tqdm
import os
import pandas as pd
from ..utils.constants import EXAMPLE_TAG
from torch.utils.data import DataLoader
from ..utils.model_utils import (
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
    gather_residual_activations, 
    get_lr,
    calculate_l1_losses
)
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import transformers, datasets
from transformers import get_scheduler
from transformers import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

@dataclass
class DataCollator(object):
    
    tokenizer: transformers.AutoTokenizer
    data_collator: transformers.DataCollator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        max_seq_len = max([len(inst["input_ids"]) for inst in instances])
        
        for inst in instances:
            non_pad_len = len(inst["input_ids"])

            _input_id_paddings = torch.tensor(
                [self.tokenizer.pad_token_id for _ in range(max_seq_len - non_pad_len)])
            inst["input_ids"] = torch.cat((inst["input_ids"], torch.tensor([self.tokenizer.pad_token_id]), _input_id_paddings)).int()

            _label_paddings = torch.tensor([-100 for _ in range(max_seq_len - non_pad_len+1)])
            inst["labels"] = torch.cat((inst["labels"], _label_paddings))
            
            inst["attention_mask"] = (inst["input_ids"] != self.tokenizer.pad_token_id).int()

        batch_inputs = self.data_collator(instances)
        return batch_inputs


def make_sft_data_module(
    tokenizer: transformers.PreTrainedTokenizer, df, 
    dataset_category="continuation",
    positions="all", # "all_prompt" or "all" or "f1+l1" (pyreft formatting)
    exclude_bos=True,
    prefix_length=1,
    **kwargs
):
    """Make dataset and collator for supervised fine-tuning with kl div loss."""
    if not exclude_bos:
        prefix_length = 0
    
    all_base_input_ids, all_output_ids = [], []
    all_prompt_lengths = []
    for _, row in df.iterrows():
        _input, _output = row["input"], row["output"]
        # prepare input ids
        base_prompt = _input
        if isinstance(_output, float):
            _output = tokenizer.eos_token
        base_input = base_prompt + _output
        base_prompt_ids = tokenizer(
            base_prompt, max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]
        base_input_ids = tokenizer(
            base_input, max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]
        base_prompt_length = len(base_prompt_ids)
        base_length = len(base_input_ids)

        # output ids with prompt token mask
        output_ids = base_input_ids.clone()
        output_ids[:base_prompt_length] = -100

        all_base_input_ids.append(base_input_ids)
        all_output_ids.append(output_ids)
        
    train_dataset = datasets.Dataset.from_dict({
        "input_ids": all_base_input_ids,
        "labels": all_output_ids,
    })
    train_dataset.set_format(
        type='torch', columns=['input_ids', 'labels'])

    data_collator_fn = transformers.DefaultDataCollator(
        return_tensors="pt"
    )
    data_collator = DataCollator(tokenizer=tokenizer, data_collator=data_collator_fn)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


class SFT(Model):

    def __init__(self, model, tokenizer, layer, training_args=None, **kwargs):
        super().__init__(model, tokenizer, layer, training_args, **kwargs)
        self.lm_model_name = kwargs.get("lm_model_name")

    def __str__(self):
        return 'SFT'
    
    def make_model(self, **kwargs):
        self.ax_model = self.model
        self.concept_id = kwargs.get("concept_id")

    def save(self, dump_dir, **kwargs):
        # folder-based saving
        dump_dir = Path(f"{dump_dir}/sft/{self.concept_id}")
        dump_dir.mkdir(parents=True, exist_ok=True)
        self.ax_model.save_pretrained(dump_dir)

    def load(self, dump_dir, **kwargs):
        # folder-based loading
        self.concept_id = kwargs.get("concept_id")
        dump_dir = Path(f"{dump_dir}/sft/{self.concept_id}")
        self.ax_model = AutoModelForCausalLM.from_pretrained(
            dump_dir, torch_dtype=torch.bfloat16)
        self.ax_model.to(self.device)

    def _train(self, examples, **kwargs):
        self.ax_model.train()
        train_dataloader = self.make_dataloader(examples, **kwargs)
        torch.cuda.empty_cache()

        # Optimizer and lr
        optimizer = torch.optim.AdamW(
            self.ax_model.parameters(), 
            lr=self.training_args.lr, weight_decay=self.training_args.weight_decay)
        num_training_steps = self.training_args.n_epochs * max(1, len(train_dataloader) // self.training_args.gradient_accumulation_steps)
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer,
            num_warmup_steps=0, num_training_steps=num_training_steps)
        norm_loss_fn = torch.nn.MSELoss()
        # Main training loop.
        rank = torch.distributed.get_rank()
        progress_bar, curr_step = tqdm(range(num_training_steps), position=rank, leave=True), 0
        
        for epoch in range(self.training_args.n_epochs):
            for step, batch in enumerate(train_dataloader):
                # prepare input
                inputs = {k: v.to(self.device) for k, v in batch.items()}
        
                # forward
                outputs = self.ax_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"]
                )
                
                # loss
                loss = outputs.loss
                loss = loss.mean()
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
                        "lr %.6f || loss %.6f" % (curr_lr, loss))
        progress_bar.close()

    def _train_fsdp(self, examples, **kwargs):
        self.ax_model.train()
        data_module = make_sft_data_module(self.tokenizer, examples, **kwargs)
        torch.cuda.empty_cache()

        # huggingface trainer with FSDP training
        training_args = TrainingArguments(
            output_dir=str(self.dump_dir),
            logging_steps=1,
            save_strategy="no",
            num_train_epochs=self.training_args.n_epochs,
            per_device_train_batch_size=self.training_args.batch_size,
            gradient_accumulation_steps=self.training_args.gradient_accumulation_steps,
            evaluation_strategy="no",  # "no" means no eval loop, only training
            learning_rate=self.training_args.lr,
            weight_decay=self.training_args.weight_decay,
            warmup_ratio=0.00,
            lr_scheduler_type="linear",
            fsdp="full_shard auto_wrap",
            fsdp_transformer_layer_cls_to_wrap="Gemma2DecoderLayer",
            bf16=True,  # whether to use bf16
            do_train=True,
            do_eval=False,
            report_to=[]
        )

        trainer = Trainer(
            model=self.ax_model, 
            tokenizer=self.tokenizer, 
            args=training_args, 
            **data_module
        )
        trainer.train()

    def train(self, examples, **kwargs):
        if "gemma-2-9b" in self.lm_model_name:
            # huggingface trainer ith FSDP training
            self._train_fsdp(examples, **kwargs)
        else:
            self._train(examples, **kwargs)

    @torch.no_grad()
    def predict_steer(self, examples, **kwargs):
        self.ax_model.eval()
        # set tokenizer padding to left
        self.tokenizer.padding_side = "left"

        # iterate rows in batch
        batch_size = kwargs.get("batch_size", 64)
        eval_output_length = kwargs.get("eval_output_length", 128)
        temperature = kwargs.get("temperature", 1.0)
        all_generations = []
        all_perplexities = []
        # Main training loop.
        rank = torch.distributed.get_rank()
        progress_bar = tqdm(range(0, len(examples), batch_size), position=rank, leave=True)
        for i in range(0, len(examples), batch_size):
            batch_examples = examples.iloc[i:i+batch_size]
            input_strings = batch_examples['input'].tolist()
            # tokenize input_strings
            inputs = self.tokenizer(
                input_strings, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            generations = self.ax_model.generate(
                **inputs, 
                max_new_tokens=eval_output_length, do_sample=True, 
                temperature=temperature,
            )

            # Decode and print only the generated text without prompt tokens
            input_lengths = [len(input_ids) for input_ids in inputs.input_ids]
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
    
