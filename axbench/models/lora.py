from pathlib import Path
from .model import Model
import peft
from peft import PeftModel, LoraConfig, get_peft_model
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
from transformers import get_scheduler
from transformers import set_seed


class LoRA(Model):
    def __str__(self):
        return 'LoRA'
    
    def make_model(self, **kwargs):
        peft_config = LoraConfig(
            r=self.training_args.low_rank_dimension,
            lora_alpha=self.training_args.lora_alpha,
            target_modules=self.training_args.lora_components,
            layers_to_transform=self.training_args.lora_layers,
            use_rslora=True, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM"
        )
        ax_model = get_peft_model(self.model, peft_config)
        ax_model.to(self.device)
        ax_model.print_trainable_parameters()
        self.ax_model = ax_model
        # lora is concept-ful due to its nature.
        self.concept_id = kwargs.get("concept_id")

    def save(self, dump_dir, **kwargs):
        # folder-based saving
        dump_dir = Path(f"{dump_dir}/lora/{self.concept_id}")
        dump_dir.mkdir(parents=True, exist_ok=True)
        self.ax_model.save_pretrained(dump_dir)

    def load(self, dump_dir, **kwargs):
        # folder-based loading
        self.concept_id = kwargs.get("concept_id")
        dump_dir = Path(f"{dump_dir}/lora/{self.concept_id}")
        self.ax_model = PeftModel.from_pretrained(
            self.model, dump_dir)

    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples, **kwargs)
        torch.cuda.empty_cache()

        # Optimizer and lr
        optimizer = torch.optim.AdamW(
            self.ax_model.parameters(), 
            lr=self.training_args.lr, weight_decay=0.0)
        num_training_steps = self.training_args.n_epochs * (len(train_dataloader) // self.training_args.gradient_accumulation_steps)
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