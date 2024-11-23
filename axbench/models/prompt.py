from .model import Model
import torch, transformers, datasets
from tqdm.auto import tqdm
import os
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union, List, Any
from torch.utils.data import DataLoader

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


class PromptSteering(Model):
    input_field = "steered_input"
    def __str__(self):
        return 'PromptSteering'

    def load(self, dump_dir=None, **kwargs):
        pass

    def make_model(self, **kwargs):
        pass
    
    @torch.no_grad()
    def predict_steer(self, examples, **kwargs):
        self.model.eval()
        # set tokenizer padding to left
        self.tokenizer.padding_side = "left"

        # iterate rows in batch
        batch_size = kwargs.get("batch_size", 64)
        eval_output_length = kwargs.get("eval_output_length", 128)
        temperature = kwargs.get("temperature", 1.0)
        all_generations = []
        all_perplexities = []
        for i in range(0, len(examples), batch_size):
            batch_examples = examples.iloc[i:i+batch_size]
            input_strings = batch_examples[self.input_field].tolist()
            # tokenize input_strings
            inputs = self.tokenizer(
                input_strings, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            generations = self.model.generate(
                **inputs, max_new_tokens=eval_output_length, do_sample=True, 
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
            batch_input_ids = self.tokenizer(
                generated_texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device)
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

        return {
            "steered_generation": all_generations,
            "perplexity": all_perplexities,
        }

    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        max_activations = {} # sae_id to max_activation
        return max_activations  

class PromptBaseline(PromptSteering):
    input_field = "input"
    def __str__(self):
        return 'PromptBaseline'

