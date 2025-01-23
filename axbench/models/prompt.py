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


# This is copied from our LLM Judge evaluator for concept relevance.
# The sentence field will be filled with the actual output.
UNIDIRECTIONAL_PAIRWISE_EVALUATION_CONCEPT_RELEVANCE_TEMPLATE = """[System]
Please act as an impartial judge and evaluate whether the specified concept is \
presented in the sentence fragment provided below. Focus solely on whether the concept is \
clearly incorporated, without regard for grammar or logical coherence.

Begin your evaluation by providing a short explanation of whether the concept is \
clearly incorporated. Point out references to the concept in the fragment, noting any \
minimal or awkward incorporations. After providing your explanation, rate the concept's \
relevance on a scale from 0 to 2, where 0 indicates the concept is not present at all, \
1 indicates the concept is somewhat present but minimally or awkwardly incorporated, \
and 2 indicates the concept is more fully and effectively incorporated, with stronger and \
more natural integration. Provide your rating using this exact format: "Rating: [[score]]".

[Concept Start]
{concept}
[Concept End]

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""


class PromptSteering(Model):
    input_field = "steered_input"
    def __str__(self):
        return 'PromptSteering'

    def load(self, dump_dir=None, **kwargs):
        pass

    def make_model(self, **kwargs):
        pass

    def save(self, dump_dir, **kwargs):
        pass # since we only need to train once for both methods.

    def train(self, examples, **kwargs):
        pass # since we only need to train once for both methods.

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


class PromptDetection(Model):
    input_field = "output"
    concept_field = "output_concept"
    def __str__(self):
        return 'PromptDetection'

    def load(self, dump_dir=None, **kwargs):
        pass

    def make_model(self, **kwargs):
        pass

    def save(self, dump_dir, **kwargs):
        pass # since we only need to train once for both methods.

    def train(self, examples, **kwargs):
        pass # since we only need to train once for both methods.

    def _get_rating_from_completion(self, completion):
        try:
            # Check if "Rating:" is in the completion
            if "Rating:" in completion:
                # Extract the part after "Rating:"
                rating_text = completion.split("Rating:")[-1].strip()
                # Take only the first line in case there's additional text
                rating_text = rating_text.split('\n')[0].strip()
                # Remove any extra characters around the number
                rating_text = rating_text.replace('[', '').replace(']', '').strip('"').strip("'").strip("*").strip()
                # Convert to float and return the rating
                rating = float(rating_text)
                
                # Ensure the rating is within the expected range
                if rating < 0 or rating > 2:
                    raise ValueError(f"Invalid rating value: {rating}")
                return rating
            else:
                # Log warning and return default if "Rating:" is missing
                logger.warning(f"Cannot find rating value: {completion}")
                return -1  # DEFAULT_RATING
        except (ValueError, IndexError) as e:
            # Catch parsing errors and log them
            logger.error(f"Error parsing rating from completion: {completion}. Error: {e}")
            return -1  # DEFAULT_RATING

    def predict_latent(self, examples, **kwargs):
        self.model.eval()
        # set tokenizer padding to left
        self.tokenizer.padding_side = "left"

        # iterate rows in batch
        batch_size = kwargs.get("batch_size", 64)
        eval_output_length = kwargs.get("eval_output_length", 512)
        temperature = kwargs.get("temperature", 1.0)
        all_max_act = []
        for i in tqdm(range(0, len(examples), batch_size), desc="Predicting latent"):
            batch_examples = examples.iloc[i:i+batch_size]
            input_strings = batch_examples[self.input_field].tolist()
            concept_strings = [kwargs["concept"]] * len(input_strings)
            # apply the template to each concept and input
            template_strings = [
                self.tokenizer.decode(self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": UNIDIRECTIONAL_PAIRWISE_EVALUATION_CONCEPT_RELEVANCE_TEMPLATE.format(
                        concept=concept, sentence=input)}], 
                    tokenize=True, add_generation_prompt=True)[1:])
                for concept, input in zip(concept_strings, input_strings)
            ]
            # tokenize template_strings
            inputs = self.tokenizer(
                template_strings, return_tensors="pt", padding=True, truncation=True
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
            for generated_text in generated_texts:
                rating = self._get_rating_from_completion(generated_text)
                all_max_act.append(rating)
        return {
            # "acts": all_acts,
            "max_act": all_max_act,
            # "max_act_idx": all_max_act_idx,
            # "max_token": all_max_token,
            # "tokens": all_tokens
        }

    def predict_latents(self, examples, **kwargs):
        return self.predict_latent(examples, **kwargs)

    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        max_activations = {}
        return max_activations