from .model import Model
import torch, transformers, datasets
from tqdm.auto import tqdm
import pyreft
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union, List, Any
from torch.utils.data import DataLoader
from ..utils.model_utils import (
    gather_residual_activations, 
    get_lr
)
from transformers import get_scheduler

from .probe import DataCollator, make_data_module
from torch.cuda.amp import autocast
import gc

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


class LMClassification(torch.nn.Module):
    def __init__(self, lm_model):
        super(LMClassification, self).__init__()
        self.lm_model = lm_model
        hidden_size = self.lm_model.config.hidden_size
        self.proj1 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.proj2 = torch.nn.Linear(hidden_size // 2, 2)
        self.dropout = torch.nn.Dropout(0.3)
    
    def forward(self, inputs):
        outputs =self.lm_model(**inputs, output_hidden_states=True)
        last_hiddens = outputs.hidden_states[-1]
        if "attention_mask" in inputs and inputs["attention_mask"] is not None:
            last_token_indices = inputs["attention_mask"].sum(dim=1) - 1
            last_token_representations = last_hiddens[torch.arange(last_hiddens.shape[0]), last_token_indices]
        else:
            last_token_representations = last_hiddens[:, -1]
        # Apply dropout after each layer
        hidden = self.dropout(last_token_representations)
        hidden = torch.relu(self.proj1(hidden))
        hidden = self.dropout(hidden)
        return torch.softmax(self.proj2(hidden), dim=-1)


class ConceptLMClassification(torch.nn.Module):
    def __init__(self, lm_model, **kwargs):
        super(ConceptLMClassification, self).__init__()
        self.lm_model = lm_model
        hidden_size = self.lm_model.config.hidden_size
        # Two projection layers per concept
        self.W_proj1 = torch.nn.Parameter(torch.zeros(
            kwargs["n_concepts"], hidden_size, hidden_size // 2))
        self.b_proj1 = torch.nn.Parameter(torch.zeros(
            kwargs["n_concepts"], hidden_size // 2))
        self.W_proj2 = torch.nn.Parameter(torch.zeros(
            kwargs["n_concepts"], hidden_size // 2, 2))
        self.b_proj2 = torch.nn.Parameter(torch.zeros(
            kwargs["n_concepts"], 2))
        
    def forward(self, last_token_representations, concept_ids=None):
        # First layer with ReLU
        hidden = torch.matmul(last_token_representations, self.W_proj1[concept_ids].permute(1, 0))
        hidden = hidden + self.b_proj1[concept_ids]
        hidden = torch.relu(hidden)
        
        # Second layer with softmax
        logits = torch.matmul(hidden, self.W_proj2[concept_ids].permute(1, 0))
        logits = logits + self.b_proj2[concept_ids]
        return torch.softmax(logits, dim=-1)


class IntegratedGradients(Model):
    def __str__(self):
        return 'IntegratedGradients'

    def make_model(self, **kwargs):
        mode = kwargs.get("mode", "latent")
        # there is one type of intervention throughout
        if mode == "train":
            ax = LMClassification(self.model)
            # Freeze the language model parameters
            for param in ax.lm_model.parameters():
                param.requires_grad = False
                ax.to(self.device)
                self.ax = ax
        else:
            ax = ConceptLMClassification(
                self.model,
                n_concepts=kwargs.get("n_concepts", 1))
            ax.to(self.device)
            self.ax = ax
    
    def save(self, dump_dir, **kwargs): 
        proj1_weight = self.ax.proj1.weight.data
        proj1_bias = self.ax.proj1.bias.data
        proj2_weight = self.ax.proj2.weight.data
        proj2_bias = self.ax.proj2.bias.data

        model_name = kwargs.get("model_name", self.__str__())
        weight_file = dump_dir / f"{model_name}_weight.pt"
        if weight_file.exists():
            existing_weight = torch.load(weight_file)
            existing_weight["proj1_weight"] = torch.cat([existing_weight["proj1_weight"], proj1_weight.cpu().unsqueeze(dim=0)], dim=0)
            existing_weight["proj2_weight"] = torch.cat([existing_weight["proj2_weight"], proj2_weight.cpu().unsqueeze(dim=0)], dim=0)

        else:
            existing_weight = {}
            existing_weight["proj1_weight"] = proj1_weight.cpu().unsqueeze(dim=0)
            existing_weight["proj2_weight"] = proj2_weight.cpu().unsqueeze(dim=0)
        torch.save(existing_weight, weight_file)

        bias_file = dump_dir / f"{model_name}_bias.pt"
        if bias_file.exists():
            existing_bias = torch.load(bias_file)
            existing_bias["proj1_bias"] = torch.cat([existing_bias["proj1_bias"], proj1_bias.cpu().unsqueeze(dim=0)], dim=0)
            existing_bias["proj2_bias"] = torch.cat([existing_bias["proj2_bias"], proj2_bias.cpu().unsqueeze(dim=0)], dim=0)
        else:
            existing_bias = {}
            existing_bias["proj1_bias"] = proj1_bias.cpu().unsqueeze(dim=0)
            existing_bias["proj2_bias"] = proj2_bias.cpu().unsqueeze(dim=0)
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
        self.ax.W_proj1.data = weight["proj1_weight"]
        self.ax.b_proj1.data = bias["proj1_bias"]
        self.ax.W_proj2.data = weight["proj2_weight"]
        self.ax.b_proj2.data = bias["proj2_bias"]

    def make_dataloader(self, examples, **kwargs):
        data_module = make_data_module(self.tokenizer, self.model, examples)
        train_dataloader = DataLoader(
            data_module["train_dataset"], shuffle=True, batch_size=self.training_args.batch_size, 
            collate_fn=data_module["data_collator"])
        return train_dataloader

    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples)
        torch.cuda.empty_cache()
        self.ax.train()
        self.ax.lm_model.eval()
        # Optimizer and lr
        optimizer = torch.optim.AdamW(self.ax.parameters(), lr=self.training_args.lr, weight_decay=self.training_args.weight_decay)
        num_training_steps = self.training_args.n_epochs * max(1, len(train_dataloader) // self.training_args.gradient_accumulation_steps)
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer,
            num_warmup_steps=0, 
            num_training_steps=num_training_steps
        )
        criterion = torch.nn.CrossEntropyLoss()

        # Main training loop.
        progress_bar, curr_step = tqdm(range(num_training_steps)), 0
        for epoch in range(self.training_args.n_epochs):
            for step, batch in enumerate(train_dataloader):
                # Prepare input
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                preds = self.ax(
                    {"input_ids": inputs["input_ids"], 
                     "attention_mask": inputs["attention_mask"]})
                labels = inputs["labels"].long()
                loss = criterion(preds.float(), labels)

                preds_labels = (preds[:, 1] >= 0.5).long()  # Use second column for positive class
                acc = (preds_labels == labels).sum().item() / len(preds)

                # Scale loss for gradient accumulation
                loss = loss / self.training_args.gradient_accumulation_steps
                loss.backward()

                # Perform optimization step every gradient_accumulation_steps
                if (step + 1) % self.training_args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    curr_step += 1  # Increment step only after optimization
                    curr_lr = get_lr(optimizer)
                    progress_bar.update(1)
                    progress_bar.set_description(
                        "lr %.6f || loss %.6f || acc %.6f" % (curr_lr, loss.item(), acc))

        progress_bar.close()
        logger.warning("Training finished.")

    def get_avg_embedding_baseline(self, shape):
        vocab_size = self.tokenizer.vocab_size
        sample_size = min(vocab_size, 1000)  # Sample subset of vocab for efficiency
        # Move random_tokens to the correct device
        random_tokens = torch.randint(0, vocab_size, (sample_size,)).to(self.device)
        embeddings = self.model.get_input_embeddings()(random_tokens)
        avg_embedding = embeddings.mean(dim=0)
        return avg_embedding.expand(shape)

    def predict_latent(self, examples, **kwargs):
        self.ax.eval()
        self.ax.lm_model.eval()

        for param in self.ax.parameters():
            param.requires_grad = False

        total_steps = kwargs.get('steps', 50)  # Total number of steps. 9B models we are using 5.
        step_batch_size = 5  # Number of steps to process at once

        all_pred_label = []
        all_acts = []
        all_max_act = []
        all_max_act_idx = []
        all_max_token = []
        all_tokens = []
        correct = 0
        for _, row in examples.iterrows(): 
            with torch.no_grad():
                inputs = self.tokenizer.encode(
                    row["input"], return_tensors="pt", add_special_tokens=True).to(self.device)
                act_in = gather_residual_activations(
                    self.model, self.layer, {"input_ids": inputs})
                act_in = act_in.detach()
                baseline = self.get_avg_embedding_baseline(act_in.shape)

                baseline = baseline.bfloat16()
                act_in = act_in.bfloat16()

            # Initialize accumulated gradients
            accumulated_grads = torch.zeros_like(act_in[:, kwargs["prefix_length"]:])

            # Process steps in batches
            for step_start in range(0, total_steps, step_batch_size):
                step_end = min(step_start + step_batch_size, total_steps)
                curr_steps = step_end - step_start

                # Calculate alpha values for this batch
                alphas = torch.linspace(
                    step_start/total_steps, 
                    (step_end-1)/total_steps, 
                    curr_steps
                ).view(-1, 1, 1).to(act_in.device)
                
                # Expand for current batch of steps
                curr_baseline = baseline.expand(curr_steps, -1, -1)
                curr_act_in = act_in.expand(curr_steps, -1, -1)
                
                interpolated_acts = curr_baseline + alphas * (curr_act_in - curr_baseline)
                interpolated_acts = interpolated_acts.squeeze(1).requires_grad_(True).to(act_in.dtype)
                
                # Expand inputs for current batch
                expanded_inputs = inputs.expand(curr_steps, -1)
                
                def set_target_act_hook(mod, inputs, outputs):
                    if isinstance(outputs, tuple):
                        new_outputs = (interpolated_acts,) + outputs[1:]
                    else:
                        new_outputs = interpolated_acts
                    return new_outputs

                handle = self.model.model.layers[self.layer].register_forward_hook(
                    set_target_act_hook, always_call=True)
                outputs = self.model.model.forward(**{"input_ids": expanded_inputs}, output_hidden_states=False)
                handle.remove()

                last_token_representations = outputs.last_hidden_state[:, -1]
                preds = self.ax(last_token_representations, row["concept_id"])
                preds = preds[..., 1]

                (grads,) = torch.autograd.grad(preds.sum(), interpolated_acts, create_graph=False)
                # Accumulate gradients for the relevant portion
                accumulated_grads += grads[:, kwargs["prefix_length"]:].sum(dim=0).detach()
                
                # Clear memory
                del grads, interpolated_acts, outputs, last_token_representations, preds
                gc.collect()
                torch.cuda.empty_cache()
        
            # Calculate final integrated gradients
            avg_grads = accumulated_grads / total_steps
            ig = (act_in[:, kwargs["prefix_length"]:] - baseline[:, kwargs["prefix_length"]:]) * avg_grads
            ax_acts = torch.abs(ig).sum(dim=-1)
            
            with torch.no_grad():
                # Get prediction using final interpolation point
                final_outputs = self.model.model.forward(**{"input_ids": inputs}, output_hidden_states=False)
                final_preds = self.ax(final_outputs.last_hidden_state[:, -1], row["concept_id"])

            pred_label = (final_preds[0, 1] >= 0.5).int().item()
            
            actual_label = 1 if row["category"] == "positive" else 0
            correct += (pred_label == actual_label)

            # Process and store results
            ax_acts = ax_acts.detach().flatten().data.float().cpu().numpy().tolist()
            ax_acts = [round(x, 3) for x in ax_acts]
            max_ax_act = max(ax_acts)
            max_ax_act_idx = ax_acts.index(max_ax_act)
            tokens = self.tokenizer.tokenize(row.input)[kwargs["prefix_length"]-1:]
            max_token = tokens[max_ax_act_idx]
            
            all_tokens.append(tokens)
            all_pred_label.append(pred_label)
            all_acts.append(ax_acts)
            all_max_act.append(max_ax_act)
            all_max_act_idx.append(max_ax_act_idx)
            all_max_token.append(max_token)
            
            # Clear memory
            del accumulated_grads, avg_grads, ig, ax_acts
            torch.cuda.empty_cache()

        acc = correct / len(examples)

        logger.warning(f"IntegratedGradients classification accuracy: {acc}")
        return {
            "pred_label": all_pred_label,
            "acts": all_acts,
            "max_act": all_max_act, 
            "max_act_idx": all_max_act_idx,
            "max_token": all_max_token,
            "tokens": all_tokens
        }


class InputXGradients(IntegratedGradients):
    def __str__(self):
        return 'InputXGradients'
    
    def load(self, dump_dir=None, **kwargs):
        # we only need to train onces for both methods.
        super().load(dump_dir, model_name="IntegratedGradients", **kwargs)

    def save(self, dump_dir, **kwargs):
        pass # since we only need to train once for both methods.

    def train(self, examples, **kwargs):
        pass # since we only need to train once for both methods.

    def predict_latent(self, examples, **kwargs):
        batch_size = kwargs.get('batch_size', 32)
        self.ax.eval()
        self.ax.lm_model.eval()
        for param in self.ax.parameters():
            param.requires_grad = False

        all_pred_label = []
        all_acts = []
        all_max_act = []
        all_max_act_idx = []
        all_max_token = []
        all_tokens = []
        correct = 0
        for i in range(0, len(examples), batch_size):
            batch = examples.iloc[i:i + batch_size]
            # Batch encode all inputs
            inputs = self.tokenizer(
                batch["input"].tolist(),
                return_tensors="pt",
                padding=True,
                add_special_tokens=True
            ).to(self.device)
            act_in = gather_residual_activations(
                self.model, self.layer, inputs)
            act_in = act_in.detach().requires_grad_(True)

            def set_target_act_hook(mod, inputs, outputs):
                if isinstance(outputs, tuple):
                    new_outputs = (act_in,) + outputs[1:]
                else:
                    new_outputs = act_in
                return new_outputs

            handle = self.model.model.layers[self.layer].register_forward_hook(
                set_target_act_hook, always_call=True)
            outputs = self.model.model.forward(
                **{"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}, 
                output_hidden_states=False)
            handle.remove()

            last_token_indices = inputs["attention_mask"].sum(dim=1) - 1
            last_token_representations = outputs.last_hidden_state[
                torch.arange(outputs.last_hidden_state.shape[0]), last_token_indices
            ]
            preds = self.ax(last_token_representations, concept_ids=batch["concept_id"].tolist()[0])
            pred_labels = (preds[:, 1] >= 0.5).int()

            # Get gradients
            (grads,) = torch.autograd.grad(preds[:, 1].sum(), act_in)            
            # Process batch results
            for idx, (pred_label, row) in enumerate(zip(pred_labels, batch.itertuples())):
                actual_label = 1 if row.category == "positive" else 0
                correct += (pred_label == actual_label).item()

                # Calculate attributions for this example
                ig = act_in[idx] * grads[idx]
                ax_acts = torch.abs(ig).sum(dim=-1)[kwargs["prefix_length"]:]
                
                # Process and store results
                ax_acts = ax_acts.detach().data.float().cpu().numpy().tolist()
                ax_acts = [round(x, 3) for x in ax_acts]
                max_ax_act = max(ax_acts)
                max_ax_act_idx = ax_acts.index(max_ax_act)
                tokens = self.tokenizer.tokenize(row.input)[kwargs["prefix_length"]-1:]
                max_token = tokens[max_ax_act_idx]
                
                all_tokens.append(tokens)
                all_pred_label.append(pred_label.item())
                all_acts.append(ax_acts)
                all_max_act.append(max_ax_act)
                all_max_act_idx.append(max_ax_act_idx)
                all_max_token.append(max_token)

            # Clear memory
            del grads, act_in, outputs
            torch.cuda.empty_cache()

        acc = correct / len(examples)
        logger.warning(f"InputXGradients classification accuracy: {acc}")
        return {
            "pred_label": all_pred_label,
            "acts": all_acts,
            "max_act": all_max_act, 
            "max_act_idx": all_max_act_idx,
            "max_token": all_max_token,
            "tokens": all_tokens
        }
