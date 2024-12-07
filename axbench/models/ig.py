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

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


class LMClassification(torch.nn.Module):
    def __init__(self, lm_model, low_rank_dimension):
        super(LMClassification, self).__init__()
        self.lm_model = lm_model
        self.proj = torch.nn.Linear(self.lm_model.config.hidden_size, low_rank_dimension)
    
    def forward(self, inputs):
        outputs =self.lm_model(**inputs, output_hidden_states=True)
        last_hiddens = outputs.hidden_states[-1]
        if "attention_mask" in inputs and inputs["attention_mask"] is not None:
            last_token_indices = inputs["attention_mask"].sum(dim=1) - 1
            last_token_representations = last_hiddens[torch.arange(last_hiddens.shape[0]), last_token_indices]
        else:
            last_token_representations = last_hiddens[:,-1]
        return torch.sigmoid(self.proj(last_token_representations))


class IntegratedGradients(Model):
    def __str__(self):
        return 'IntegratedGradients'

    def make_model(self, **kwargs):
        ax = LMClassification(self.model, kwargs.get("low_rank_dimension", 1))
        # Freeze the language model parameters
        for param in ax.lm_model.parameters():
            param.requires_grad = False
        ax.to(self.device)
        self.ax = ax
    
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
        # Optimizer and lr
        optimizer = torch.optim.AdamW(self.ax.parameters(), lr=self.training_args.lr, weight_decay=self.training_args.weight_decay)
        num_training_steps = self.training_args.n_epochs * len(train_dataloader) // self.training_args.gradient_accumulation_steps
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer,
            num_warmup_steps=0, 
            num_training_steps=num_training_steps
        )
        criterion = torch.nn.BCELoss()

        # Main training loop.
        progress_bar, curr_step = tqdm(range(num_training_steps)), 0
        for epoch in range(self.training_args.n_epochs):
            for step, batch in enumerate(train_dataloader):
                # Prepare input
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                preds = self.ax(
                    {"input_ids": inputs["input_ids"], 
                     "attention_mask": inputs["attention_mask"]})
                labels = inputs["labels"].unsqueeze(1).float()
                loss = criterion(preds.float(), labels)

                preds_labels = (preds >= 0.5).float()  # Apply threshold
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

    def predict_latent(self, examples, **kwargs):
        self.ax.eval()
        for param in self.ax.parameters():
            param.requires_grad = False

        steps = kwargs.get('steps', 10)  # Default to 10 steps if not specified

        # Get the token embedding of a single space, excluding special tokens
        space_input = self.tokenizer(" ", return_tensors="pt", add_special_tokens=False).to(self.device)
        space_embedding = self.model.get_input_embeddings()(space_input.input_ids)

        all_pred_label = []
        all_acts = []
        all_max_act = []
        all_max_act_idx = []
        all_max_token = []
        all_tokens = []
        correct = 0
        for _, row in examples.iterrows(): 
            inputs = self.tokenizer.encode(
                row["input"], return_tensors="pt", add_special_tokens=True).to(self.device)
            act_in = gather_residual_activations(
                self.model, self.layer, {"input_ids": inputs})
            act_in = act_in.detach()
            torch.cuda.empty_cache()
            # Create a baseline using the space embedding, excluding the first token if it's special
            baseline = space_embedding.expand_as(act_in)
            
            # Create interpolated inputs for all steps at once
            alphas = torch.linspace(0, 1, steps).view(-1, 1, 1, 1).to(act_in.device)
            interpolated_acts = baseline + alphas * (act_in - baseline)
            interpolated_acts = interpolated_acts.squeeze(1).requires_grad_(True).to(
                act_in.dtype) # step, seq_len, hidden_size
            # Expand inputs to match the number of steps
            expanded_inputs = inputs.expand(steps, -1)
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
            preds = torch.sigmoid(self.ax.proj(last_token_representations))
            preds = preds[..., row["concept_id"]]  # only consider the target concept
            (grads,) = torch.autograd.grad(preds.sum(), interpolated_acts, create_graph=True)
            # Average gradients and multiply by (input - baseline)
            ax_acts = torch.abs((act_in - baseline) * grads.mean(dim=0)).sum(dim=-1)[:,kwargs["prefix_length"]:]

            pred_label = (preds[-1].flatten() >= 0.5).int().tolist()[0]
            actual_label = 1 if row["category"] == "positive" else 0
            correct += (pred_label == actual_label)

            ax_acts = ax_acts.detach().flatten().data.float().cpu().numpy().tolist()
            ax_acts = [round(x, 3) for x in ax_acts]
            max_ax_act = max(ax_acts)
            max_ax_act_idx = ax_acts.index(max_ax_act)
            tokens = self.tokenizer.tokenize(row.input)[kwargs["prefix_length"]-1:] # -1 is because it does not prepend BOS token
            max_token = tokens[max_ax_act_idx]
            all_tokens.append(tokens)

            all_pred_label += [pred_label]
            all_acts += [ax_acts]
            all_max_act += [max_ax_act]
            all_max_act_idx += [max_ax_act_idx]
            all_max_token += [max_token]
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

    def predict_latent(self, examples, **kwargs):
        self.ax.eval()
        batch_size = kwargs.get('batch_size', 32)

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
            # simulate forward pass for the rest of the layers in the model
            def set_target_act_hook(mod, inputs, outputs):
                if isinstance(outputs, tuple):
                    new_outputs = (act_in,) + outputs[1:]
                else:
                    new_outputs = act_in
                return new_outputs
            handle = self.model.model.layers[self.layer].register_forward_hook(
                set_target_act_hook, always_call=True)
            pred = self.ax(
                {"input_ids": inputs["input_ids"], 
                 "attention_mask": inputs["attention_mask"]}
            )
            handle.remove()
            pred = pred[torch.arange(pred.shape[0]), batch["concept_id"].tolist()] # only consider the target concept
            # get gradient
            (grad,) = torch.autograd.grad(pred.sum(), act_in)
            pred_labels = (pred.flatten() >= 0.5).int().tolist()
            
            seq_lens = inputs["attention_mask"].sum(dim=1) - kwargs["prefix_length"] # no bos token
            # Handle batch of examples
            for idx, (pred_label, row) in enumerate(zip(pred_labels, batch.itertuples())):
                actual_label = 1 if row.category == "positive" else 0
                correct += (pred_label == actual_label)
                
                # Get attributions for this example
                ax_acts_single = torch.abs(act_in[idx] * grad[idx]).sum(dim=-1)[kwargs["prefix_length"]:]  # Remove first token
                ax_acts = ax_acts_single[:seq_lens[idx]].flatten().data.float().cpu().numpy().tolist()
                ax_acts = [round(x, 3) for x in ax_acts]
                max_ax_act = max(ax_acts)
                max_ax_act_idx = ax_acts.index(max_ax_act)
                tokens = self.tokenizer.tokenize(row.input)[kwargs["prefix_length"]-1:] # -1 is because it does not prepend BOS token
                max_token = tokens[max_ax_act_idx]

                all_pred_label.append(pred_label)
                all_acts.append(ax_acts)
                all_max_act.append(max_ax_act)
                all_max_act_idx.append(max_ax_act_idx)
                all_max_token.append(max_token)
                all_tokens.append(tokens)
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
