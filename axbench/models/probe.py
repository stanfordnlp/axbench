from .model import Model
import torch, transformers, datasets
from tqdm.auto import tqdm

try:
    # This library is our indicator that the required installs
    # need to be done.
    import pyreax

except ModuleNotFoundError:
    # relative import; better to pip install subctrl
    import sys
    sys.path.append("../../../pyreax")
    import pyreax

import os
import pandas as pd
from pyvene import (
    IntervenableConfig,
    IntervenableModel
)
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union, List, Any
from torch.utils.data import DataLoader
from pyreax import (
    AdditionIntervention
)
from pyreax import (
    set_decoder_norm_to_unit_norm, 
    remove_gradient_parallel_to_decoder_directions,
    gather_residual_activations, 
    get_lr
)
from pyreax.utils.model_utils import calculate_l1_losses
from transformers import get_scheduler

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


@dataclass
class DataCollator(object):
    """Collate examples for ReFT."""
    
    tokenizer: transformers.AutoTokenizer
    data_collator: transformers.DataCollator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        max_seq_len = max([len(inst["input_ids"]) for inst in instances])
        
        for inst in instances:
            non_pad_len = len(inst["input_ids"])
            _input_id_paddings = torch.tensor(
                [self.tokenizer.pad_token_id for _ in range(max_seq_len - non_pad_len)])
            inst["input_ids"] = torch.cat((inst["input_ids"], torch.tensor([self.tokenizer.pad_token_id]), _input_id_paddings)).int()
            inst["attention_mask"] = (inst["input_ids"] != self.tokenizer.pad_token_id).int()
            inst["labels"] = inst["labels"].int()
        batch_inputs = self.data_collator(instances)
        return batch_inputs


def make_data_module(
    tokenizer: transformers.PreTrainedTokenizer, model, df,
):
    all_input_ids, all_labels = [], []
    for _, row in df.iterrows():
        input_ids = tokenizer(
            row["input"], max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]
        all_input_ids.append(input_ids)
        all_labels.append(row["labels"])
        
    train_dataset = datasets.Dataset.from_dict({
        "input_ids": all_input_ids,
        "labels": all_labels
    })
    train_dataset.set_format(type='torch', columns=['input_ids', 'labels'])

    data_collator_fn = transformers.DefaultDataCollator(
        return_tensors="pt"
    )
    data_collator = DataCollator(tokenizer=tokenizer, data_collator=data_collator_fn)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, low_rank_dimension):
        super(LogisticRegressionModel, self).__init__()
        # Linear layer: input_dim -> 1 output (since binary classification)
        self.proj = torch.nn.Linear(input_dim, low_rank_dimension)
    
    def forward(self, x):
        # Forward pass: apply linear transformation followed by sigmoid activation
        return torch.sigmoid(self.proj(x))


class LinearProbe(Model):
    
    def __str__(self):
        return 'LinearProbe'

    def make_model(self, **kwargs):
        mode = kwargs.get("mode", "latent")
        if mode == "latent":
            ax = LogisticRegressionModel(
                self.model.config.hidden_size, kwargs.get("low_rank_dimension", 1))
            ax.to(self.device)
            self.ax = ax
        elif mode == "steering":
            ax = AdditionIntervention(
                embed_dim=self.model.config.hidden_size, 
                low_rank_dimension=kwargs.get("low_rank_dimension", 1),
            )
            self.ax = ax
            self.ax.train()
            ax_config = IntervenableConfig(representations=[{
                "layer": l,
                "component": f"model.layers[{l}].output",
                "low_rank_dimension": kwargs.get("low_rank_dimension", 1),
                "intervention": self.ax} for l in [self.layer]])
            ax_model = IntervenableModel(ax_config, self.model)
            ax_model.set_device(self.device)
            self.ax_model = ax_model

    def make_dataloader(self, examples, **kwargs):
        data_module = make_data_module(self.tokenizer, self.model, examples)
        train_dataloader = DataLoader(
            data_module["train_dataset"], shuffle=True, batch_size=self.training_args.batch_size, 
            collate_fn=data_module["data_collator"])
        return train_dataloader
    
    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples)
        self.make_model(**kwargs)
        torch.cuda.empty_cache()
        self.ax.train()
        # Optimizer and lr
        optimizer = torch.optim.AdamW(self.ax.parameters(), lr=self.training_args.lr)
        num_training_steps = self.training_args.n_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer,
            num_warmup_steps=0, num_training_steps=num_training_steps)
        criterion = torch.nn.BCELoss()

        # Main training loop.
        progress_bar, curr_step = tqdm(range(num_training_steps)), 0
        for epoch in range(self.training_args.n_epochs):
            for batch in train_dataloader:
                # prepare input
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                activations = gather_residual_activations(
                    self.model, self.layer, 
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
                )
                nonbos_mask = inputs["attention_mask"][:,1:]
                activations = activations[:,1:][nonbos_mask.bool()]
                labels = inputs["labels"].unsqueeze(1).repeat(1, inputs["input_ids"].shape[1] - 1)
                labels = labels[nonbos_mask.bool()].unsqueeze(1).float()

                preds = self.ax(activations)
                loss = criterion(preds, labels)

                loss.backward()
                set_decoder_norm_to_unit_norm(self.ax)
                remove_gradient_parallel_to_decoder_directions(self.ax)
                curr_lr = get_lr(optimizer)
                curr_step += 1
                # optim
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_description(
                    "lr %.6f || loss %.6f" % (curr_lr, loss))
        progress_bar.close()
        logger.warning("Training finished.")


class L1LinearProbe(LinearProbe):
    
    def __str__(self):
        return 'L1LinearProbe'

    def train(self, examples, **kwargs):
        """with a L1 penalty on the activations"""
        train_dataloader = self.make_dataloader(examples)
        self.make_model(**kwargs)
        torch.cuda.empty_cache()
        self.ax.train()
        # Optimizer and lr
        optimizer = torch.optim.AdamW(self.ax.parameters(), lr=self.training_args.lr)
        num_training_steps = self.training_args.n_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer,
            num_warmup_steps=0, num_training_steps=num_training_steps)
        criterion = torch.nn.BCELoss()

        # Main training loop.
        progress_bar, curr_step = tqdm(range(num_training_steps)), 0
        for epoch in range(self.training_args.n_epochs):
            for batch in train_dataloader:
                # prepare input
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                nonbos_mask = inputs["attention_mask"][:,1:]
                activations = gather_residual_activations(
                    self.model, self.layer, 
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
                )
                latent = self.ax(activations).squeeze(-1)  # bs, seq
                loss = criterion(
                    latent[:,1:][nonbos_mask.bool()], 
                    inputs["labels"].unsqueeze(1).repeat(
                        1, inputs["input_ids"].shape[1] - 1)[nonbos_mask.bool()].float()
                )
                null_loss, l1_loss = calculate_l1_losses(
                    latent[:,1:],
                    labels=inputs["labels"],
                    mask=nonbos_mask,
                    k_latent_null_loss=self.training_args.k_latent_null_loss
                )

                coeff = curr_step/num_training_steps
                loss += coeff*self.training_args.coeff_l1_loss_null*null_loss + coeff*self.training_args.coeff_l1_loss*l1_loss

                # grads
                loss.backward()
                set_decoder_norm_to_unit_norm(self.ax)
                remove_gradient_parallel_to_decoder_directions(self.ax)
                curr_lr = get_lr(optimizer)
                curr_step += 1
                # optim
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_description(
                    "lr %.6f || loss %.6f || null l1 loss %.6f" % (curr_lr, loss, null_loss))
        progress_bar.close()
        logger.warning("Training finished.")
