from .model import Model
import torch, transformers, datasets
from tqdm.auto import tqdm
import os
import pandas as pd
from pyvene import (
    IntervenableConfig,
    IntervenableModel
)
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union, List, Any
from torch.utils.data import DataLoader
from .interventions import (
    AdditionIntervention,
    SubspaceIntervention,
    JumpReLUSAECollectIntervention
)
from ..utils.model_utils import (
    set_decoder_norm_to_unit_norm, 
    remove_gradient_parallel_to_decoder_directions,
    gather_residual_activations, 
    get_lr
)
from ..utils.model_utils import calculate_l1_losses
from transformers import get_scheduler
import sklearn.decomposition
import numpy as np

from .probe import DataCollator, make_data_module

import logging
import random
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, low_rank_dimension):
        super(LogisticRegressionModel, self).__init__()
        # Linear layer: input_dim -> 1 output (since binary classification)
        self.proj = torch.nn.Linear(input_dim, low_rank_dimension)
        with torch.no_grad():
            self.proj.bias.fill_(0)

    def forward(self, x):
        return self.proj(x)


class MeanEmbedding(Model):
    def __str__(self):
        return 'MeanEmbedding'

    def make_model(self, **kwargs):
        mode = kwargs.get("mode", "train")
        intervention_type = kwargs.get("intervention_type", "addition")
        if mode == "steering":
            if intervention_type == "addition":
                ax = AdditionIntervention(
                embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            elif intervention_type == "clamping":
                ax = SubspaceIntervention(
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
        else:
            ax = LogisticRegressionModel(
                self.model.config.hidden_size, kwargs.get("low_rank_dimension", 1))
            ax.to(self.device)
            self.ax = ax
    
    def make_dataloader(self, examples, **kwargs):
        data_module = make_data_module(self.tokenizer, self.model, examples)
        train_dataloader = DataLoader(
            data_module["train_dataset"], shuffle=True, batch_size=self.training_args.batch_size, 
            collate_fn=data_module["data_collator"])
        return train_dataloader

    def train(self, examples, **kwargs):
        torch.cuda.empty_cache()
        # set the decoder weights to be the mean of the embeddings
        W_U = self.model.lm_head.weight.mean(dim=0).detach().clone().unsqueeze(0)
        self.ax.proj.weight.data = W_U.data
        set_decoder_norm_to_unit_norm(self.ax)
        logger.warning("Training finished.")


class MeanActivation(MeanEmbedding):
    """take the mean of all activations"""
    def __str__(self):
        return 'MeanActivation'

    @torch.no_grad()
    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples)
        torch.cuda.empty_cache()
        self.ax.eval()
        # Main training loop.
        all_activations = []
        num_training_steps = self.training_args.n_epochs * len(train_dataloader)
        for epoch in range(self.training_args.n_epochs):
            for batch in train_dataloader:
                # prepare input
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                activations = gather_residual_activations(
                    self.model, self.layer, 
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
                ).detach()
                nonbos_mask = inputs["attention_mask"][:,kwargs["prefix_length"]:]
                activations = activations[:,kwargs["prefix_length"]:][nonbos_mask.bool()]
                all_activations.append(activations)
        all_activations = torch.cat(all_activations, dim=0)
        mean_activation = all_activations.mean(dim=0)
        self.ax.proj.weight.data = mean_activation.unsqueeze(0)
        set_decoder_norm_to_unit_norm(self.ax)
        logger.warning("Training finished.")


class DiffMean(MeanActivation):
    """
    difference in means of positive and negative classes
    - https://arxiv.org/abs/2310.06824
    - https://blog.eleuther.ai/diff-in-means/
    """
    
    def __str__(self):
        return 'DiffMean'

    @torch.no_grad()
    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples)
        torch.cuda.empty_cache()
        self.ax.eval()
        self.ax.to(self.device)
        # Main training loop.
        positive_activations = []
        negative_activations = []
        for _ in range(self.training_args.n_epochs):
            for batch in train_dataloader:
                # prepare input
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                activations = gather_residual_activations(
                    self.model, self.layer, 
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
                ).detach()
                nonbos_mask = inputs["attention_mask"][:,kwargs["prefix_length"]:]
                activations = activations[:,kwargs["prefix_length"]:][nonbos_mask.bool()]
                labels = inputs["labels"].unsqueeze(1).repeat(
                    1, inputs["input_ids"].shape[1] - kwargs["prefix_length"])
                positive_activations.append(activations[labels[nonbos_mask.bool()] == 1])
                negative_activations.append(activations[labels[nonbos_mask.bool()] != 1])

        mean_positive_activation = torch.cat(positive_activations, dim=0).mean(dim=0)
        mean_negative_activation = torch.cat(negative_activations, dim=0).mean(dim=0)
        self.ax.proj.weight.data = mean_positive_activation.unsqueeze(0) - mean_negative_activation.unsqueeze(0)
        set_decoder_norm_to_unit_norm(self.ax)
        logger.warning("Training finished.")


class PCA(MeanActivation):
    
    def __str__(self):
        return 'PCA'

    @torch.no_grad()
    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples)
        torch.cuda.empty_cache()
        self.ax.eval()
        self.ax.to(self.device)
        # Main training loop.
        all_activations = []
        
        for _ in range(self.training_args.n_epochs):
            for batch in train_dataloader:
                # prepare input
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                activations = gather_residual_activations(
                    self.model, self.layer, 
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
                ).detach()
                nonbos_mask = inputs["attention_mask"][:,kwargs["prefix_length"]:]
                activations = activations[:,kwargs["prefix_length"]:][nonbos_mask.bool()]
                labels = inputs["labels"].unsqueeze(1).repeat(1, inputs["input_ids"].shape[1] - kwargs["prefix_length"])
                label_mask = labels[nonbos_mask.bool()] == 1 # only positive examples
                all_activations.append(activations[label_mask].detach().cpu().float().numpy())

        all_activations = np.concatenate(all_activations)
        pca = sklearn.decomposition.PCA(n_components=2)
        pca.fit(all_activations)
        variance = pca.explained_variance_ratio_[0]
        logger.warning(f"PCA explains {variance:.5%} of the variance")
        first_principal_component = torch.tensor(pca.components_[0])
        self.ax.proj.weight.data = first_principal_component.unsqueeze(0)
        set_decoder_norm_to_unit_norm(self.ax)
        logger.warning("Training finished.")


class LAT(MeanActivation):
    """
    LAT is just PCA over normed differences of random pairs of activations
    - https://arxiv.org/abs/2310.01405
    """
    
    def __str__(self):
        return 'LAT'

    @torch.no_grad()
    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples)
        torch.cuda.empty_cache()
        self.ax.eval()
        self.ax.to(self.device)
        # Main training loop.
        all_activations = []
        
        for _ in range(self.training_args.n_epochs):
            for batch in train_dataloader:
                # prepare input
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                activations = gather_residual_activations(
                    self.model, self.layer, 
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
                ).detach()
                nonbos_mask = inputs["attention_mask"][:,kwargs["prefix_length"]:]
                activations = activations[:,kwargs["prefix_length"]:][nonbos_mask.bool()]
                labels = inputs["labels"].unsqueeze(1).repeat(1, inputs["input_ids"].shape[1] - kwargs["prefix_length"])
                label_mask = labels[nonbos_mask.bool()] == 1 # only positive examples
                all_activations.append(activations[label_mask].detach().cpu().float().numpy())

        # shuffle and take diffs of random pairs
        all_activations = np.concatenate(all_activations)
        logger.warning(f"Shuffling {all_activations.shape[0]} activations")
        np.random.shuffle(all_activations)
        length = all_activations.shape[0] // 2
        all_activations = all_activations[:length] - all_activations[length:length * 2]
        logger.warning(f"Shuffled and diff'd:  {all_activations.shape[0]} ")
        logger.warning(f"Potential NaNs: {np.isnan(all_activations).sum()}")
        logger.warning(f"Potential Infs: {np.isinf(all_activations).sum()}")
        logger.warning(f"Range: {all_activations.min()} to {all_activations.max()}")

        # normalize the diffs, avoiding division by zero
        norms = np.linalg.norm(all_activations, axis=1, keepdims=True)
        all_activations = np.where(norms == 0, 0, all_activations / norms)

        # fit PCA on the diffs
        pca = sklearn.decomposition.PCA(n_components=2)
        pca.fit(all_activations)
        variance = pca.explained_variance_ratio_[0]
        logger.warning(f"LAT explains {variance:.5%} of the variance")
        first_principal_component = torch.tensor(pca.components_[0])
        self.ax.proj.weight.data = first_principal_component.unsqueeze(0)
        set_decoder_norm_to_unit_norm(self.ax)
        logger.warning("Training finished.")


class GemmaScopeSAEDiffMean(MeanActivation):
    """
    SAE mean difference
    """
    def __str__(self):
        return 'GemmaScopeSAEDiffMean'

    def make_model(self, **kwargs):
        if kwargs.get("mode", "latent") == "train":
            # load the entire SAE
            self.sae_params = kwargs.get("sae_params", None)
            self.metadata_path = kwargs.get("metadata_path", None)
            self.sae_width = self.sae_params['W_dec'].shape[0]
            self.sae = JumpReLUSAECollectIntervention(
                embed_dim=self.model.config.hidden_size, 
                low_rank_dimension=self.sae_width,
            )
            sae_pt_params = {k: torch.from_numpy(v) for k, v in self.sae_params.items()}
            self.sae.load_state_dict(sae_pt_params, strict=False)
            self.sae.eval()
            self.sae.to(self.device)
        super().make_model(**kwargs)
    
    @torch.inference_mode()
    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples)
        torch.cuda.empty_cache()
        prefix_length = kwargs.get("prefix_length", 1)

        sum_positive_acts = torch.zeros(self.sae_width).to(self.device)
        sum_negative_acts = torch.zeros(self.sae_width).to(self.device)
        positive_count = 0
        negative_count = 0
        num_training_steps = self.training_args.n_epochs * len(train_dataloader)
        rank = torch.distributed.get_rank()
        progress_bar, curr_step = tqdm(range(num_training_steps), position=rank, leave=True), 0
        
        for epoch in range(self.training_args.n_epochs):
            for step, batch in enumerate(train_dataloader):
                # prepare input
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                inputs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                }

                # get SAE latents
                act_in = gather_residual_activations(
                    self.model, self.layer, inputs).to(dtype=torch.float32)
                ax_acts_batch = self.sae(act_in[:, prefix_length:]) # no bos token
                seq_lens = inputs["attention_mask"].sum(dim=1) - prefix_length # no bos token

                # add avg latents for each sequence
                for i in range(len(batch)):
                    acts = ax_acts_batch[i, :seq_lens[i]]
                    label = batch["labels"][i]
                    avg_acts = acts.mean(dim=0)
                    if label == 1:
                        sum_positive_acts += avg_acts
                        positive_count += 1
                    else:
                        sum_negative_acts += avg_acts
                        negative_count += 1
                
                del ax_acts_batch
                del act_in
                torch.cuda.empty_cache()
                progress_bar.update(1)
        progress_bar.close()

        # get latent activations
        mean_positive_activation = sum_positive_acts / positive_count
        mean_negative_activation = sum_negative_acts / negative_count
        mean_diff = (mean_positive_activation - mean_negative_activation) @ self.sae.W_dec
        self.ax.proj.weight.data = mean_diff.unsqueeze(0)
        self.ax.proj.bias.data = torch.zeros(1)
        set_decoder_norm_to_unit_norm(self.ax)

        # print top 10 features
        top = (mean_positive_activation - mean_negative_activation).topk(10)
        for i in range(10):
            logger.warning(f"Feature {top.indices[i].item()}: {top.values[i].item()}")
        logger.warning("Training finished.")