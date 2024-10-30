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
    SubspaceAdditionIntervention
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


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, low_rank_dimension):
        super(LogisticRegressionModel, self).__init__()
        # Linear layer: input_dim -> 1 output (since binary classification)
        self.proj = torch.nn.Linear(input_dim, low_rank_dimension)
    
    def forward(self, x):
        return self.proj(x)


class Random(Model):
    
    def __str__(self):
        return 'Random'

    def make_model(self, **kwargs):
        mode = kwargs.get("mode", "latent")
        if mode == "latent":
            ax = LogisticRegressionModel(
                self.model.config.hidden_size, kwargs.get("low_rank_dimension", 1))
            ax.to("cuda")
            self.ax = ax
        elif mode == "steering":
            ax = SubspaceAdditionIntervention(
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
            ax_model.set_device("cuda")
            self.ax_model = ax_model
    
    def train(self, examples, **kwargs):
        self.make_model(**kwargs)
        torch.cuda.empty_cache()
        set_decoder_norm_to_unit_norm(self.ax)
        logger.warning("Dummy training finished :) I'm a random baseline.")

    @torch.no_grad()
    def predict_latent(self, examples, **kwargs):
        self.ax.eval()

        all_acts = []
        all_max_act = []
        all_max_act_idx = []
        all_max_token = []
        for _, row in examples.iterrows():
            inputs = self.tokenizer.encode(
                row["input"], return_tensors="pt", add_special_tokens=True).to("cuda")
            act_in = gather_residual_activations(
                self.model, self.layer, {"input_ids": inputs})
            ax_acts = self.ax(act_in[:,1:])
            ax_acts = ax_acts[..., row["concept_id"]]
            ax_acts = ax_acts.flatten().data.cpu().numpy().tolist()
            ax_acts = [round(x, 3) for x in ax_acts]
            max_ax_act = max(ax_acts)
            max_ax_act_idx = ax_acts.index(max_ax_act)
            max_token = self.tokenizer.tokenize(row["input"])[max_ax_act_idx]

            all_acts += [ax_acts]
            all_max_act += [max_ax_act] 
            all_max_act_idx += [max_ax_act_idx]
            all_max_token += [max_token]
        return {
            "acts": all_acts,
            "max_act": all_max_act, 
            "max_act_idx": all_max_act_idx,
            "max_token": all_max_token}

    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        # For ReAX, we need to look into the concept in the same group, since they are used in training.
        max_activations = {} # sae_id to max_activation
        # Loop over saved latent files in dump_dir.
        for file in os.listdir(dump_dir):
            if file.startswith("latent_") and file.endswith(".csv"):
                latent_path = os.path.join(dump_dir, file)
                latent = pd.read_csv(latent_path)
                # loop through unique sorted concept_id
                for concept_id in sorted(latent["concept_id"].unique()):
                    concept_latent = latent[latent["concept_id"] == concept_id]
                    # group id if this concept
                    group_id = concept_latent["group_id"].iloc[0]
                    # get the mean activation of this group but not with this concept_id
                    group_latent = latent[latent["group_id"] == group_id]
                    group_latent = group_latent[group_latent["concept_id"] != concept_id]
                    # load ReAX as the approximation of the max activation for probes
                    max_act = group_latent["ReAX_max_act"].max()
                    max_activations[concept_id] = max_act
        self.max_activations = max_activations
        return max_activations  