from .model import Model
from pyvene import (
    IntervenableConfig,
    IntervenableModel
)

import os, requests, torch
import json
import tqdm
import numpy as np
import pandas as pd
from .interventions import (
    JumpReLUSAECollectIntervention, 
    AdditionIntervention,
    SubspaceIntervention,
    DictionaryAdditionIntervention # please try this one
)
from ..utils.model_utils import (
    gather_residual_activations, 
)
from huggingface_hub import hf_hub_download

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


class GemmaScopeSAE(Model):
    def __str__(self):
        return 'GemmaScopeSAE'
    
    def make_model(self, **kwargs):
        mode = kwargs.get("mode", "latent")
        intervention_type = kwargs.get("intervention_type", "addition")
        if mode == "steering":
            # if intervention_type == "addition":
            #     ax = AdditionIntervention(
            #         embed_dim=self.model.config.hidden_size, 
            #         low_rank_dimension=kwargs.get("low_rank_dimension", 1),
            #     )
            # elif intervention_type == "clamping":

            # we default to clamping for SAE following Anthropic's implementation
            ax = DictionaryAdditionIntervention(
                embed_dim=self.model.config.hidden_size, 
                low_rank_dimension=kwargs.get("low_rank_dimension", 1),
            )
        else:
            ax = JumpReLUSAECollectIntervention(
                embed_dim=self.model.config.hidden_size, 
                low_rank_dimension=kwargs.get("low_rank_dimension", 1),
            )
        ax = ax.train()
        ax_config = IntervenableConfig(representations=[{
            "layer": l,
            "component": f"model.layers[{l}].output",
            "low_rank_dimension": kwargs.get("low_rank_dimension", 1),
            "intervention": ax} for l in [self.layer]])
        ax_model = IntervenableModel(ax_config, self.model)
        ax_model.set_device(self.device)
        self.ax = ax
        self.ax_model = ax_model

    def load(self, dump_dir=None, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        params = torch.load(
            f"{dump_dir}/{model_name}.pt"
        )
        pt_params = {k: v.to(self.device) for k, v in params.items()}
        self.make_model(low_rank_dimension=params['W_enc'].shape[1], **kwargs)
        if isinstance(self.ax, SubspaceIntervention) or isinstance(self.ax, AdditionIntervention):
            self.ax.proj.weight.data = pt_params['W_dec']
        else:
            try:
                self.ax.load_state_dict(pt_params, strict=True)
            except Exception as e:
                # let it passing
                logger.warning(f"Error loading state dict: {e}")
                self.ax.load_state_dict(pt_params, strict=False)

    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        # Loop over all praqut files in dump_dir.
        sae_links = []
        for file in os.listdir(dump_dir):
            if file.endswith(".parquet") and file.startswith("latent_data"):
                df = pd.read_parquet(os.path.join(dump_dir, file))
                # sort by concept_id from small to large and enumerate through all concept_ids.
                for sae_link in sorted(df["sae_link"].unique()):
                    sae_links += [sae_link]

        model_name, sae_name = sae_links[0].split("/")[-3], sae_links[0].split("/")[-2]
        max_activations = {} # sae_id to max_activation

        # Load existing max activations file and skip if exists.
        max_activations_file = os.path.join(
            kwargs.get("master_data_dir", "axbench/data"), 
            f"{model_name}_{sae_name}_max_activations.json")
        if os.path.exists(max_activations_file):
            with open(max_activations_file, "r") as f:
                max_activations = json.load(f)
            max_activations = {int(k): v for k, v in max_activations.items()}
        
        has_new = False
        for sae_link in tqdm.tqdm(sae_links):
            sae_path = sae_link.split("https://www.neuronpedia.org/")[-1]
            sae_id = int(sae_link.split("/")[-1])
            if sae_id in max_activations:
                continue
            url = f"https://www.neuronpedia.org/api/feature/{sae_path}"
            headers = {"X-Api-Key": os.environ["NP_API_KEY"]}
            response = requests.get(url, headers=headers)
            max_activation = response.json()["activations"][0]["maxValue"]
            max_activations[sae_id] = max_activation if max_activation > 0 else 50
            has_new = True

        if has_new:
            with open(max_activations_file, "w") as f:
                json.dump(max_activations, f)

        self.max_activations = max_activations
        return max_activations