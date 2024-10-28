from .model import Model

try:
    # This library is our indicator that the required installs
    # need to be done.
    import pyreax

except ModuleNotFoundError:
    # relative import; better to pip install subctrl
    import sys
    sys.path.append("../../../pyreax")
    import pyreax
from pyvene import (
    IntervenableConfig,
    IntervenableModel
)

import os, requests, torch
import json
import tqdm
import numpy as np
import pandas as pd
from pyreax import (
    JumpReLUSAECollectIntervention, 
    SubspaceAdditionIntervention
)
from huggingface_hub import hf_hub_download


class GemmaScopeSAE(Model):
    def __str__(self):
        return 'GemmaScopeSAE'
    
    def make_model(self, **kwargs):
        mode = kwargs.get("mode", "latent")
        if mode == "latent":
            ax = JumpReLUSAECollectIntervention(
            embed_dim=self.model.config.hidden_size, 
                low_rank_dimension=kwargs.get("low_rank_dimension", 2),
            )
        elif mode == "steering":
            ax = SubspaceAdditionIntervention(
                embed_dim=self.model.config.hidden_size, 
                low_rank_dimension=kwargs.get("low_rank_dimension", 2),
            )
        ax = ax.train()
        ax_config = IntervenableConfig(representations=[{
            "layer": l,
            "component": f"model.layers[{l}].output",
            "low_rank_dimension": kwargs.get("low_rank_dimension", 2),
            "intervention": ax} for l in [self.layer]])
        ax_model = IntervenableModel(ax_config, self.model)
        ax_model.set_device("cuda")
        self.ax = ax
        self.ax_model = ax_model

    def load(self, dump_dir=None, **kwargs):
        sae_path = kwargs["sae_path"].split("https://www.neuronpedia.org/")[-1]
        sae_url = f"https://www.neuronpedia.org/api/feature/{sae_path}"
        
        headers = {"X-Api-Key": os.environ.get("NP_API_KEY")}
        response = requests.get(sae_url, headers=headers).json()
        hf_repo = response["source"]["hfRepoId"]
        hf_folder = response["source"]["hfFolderId"]
        path_to_params = hf_hub_download(
            repo_id=hf_repo,
            filename=f"{hf_folder}/params.npz",
            force_download=False,
        )
        params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}
        self.make_model(low_rank_dimension=params['W_enc'].shape[1], **kwargs)
        if isinstance(self.ax, JumpReLUSAECollectIntervention):
            self.ax.load_state_dict(pt_params, strict=False)
        else:
            self.ax.proj.weight.data = pt_params['W_dec']
    
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
            ax_acts = self.ax_model.forward(
                {"input_ids": inputs}, return_dict=True
            ).collected_activations[0][1:, row["sae_id"]].data.cpu().numpy().tolist() # no bos token
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

        # Loop over all csv files in dump_dir.
        sae_links = []
        for file in os.listdir(dump_dir):
            if file.endswith(".csv") and file.startswith("latent_data_fragment"):
                df = pd.read_csv(os.path.join(dump_dir, file))
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
            max_activations[sae_id] = max_activation
            has_new = True

        if has_new:
            with open(max_activations_file, "w") as f:
                json.dump(max_activations, f)

        self.max_activations = max_activations
        return max_activations