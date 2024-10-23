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

import os, requests, torch
import numpy as np
import pyreft
from pyreax import (
    JumpReLUSAECollectIntervention, 
)
from huggingface_hub import hf_hub_download


class GemmaScopeSAE(Model):

    def __init__(self, model, tokenizer, layer, training_args=None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        # abstracting layer
        self.layer = layer
        self.training_args = training_args

    def __str__(self):
        return 'GemmaScopeSAE'
    
    def make_model(self, **kwargs):
        sae_intervention = JumpReLUSAECollectIntervention(
            embed_dim=self.model.config.hidden_size, 
            low_rank_dimension=kwargs.get("low_rank_dimension", 2),
        )
        sae_intervention = sae_intervention.train()
        reft_config = pyreft.ReftConfig(representations=[{
            "layer": l,
            "component": f"model.layers[{l}].output",
            "low_rank_dimension": kwargs.get("low_rank_dimension", 2),
            "intervention": sae_intervention} for l in [self.layer]])
        reft_model = pyreft.get_reft_model(self.model, reft_config)
        reft_model.set_device("cuda")
        reft_model.print_trainable_parameters()
        self.sae_intervention = sae_intervention
        self.reft_model = reft_model

    def make_dataloader(self, examples, **kwargs):
        pass
    
    def train(self, examples, **kwargs):
        pass
        
    def save(self, dump_dir, **kwargs):
        pass

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
        self.make_model(low_rank_dimension=params['W_enc'].shape[1])
        self.sae_intervention.load_state_dict(pt_params, strict=False)
    
    def predict_latent(self, examples, **kwargs):
        all_acts = []
        all_max_act = []
        for _, row in examples.iterrows():
            inputs = self.tokenizer.encode(
                row["input"], return_tensors="pt", add_special_tokens=True).to("cuda")
            sae_acts = self.reft_model.forward(
                {"input_ids": inputs}, return_dict=True
            ).collected_activations[0][1:, row["sae_id"]].data.cpu().numpy().tolist() # no bos token
            sae_acts = [round(x, 3) for x in sae_acts]
            max_sae_act = max(sae_acts)

            all_acts += [sae_acts]
            all_max_act += [max_sae_act]
        return {
            "acts": all_acts,
            "max_act": all_max_act}

    def predict_steer(self, examples, **kwargs):
        pass

