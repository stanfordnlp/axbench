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
import numpy as np
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
        self.ax.load_state_dict(pt_params, strict=False)
    
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

    def predict_steer(self, examples, **kwargs):
        self.ax.eval()
        # set tokenizer padding to left
        self.tokenizer.padding_side = "left"

        # iterate rows in batch
        batch_size = kwargs.get("batch_size", 16)
        all_generations = []
        for i in range(0, len(examples), batch_size):
            batch_examples = examples.iloc[i:i+batch_size]
            input_strings = batch_examples['input'].tolist()
            mag = torch.tensor(batch_examples['factor'].tolist()).to("cuda")
            idx = torch.tensor(batch_examples['sae_id'].tolist()).to("cuda")
            # tokenize input_strings
            inputs = self.tokenizer(
                input_strings, return_tensors="pt", padding=True, truncation=True
            ).to("cuda")
            _, generations = self.ax_model.generate(
                inputs, 
                unit_locations=None, intervene_on_prompt=True, 
                subspaces=[{"idx": idx, "mag": mag}],
                max_new_tokens=128, do_sample=True, 
                early_stopping=True, 
                temperature=0.7, 
            )
            # Decode and print only the generated text without prompt tokens
            input_lengths = [len(input_ids) for input_ids in inputs.input_ids]
            generated_texts = [
                self.tokenizer.decode(generation[input_length:], skip_special_tokens=True)
                for generation, input_length in zip(generations, input_lengths)
            ]
            all_generations += generated_texts
        return {
            "steered_generation": all_generations,
        }