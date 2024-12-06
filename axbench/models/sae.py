from .model import Model
from pyvene import (
    IntervenableConfig,
    IntervenableModel
)

import os, requests, torch
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from .interventions import (
    JumpReLUSAECollectIntervention, 
    AdditionIntervention,
    SubspaceIntervention,
    DictionaryAdditionIntervention, # please try this one
    SigmoidMaskAdditionIntervention,
)
from ..utils.model_utils import (
    remove_gradient_parallel_to_decoder_directions,
    gather_residual_activations, 
    get_lr,
    calculate_l1_losses
)
from huggingface_hub import hf_hub_download
from transformers import get_scheduler

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)

# using pyreft out-of-the-box
import pyreft


class GemmaScopeSAE(Model):
    def __str__(self):
        return 'GemmaScopeSAE'
    
    def make_model(self, **kwargs):
        mode = kwargs.get("mode", "latent")
        intervention_type = kwargs.get("intervention_type", "addition")
        if mode == "steering":
            if intervention_type == "addition":
                ax = AdditionIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            elif intervention_type == "clamping":
                ax = DictionaryAdditionIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            elif intervention_type == "sigmoid_mask":
                ax = SigmoidMaskAdditionIntervention(
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
        for sae_link in tqdm(sae_links):
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


class GemmaScopeSAEBinaryMask(Model):
    def __str__(self):
        return 'GemmaScopeSAE+DBM'

    def make_model(self, **kwargs):
        mode = kwargs.get("mode", "latent")
        if mode == "train":
            sae_params = kwargs.get("sae_params", None)
            if sae_params is not None:
                logger.warning(f"Setting up SAE for binary mask with shape {sae_params['W_dec'].shape}")
                ax = SigmoidMaskAdditionIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                    sae_width=sae_params['W_dec'].shape[0],
                )
                ax.proj.weight.data = torch.from_numpy(sae_params['W_dec']).t()
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
    
    def save(self, dump_dir, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        saved_masks = None
        saved_sources = None
        if os.path.exists(dump_dir / f"{model_name}_masks.pt"):
            saved_masks = torch.load(dump_dir / f"{model_name}_masks.pt")
        saved_masks = torch.cat([saved_masks, self.ax.mask.data], dim=0) if saved_masks is not None else self.ax.mask.data
        torch.save(saved_masks, dump_dir / f"{model_name}_masks.pt")
        if os.path.exists(dump_dir / f"{model_name}_sources.pt"):
            saved_sources = torch.load(dump_dir / f"{model_name}_sources.pt")
        saved_sources = torch.cat([saved_sources, self.ax.source.data], dim=0) if saved_sources is not None else self.ax.source.data
        torch.save(saved_sources, dump_dir / f"{model_name}_sources.pt")

    def load(self, dump_dir, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        self.ax.mask.data = torch.load(dump_dir / f"{model_name}_masks.pt")
        self.ax.source.data = torch.load(dump_dir / f"{model_name}_sources.pt")
    
    def train(self, examples,**kwargs):
        train_dataloader = self.make_dataloader(examples, **kwargs)
        torch.cuda.empty_cache()

        # Optimizer and lr
        optimizer = torch.optim.AdamW(
            self.ax_model.parameters(), 
            lr=self.training_args.lr, weight_decay=self.training_args.weight_decay)
        num_training_steps = self.training_args.n_epochs * (len(train_dataloader) // self.training_args.gradient_accumulation_steps)
        temperature_start = 1e-2
        temperature_end = 1e-7
        temperature_schedule = (
            torch.linspace(
                temperature_start, temperature_end, num_training_steps
            ).to(torch.bfloat16).to("cuda")
        )
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer,
            num_warmup_steps=0, num_training_steps=num_training_steps)
        norm_loss_fn = torch.nn.MSELoss()
        # Main training loop.
        rank = torch.distributed.get_rank()
        progress_bar, curr_step = tqdm(range(num_training_steps), position=rank, leave=True), 0
        
        for epoch in range(self.training_args.n_epochs):
            for step, batch in enumerate(train_dataloader):
                # prepare input
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                unit_locations={"sources->base": (
                    None,
                    inputs["intervention_locations"].permute(1, 0, 2).tolist()
                )}
        
                # forward
                _, cf_outputs = self.ax_model(
                    base={
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"]
                    }, unit_locations=unit_locations, labels=inputs["labels"],
                    use_cache=False, subspaces=[{"idx": 0}]*self.num_of_layers)

                # loss
                loss = cf_outputs.loss.mean()
                loss /= self.training_args.gradient_accumulation_steps
                # grads
                loss.backward()
                
                # Perform optimization step every gradient_accumulation_steps
                if (step + 1) % self.training_args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    temp = temperature_schedule[curr_step] if len(temperature_schedule) > curr_step else temperature_end
                    self.ax_model.set_temperature(temp)
                    torch.nn.utils.clip_grad_norm_(self.ax_model.parameters(), 1.0)
                    curr_step += 1
                    curr_lr = get_lr(optimizer)
                    # optim
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        lr=f"{curr_lr:.6f}", loss=f"{loss:.6f}", temp=f"{temp:.6f}")
        progress_bar.close()
