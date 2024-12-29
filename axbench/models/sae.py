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

from torch.utils.data import DataLoader
from .probe import DataCollator, make_data_module

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)

# using pyreft out-of-the-box
import pyreft


def load_metadata_flatten(metadata_path):
    """
    Load flatten metadata from a JSON lines file.
    """
    metadata = []
    concept_id = 0
    with open(metadata_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            concept, ref =data["concept"], data["ref"]
            concept_genres_map = data["concept_genres_map"][concept]
            ref = data["ref"]
            flatten_data = {
                "concept": concept,
                "ref": ref,
                "concept_genres_map": {concept: concept_genres_map},
                "concept_id": concept_id
            }
            metadata += [flatten_data]  # Return the metadata as is
            concept_id += 1
    return metadata


def save_pruned_sae(
    metadata_path, dump_dir, modified_refs=None, savefile="GemmaScopeSAE.pt", sae_params=None
):
    # Save SAE weights and biases for inference
    logger.warning("Saving SAE weights and biases for inference")
    flatten_metadata = load_metadata_flatten(metadata_path)
    logger.warning(flatten_metadata)

    # mess with SAE refs if needed (for DBM stuff)
    if modified_refs is not None:
        for i, ref in enumerate(modified_refs):
            flatten_metadata[i]["ref"] = "/".join(flatten_metadata[i]["ref"].split("/")[:-1]) + "/" + str(ref)

    # Save pruned SAE weights and biases
    sae_path = flatten_metadata[0]["ref"].split("https://www.neuronpedia.org/")[-1]
    sae_url = f"https://www.neuronpedia.org/api/feature/{sae_path}"
    headers = {"X-Api-Key": os.environ.get("NP_API_KEY")}
    response = requests.get(sae_url, headers=headers).json()
    hf_repo = response["source"]["hfRepoId"]
    hf_folder = response["source"]["hfFolderId"]
    if sae_params is None:
        path_to_params = hf_hub_download(
            repo_id=hf_repo,
            filename=f"{hf_folder}/params.npz",
            force_download=False,
        )
        sae_params = np.load(path_to_params)
    sae_pt_params = {k: torch.from_numpy(v) for k, v in sae_params.items()}
    pruned_sae_pt_params = {
        "b_dec": sae_pt_params["b_dec"],
        "W_dec": [],
        "W_enc": [],
        "b_enc": [],
        "threshold": []
    }
    for concept_id, m in enumerate(flatten_metadata):
        sae_id = int(m["ref"].split("/")[-1])
        pruned_sae_pt_params["W_dec"].append(sae_pt_params["W_dec"][[sae_id], :])
        pruned_sae_pt_params["W_enc"].append(sae_pt_params["W_enc"][:, [sae_id]])
        pruned_sae_pt_params["b_enc"].append(sae_pt_params["b_enc"][[sae_id]])
        pruned_sae_pt_params["threshold"].append(sae_pt_params["threshold"][[sae_id]])
    for k, v in pruned_sae_pt_params.items():
        if k == "b_dec":
            continue
        if k == "W_enc":
            pruned_sae_pt_params[k] = torch.cat(v, dim=1)
        else:
            pruned_sae_pt_params[k] = torch.cat(v, dim=0)
    torch.save(pruned_sae_pt_params, dump_dir / savefile) # sae only has one file
    return sae_params


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
            else:
                raise ValueError(f"Invalid intervention type for steering: {intervention_type}")
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
        logger.warning(f"Loading SAE from {dump_dir}/{model_name}.pt")
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


class GemmaScopeSAEMaxDiff(GemmaScopeSAE):
    """
    pick the SAE feature with the largest difference in activation between the two classes
    """
    def __str__(self):
        return 'GemmaScopeSAEMaxDiff'

    def make_model(self, **kwargs):
        if kwargs.get("mode", "latent") == "train":
            # load the entire SAE
            self.sae_params = kwargs.get("sae_params", None)
            metadata_path = kwargs.get("metadata_path", None)
            self.sae_width = self.sae_params['W_dec'].shape[0]
            self.metadata_path = metadata_path
            kwargs["low_rank_dimension"] = self.sae_width
            sae_pt_params = {k: torch.from_numpy(v) for k, v in self.sae_params.items()}
            super().make_model(**kwargs)
            self.ax.load_state_dict(sae_pt_params, strict=False)
            self.ax.eval()
            self.ax.to(self.device)
        else:
            super().make_model(**kwargs)
    
    def make_dataloader(self, examples, **kwargs):
        data_module = make_data_module(self.tokenizer, self.model, examples)
        train_dataloader = DataLoader(
            data_module["train_dataset"], shuffle=True, batch_size=self.training_args.batch_size, 
            collate_fn=data_module["data_collator"])
        return train_dataloader
    
    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples)
        torch.cuda.empty_cache()
        prefix_length = kwargs.get("prefix_length", 1)

        sum_positive_acts = torch.zeros(self.sae_width).to(self.device)
        sum_negative_acts = torch.zeros(self.sae_width).to(self.device)
        positive_count = 0
        negative_count = 0
        
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
                    self.model, self.layer, inputs)
                ax_acts_batch = self.ax(act_in[:, prefix_length:])  # no bos token
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

        # get latent activations
        positive_acts = sum_positive_acts / positive_count
        negative_acts = sum_negative_acts / negative_count
        max_diff = (positive_acts - negative_acts).argmax().item()
        self.top_feature = max_diff

        # print top 10 features
        top = (positive_acts - negative_acts).topk(10)
        for i in range(10):
            print(f"Feature {top.indices[i].item()}: {top.values[i].item()}")
    
    def save(self, dump_dir, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        top_feature = self.top_feature
        top_features = []
        if os.path.exists(dump_dir / f"{model_name}_top_features.json"):
            with open(dump_dir / f"{model_name}_top_features.json", "r") as f:
                top_features = json.load(f)
        top_features += [top_feature]   
        with open(dump_dir / f"{model_name}_top_features.json", "w") as f:
            json.dump(top_features, f)

        # save the pruned SAE
        save_pruned_sae(
            self.metadata_path, dump_dir, modified_refs=top_features, savefile="GemmaScopeSAEMaxDiff.pt", sae_params=self.sae_params
        )

class GemmaScopeSAEBinaryMask(GemmaScopeSAE):
    """
    basic idea:
    - in "train" mode, learn a DBM on top of SAE
    - use this to identify the most important SAE feature for inference stuff
    - save the decoder vector for this feature
    - in inference modes, its a normal SAE intervention but with the decoder vector fixed to the one identified in training
    - unclear how else to *cheaply* identify good SAE features for steering
    """
    def __str__(self):
        return 'GemmaScopeSAEBinaryMask'

    def make_model(self, **kwargs):
        mode = kwargs.get("mode", "latent")
        if mode == "train":
            self.sae_params = kwargs.get("sae_params", None)
            metadata_path = kwargs.get("metadata_path", None)
            if self.sae_params is not None:
                logger.warning(f"Setting up SAE for binary mask with shape {self.sae_params['W_dec'].shape}")
                ax = SigmoidMaskAdditionIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                    sae_width=self.sae_params['W_dec'].shape[0],
                )
                ax.proj.weight.data = torch.from_numpy(self.sae_params['W_dec']).t()
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
                self.metadata_path = metadata_path
        else:
            # defer to parent class
            super().make_model(**kwargs)
    
    def save(self, dump_dir, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        top_feature = self.ax.get_latent_weights().argmax().item()
        # log the top 10 features
        logger.warning(f"Top 10 features: {self.ax.get_latent_weights().topk(10).indices.tolist()}")
        top_features = []
        if os.path.exists(dump_dir / f"{model_name}_top_features.json"):
            with open(dump_dir / f"{model_name}_top_features.json", "r") as f:
                top_features = json.load(f)
        top_features += [top_feature]   
        with open(dump_dir / f"{model_name}_top_features.json", "w") as f:
            json.dump(top_features, f)

        # save the pruned SAE
        save_pruned_sae(
            self.metadata_path, dump_dir, modified_refs=top_features, savefile="GemmaScopeSAEBinaryMask.pt", sae_params=self.sae_params
        )
    
    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples, **kwargs)
        torch.cuda.empty_cache()

        # Optimizer and lr
        optimizer = torch.optim.AdamW(
            self.ax_model.parameters(), 
            lr=self.training_args.lr, weight_decay=self.training_args.weight_decay)
        num_training_steps = self.training_args.n_epochs * (len(train_dataloader) // self.training_args.gradient_accumulation_steps)
        temperature_start = float(self.training_args.temperature_start)
        temperature_end = float(self.training_args.temperature_end)
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
