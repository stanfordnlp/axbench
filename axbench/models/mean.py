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
        return self.proj(x)


class MeanEmbedding(Model):
    
    def __str__(self):
        return 'MeanEmbedding'

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
        # set the decoder weights to be the mean of the embeddings
        W_U = self.model.lm_head.weight.mean(dim=0).detach().clone().unsqueeze(0)
        self.ax.proj.weight.data = W_U.data
        set_decoder_norm_to_unit_norm(self.ax)
        logger.warning("Training finished.")

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
    

class MeanActivation(MeanEmbedding):
    
    def __str__(self):
        return 'MeanActivation'

    def make_dataloader(self, examples, **kwargs):
        data_module = make_data_module(self.tokenizer, self.model, examples)
        train_dataloader = DataLoader(
            data_module["train_dataset"], shuffle=True, batch_size=self.training_args.batch_size, 
            collate_fn=data_module["data_collator"])
        return train_dataloader

    @torch.no_grad()
    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples)
        self.make_model(**kwargs)
        torch.cuda.empty_cache()
        self.ax.eval()
        # Main training loop.
        all_activations = []
        num_training_steps = self.training_args.n_epochs * len(train_dataloader)
        for epoch in range(self.training_args.n_epochs):
            for batch in train_dataloader:
                # prepare input
                inputs = {k: v.to("cuda") for k, v in batch.items()}
                activations = gather_residual_activations(
                    self.model, self.layer, 
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
                ).detach()
                nonbos_mask = inputs["attention_mask"][:,1:]
                activations = activations[:,1:][nonbos_mask.bool()]
                all_activations.append(activations)
        all_activations = torch.cat(all_activations, dim=0)
        mean_activation = all_activations.mean(dim=0)
        self.ax.proj.weight.data = mean_activation.unsqueeze(0)
        set_decoder_norm_to_unit_norm(self.ax)
        logger.warning("Training finished.")


class MeanPositiveActivation(MeanActivation):
    
    def __str__(self):
        return 'MeanPositiveActivation'

    @torch.no_grad()
    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples)
        self.make_model(**kwargs)
        torch.cuda.empty_cache()
        self.ax.eval()
        # Main training loop.
        all_activations = []
        num_training_steps = self.training_args.n_epochs * len(train_dataloader)
        for epoch in range(self.training_args.n_epochs):
            for batch in train_dataloader:
                # prepare input
                inputs = {k: v.to("cuda") for k, v in batch.items()}
                activations = gather_residual_activations(
                    self.model, self.layer, 
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
                ).detach()
                nonbos_mask = inputs["attention_mask"][:,1:]
                activations = activations[:,1:][nonbos_mask.bool()]
                labels = inputs["labels"].unsqueeze(1).repeat(1, inputs["input_ids"].shape[1] - 1)
                label_mask = labels[nonbos_mask.bool()] == 1 # only positive examples
                all_activations.append(activations[label_mask])
        all_activations = torch.cat(all_activations, dim=0)
        mean_activation = all_activations.mean(dim=0)
        self.ax.proj.weight.data = mean_activation.unsqueeze(0)
        set_decoder_norm_to_unit_norm(self.ax)
        logger.warning("Training finished.")

