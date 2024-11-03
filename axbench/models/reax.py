from .model import Model
import torch
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
from pyreax import (
    EXAMPLE_TAG, 
    MaxReLUIntervention, 
    AdditionIntervention,
    SubspaceAdditionIntervention,
    make_data_module, 
)
from torch.utils.data import DataLoader
from pyreax import (
    set_decoder_norm_to_unit_norm, 
    remove_gradient_parallel_to_decoder_directions,
    gather_residual_activations, 
    get_lr
)
from transformers import get_scheduler
from pyreax.utils.model_utils import calculate_l1_losses

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


class ReAX(Model):
    def __str__(self):
        return 'ReAX'

    def make_model(self, **kwargs):
        mode = kwargs.get("mode", "latent")
        if mode == "latent":
            ax = MaxReLUIntervention(
                embed_dim=self.model.config.hidden_size, 
                low_rank_dimension=kwargs.get("low_rank_dimension", 2),
            )
        elif mode == "steering":
            ax = SubspaceAdditionIntervention(
                embed_dim=self.model.config.hidden_size, 
                low_rank_dimension=kwargs.get("low_rank_dimension", 2),
            )
        self.ax = ax.to(self.device)
        self.ax.train()
        ax_config = IntervenableConfig(representations=[{
            "layer": l,
            "component": f"model.layers[{l}].output",
            "low_rank_dimension": kwargs.get("low_rank_dimension", 2),
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

        # Optimizer and lr
        optimizer = torch.optim.AdamW(self.ax_model.parameters(), lr=self.training_args.lr)
        num_training_steps = self.training_args.n_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer,
            num_warmup_steps=0, num_training_steps=num_training_steps)
    
        # Main training loop.
        progress_bar, curr_step = tqdm(range(num_training_steps)), 0
        for epoch in range(self.training_args.n_epochs):
            for batch in train_dataloader:
                # prepare input
                inputs = {k: v.to("cuda") for k, v in batch.items()}
                unit_locations={"sources->base": (
                    None,
                    inputs["intervention_locations"].permute(1, 0, 2).tolist()
                )}
                subspaces = [{
                    "input_subspaces": inputs["input_subspaces"],
                    "output_subspaces": inputs["output_subspaces"]}]
        
                # forward
                _, cf_outputs = self.ax_model(
                    base={
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"]
                    }, unit_locations=unit_locations, labels=inputs["labels"],
                    subspaces=subspaces, use_cache=False)
        
                # loss
                loss = cf_outputs.loss
                latent = self.ax_model.full_intervention_outputs[0].latent

                null_loss, l1_loss = calculate_l1_losses(
                    latent, 
                    labels=inputs["groups"] != EXAMPLE_TAG.CONTROL.value,
                    mask=inputs["intervention_masks"],
                    k_latent_null_loss=self.training_args.k_latent_null_loss
                )
        
                coeff = curr_step/num_training_steps
                loss += coeff*self.training_args.coeff_l1_loss_null*null_loss + coeff*self.training_args.coeff_l1_loss*l1_loss
                
                # grads
                loss.backward()
                set_decoder_norm_to_unit_norm(self.ax)
                remove_gradient_parallel_to_decoder_directions(self.ax)
                curr_step += 1
                curr_lr = get_lr(optimizer)
                # optim
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_description(
                    "lr %.6f || loss %.6f || null l1 loss %.6f" % (curr_lr, loss, null_loss))
        progress_bar.close()
        logger.warning("Training finished.")
    
    @torch.no_grad()
    def predict_latent(self, examples, **kwargs):
        self.ax.eval()
        batch_size = kwargs.get('batch_size', 32)
        
        all_acts = []
        all_max_act = []
        all_max_act_idx = []
        all_max_token = []
        
        # Process in batches
        for i in range(0, len(examples), batch_size):
            batch = examples.iloc[i:i + batch_size]
            # Batch encode all inputs
            inputs = self.tokenizer(
                batch["input"].tolist(), return_tensors="pt", 
                add_special_tokens=True, padding=True, truncation=True).to(self.device)
            
            gather_acts = gather_residual_activations(
                self.model, self.layer, inputs)
            _, ax_acts = self.ax.encode(
                gather_acts[:, 1:],  # no bos token
                subspaces={
                    "input_subspaces": torch.tensor(batch["concept_id"].tolist()).to(self.device)
                }, k=1)
            seq_lens = inputs["attention_mask"].sum(dim=1) - 1 # no bos token
            # Process each sequence in the batch
            for seq_idx, ax_seq in enumerate(ax_acts):
                acts = ax_seq[:seq_lens[seq_idx]].flatten().data.cpu().numpy().tolist()
                acts = [round(x, 3) for x in acts]
                max_act = max(acts)
                max_act_indices = [i for i, x in enumerate(acts) if x == max_act]
                max_act_idx = max_act_indices[0]
                # Get tokens for this specific sequence
                tokens = self.tokenizer.tokenize(batch.iloc[seq_idx]["input"])
                max_token = tokens[max_act_idx]
                all_acts.append(acts)
                all_max_act.append(max_act)
                all_max_act_idx.append(max_act_idx)
                all_max_token.append(max_token)
        return {
            "acts": all_acts,
            "max_act": all_max_act,
            "max_act_idx": all_max_act_idx,
            "max_token": all_max_token
        }
