from .model import Model
import torch, einops
from tqdm.auto import tqdm
import os
import pandas as pd
from pyvene import (
    IntervenableConfig,
    IntervenableModel
)
from .interventions import (
    TopKReLUIntervention,
    TopKReLUSubspaceIntervention,
    AdditionIntervention,
    SubspaceIntervention,
    AdditionGatingIntervention,
    TopKReLUGatingIntervention,
    TopKReLUGatingAnnealIntervention,
    AdditionGatingAnnealIntervention,
    AdditionFlexibleIntervention,
    FlexibleFactorIntervention,
    TopKReLUNoiseIntervention
)
from ..utils.constants import EXAMPLE_TAG
from torch.utils.data import DataLoader
from ..utils.model_utils import (
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
    gather_residual_activations, 
    get_lr,
    calculate_l1_losses,
    calculate_F_latent
)
from transformers import get_scheduler
from transformers import set_seed


class LsReFT(Model):
    def __str__(self):
        return 'LsReFT'

    def make_model(self, **kwargs):
        mode = kwargs.get("mode", "latent")
        if mode == "steering":
            intervention_type = kwargs.get("intervention_type", "addition")
            if intervention_type == "addition" or intervention_type == "noise":
                ax = AdditionIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            elif intervention_type == "clamping":
                ax = SubspaceIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            elif intervention_type == "gating":
                ax = AdditionGatingIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            elif intervention_type == "anneal":
                ax = AdditionGatingAnnealIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            elif intervention_type == "factor_individual" or intervention_type == "factor":
                ax = AdditionFlexibleIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            
        else:
            intervention_type = kwargs.get("intervention_type", "addition")
            if intervention_type == "addition":
                ax = TopKReLUIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            elif intervention_type == "clamping":
                ax = TopKReLUSubspaceIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            elif intervention_type == "gating":
                ax = TopKReLUGatingIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            elif intervention_type == "anneal":
                ax = TopKReLUGatingAnnealIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            elif intervention_type == "factor_individual" or intervention_type == "factor":
                ax = TopKReLUNoiseIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
                ax2 = FlexibleFactorIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )

            elif intervention_type == "noise":
                ax = TopKReLUNoiseIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )

        layers = self.steering_layers if self.steering_layers else [self.layer]
        self.ax = ax.to(self.device)
        self.ax.train()
        ax_config = IntervenableConfig(representations=[{
            "layer": l,
            "component": f"model.layers[{l}].output",
            "low_rank_dimension": kwargs.get("low_rank_dimension", 1),
            "intervention": self.ax} for l in layers])
        ax_model = IntervenableModel(ax_config, self.model)
        ax_model.set_device(self.device)
        self.ax_model = ax_model
        
        if mode != "steering":
            if intervention_type == "factor_individual" or intervention_type == "factor":
                self.ax2 = ax2.to(self.device)
                self.ax2.train()
                ax2_config = IntervenableConfig(representations=[{
                    "layer": l,
                    "component": f"model.layers[{l}].output",
                    "low_rank_dimension": kwargs.get("low_rank_dimension", 1),
                    "intervention": self.ax2} for l in layers])
                ax2_model = IntervenableModel(ax2_config, self.model)
                ax2_model.set_device(self.device)
                self.ax2_model = ax2_model

    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples, **kwargs)
        torch.cuda.empty_cache()

        # Optimizer and lr
        optimizer = torch.optim.AdamW(
            self.ax_model.parameters(), 
            lr=self.training_args.lr, weight_decay=self.training_args.weight_decay)
        num_training_steps = self.training_args.n_epochs * (len(train_dataloader) // self.training_args.gradient_accumulation_steps)
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
                subspaces = [{
                    "k": self.training_args.topk,
                    
                }]
                # forward
                _, cf_outputs = self.ax_model(
                    base={
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"]
                    }, unit_locations=unit_locations, labels=inputs["labels"],
                    subspaces=subspaces, use_cache=False)
                
                # loss
                loss = cf_outputs.loss
                latent, non_topk_latent = self.ax_model.full_intervention_outputs[0].latent
                l1_loss = calculate_l1_losses(
                    latent, non_topk_latent,
                    mask=inputs["intervention_masks"],
                )
                coeff = curr_step/num_training_steps
                loss += coeff*self.training_args.coeff_latent_l1_loss*l1_loss
                loss = loss.mean()
                loss /= self.training_args.gradient_accumulation_steps
                # grads
                loss.backward()

                # Perform optimization step every gradient_accumulation_steps
                if (step + 1) % self.training_args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    torch.nn.utils.clip_grad_norm_(self.ax_model.parameters(), 1.0)
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
                        "lr %.6f || loss %.6f || l1 loss %.6f" % (
                            curr_lr, loss, l1_loss))
        progress_bar.close()

    def train_anneal(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples, **kwargs)
        torch.cuda.empty_cache()

        # Optimizer and lr
        optimizer = torch.optim.AdamW(
            self.ax_model.parameters(), 
            lr=self.training_args.lr, weight_decay=self.training_args.weight_decay)
        num_training_steps = self.training_args.n_epochs * (len(train_dataloader) // self.training_args.gradient_accumulation_steps)
        
        temperature_start = float(1)
        temperature_end = float(0.001)
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
                subspaces = [{
                    "k": self.training_args.topk
                }]
        
                # forward
                _, cf_outputs = self.ax_model(
                    base={
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"]
                    }, unit_locations=unit_locations, labels=inputs["labels"],
                    subspaces=subspaces, use_cache=False)
                
                # loss
                loss = cf_outputs.loss
                latent, non_topk_latent = self.ax_model.full_intervention_outputs[0].latent
                l1_loss = calculate_l1_losses(
                    latent, non_topk_latent,
                    mask=inputs["intervention_masks"],
                )
                coeff = curr_step/num_training_steps
                loss += coeff*self.training_args.coeff_latent_l1_loss*l1_loss
                loss = loss.mean()
                loss /= self.training_args.gradient_accumulation_steps
                # grads
                loss.backward()

                # Perform optimization step every gradient_accumulation_steps
                if (step + 1) % self.training_args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    temp = temperature_schedule[curr_step] if len(temperature_schedule) > curr_step else temperature_end
                    self.ax.set_temperature(temp)
                    torch.nn.utils.clip_grad_norm_(self.ax_model.parameters(), 1.0)
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
                        "lr %.6f || loss %.6f || l1 loss %.6f" % (
                            curr_lr, loss, l1_loss))
        progress_bar.close()

    def train_noise(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples, **kwargs)
        torch.cuda.empty_cache()

        # Optimizer and lr
        optimizer = torch.optim.AdamW(
            self.ax_model.parameters(), 
            lr=self.training_args.lr, weight_decay=self.training_args.weight_decay)
        num_training_steps = self.training_args.n_epochs * (len(train_dataloader) // self.training_args.gradient_accumulation_steps)
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
                
                subspaces = [{
                    "k": self.training_args.topk
                }]
                self.ax.set_seq_len_mask(inputs["attention_mask"])
                # forward
                _, cf_outputs = self.ax_model(
                    base={
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"]
                    }, unit_locations=unit_locations, labels=inputs["labels"],
                    subspaces=subspaces, use_cache=False)
                
                # loss
                loss = cf_outputs.loss
                latent, non_topk_latent = self.ax_model.full_intervention_outputs[0].latent
                l1_loss = calculate_l1_losses(
                    latent, non_topk_latent,
                    mask=inputs["intervention_masks"],
                )
                coeff = curr_step/num_training_steps
                loss += coeff*self.training_args.coeff_latent_l1_loss*l1_loss
                loss = loss.mean()
                loss /= self.training_args.gradient_accumulation_steps
                # grads
                loss.backward()

                # Perform optimization step every gradient_accumulation_steps
                if (step + 1) % self.training_args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    torch.nn.utils.clip_grad_norm_(self.ax_model.parameters(), 1.0)
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
                        "lr %.6f || loss %.6f || l1 loss %.6f" % (
                            curr_lr, loss, l1_loss))
        progress_bar.close()

    def train_factor(self, examples, dir_name, model_name, **kwargs):

        #examples  = examples[examples["category"] == "positive"]
        train_dataloader = self.make_dataloader(examples, **kwargs)
        torch.cuda.empty_cache()

        # Optimizer and lr
        optimizer = torch.optim.AdamW(
            self.ax2_model.parameters(), 
            lr=self.training_args.lr, weight_decay=self.training_args.weight_decay)

        num_training_steps = self.training_args.n_epochs * (len(train_dataloader) // self.training_args.gradient_accumulation_steps)
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
                    inputs["intervention_locations"].clone().permute(1, 0, 2).tolist()
                )}           
                subspaces = [{
                    "k": self.training_args.topk,
                    "subspaces": torch.tensor(inputs["concept_id"]).to(self.device),                
                }]

                self.ax2.load_steer_vector(dir_name, model_name)                
                # forward
                _, cf_outputs = self.ax2_model(
                    base={
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"]
                    }, unit_locations=unit_locations, labels=inputs["labels"],
                    subspaces=subspaces, use_cache=False)                
                # loss

                loss = cf_outputs.loss
                pushdown = self.ax2_model.full_intervention_outputs[0].latent
                #l1_loss = calculate_l1_losses(
                #    latent, non_topk_latent,
                #    mask=inputs["intervention_masks"],
                #)          
                coeff = curr_step/num_training_steps
                F_latent_loss = calculate_F_latent(pushdown, inputs["intervention_masks"])
                loss += coeff* 0.1* F_latent_loss
                loss = loss.mean()
                loss /= self.training_args.gradient_accumulation_steps
                # grads
                loss.backward()
                # Perform optimization step every gradient_accumulation_steps
                if (step + 1) % self.training_args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    torch.nn.utils.clip_grad_norm_(self.ax2_model.parameters(), 1.0)
                    curr_step += 1
                    curr_lr = get_lr(optimizer)
                    # optim
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    progress_bar.set_description(
                        "lr %.6f || loss %.6f" % (
                            curr_lr, loss))
        progress_bar.close()
    @torch.no_grad()
    def predict_latent(self, examples, **kwargs):
        self.ax.eval()
        batch_size = kwargs.get('batch_size', 32)
        
        all_acts = []
        all_max_act = []
        all_max_act_idx = []
        all_max_token = []
        all_tokens = []
        # Process in batches
        for i in range(0, len(examples), batch_size):
            batch = examples.iloc[i:i + batch_size]
            # Batch encode all inputs
            inputs = self.tokenizer(
                batch["input"].tolist(), return_tensors="pt", 
                add_special_tokens=True, padding=True, truncation=True).to(self.device)
            gather_acts = gather_residual_activations(
                self.model, self.layer, inputs)
            outputs = self.ax(
                gather_acts[:, kwargs["prefix_length"]:],  # no bos token
                subspaces={
                    "subspaces": torch.tensor(batch["concept_id"].tolist()).to(self.device),
                    "k": 1
                })
            ax_acts = outputs.latent[0]
            seq_lens = inputs["attention_mask"].sum(dim=1) - kwargs["prefix_length"] # no bos token
            # Process each sequence in the batch
            for seq_idx, ax_seq in enumerate(ax_acts):
                acts = ax_seq[:seq_lens[seq_idx]].flatten().data.float().cpu().numpy().tolist()
                acts = [round(x, 3) for x in acts]
                max_act = max(acts)
                max_act_indices = [i for i, x in enumerate(acts) if x == max_act]
                max_act_idx = max_act_indices[0]
                # Get tokens for this specific sequence
                tokens = self.tokenizer.tokenize(batch.iloc[seq_idx]["input"])[kwargs["prefix_length"]-1:] # -1 is because it does not prepend BOS token
                max_token = tokens[max_act_idx]
                all_acts.append(acts)
                all_max_act.append(max_act)
                all_max_act_idx.append(max_act_idx)
                all_max_token.append(max_token)
                all_tokens.append(tokens)
        return {
            "acts": all_acts,
            "max_act": all_max_act,
            "max_act_idx": all_max_act_idx,
            "max_token": all_max_token,
            "tokens": all_tokens
        }