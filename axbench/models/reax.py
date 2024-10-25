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

import pyreft
from pyreax import (
    EXAMPLE_TAG, 
    ReAXFactory, 
    MaxReLUIntervention, 
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

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


class ReAX(Model):
    
    def __init__(self, model, tokenizer, layer, training_args=None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        # abstracting layer
        self.layer = layer
        self.training_args = training_args

        self.make_model(**kwargs)
    
    def __str__(self):
        return 'ReAX'

    def make_model(self, **kwargs):
        reax_intervention = MaxReLUIntervention(
            embed_dim=self.model.config.hidden_size, 
            low_rank_dimension=kwargs.get("low_rank_dimension", 2),
        )
        reax_intervention = reax_intervention.train()
        reft_config = pyreft.ReftConfig(representations=[{
            "layer": l,
            "component": f"model.layers[{l}].output",
            "low_rank_dimension": kwargs.get("low_rank_dimension", 2),
            "intervention": reax_intervention} for l in [self.layer]])
        reft_model = pyreft.get_reft_model(self.model, reft_config)
        reft_model.set_device("cuda")
        self.reax_intervention = reax_intervention
        self.reft_model = reft_model
    
    def make_dataloader(self, examples, **kwargs):
        data_module = make_data_module(self.tokenizer, self.model, examples)
        train_dataloader = DataLoader(
            data_module["train_dataset"], shuffle=True, batch_size=self.training_args.batch_size, 
            collate_fn=data_module["data_collator"])
        return train_dataloader
        
    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples)
        torch.cuda.empty_cache()

        self.reft_model.print_trainable_parameters()
        
        # Optimizer and lr
        optimizer = torch.optim.AdamW(self.reft_model.parameters(), lr=self.training_args.lr)
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
                _, cf_outputs = self.reft_model(
                    base={
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"]
                    }, unit_locations=unit_locations, labels=inputs["labels"],
                    subspaces=subspaces, use_cache=False)
        
                # loss
                loss = cf_outputs.loss
                latent = self.reft_model.full_intervention_outputs[0].latent * inputs["intervention_masks"]
                topk_latent, _ = torch.topk(latent, self.training_args.k_latent_null_loss, dim=-1)
                null_loss = (topk_latent.mean(dim=-1)*(inputs["groups"]==EXAMPLE_TAG.CONTROL.value))
                null_loss = null_loss.sum()
        
                l1_loss = (latent.mean(dim=-1)*(inputs["groups"]!=EXAMPLE_TAG.CONTROL.value))
                l1_loss = l1_loss.sum()
                
                coeff = curr_step/num_training_steps
                loss += coeff*self.training_args.coeff_l1_loss_null*null_loss + coeff*self.training_args.coeff_l1_loss*l1_loss
                
                # grads
                loss.backward()
                set_decoder_norm_to_unit_norm(self.reax_intervention)
                remove_gradient_parallel_to_decoder_directions(self.reax_intervention)
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
    
    def save(self, dump_dir, **kwargs):
        weight_file = dump_dir / f"ReAX_weight.pt"
        weight = self.reax_intervention.proj.weight.data.cpu()
        if weight_file.exists():
            weight = torch.cat([torch.load(weight_file), weight], dim=0)
        torch.save(weight, weight_file)
        
        bias_file = dump_dir / f"ReAX_bias.pt"
        bias = self.reax_intervention.proj.bias.data.cpu()
        if bias_file.exists():
            bias = torch.cat([torch.load(bias_file), bias], dim=0)
        torch.save(bias, bias_file)

    def load(self, dump_dir=None, **kwargs):
        weight = torch.load(
            f"{dump_dir}/ReAX_weight.pt"
        )
        bias = torch.load(
            f"{dump_dir}/ReAX_bias.pt"
        )
        self.reax_intervention.proj.weight.data = weight.cuda()
        self.reax_intervention.proj.bias.data = bias.cuda()
    
    def predict_latent(self, examples, **kwargs):
        self.reax_intervention.eval()
        self.reft_model.print_trainable_parameters()
        
        all_acts = []
        all_max_act = []
        for _, row in examples.iterrows():
            inputs = self.tokenizer.encode(
                row["input"], return_tensors="pt", add_special_tokens=True).to("cuda")
            reax_in = gather_residual_activations(
                self.model, self.layer, {"input_ids": inputs})
            _, reax_acts = self.reax_intervention.encode(
                reax_in[:,1:], # no bos token
                subspaces={
                    "input_subspaces": torch.tensor([row["concept_id"]])}, k=1)
            reax_acts = reax_acts.flatten().data.cpu().numpy().tolist()
            reax_acts = [round(x, 3) for x in reax_acts]
            max_reax_act = max(reax_acts)

            all_acts += [reax_acts]
            all_max_act += [max_reax_act]
        return {
            "acts": all_acts,
            "max_act": all_max_act}

    def predict_steer(self, examples, **kwargs):
        pass
