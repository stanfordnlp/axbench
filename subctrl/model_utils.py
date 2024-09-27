#################################
#
# Model utils.
#
#################################
import torch, einops

import pyreft
import pyvene
from pyvene import (
    IntervenableModel,
    ConstantSourceIntervention,
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
    CollectIntervention,
    InterventionOutput
)


def get_model_continues(
    model, tokenizer, examples, max_new_tokens, chat_template=None
):
    """we ground examples with the model's original generation."""
    continued_examples = []
    if chat_template is not None:
        pass # TODO: to handle chat models
    for e in examples:
        batch_inputs = tokenizer(e, padding=True, return_tensors="pt").to(model.device)
        response = model.generate(
            **batch_inputs, max_new_tokens=max_new_tokens, do_sample=False)
        r = tokenizer.batch_decode(
            response[:,batch_inputs["input_ids"].shape[1]:], 
            skip_special_tokens=False)[0]
        continued_examples += [r]
    
    return continued_examples


def gather_residual_activations(model, target_layer, inputs):
  target_act = None
  def gather_target_act_hook(mod, inputs, outputs):
    nonlocal target_act # make sure we can modify the target_act from the outer scope
    target_act = outputs[0]
    return outputs
  handle = model.model.layers[target_layer].register_forward_hook(gather_target_act_hook)
  _ = model.forward(inputs)
  handle.remove()
  return target_act


@torch.no_grad()
def set_decoder_norm_to_unit_norm(model):
    assert model.proj.weight is not None, "Decoder weight was not initialized."

    eps = torch.finfo(model.proj.weight.dtype).eps
    norm = torch.norm(model.proj.weight.data, dim=1, keepdim=True)
    model.proj.weight.data /= norm + eps


@torch.no_grad()
def remove_gradient_parallel_to_decoder_directions(model):
    assert model.proj.weight is not None, "Decoder weight was not initialized."
    assert model.proj.weight.grad is not None  # keep pyright happy

    parallel_component = einops.einsum(
        model.proj.weight.grad,
        model.proj.weight.data,
        "d_out d_in, d_out d_in -> d_out",
    )
    model.proj.weight.grad -= einops.einsum(
        parallel_component,
        model.proj.weight.data,
        "d_out, d_out d_in -> d_out d_in",
    )


class MaxReLUIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]*2, bias=False)
        
    def encode(
        self, base, source=None, subspaces=None, k=1
    ):
        _weight = []
        for subspace in subspaces["input_subspaces"]:
            _weight += [self.proj.weight[subspace]]
        W_c = torch.stack(_weight, dim=0).unsqueeze(dim=-1)

        # latent
        # base : [b, s, h]
        latent = torch.relu(torch.bmm(base, W_c)).squeeze(dim=-1)
        
        # topk over a seq
        topk_acts, topk_indices = latent.topk(k=k, dim=-1, sorted=False)

        topk_latent = torch.zeros_like(latent)
        topk_latent.scatter_(-1, topk_indices, topk_acts)
        topk_latent = topk_latent.unsqueeze(dim=-1)
        return topk_latent, latent

    def forward(
        self, base, source=None, subspaces=None
    ):
        ctrl_weight = []
        for subspace in subspaces["output_subspaces"]:
            ctrl_weight += [self.proj.weight[subspace]]
        W_ctrl = torch.stack(ctrl_weight, dim=0).unsqueeze(dim=-1).permute(0, 2, 1)
        
        topk_latent, latent = self.encode(base, source, subspaces)
        topk_latent = topk_latent.squeeze(dim=-1)
        max_latent = topk_latent.max(dim=-1, keepdim=True)[0]
        steer_dir = torch.bmm(max_latent.unsqueeze(dim=-1), W_ctrl) # bs, 1, dim
        output = base + steer_dir

        return InterventionOutput(
            output=output.to(base.dtype),
            latent=latent
        )


class SubspaceAdditionIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    def __init__(self, **kwargs):
        # Note that we initialise these to zeros because we're loading in pre-trained weights.
        # If you want to train your own SAEs then we recommend using blah
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj = torch.nn.Linear(
                self.embed_dim, kwargs["low_rank_dimension"]*2, bias=True)

    def forward(self, base, source=None, subspaces=None):
        steering_vec = torch.tensor(subspaces["mag"]) * self.proj.weight[subspaces["idx"]].unsqueeze(dim=0)
        output = base + steering_vec
        return output

