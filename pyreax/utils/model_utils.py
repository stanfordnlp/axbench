#################################
#
# Model utils.
#
#################################
import torch, einops
from torch import nn
import pyreft
import pyvene


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_model_continues(
    model, tokenizer, prompts, max_new_tokens, chat_template=None
):
    """we ground examples with the model's original generation."""
    outputs = []
    tokenizer.padding_side = "left"
    if chat_template is not None:
        pass # TODO: to handle chat models
    encoding = tokenizer(prompts, return_tensors='pt', padding=True).to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids = generated_ids[:, encoding.input_ids.shape[1]:]
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    return generated_texts


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

