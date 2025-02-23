#################################
#
# Model utils.
#
#################################
import torch, einops
from torch import nn
from tqdm.auto import tqdm


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_model_continues(
    model, tokenizer, prompts, max_new_tokens, is_chat_model=True, batch_size=8, include_system_prompt=False, verbose=False
):
    """we ground examples with the model's original generation."""
    tokenizer.padding_side = "left"
    if is_chat_model:
        if include_system_prompt:
            def apply_chat_template(prompt):
                messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
                nobos = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)[1:]
                return tokenizer.decode(nobos)
        else:
            def apply_chat_template(prompt):
                messages = [{"role": "user", "content": prompt}]
                nobos = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)[1:]
                return tokenizer.decode(nobos)
        prompts = [apply_chat_template(prompt) for prompt in prompts]
    
    # Process prompts in batches
    all_generated_texts = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating responses", disable=not verbose):
        batch_prompts = prompts[i:i + batch_size]
        encoding = tokenizer(batch_prompts, return_tensors='pt', padding=True).to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(
                **encoding, max_new_tokens=max_new_tokens, do_sample=False)
            generated_ids = generated_ids[:, encoding.input_ids.shape[1]:]
        batch_generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        all_generated_texts.extend(batch_generated_texts)
    
    return all_generated_texts


def gather_residual_activations(model, target_layer, inputs):
  target_act = None
  def gather_target_act_hook(mod, inputs, outputs):
    nonlocal target_act # make sure we can modify the target_act from the outer scope
    target_act = outputs[0]
    return outputs
  handle = model.model.layers[target_layer].register_forward_hook(
      gather_target_act_hook, always_call=True)
  _ = model.forward(**inputs)
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


def calculate_l1_losses(latent, non_topk_latent, labels=None, mask=None):
    """
    Calculate L1 losses with masked mean.
    
    Parameters:
    - latent: latent representation, shape [batch_size, seq_len]
    - non_topk_latent: non-topk latent representation, shape [batch_size, seq_len]
    - labels: labels, shape [batch_size]
    - mask: long mask, shape [batch_size, seq_len]
    """
    if mask is None:
        mask = torch.ones_like(latent, dtype=torch.long)
    
    mask = mask.bool()
    
    valid_counts = mask.sum(dim=-1)  # [batch_size]
    eps = torch.finfo(latent.dtype).eps
    if non_topk_latent is not None:
        masked_non_topk_sum = (non_topk_latent * mask).sum(dim=-1)  # [batch_size]
        mean_non_topk = masked_non_topk_sum / (valid_counts + eps)
        l1_loss = mean_non_topk.mean() # mean across batch
    else:
        masked_sum = (latent * mask).sum(dim=-1)  # [batch_size]
        mean_all = masked_sum / (valid_counts + eps)
        l1_loss = mean_all.mean() # mean across batch
    return l1_loss


def get_prefix_length(tokenizer, common_prefix=None):
    if common_prefix is None:
        message_a = [{"role": "user", "content": "1"}]
        message_b = [{"role": "user", "content": "2"}]
        tokens_a = tokenizer.apply_chat_template(message_a, tokenize=True)
        tokens_b = tokenizer.apply_chat_template(message_b, tokenize=True)
        print("Detecting sequence a:", tokens_a)
        print("Detecting sequence b:", tokens_b)
        prefix_length = 0
        for i, (ta, tb) in enumerate(zip(tokens_a, tokens_b)):
            if ta != tb:
                prefix_length = i
                break
    else:
        message = [{"role": "user", "content": common_prefix}]
        tokens = tokenizer.apply_chat_template(
            message, tokenize=True, add_generation_prompt=True)
        prefix_length = len(tokens)
    return prefix_length


def get_suffix_length(tokenizer):
    message_a = [{"role": "user", "content": "1"}]
    message_b = [{"role": "user", "content": "2"}]
    tokens_a = tokenizer.apply_chat_template(message_a, tokenize=True)
    tokens_b = tokenizer.apply_chat_template(message_b, tokenize=True)
    suffix_length = 0
    for i, (ta, tb) in enumerate(zip(reversed(tokens_a), reversed(tokens_b))):
        if ta != tb:
            suffix_length = i
            break
    return suffix_length, tokenizer.decode(tokens_a[-suffix_length:])