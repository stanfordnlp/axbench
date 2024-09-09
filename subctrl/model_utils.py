#################################
#
# Model utils.
#
#################################


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