try:
    # This library is our indicator that the required installs
    # need to be done.
    import pyreax

except ModuleNotFoundError:
    # relative import; better to pip install subctrl
    import sys
    sys.path.append("../../../pyreax")
    import pyreax

import pandas as pd
import os, asyncio
from pyreax import LanguageModel


def OUATPrefix_steering_factors(n):
    # return [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    return [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 5.0, 10.0]

T_PROMPT_STEERING = """You must answer the question with content \
related to %s even if it is not related to the question or it does not make sense."""


T_GENERATE_STEERING_PROMPT = """Generate a prompt to guide a language \
model in producing responses. 

Objective: 
Direct the model to include content related to %s (the concept) in its responses. 
Ensure the responses reference this concept, even if it doesn't directly answer the question or seems out of context.
Optionally, provide in-context examples to reinforce this behavior.
        
Return only the final prompt without any additional text."""


async def get_steering_prompts(client, concepts):
    prompts = []
    for concept in concepts:
        prompts += [T_GENERATE_STEERING_PROMPT % (concept)]
    responses = await client.chat_completions("get_steering_prompts", prompts)
    return responses


class SteeringDatasetFactory(object):
    def __init__(
        self, model, tokenizer, dump_dir, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.master_data_dir = kwargs.get("master_data_dir", None)
        if kwargs.get("lm_client", None):
            self.lm_model = LanguageModel(
                kwargs.get("lm_model", "gpt-4o"), kwargs["lm_client"], dump_dir, 
                use_cache=True, master_data_dir=self.master_data_dir
            )

    def create_eval_df(self, concepts, subset_n, n_steering_factors, steering_datasets):
        for dataset_name in steering_datasets:
            if dataset_name == "OUATPrefix":
                common_steering_factors = OUATPrefix_steering_factors(n_steering_factors)
                # we generate subset_n * n_steering_factors examples for OUATPrefix.
                # OUATPrefix is basically a prefix dataset.
                # "Once upon a time, " is the prefix, and there is no other labels.
                # we also need to label these in groups:
                # each one of subset_n group has the same group id.
                all_examples = []
                for idx, concept in enumerate(concepts):
                    for i in range(subset_n):
                        for factor in common_steering_factors:
                            all_examples += [
                                [dataset_name, idx, concept, i, factor, "Once upon a time, there was a ", ]
                            ]
                df = pd.DataFrame(
                    all_examples, 
                    columns = [
                        'dataset_name', 'concept_id', 'input_concept', 'input_id', 'factor', 'input'])
                return df
            elif dataset_name == "AlpacaEval":
                # load alpaca eval dataset.
                assert self.master_data_dir is not None, "Master data dir is required for AlpacaEval."
                alpaca_eval_path = os.path.join(self.master_data_dir, "alpaca_eval.json")
                alpaca_eval_df = pd.read_json(alpaca_eval_path)
                common_steering_factors = OUATPrefix_steering_factors(n_steering_factors)

                # get gpt-4o boosted steering prompts.
                steering_prompts = asyncio.run(get_steering_prompts(self.lm_model, concepts))
                steering_prompts = [prompt.strip() for prompt in steering_prompts]
                all_examples = []
                for idx, concept in enumerate(concepts):
                    for i in range(subset_n):
                        # sample a random example from alpaca eval dataset.
                        sampled_prompt = alpaca_eval_df.sample(1)["instruction"].tolist()[0]
                        # for prompt-based steering ONLY.
                        steering_prompt = steering_prompts[idx] \
                            if steering_prompts[idx] != "" else T_PROMPT_STEERING % (concept)
                        steered_prompt = f" {steering_prompt}\n\nQuestion: {sampled_prompt}"
                        formatted_steered_prompt = self.tokenizer.apply_chat_template(
                            [{"role": "user", "content": steered_prompt}], 
                            tokenize=False, add_generation_prompt=True)
                        # apply the tokenizer chat format to the prompt.
                        formatted_prompt = self.tokenizer.apply_chat_template(
                            [{"role": "user", "content": sampled_prompt}], 
                            tokenize=False, add_generation_prompt=True)
                        for factor in common_steering_factors:
                            all_examples += [[
                                dataset_name, idx, concept, i, factor, 
                                sampled_prompt, formatted_steered_prompt, formatted_prompt
                            ]]
                df = pd.DataFrame(
                    all_examples, 
                    columns = [
                        'dataset_name', 'concept_id', 'input_concept', 
                        'input_id', 'factor', 'original_prompt', 'steered_input', 'input'])
                return df
            else:
                # not implemented yet.
                raise NotImplementedError(f"Steering dataset {dataset_name} not implemented.")