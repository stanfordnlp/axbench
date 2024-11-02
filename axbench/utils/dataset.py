import pandas as pd
import os

def OUATPrefix_steering_factors(n):
    return [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]


PROMPT_STEERING_TEMPLATE = """You must answer the question with content \
related to %s even if it is not related to the question or it does not make sense."""


class SteeringDatasetFactory(object):
    def __init__(
        self, model, tokenizer, dump_dir, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.master_data_dir = kwargs.get("master_data_dir", None)

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
                all_examples = []
                for idx, concept in enumerate(concepts):
                    for i in range(subset_n):
                        # sample a random example from alpaca eval dataset.
                        sampled_prompt = alpaca_eval_df.sample(1)["instruction"].tolist()[0]
                        # for prompt-based steering ONLY.
                        steering_prompt = PROMPT_STEERING_TEMPLATE % (concept)
                        steered_prompt = f" {steering_prompt} {sampled_prompt}"
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