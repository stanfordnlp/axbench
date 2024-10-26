import pandas as pd


class SteeringDatasetFactory(object):
    def __init__(
        self, model, tokenizer, dump_dir, **kwargs):
        self.model = model
        self.tokenizer = tokenizer

    def create_eval_df(self, concepts, subset_n, n_steering_factors, steering_datasets):
        common_steering_factors = [f*20.0 for f in range(n_steering_factors)]
        for dataset_name in steering_datasets:
            print(f"Creating steering dataset: {dataset_name}")
            if dataset_name == "OUATPrefix":
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
            else:
                # not implemented yet.
                raise NotImplementedError(f"Steering dataset {dataset_name} not implemented.")