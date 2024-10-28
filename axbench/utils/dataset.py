import pandas as pd


def OUATPrefix_steering_factors(n):
    if n < 3:
        return [i * 0.5 for i in range(n)]
    # Allocate points: 60% to 0-2 range, 40% to >2 range
    n_small = int(n * 0.6)
    n_large = n - n_small
    # Dense sampling from 0 to 2
    small_factors = [i * (2.0 / (n_small - 1)) for i in range(n_small)]
    # Exponential spacing for larger values
    large_factors = [2.0 + (i * 0.5) for i in range(1, n_large + 1)]
    return small_factors + large_factors


class SteeringDatasetFactory(object):
    def __init__(
        self, model, tokenizer, dump_dir, **kwargs):
        self.model = model
        self.tokenizer = tokenizer

    def create_eval_df(self, concepts, subset_n, n_steering_factors, steering_datasets):
        for dataset_name in steering_datasets:
            common_steering_factors = OUATPrefix_steering_factors(n_steering_factors)
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