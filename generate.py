# generate script for creating the training dataset for concepts.
# we assume we group two concepts into a learning group.
# it is possible to extend to more concepts into the same group,
# although more training data will likely to be needed.
# 
# example launch command:
#    python generate.py train ./tmp ./demo/concepts.csv 72 10

import sys
import argparse

try:
    import pyreax
except ModuleNotFoundError:
    sys.path.append("../pyreax")
    import pyreax
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from pyreax import ReAXFactory, load_concepts

model_name_map = {
    "gemma-2-2b": "google/gemma-2-2b",
}

def partition_lists(list1, list2, step=2):
    """
    Partitions two lists into groups of adjacent elements based on the step.
    
    Args:
        list1 (list): The first list to partition.
        list2 (list): The second list to partition.
        step (int): The step size for partitioning. Defaults to 2.
    
    Returns:
        list: A list containing the partitioned groups.
    """
    partitions = []
    for i in range(0, len(list1), step):
        group = [list1[i:i+step], list2[i:i+step]]
        partitions.append(group)
    return partitions


def main(partition, dump_dir, concept_path, num_of_examples, rotation_freq):
    """
    Main function.
    """
    assert partition in {"train", "eval"}
    
    all_concepts, all_refs = load_concepts(concept_path)
    concept2id = {concept: i for i, concept in enumerate(all_concepts)}
    concept_groups = partition_lists(all_concepts, all_refs)

    # Load lm and tokenizer.
    model_name = model_name_map[all_refs[0].split("/")[3]]
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
    model.config.use_cache = False
    model = model.cuda()
    tokenizer =  AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"

    # Init the dataset factory.
    dataset_factory = ReAXFactory(model, tokenizer, dump_dir)
    for group_idx, (concepts, refs) in enumerate(concept_groups):
        # prepare concept related data.
        concept_genres_map, contrast_concepts_map = \
            dataset_factory.prepare_concepts(concepts)
        # generate.
        if partition == "train":
            current_df = dataset_factory.create_train_df(
                concepts, num_of_examples, concept_genres_map, contrast_concepts_map)
            
        elif partition == "eval":
            pass


if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Generate script for creating the training dataset for concepts..")
    
    # Define the arguments
    parser.add_argument("partition", type=str, help="The partition to generate")
    parser.add_argument("dump_dir", type=str, help="Path to the dump directory")
    parser.add_argument("concept_path", type=str, help="Path to the concept file")
    parser.add_argument("num_of_examples", type=int, help="The number of examples")
    parser.add_argument("rotation_freq", type=int, help="Frequency for chunking files")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Run the main function with parsed arguments
    main(args.partition, args.dump_dir, args.concept_path, args.num_of_examples, args.rotation_freq)

