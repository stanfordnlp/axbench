# generate script for creating the training dataset for concepts.
# we assume we group two concepts into a learning group.
# it is possible to extend to more concepts into the same group,
# although more training data will likely to be needed.
# 
# example launch command:
#    python generate.py --dump_dir demo --concept_path demo/concepts.csv --num_of_examples 72 --rotation_freq 1000

import sys
import argparse
import time
import os
import pickle
import random
import json

try:
    import pyreax
except ModuleNotFoundError:
    sys.path.append("../pyreax")
    import pyreax
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from pyreax import ReAXFactory, load_concepts

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)

model_name_map = {
    "gemma-2-2b": "google/gemma-2-2b",
}

MAX_RETRIES = 5
RETRY_DELAY = 1  # in seconds
STATE_FILE = "dataset_state.pkl"
METADATA_FILE = "metadata.jsonl"
DEFAULT_ROTATION_FREQ = 1000

def retry_with_backoff(func, *args, **kwargs):
    """
    Generic retry mechanism with exponential backoff.
    
    Args:
        func (callable): The function to retry.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.
    
    Returns:
        The result of the function if successful.
    
    Raises:
        Exception: The last exception raised if all retries fail.
    """
    retries = 0
    while retries < MAX_RETRIES:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            retries += 1
            if retries >= MAX_RETRIES:
                logger.warning(f"Failed after {MAX_RETRIES} retries: {e}")
                raise
            logger.warning(f"Retrying ({retries}/{MAX_RETRIES}) after failure: {e}")
            time.sleep(RETRY_DELAY * retries)

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

def save(dump_dir, state, group_idx, concepts, refs, partition, current_df, rotation_freq):
    """
    Save the current state, metadata, and DataFrame.
    
    Args:
        dump_dir (str): The directory to save the files.
        state (dict): The state dictionary to save.
        group_idx (int): The group index.
        concepts (list): The list of concepts.
        refs (list): The list of references.
        partition (str): The partition type (train or eval).
        current_df (DataFrame): The DataFrame to save.
        rotation_freq (int): Frequency for chunking files.
    """
    # Save state
    state_path = os.path.join(dump_dir, STATE_FILE)
    with open(state_path, "wb") as f:
        pickle.dump(state, f)
    
    # Save metadata
    metadata_path = os.path.join(dump_dir, METADATA_FILE)
    metadata_entry = {
        "group_idx": group_idx,
        "concepts": concepts,
        "refs": refs
    }
    with open(metadata_path, "a") as f:
        f.write(json.dumps(metadata_entry) + "\n")
    
    # Save DataFrame
    fragment_index = group_idx // rotation_freq
    df_path = os.path.join(dump_dir, f"{partition}_data_fragment_{fragment_index}.csv")
    if os.path.exists(df_path):
        existing_df = pd.read_csv(df_path)
        combined_df = pd.concat([existing_df, current_df], ignore_index=True)
    else:
        combined_df = current_df
    combined_df.to_csv(df_path, index=False)

def load_state(dump_dir):
    """
    Load the state from a file if it exists.
    
    Args:
        dump_dir (str): The directory to load the state file from.
    
    Returns:
        dict: The loaded state dictionary, or None if no state file exists.
    """
    state_path = os.path.join(dump_dir, STATE_FILE)
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            return pickle.load(f)
    return None

def main(dump_dir, concept_path, num_of_examples, rotation_freq, seed, max_concepts):
    """
    Main function.
    """
    # Load and optionally shuffle concepts
    all_concepts, all_refs = load_concepts(concept_path)
    if seed is not None:
        random.seed(seed)
        combined = list(zip(all_concepts, all_refs))
        random.shuffle(combined)
        all_concepts, all_refs = zip(*combined)
    
    # Limit the number of concepts if specified
    if max_concepts is not None:
        all_concepts = all_concepts[:max_concepts]
        all_refs = all_refs[:max_concepts]
    
    concept2id = {concept: i for i, concept in enumerate(all_concepts)}
    concept_groups = partition_lists(all_concepts, all_refs)

    # Load lm and tokenizer.
    model_name = model_name_map[all_refs[0].split("/")[3]]
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
    model.config.use_cache = False
    model = model.cuda()
    tokenizer =  AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"

    # Load the state if it exists.
    state = load_state(dump_dir)
    start_group_idx = state.get("group_idx", 0) if state else 0

    logger.warning(f"Starting group index: {start_group_idx}")
    
    # Init the dataset factory.
    dataset_factory = ReAXFactory(model, tokenizer, dump_dir)
    progress_bar = tqdm(range(start_group_idx, len(concept_groups)), desc="Processing concept groups")
    for group_idx in progress_bar:
        concepts, refs = concept_groups[group_idx]
        
        # prepare concept related data.
        concept_genres_map, contrast_concepts_map = \
            dataset_factory.prepare_concepts(concepts)
        
        # generate with retry mechanism.
        try:
            current_df = retry_with_backoff(
                dataset_factory.create_train_df,
                concepts, num_of_examples, concept_genres_map, contrast_concepts_map
            )
            current_df["group_id"] = group_idx
        except Exception as e:
            logger.warning(f"Failed to create training data for group {group_idx}: {e}")
            return
        
        # Save the generated DataFrame, metadata, and current state
        save(dump_dir, {"group_idx": group_idx + 1}, group_idx, concepts, refs, "train", current_df, rotation_freq)

    logger.warning(f"Finished creating dataset.")

if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Generate script for creating the training dataset for concepts.")
    
    # Define the arguments
    parser.add_argument("--dump_dir", type=str, help="Path to the dump directory")
    parser.add_argument("--concept_path", type=str, help="Path to the concept file")
    parser.add_argument("--num_of_examples", type=int, help="The number of examples")
    parser.add_argument("--rotation_freq", type=int, help="Frequency for chunking files", default=DEFAULT_ROTATION_FREQ)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for shuffling concepts")
    parser.add_argument("--max_concepts", type=int, default=None, help="Maximum number of concepts to use")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Run the main function with parsed arguments
    main(
        args.dump_dir,
        args.concept_path,
        args.num_of_examples,
        args.rotation_freq,
        args.seed,
        args.max_concepts
    )