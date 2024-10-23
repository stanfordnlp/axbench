# generate script for creating the training dataset for concepts.
# we assume we group two concepts into a learning group.
# it is possible to extend to more concepts into the same group,
# although more training data will likely to be needed.
# 
# example launch command:
#    python benchmark/scripts/generate.py --config benchmark/demo/sweep/generate.yaml

try:
    # This library is our indicator that the required installs
    # need to be done.
    import pyreax

except ModuleNotFoundError:
    # relative import; better to pip install subctrl
    import sys
    sys.path.append("../../pyreax")
    import pyreax

import sys
import argparse
import time
import os
import pickle
import random
import json
import csv

import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from pyreax import ReAXFactory
from args.dataset_args import DatasetArgs
from pathlib import Path

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
STATE_FILE = "generate_state.pkl"
METADATA_FILE = "metadata.jsonl"
DEFAULT_ROTATION_FREQ = 1000


def load_concepts(dump_dir):
    sae_concepts = []
    if ".txt" in dump_dir:
        with open(dump_dir, 'r') as file:
            concepts = [line.strip() for line in file.readlines()]
        if concepts[0].startswith("http://") or concepts[0].startswith("https://"):
            logger.warning("Detect external links. Pull concept info from the link.")
            for concept in concepts:
                if "www.neuronpedia.org" not in concept:
                    raise ValueError(f"Pulling from {concept} is not supported.")
                sae_path = concept.split("https://www.neuronpedia.org/")[-1]
                sae_url = f"https://www.neuronpedia.org/api/feature/{sae_path}"
                headers = {"X-Api-Key": os.environ.get("NP_API_KEY")}
                response = requests.get(sae_url, headers=headers).json()
                explanation = response["explanations"][0]["description"]
                sae_concepts += [explanation.strip()]
            return sae_concepts, concepts
        return concepts, ["null"]*len(concepts)
    elif ".csv" in dump_dir:
        # for csv, then the format is <concept>,<url>
        # no http connection is needed
        concepts = []
        with open(dump_dir, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                sae_concepts += [row[0]]
                concepts += [row[1]]
        return sae_concepts, concepts
    elif ".json" in dump_dir:
        concepts = []
        # this must be a neuropedia export.
        with open(dump_dir, 'r') as file:
            json_concepts = json.load(file)
        for concept in json_concepts:
            sae_concepts += [concept["description"].strip()]
            model = concept["modelId"]
            sae_model = concept["layer"]
            subspace_id = concept["index"]
            concepts += [f"https://www.neuronpedia.org/{model}/{sae_model}/{subspace_id}"]
        return sae_concepts, concepts
    else:
        raise ValueError(f"Unsupported file type: {dump_dir}.")  


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

def save(
    dump_dir, state, group_id, 
    concepts, concept_genres_map, contrast_concepts_map, 
    refs, partition, current_df, rotation_freq):
    """
    Save the current state, metadata, and DataFrame.
    """
    # handle training df first
    dump_dir = Path(dump_dir) / "generate"
    dump_dir.mkdir(parents=True, exist_ok=True)
    
    # Save state
    state_path = os.path.join(dump_dir, STATE_FILE)
    with open(state_path, "wb") as f:
        pickle.dump(state, f)
    
    # Save metadata
    metadata_path = os.path.join(dump_dir, METADATA_FILE)
    metadata_entry = {
        "group_id": group_id,
        "concepts": concepts,
        "refs": refs,
        "concept_genres_map": concept_genres_map,
        "contrast_concepts_map": contrast_concepts_map,
    }
    with open(metadata_path, "a") as f:
        f.write(json.dumps(metadata_entry) + "\n")
    
    # Save DataFrame
    fragment_index = group_id // rotation_freq
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

def main():
    args = DatasetArgs()
    logger.warning("Generating datasets with the following configuration:")
    logger.warning(args)

    dump_dir = args.dump_dir
    concept_path = args.concept_path
    num_of_examples = args.num_of_examples
    rotation_freq = args.rotation_freq
    seed = args.seed
    max_concepts = args.max_concepts
    
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
    start_group_id = state.get("group_id", 0) if state else 0
    logger.warning(f"Starting group index: {start_group_id}")
    
    # Init the dataset factory.
    dataset_factory = ReAXFactory(model, tokenizer, dump_dir)
    progress_bar = tqdm(range(start_group_id, len(concept_groups)), desc="Processing concept groups")
    for group_id in progress_bar:
        concepts, refs = concept_groups[group_id]
        
        # prepare concept related data.
        concept_genres_map, contrast_concepts_map = \
            dataset_factory.prepare_concepts(concepts)
        
        # generate with retry mechanism.
        try:
            current_df = retry_with_backoff(
                dataset_factory.create_train_df,
                concepts, num_of_examples, concept_genres_map, contrast_concepts_map,
                input_length=args.input_length, output_length=args.output_length
            )
            current_df["group_id"] = group_id
        except Exception as e:
            logger.warning(f"Failed to create training data for group {group_id}: {e}")
            return
        
        # Save the generated DataFrame, metadata, and current state
        save(
            dump_dir, {"group_id": group_id + 1}, group_id, 
            concepts, concept_genres_map, contrast_concepts_map, 
            refs, "train", current_df, rotation_freq)

    logger.warning(f"Finished creating dataset.")

if __name__ == "__main__":
    main()

