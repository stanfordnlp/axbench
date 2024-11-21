# generate script for creating the training dataset for concepts.
# we assume we group two concepts into a learning group.
# it is possible to extend to more concepts into the same group,
# although more training data will likely to be needed.
# 
# example launch command:
#    python axbench/scripts/generate.py --config axbench/demo/sweep/generate.yaml

import shutil
import sys
import argparse
import time
import os
import pickle
import random
import json
import csv
import atexit

import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from axbench.utils.dataset import DatasetFactory
from args.dataset_args import DatasetArgs
from pathlib import Path
from openai import AsyncOpenAI
import httpx, asyncio
from transformers import set_seed

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
        seen_index = set()
        for concept in json_concepts:
            model = concept["modelId"]
            sae_model = concept["layer"]
            subspace_id = concept["index"]
            if subspace_id in seen_index:
                continue # if there are multiple descriptions, we only take the first one.
            seen_index.add(subspace_id)
            sae_concepts += [concept["description"].strip()]
            concepts += [f"https://www.neuronpedia.org/{model}/{sae_model}/{subspace_id}"]
        return sae_concepts, concepts
    else:
        raise ValueError(f"Unsupported file type: {dump_dir}.")  


def save(
    dump_dir, state, concept_id, 
    concept, concept_genres_map, 
    ref, partition, current_df):
    """
    Save the current state, metadata, and DataFrame using Parquet format.
    """    
    # Save state
    state_path = os.path.join(dump_dir, STATE_FILE)
    with open(state_path, "wb") as f:
        pickle.dump(state, f)
    
    # Save metadata
    metadata_path = os.path.join(dump_dir, METADATA_FILE)
    metadata_entry = {
        "concept_id": concept_id,
        "concept": concept,
        "ref": ref,
        "concept_genres_map": concept_genres_map,
    }
    with open(metadata_path, "a") as f:
        f.write(json.dumps(metadata_entry) + "\n")
    
    # Save DataFrame using Parquet
    df_path = os.path.join(dump_dir, f"{partition}_data.parquet")
    if os.path.exists(df_path):
        existing_df = pd.read_parquet(df_path)
        combined_df = pd.concat([existing_df, current_df], ignore_index=True)
    else:
        combined_df = current_df
    combined_df.to_parquet(df_path, index=False)


def load_state(dump_dir):
    """
    Load the state from a file if it exists.
    
    Args:
        dump_dir (str): The directory to load the state file from.
    
    Returns:
        dict: The loaded state dictionary, or None if no state file exists.
    """
    state_path = os.path.join(Path(dump_dir), STATE_FILE)
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            state = pickle.load(f)
            return state
    return None

def main():
    args = DatasetArgs(section="generate")
    logger.warning("Generating datasets with the following configuration:")
    logger.warning(args)

    dump_dir = args.dump_dir
    dump_dir = Path(dump_dir) / "generate"
    dump_dir.mkdir(parents=True, exist_ok=True)

    concept_path = args.concept_path
    num_of_examples = args.num_of_examples
    max_concepts = args.max_concepts

    # Load and optionally shuffle concepts
    set_seed(args.seed)
    all_concepts, all_refs = load_concepts(concept_path)
    
    # Limit the number of concepts if specified
    if max_concepts is not None:
        combined = list(zip(all_concepts, all_refs))
        random.shuffle(combined)
        all_concepts, all_refs = zip(*combined)
        all_concepts = list(all_concepts)[:max_concepts]
        all_refs = list(all_refs)[:max_concepts]
    
    concept2id = {concept: i for i, concept in enumerate(all_concepts)}
    concepts = list(zip(all_concepts, all_refs))

    # Load the state if it exists.
    state = load_state(dump_dir)
    start_concept_id = state.get("concept_id", 0) if state else 0
    logger.warning(f"Starting group index: {start_concept_id}")
    if start_concept_id >= len(concepts):
        logger.warning(f"Datasets for all concepts have been generated. Exiting.")
        return

    # Create a new OpenAI client.
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=60.0,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=100, 
                max_connections=1000
            ),
            headers={"Connection": "close"},
        ),
        max_retries=3,
    )

    # Load lm and tokenizer.
    model_name = model_name_map[all_refs[0].split("/")[3]]
    tokenizer =  AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    tokenizer.padding_side = "right"

    # Init the dataset factory.
    dataset_factory = DatasetFactory(
        None, client, tokenizer, dump_dir, 
        use_cache=True, master_data_dir=args.master_data_dir,
        seed=args.seed, lm_model=args.lm_model
    )
    atexit.register(dataset_factory.save_cache)
    atexit.register(dataset_factory.reset_stats)

    progress_bar = tqdm(range(start_concept_id, len(concepts)), desc="Processing concept groups")
    data_concept_id = start_concept_id
    for concept_id in progress_bar:
        print(f"Generating concept {concept_id}...")
        concept, ref = concepts[concept_id]
        
        # prepare concept related data.
        concept_genres_map = \
            dataset_factory.prepare_genre_concepts([concept])
        # generate with retry mechanism.
        # try:
        current_df = dataset_factory.create_train_df(
            concept, num_of_examples, concept_genres_map,
            input_length=args.input_length, output_length=args.output_length,
            current_concept_id=data_concept_id
        )
        current_df["concept_id"] = data_concept_id
        # except Exception as e:
        #     logger.warning(f"Failed to create training data for group {concept_id}: {e}")
        #     continue # continue to the next group.
        
        # Save the generated DataFrame, metadata, and current state
        save(
            dump_dir, {"concept_id": concept_id + 1}, data_concept_id,
            concept, concept_genres_map, 
            ref, "train", current_df)
        data_concept_id += 1

    logger.warning(f"Finished creating dataset.")

if __name__ == "__main__":
    main()

