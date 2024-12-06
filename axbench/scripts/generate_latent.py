# inference.py: Inference with existing subspaces.
#
# example launch command:
#     torchrun --nproc_per_node=NUM_GPUS axbench/scripts/inference.py --config axbench/demo/sweep/inference.yaml --mode latent
import os, argparse, yaml, json, glob, pickle, time, itertools
import shutil
import pandas as pd
from tqdm.auto import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import atexit

from axbench.utils.dataset import (
    DatasetFactory,
    SteeringDatasetFactory
)
from axbench.utils.constants import * 
from axbench.utils.model_utils import get_prefix_length, get_suffix_length
from args.dataset_args import DatasetArgs
from args.training_args import TrainingArgs
from transformers import set_seed

# all supported methods
import axbench
from openai import AsyncOpenAI
import httpx, asyncio

import logging
import torch.distributed as dist
import sys

# Initialize the logger
logger = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_DELAY = 1  # in seconds
STATE_FILE = "inference_state.pkl"
CONFIG_FILE = "config.json"
METADATA_FILE = "metadata.jsonl"
STEERING_EXCLUDE_MODELS = {}
LATENT_EXCLUDE_MODELS = {"PromptSteering", "PromptBaseline", "DiReFT", "LoReFT", "LoRA", "SFT"}
LATENT_PROMPT_PREFIX = "Generate a random sentence."


def load_config(config_path):
    """
    Load metadata from a JSON lines file.
    """
    if not os.path.exists(Path(config_path) / CONFIG_FILE):
        return None
    with open(Path(config_path) / CONFIG_FILE) as f:
        d = json.load(f)
    return d


def load_state(dump_dir, mode):
    """
    Load the state from a file if it exists.
    """
    state_path = os.path.join(f"{dump_dir}/inference", f"{mode}_{STATE_FILE}")
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            return pickle.load(f)
    return None


def save_state(dump_dir, state, partition):
    dump_dir = Path(dump_dir) / "inference"
    dump_dir.mkdir(parents=True, exist_ok=True)
    # Save state
    state_path = os.path.join(dump_dir, f"{partition}_{STATE_FILE}")
    with open(state_path, "wb") as f:
        pickle.dump(state, f)


def load_metadata_flatten(metadata_path):
    """
    Load flatten metadata from a JSON lines file.
    """
    metadata = []
    with open(Path(metadata_path) / METADATA_FILE, 'r') as f:
        for line in f:
            data = json.loads(line)
            concept, ref =data["concept"], data["ref"]
            concept_genres_map = data["concept_genres_map"][concept]
            ref = data["ref"]
            flatten_data = {
                "concept": concept,
                "ref": ref,
                "concept_genres_map": {concept: concept_genres_map},
                "concept_id": data["concept_id"]
            }
            metadata += [flatten_data]  # Return the metadata as is
    return metadata


def save(
    dump_dir, concept_id, partition,
    current_df):
    # This function saves DataFrames per rank per partition (latent or steering)
    dump_dir = Path(dump_dir) / "inference"
    dump_dir.mkdir(parents=True, exist_ok=True)
    
    # Save DataFrame using Parquet
    rotation_freq = 500
    file_index = concept_id // rotation_freq
    if file_index == 0:
        df_path = os.path.join(dump_dir, f"{partition}_eval_data.parquet")
    else:
        df_path = os.path.join(dump_dir, f"{partition}_eval_data_{file_index}.parquet")
    if os.path.exists(df_path):
        existing_df = pd.read_parquet(df_path)
        combined_df = pd.concat([existing_df, current_df], ignore_index=True)
    else:
        combined_df = current_df
    combined_df.to_parquet(df_path, index=False)


def create_data_latent(dataset_factory, metadata, concept_id, num_of_examples, args):
    # prepare concept related data.
    concept = metadata[concept_id]["concept"]
    sae_link = metadata[concept_id]["ref"]
    sae_id = int(sae_link.split("/")[-1]) 
    concept_genres_map = metadata[concept_id]["concept_genres_map"]
    _, eval_contrast_concepts_map = \
        dataset_factory.prepare_concepts(
            [concept], 
            concept_genres_map=concept_genres_map,
            contrast_concepts_map={}, api_tag="inference")
    current_df = dataset_factory.create_eval_df(
        [concept], num_of_examples, concept_genres_map, {},
        eval_contrast_concepts_map, input_length=args.input_length, 
        output_length=args.output_length
    )
    current_df["concept_id"] = concept_id
    current_df["sae_link"] = sae_link
    current_df["sae_id"] = sae_id
    return current_df


def generate_latent(args, logger, generate_args):
    data_dir = args.data_dir
    dump_dir = args.dump_dir
    num_of_examples = args.latent_num_of_examples
    metadata = load_metadata_flatten(data_dir)
    # Get list of all concept_ids
    concept_ids = list(range(len(metadata)))

    # Load the state if it exists.
    state = load_state(args.dump_dir, "latent")
    start_concept_id = state.get("concept_id", 0) if state else 0
    logger.warning(f"Starting concept index: {start_concept_id}")
    if start_concept_id >= len(concept_ids):
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

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, model_max_length=512)
    tokenizer.padding_side = "right"

    # Load dataset factory for evals.
    dataset_factory = DatasetFactory(
        None, client, tokenizer, generate_args.dataset_category, None, None, dump_dir,
        use_cache=True, master_data_dir=args.master_data_dir,
        lm_model=args.lm_model, logger=logger, is_inference=True
    )
    atexit.register(dataset_factory.save_cache)
    atexit.register(dataset_factory.reset_stats)

    progress_bar = tqdm(range(start_concept_id, len(metadata)), desc="Processing concept")
    for start_idx in progress_bar:
        concept_id = metadata[start_idx]["concept_id"]
        current_df = create_data_latent(
            dataset_factory, metadata, concept_id, num_of_examples, args)

        save(dump_dir, concept_id, 'latent', current_df)
        logger.warning(f"Saved inference dataset for concept {concept_id} to latent_eval_data.parquet")
        # After processing, save state
        current_state = {'concept_id': concept_id}
        save_state(args.dump_dir, current_state, 'latent')


def main():
    custom_args = [
        {
            'args': ['--mode'],
            'kwargs': {
                'type': str,
                'default': "all",
                'help': 'The inference mode.'
            }
        }
    ]
    generate_args = DatasetArgs(custom_args=custom_args, section="generate")
    args = DatasetArgs(custom_args=custom_args, section="inference")
    args.data_dir = f"{args.dump_dir}/generate"
    logger.warning("Inferencing with following configuration:")
    logger.warning(args)
    set_seed(args.seed)

    # Configure the logger per rank
    logger.setLevel(logging.WARNING)  # Set the logging level as desired

    # Create a logging formatter that includes the rank
    formatter = logging.Formatter(
        fmt=f'%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S'
    )

    # Create a console handler and set its formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    if not logger.handlers:
        logger.addHandler(console_handler)

    # Optionally, create a file handler per rank
    """
    log_file = f'log_rank_{rank}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    """
    generate_latent(args, logger, generate_args)

if __name__ == "__main__":
    main()

