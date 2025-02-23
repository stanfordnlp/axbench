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
import torch

import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from axbench.utils.dataset import DatasetFactory
from args.dataset_args import DatasetArgs
from pathlib import Path
from openai import AsyncOpenAI
import httpx, asyncio
from transformers import set_seed
from axbench.utils.constants import * 

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)

model_name_map = {
    "gemma-2-2b": "google/gemma-2-2b-it",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
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


def save_df_to_parquet_safely(df, final_path):
    import tempfile
    import os
    
    # Create temporary file in the same directory as the target
    dirname = os.path.dirname(os.path.abspath(final_path))
    with tempfile.NamedTemporaryFile(delete=False, dir=dirname, suffix='.parquet.tmp') as tmp:
        temp_path = tmp.name
        try:
            # Write to temporary file first
            df.to_parquet(temp_path, index=False)
            # Ensure data is written to disk
            os.fsync(tmp.fileno())
        except Exception as e:
            os.unlink(temp_path)  # Clean up temp file
            raise e
    
    try:
        # Atomic rename operation
        os.rename(temp_path, final_path)
    except Exception as e:
        os.unlink(temp_path)  # Clean up temp file
        raise e
    

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
    dump_dir, state, concept_id, 
    concept, concept_genres_map, 
    ref, partition, current_df, dataset_factory):
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
    rotation_freq = 500
    file_index = concept_id // rotation_freq
    if file_index == 0:
        df_path = os.path.join(dump_dir, f"{partition}_data.parquet")
    else:
        df_path = os.path.join(dump_dir, f"{partition}_data_{file_index}.parquet")
    if os.path.exists(df_path):
        existing_df = pd.read_parquet(df_path)
        combined_df = pd.concat([existing_df, current_df], ignore_index=True)
    else:
        # first time cache, we need to add global negative examples.
        if concept_id == 0:
            combined_df = pd.concat([dataset_factory.negative_df, current_df], ignore_index=True)
        else:
            combined_df = current_df
    save_df_to_parquet_safely(combined_df, df_path)


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


def load_state_latent(dump_dir, mode):
    """
    Load the state from a file if it exists.
    """
    state_path = os.path.join(f"{dump_dir}/inference", f"{mode}_{STATE_FILE}")
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            return pickle.load(f)
    return None


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


def save_state_latent(dump_dir, state, partition):
    dump_dir = Path(dump_dir) / "inference"
    dump_dir.mkdir(parents=True, exist_ok=True)
    # Save state
    state_path = os.path.join(dump_dir, f"{partition}_{STATE_FILE}")
    with open(state_path, "wb") as f:
        pickle.dump(state, f)


def save_latent(
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


def generate_latent(generate_args, args):
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
    data_dir = args.data_dir
    dump_dir = args.dump_dir
    num_of_examples = args.latent_num_of_examples
    metadata = load_metadata_flatten(data_dir)
    # Get list of all concept_ids
    concept_ids = list(range(len(metadata)))

    # Load the state if it exists.
    state = load_state_latent(args.dump_dir, "latent")
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

        save_latent(dump_dir, concept_id, 'latent', current_df)
        logger.warning(f"Saved inference dataset for concept {concept_id} to latent_eval_data.parquet")
        # After processing, save state
        current_state = {'concept_id': concept_id}
        save_state_latent(args.dump_dir, current_state, 'latent')


def generate_training(args, generate_args):
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
    logger.warning(f"Starting concept index: {start_concept_id}")
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
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16)
    is_chat_model = True if model_name in CHAT_MODELS else False
    include_system_prompt = True if model_name == "meta-llama/Llama-3.1-8B-Instruct" else False
    model = model.cuda()

    tokenizer =  AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    tokenizer.padding_side = "right"

    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        need_resize = True
    else:
        need_resize = False
    if need_resize:
        model.resize_token_embeddings(len(tokenizer))

    # Init the dataset factory.
    dataset_factory = DatasetFactory(
        model, client, tokenizer, args.dataset_category, num_of_examples, args.output_length, 
        dump_dir, use_cache=args.lm_use_cache, master_data_dir=args.master_data_dir,
        seed=args.seed, lm_model=args.lm_model, start_concept_id=start_concept_id, is_chat_model=is_chat_model,
        include_system_prompt=include_system_prompt,
    )
    atexit.register(dataset_factory.save_cache)
    atexit.register(dataset_factory.reset_stats)

    progress_bar = tqdm(range(start_concept_id, len(concepts)), desc="Processing concept")
    only_one_concept = True if len(concepts) == 1 else False
    data_concept_id = start_concept_id
    for concept_id in progress_bar:
        concept, ref = concepts[concept_id]
        print(f"Generating for concept: {concept}...")

        # prepare concept related data.
        concept_genres_map = \
            dataset_factory.prepare_genre_concepts([concept])
        # generate with retry mechanism.
        # try:
        current_df = dataset_factory.create_train_df(
            concept, num_of_examples, concept_genres_map,
            output_length=args.output_length,
            current_concept_id=data_concept_id,
            only_one_concept=only_one_concept,
        )
        current_df["concept_id"] = data_concept_id
        # except Exception as e:
        #     logger.warning(f"Failed to create training data for group {concept_id}: {e}")
        #     continue # continue to the next group.
        
        # Save the generated DataFrame, metadata, and current state
        save(
            dump_dir, {"concept_id": concept_id + 1}, data_concept_id,
            concept, concept_genres_map, 
            ref, "train", current_df, dataset_factory)
        data_concept_id += 1

    logger.warning(f"Finished creating dataset.")


def save_dpo(
    dump_dir, concept_id, partition,
    current_df):
    # This function saves DataFrames per rank per partition (latent or steering)
    dump_dir = Path(dump_dir) / "generate"
    dump_dir.mkdir(parents=True, exist_ok=True)
    
    # Save DataFrame using Parquet
    rotation_freq = 500
    file_index = concept_id // rotation_freq if concept_id != -1 else 0
    if file_index == 0:
        df_path = os.path.join(dump_dir, f"{partition}_train_data.parquet")
    else:
        df_path = os.path.join(dump_dir, f"{partition}_train_data_{file_index}.parquet")
    if os.path.exists(df_path):
        existing_df = pd.read_parquet(df_path)
        combined_df = pd.concat([existing_df, current_df], ignore_index=True)
    else:
        combined_df = current_df
    combined_df.to_parquet(df_path, index=False)


def save_state_dpo(dump_dir, state, partition):
    dump_dir = Path(dump_dir) / "generate"
    dump_dir.mkdir(parents=True, exist_ok=True)
    # Save state
    state_path = os.path.join(dump_dir, f"{partition}_{STATE_FILE}")
    with open(state_path, "wb") as f:
        pickle.dump(state, f)


def generate_dpo_training(args, generate_args):
    dump_dir = args.dump_dir
    dump_dir = Path(dump_dir) / "generate"
    # check the generate directory exists.
    if not os.path.exists(dump_dir):
        raise ValueError(f"Generate directory does not exist: {dump_dir}")
    # check the train_data.parquet exists.
    if not os.path.exists(os.path.join(dump_dir, "train_data.parquet")):
        raise ValueError(f"Train data does not exist: {os.path.join(dump_dir, 'train_data.parquet')}")

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
    state = load_state_latent(dump_dir, "dpo")
    start_concept_id = state.get("concept_id", 0) if state else 0
    logger.warning(f"Starting concept index: {start_concept_id}")
    if start_concept_id >= len(concepts):
        logger.warning(f"Datasets for all concepts have been generated. Exiting.")
        return

    # Load lm and tokenizer.
    model_name = model_name_map[all_refs[0].split("/")[3]]
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16)
    is_chat_model = True if model_name in CHAT_MODELS else False
    include_system_prompt = True if model_name == "meta-llama/Llama-3.1-8B-Instruct" else False
    model = model.cuda()

    tokenizer =  AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    tokenizer.padding_side = "right"

    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        need_resize = True
    else:
        need_resize = False
    if need_resize:
        model.resize_token_embeddings(len(tokenizer))

    # Init the dataset factory.
    dataset_factory = DatasetFactory(
        model, None, tokenizer, args.dataset_category, num_of_examples, args.output_length, 
        dump_dir, use_cache=args.lm_use_cache, master_data_dir=args.master_data_dir,
        seed=args.seed, lm_model=args.lm_model, start_concept_id=start_concept_id, is_chat_model=is_chat_model,
        include_system_prompt=include_system_prompt,
    )
    atexit.register(dataset_factory.save_cache)
    atexit.register(dataset_factory.reset_stats)
    
    # get negative and do nothing on them just renaming.
    existing_df = pd.read_parquet(os.path.join(dump_dir, "train_data.parquet"))
    existing_df = existing_df.rename(
        columns={
            'output': 'winning_output', 
            'output_concept': 'winning_output_concept',
        }
    )
    negative_df = existing_df[existing_df["category"] == "negative"] # keep later to concate.
    negative_df["losing_output"] = negative_df["winning_output"]
    negative_df["losing_output_concept"] = negative_df["winning_output_concept"]
    save_dpo(dump_dir, -1, 'dpo', negative_df)

    progress_bar = tqdm(range(start_concept_id, len(concepts)), desc="Processing concept")
    for concept_id in progress_bar:
        concept, ref = concepts[concept_id]
        print(f"Generating for concept: {concept}...")
        existing_df = existing_df[existing_df["concept_id"] == concept_id].copy()
        dpo_df = dataset_factory.create_dpo_df(
            existing_df,
            output_length=args.output_length,
            is_chat_model=is_chat_model,
            include_system_prompt=include_system_prompt,
            batch_size=args.inference_batch_size,
        )

        save_dpo(dump_dir, concept_id, 'dpo', dpo_df)
        logger.warning(f"Saved inference dataset for concept {concept_id} to latent_eval_data.parquet")
        # After processing, save state
        current_state = {'concept_id': concept_id}
        save_state_dpo(args.dump_dir, current_state, 'latent')

    logger.warning(f"Finished creating DPO dataset.")


def main():
    custom_args = [
        {
            'args': ['--mode'],
            'kwargs': {
                'type': str,
                'default': "training",
                'help': 'The generation mode.'
            }
        }
    ]

    generate_args = DatasetArgs(custom_args=custom_args, section="generate")
    inference_args = DatasetArgs(custom_args=custom_args, section="inference")
    logger.warning("Generating datasets with the following configuration:")
    logger.warning(generate_args)

    if generate_args.mode == "training":
        generate_training(generate_args, inference_args)
    elif generate_args.mode == "latent":
        generate_latent(generate_args, inference_args)   
    elif generate_args.mode == "dpo_training":
        generate_dpo_training(generate_args, inference_args)
    else:
        raise ValueError(f"Invalid mode: {generate_args.mode}")


if __name__ == "__main__":
    main()

