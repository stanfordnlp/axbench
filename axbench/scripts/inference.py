# inference with existing subspaces.
#
# example launch command:
#     python axbench/scripts/inference.py --config axbench/demo/sweep/inference.yaml --mode latent

try:
    # This library is our indicator that the required installs
    # need to be done.
    import pyreax

except ModuleNotFoundError:
    # relative import; better to pip install subctrl
    import sys
    sys.path.append("../../pyreax")
    import pyreax


import os, argparse, yaml, json, glob, pickle, time, itertools
import shutil
import pandas as pd
from tqdm.auto import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import atexit
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

from pyreax import (
    EXAMPLE_TAG, 
    ReAXFactory,
)

from args.dataset_args import DatasetArgs
from transformers import set_seed

# all supported methods
import axbench
from axbench import SteeringDatasetFactory
from openai import AsyncOpenAI
import httpx, asyncio

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_DELAY = 1  # in seconds
STATE_FILE = "inference_state.pkl"
CONFIG_FILE = "config.json"
METADATA_FILE = "metadata.jsonl"
STEERING_EXCLUDE_MODELS = {}
LATENT_EXCLUDE_MODELS = {"PromptSteering"}


def load_config(config_path):
    """
    Load metadata from a JSON lines file.
    """
    with open(Path(config_path) / CONFIG_FILE) as f:
        d = json.load(f)
    return d


def load_state(dump_dir, mode):
    """
    Load the state from a file if it exists.
    
    Args:
        dump_dir (str): The directory to load the state file from.
    
    Returns:
        dict: The loaded state dictionary, or None if no state file exists.
    """
    state_path = os.path.join(f"{dump_dir}/inference", f"{mode}_{STATE_FILE}")
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            return pickle.load(f)
    return None


def load_metadata_flatten(metadata_path):
    """
    Load flatten metadata from a JSON lines file.
    """
    metadata = []
    group_id = 0
    with open(Path(metadata_path) / METADATA_FILE, 'r') as f:
        for line in f:
            data = json.loads(line)
            for concept_id, concept in enumerate(data["concepts"]):
                concept_genres_map = data["concept_genres_map"][concept]
                contrast_concepts_map = data["contrast_concepts_map"][concept]
                ref = data["refs"][concept_id]
                flatten_data = {
                    "concept": concept,
                    "ref": ref,
                    "concept_genres_map": {concept: concept_genres_map},
                    "contrast_concepts_map": {concept: contrast_concepts_map},
                    "group_id": group_id
                }
                metadata += [flatten_data]  # Return the metadata as is
            group_id += 1
    return metadata


def save(
    dump_dir, state, partition,
    current_df):
    dump_dir = Path(dump_dir) / "inference"
    dump_dir.mkdir(parents=True, exist_ok=True)
    
    # Save state
    state_path = os.path.join(dump_dir, f"{partition}_{STATE_FILE}")
    with open(state_path, "wb") as f:
        pickle.dump(state, f)
    
    # Save DataFrame
    df_path = os.path.join(dump_dir, f"{partition}_data.parquet")
    
    if os.path.exists(df_path):
        existing_df = pd.read_parquet(df_path)
        combined_df = pd.concat([existing_df, current_df], ignore_index=True)
    else:
        combined_df = current_df
    
    combined_df.to_parquet(df_path, engine='pyarrow')


def create_data_latent(dataset_factory, metadata, concept_id, num_of_examples, args):
    # prepare concept related data.
    concept = metadata[concept_id]["concept"]
    sae_link = metadata[concept_id]["ref"]
    group_id = metadata[concept_id]["group_id"]
    sae_id = int(sae_link.split("/")[-1]) 
    concept_genres_map = metadata[concept_id]["concept_genres_map"]
    contrast_concepts_map = metadata[concept_id]["contrast_concepts_map"]
    _, eval_contrast_concepts_map = \
        dataset_factory.prepare_concepts(
            [concept], 
            concept_genres_map=concept_genres_map,
            contrast_concepts_map=contrast_concepts_map, api_tag="inference")
    current_df = dataset_factory.create_eval_df(
        [concept], num_of_examples, concept_genres_map, contrast_concepts_map,
        eval_contrast_concepts_map, input_length=args.input_length,
    )
    current_df["concept_id"] = concept_id
    current_df["sae_link"] = sae_link
    current_df["sae_id"] = sae_id
    current_df["group_id"] = group_id
    return current_df


def create_data_steering(
    dataset_factory, metadata, concept_id, num_of_examples, 
    n_steering_factors, steering_datasets, args):
    # prepare concept related data.
    concept = metadata[concept_id]["concept"]
    sae_link = metadata[concept_id]["ref"]
    sae_id = int(sae_link.split("/")[-1]) 

    current_df = dataset_factory.create_eval_df(
        [concept], num_of_examples, n_steering_factors, steering_datasets,
    )
    current_df["concept_id"] = concept_id
    current_df["sae_link"] = sae_link
    current_df["sae_id"] = sae_id

    return current_df, (concept_id, sae_link, sae_id)


def infer_steering(args):

    data_dir = args.data_dir
    train_dir = args.train_dir
    dump_dir = args.dump_dir
    num_of_examples = args.steering_num_of_examples
    config = load_config(train_dir)
    metadata = load_metadata_flatten(data_dir)
    layer = config["layer"]
    steering_factors = args.steering_factors
    steering_datasets = args.steering_datasets

    # Create a new OpenAI client.
    lm_client = AsyncOpenAI(
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

    state = load_state(args.dump_dir, "steering")
    start_concept_id = state.get("concept_id", 0) if state else 0
    logger.warning(f"Starting concept index: {start_concept_id}")
    concept_ids = list(range(start_concept_id, len(metadata)))
    if len(concept_ids) == 0:
        logger.warning(f"No concept ids to infer. Exiting.")
        return
    
    # Initialize the dataset factory with the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.steering_model_name, use_fast=False)
    tokenizer.padding_side = "right"
    dataset_factory = SteeringDatasetFactory(
        tokenizer, dump_dir,
        master_data_dir=args.master_data_dir, lm_client=lm_client,
        lm_model=args.lm_model
    )

    # Initialize the available devices queue.
    available_devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    device_queue = queue.Queue()
    for device in available_devices:
        device_queue.put(device)

    device_lm_models = {}
    for device in available_devices:
        model_instance = AutoModelForCausalLM.from_pretrained(
            args.steering_model_name if args.steering_model_name else args.model_name
        )
        model_instance.config.use_cache = False
        model_instance = model_instance.eval()
        model_instance = model_instance.to(device)
        device_lm_models[device] = model_instance

    # Combine datasets, sae_links, and sae_ids into a single dictionary
    data_per_concept = {}
    for concept_id in concept_ids:
        current_df, (_, sae_link, sae_id) = create_data_steering(
            dataset_factory, metadata, concept_id, num_of_examples,
            steering_factors, steering_datasets, args
        )
        data_per_concept[concept_id] = (current_df, sae_link, sae_id)

    # Prepare tasks: list of (model_idx, model_name, concept_id)
    tasks = []
    for concept_id in concept_ids:
        for model_idx, model_name in enumerate(args.models):
            if model_name in STEERING_EXCLUDE_MODELS:
                continue
            tasks.append((model_idx, model_name, concept_id))

    # Dictionary to collect results per concept_id and model_name
    torch.cuda.empty_cache()
    results_per_concept = {}
    def run_predict_steer(task):
        model_idx, model_name, concept_id = task
        # Get an available device from the queue
        device = device_queue.get()
        try:
            model_class = getattr(axbench, model_name)
            logger.warning(f"Inference steering with {model_class} on {device} for concept {concept_id}.\n")
            # Use the pre-loaded model_instance
            model_instance = device_lm_models[device]
            # Instantiate a new tokenizer inside the thread to avoid sharing across threads
            thread_tokenizer = AutoTokenizer.from_pretrained(
                args.steering_model_name, use_fast=False
            )
            thread_tokenizer.padding_side = "right"
            benchmark_model = model_class(
                model_instance, thread_tokenizer, layer=layer,
                low_rank_dimension=len(metadata),
                device=device
            )
            benchmark_model.load(
                dump_dir=train_dir, sae_path=metadata[0]["ref"], mode="steering"
            )
            benchmark_model.to(device)
            # Pre-compute mean activations for steering evaluation.
            benchmark_model.pre_compute_mean_activations(
                os.path.join(dump_dir, "inference"), master_data_dir=args.master_data_dir
            )
            # Get the data for this concept_id
            current_df, sae_link, sae_id = data_per_concept[concept_id]
            # Run prediction
            results = benchmark_model.predict_steer(
                current_df, concept_id=concept_id, sae_link=sae_link, sae_id=sae_id,
                batch_size=args.steering_batch_size,
                eval_output_length=args.steering_output_length
            )
            # Collect results
            return (concept_id, model_name, results)
        finally:
            # Release the device back to the queue
            device_queue.put(device)
            # Clean up
            del benchmark_model
            torch.cuda.empty_cache()

    # Run the tasks in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(run_predict_steer, task): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            model_idx, model_name, concept_id = task
            concept_id_result, model_name_result, results = future.result()
            # Store the results
            if concept_id_result not in results_per_concept:
                results_per_concept[concept_id_result] = {}
            results_per_concept[concept_id_result][model_name_result] = results

    # After all tasks are done, update datasets with results and save
    for concept_id in concept_ids:
        current_df, _, _ = data_per_concept[concept_id]
        # Get the results for this concept_id
        model_results = results_per_concept[concept_id]
        for model_name, results in model_results.items():
            for k, v in results.items():
                current_df[f"{model_name}_{k}"] = v
        # Save the combined results
        save(dump_dir, {"concept_id": concept_id + 1}, "steering",
             current_df)


def infer_latent(args):
    data_dir = args.data_dir
    train_dir = args.train_dir
    dump_dir = args.dump_dir
    num_of_examples = args.latent_num_of_examples
    config = load_config(train_dir)
    metadata = load_metadata_flatten(data_dir)
    layer = config["layer"]

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

    state = load_state(args.dump_dir, "latent")
    start_concept_id = state.get("concept_id", 0) if state else 0
    logger.warning(f"Starting concept index: {start_concept_id}")
    concept_ids = list(range(start_concept_id, len(metadata)))
    if len(concept_ids) == 0:
        logger.warning(f"No concept ids to infer. Exiting.")
        return

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, model_max_length=512)
    tokenizer.padding_side = "right"

    # Load dataset factory for evals.
    dataset_factory = ReAXFactory(
        None, client, tokenizer, dump_dir,
        use_cache=True, master_data_dir=args.master_data_dir,
        lm_model=args.lm_model
    )
    atexit.register(dataset_factory.save_cache)
    atexit.register(dataset_factory.reset_stats)

    # Initialize the available devices.
    available_devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    num_devices = len(available_devices)
    device_queue = queue.Queue()
    for device in available_devices:
        device_queue.put(device)

    # Pre-load model instances per device
    device_lm_models = {}
    for device in available_devices:
        model_instance = AutoModelForCausalLM.from_pretrained(args.model_name)
        model_instance.config.use_cache = False
        model_instance = model_instance.eval()
        model_instance = model_instance.to(device)
        device_lm_models[device] = model_instance

    # Prepare tasks: list of (model_name, concept_id)
    datasets = {}
    tasks = []
    for concept_id in concept_ids:
        current_df = create_data_latent(
            dataset_factory, metadata, concept_id, num_of_examples, args)
        datasets[concept_id] = current_df
    for concept_id in concept_ids:
        for model_name in args.models:
            if model_name in LATENT_EXCLUDE_MODELS:
                continue
            tasks.append((model_name, concept_id))

    # Dictionary to collect results per concept_id and model_name
    torch.cuda.empty_cache()
    results_per_concept = {}
    def run_predict_latent(task):
        try:
            model_name, concept_id = task
            # Get an available device from the queue
            device = device_queue.get()

            model_class = getattr(axbench, model_name)
            logger.warning(f"Inference latent with {model_class} on {device} for concept {concept_id}.")
            # Use the pre-loaded model_instance
            model_instance = device_lm_models[device]
            # Instantiate a new tokenizer inside the thread to avoid sharing across threads
            thread_tokenizer = AutoTokenizer.from_pretrained(
                args.model_name, model_max_length=512)
            thread_tokenizer.padding_side = "right"
            benchmark_model = model_class(
                model_instance, thread_tokenizer, layer=layer,
                low_rank_dimension=len(metadata),
                device=device
            )
            benchmark_model.load(
                dump_dir=train_dir, sae_path=metadata[0]["ref"]
            )
            benchmark_model.to(device)
            # Get the dataset for this concept_id
            current_df = datasets[concept_id]
            # Run prediction
            results = benchmark_model.predict_latent(
                current_df, batch_size=args.latent_batch_size
            )
            return (concept_id, model_name, results)
        finally:
            # Release the device back to the queue
            device_queue.put(device)
            # Clean up
            del benchmark_model
            torch.cuda.empty_cache()

    # Run the tasks in parallel on available GPUs
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(run_predict_latent, task): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            model_name, concept_id = task
            concept_id_result, model_name_result, results = future.result()
            # Store the results
            if concept_id_result not in results_per_concept:
                results_per_concept[concept_id_result] = {}
            results_per_concept[concept_id_result][model_name_result] = results

    # After all tasks are done, update datasets with results and save
    for concept_id in concept_ids:
        current_df = datasets[concept_id]
        # Get the results for this concept_id
        model_results = results_per_concept[concept_id]
        for model_name, results in model_results.items():
            for k, v in results.items():
                current_df[f"{model_name}_{k}"] = v
        # Save the combined results
        save(dump_dir, {"concept_id": concept_id + 1}, "latent",
             current_df)


def main():
    custom_args = [
        {
            'args': ['--mode'],
            'kwargs': {
                'type': str,
                'default': "latent",
                'help': 'The inference mode.'
            }
        }
    ]
    args = DatasetArgs(custom_args=custom_args, section="inference")
    args.data_dir = f"{args.dump_dir}/generate"
    args.train_dir = f"{args.dump_dir}/train"
    logger.warning("Inferencing with following configuration:")
    logger.warning(args)
    set_seed(args.seed)

    def check_latent_eval_done(args):
        # Check if at least one latent eval fragment exists.
        if os.path.exists(os.path.join(
            args.dump_dir, "inference", "latent_data.parquet")):
            return True
        return False

    if args.mode == "latent":
        infer_latent(args)
    elif args.mode == "steering":
        # steering eval must be done after latent eval.
        if not check_latent_eval_done(args):
            raise ValueError("Latent eval must be done before steering eval.")
        infer_steering(args)


if __name__ == "__main__":
    main()