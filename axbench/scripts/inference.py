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


import os, sys, argparse, yaml, json, glob, pickle, time
import pandas as pd
from tqdm.auto import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import atexit

from pyreax import (
    EXAMPLE_TAG,
    ReAXFactory,
)

from args.dataset_args import DatasetArgs
from transformers import set_seed

import axbench
from axbench import SteeringDatasetFactory
from openai import AsyncOpenAI
import httpx, asyncio

import torch.multiprocessing as mp
from torch.multiprocessing import Queue, Process
from queue import Empty

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
DEFAULT_ROTATION_FREQ = 1000
STEERING_EXCLUDE_MODELS = {}
LATENT_EXCLUDE_MODELS = {"PromptSteering"}

def load_config(config_path):
    with open(Path(config_path) / CONFIG_FILE) as f:
        d = json.load(f)
    return d

def load_state(dump_dir, mode):
    state_path = os.path.join(f"{dump_dir}/inference", f"{mode}_{STATE_FILE}")
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            return pickle.load(f)
    return None

def load_metadata_flatten(metadata_path):
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
                metadata += [flatten_data]
            group_id += 1
    return metadata

def save(
    dump_dir, state, concept_id, partition,
    current_df, rotation_freq):
    """
    Save the current state, metadata, and DataFrame using Parquet format.
    
    Args:
        dump_dir (str): Directory to save the files
        state (dict): State dictionary to save
        concept_id (int): Current concept ID
        partition (str): Partition name (e.g., 'latent' or 'steering')
        current_df (pd.DataFrame): Current DataFrame to save
        rotation_freq (int): Frequency to rotate fragment files
    """
    dump_dir = Path(dump_dir) / "inference"
    dump_dir.mkdir(parents=True, exist_ok=True)

    # Save state
    state_path = os.path.join(dump_dir, f"{partition}_{STATE_FILE}")
    with open(state_path, "wb") as f:
        pickle.dump(state, f)

    # Save DataFrame
    fragment_index = concept_id // rotation_freq
    df_path = os.path.join(dump_dir, f"{partition}_data_fragment_{fragment_index}.parquet")

    if os.path.exists(df_path):
        existing_df = pd.read_parquet(df_path)
        combined_df = pd.concat([existing_df, current_df], ignore_index=True)
    else:
        combined_df = current_df

    combined_df.to_parquet(df_path, engine='pyarrow')

def create_data_latent(dataset_factory, metadata, concept_id, num_of_examples, args):
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

def create_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.steering_model_name, model_max_length=512)
    tokenizer.padding_side = "right"
    return tokenizer

def create_base_model(args, device):
    logger.warning(f"Loading base model to {device}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.steering_model_name,
        device_map=device
    )
    base_model.config.use_cache = False
    return base_model.eval()

def setup_benchmark_model(model_class, base_model, tokenizer, layer, metadata, mode, args):
    model = model_class(
        base_model, tokenizer, layer=layer,
        low_rank_dimension=len(metadata)
    ).to(base_model.device)

    load_kwargs = {"mode": mode} if mode == "steering" else {}
    model.load(
        dump_dir=args.train_dir,
        sae_path=metadata[0]["ref"],
        **load_kwargs
    )

    if mode == "steering":
        model.pre_compute_mean_activations(
            os.path.join(args.dump_dir, "inference"),
            master_data_dir=args.master_data_dir
        )

    return model

def infer_latent(args):
    data_dir = args.data_dir
    train_dir = args.train_dir
    dump_dir = args.dump_dir
    num_of_examples = args.latent_num_of_examples
    rotation_freq = args.rotation_freq
    config = load_config(train_dir)
    metadata = load_metadata_flatten(data_dir)
    layer = config["layer"]

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

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    base_model = create_base_model(args, device)
    tokenizer = create_tokenizer(args)

    benchmark_models = {}
    for model_name in args.models:
        if model_name in LATENT_EXCLUDE_MODELS:
            continue
        model_class = getattr(axbench, model_name)
        model = setup_benchmark_model(
            model_class, base_model, tokenizer, layer,
            metadata, "latent", args
        )
        logger.info(f"Model {model_name} initialized on {device}")

        benchmark_models[model_name] = model

    dataset_factory = ReAXFactory(
        base_model, client, tokenizer, dump_dir,
        use_cache=True, master_data_dir=args.master_data_dir,
        lm_model=args.lm_model
    )
    atexit.register(dataset_factory.save_cache)
    atexit.register(dataset_factory.reset_stats)

    state = load_state(args.dump_dir, "latent")
    start_concept_id = state.get("concept_id", 0) if state else 0

    progress_bar = tqdm(
        range(start_concept_id, len(metadata)),
        initial=start_concept_id,
        total=len(metadata),
        desc="Infer Latent"
    )

    for concept_id in progress_bar:
        # Create data for current concept
        current_df = create_data_latent(
            dataset_factory, metadata, concept_id, num_of_examples, args)

        # Process concept
        for model_name, model in benchmark_models.items():
            results = model.predict_latent(current_df)
            for k, v in results.items():
                current_df[f"{model_name}_{k}"] = v

        # Save results
        save(dump_dir, {"concept_id": concept_id + 1}, concept_id, "latent",
                current_df, rotation_freq)
    # Clean up GPU memory
    torch.cuda.empty_cache()

def infer_steering(args):
    data_dir = args.data_dir
    train_dir = args.train_dir
    dump_dir = args.dump_dir
    num_of_examples = args.steering_num_of_examples
    rotation_freq = args.rotation_freq
    config = load_config(train_dir)
    metadata = load_metadata_flatten(data_dir)
    layer = config["layer"]
    n_steering_factors = args.n_steering_factors
    steering_datasets = args.steering_datasets

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

    # Create base model and tokenizer
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    base_model = create_base_model(args, device)
    tokenizer = create_tokenizer(args)

    # Create benchmark models
    benchmark_models = {}
    for model_name in args.models:
        if model_name in STEERING_EXCLUDE_MODELS:
            continue
        model_class = getattr(axbench, model_name)
        model = setup_benchmark_model(
            model_class, base_model, tokenizer, layer,
            metadata, "steering", args
        )
        benchmark_models[model_name] = model

    # Create dataset factory
    dataset_factory = SteeringDatasetFactory(
        base_model, tokenizer, dump_dir,
        master_data_dir=args.master_data_dir,
        lm_client=client,
        lm_model=args.lm_model
    )

    state = load_state(args.dump_dir, "steering")
    start_concept_id = state.get("concept_id", 0) if state else 0

    progress_bar = tqdm(
        range(start_concept_id, len(metadata)),
        initial=start_concept_id,
        total=len(metadata),
        desc="Infer Steering"
    )

    for concept_id in progress_bar:
        # Create data for current concept
        current_df, (_, sae_link, sae_id) = create_data_steering(
            dataset_factory, metadata, concept_id,
            num_of_examples,
            n_steering_factors,
            steering_datasets, args)

        # Process concept
        for model_name, model in benchmark_models.items():
            results = model.predict_steer(
                current_df,
                concept_id=concept_id,
                sae_link=sae_link,
                sae_id=sae_id,
                batch_size=args.steering_batch_size,
                eval_output_length=args.steering_output_length
            )
            for k, v in results.items():
                current_df[f"{model_name}_{k}"] = v

        # Save results
        save(dump_dir, {"concept_id": concept_id + 1}, concept_id, "steering",
                current_df, rotation_freq)
    # Clean up GPU memory
    torch.cuda.empty_cache()


def worker_process(model_name, gpu_id, task_queue, result_queue, args, metadata, layer, shared_base_model=None):
    # Set GPU environment
    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'

    # Load base model and tokenizer
    if shared_base_model is not None:
        base_model = shared_base_model
    else:
        base_model = create_base_model(args, device)
        base_model.eval()
    tokenizer = create_tokenizer(args)

    # Create benchmark model
    model_class = getattr(axbench, model_name)
    model = setup_benchmark_model(
        model_class, base_model, tokenizer, layer,
        metadata, "steering", args
    )

    while True:
        task = task_queue.get()
        if task is None:  # Exit signal
            break
        concept_id, current_df, sae_link, sae_id = task
        results = model.predict_steer(
            current_df,
            concept_id=concept_id,
            sae_link=sae_link,
            sae_id=sae_id,
            batch_size=args.steering_batch_size,
            eval_output_length=args.steering_output_length
        )
        result_queue.put((model_name, concept_id, results))

def infer_steering_multi_gpu(args):
    data_dir = args.data_dir
    train_dir = args.train_dir
    dump_dir = args.dump_dir
    num_of_examples = args.latent_num_of_examples
    rotation_freq = args.rotation_freq
    config = load_config(train_dir)
    metadata = load_metadata_flatten(data_dir)
    layer = config["layer"]
    n_steering_factors = args.n_steering_factors
    steering_datasets = args.steering_datasets

    factory_tokenizer = create_tokenizer(args)

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

    # Create dataset factory and base model on GPU 0
    base_model = create_base_model(args, "cuda:0")
    dataset_factory = SteeringDatasetFactory(
        base_model, factory_tokenizer, dump_dir,
        master_data_dir=args.master_data_dir,
        lm_client=client,
        lm_model=args.lm_model
    )

    # Set up multiprocessing
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        num_gpus = 1  # Use CPU if no GPU is available

    task_queue = Queue()
    result_queue = Queue()

    processes = []
    model_names = [m for m in args.models if m not in STEERING_EXCLUDE_MODELS]
    num_workers = len(model_names)

    for idx, model_name in enumerate(model_names):
        gpu_id = idx % num_gpus if torch.cuda.is_available() else -1
        p = Process(target=worker_process, args=(
            model_name, gpu_id, task_queue, result_queue, args, metadata, layer,
            base_model if gpu_id == 0 else None))
        p.start()
        processes.append(p)

    state = load_state(args.dump_dir, "steering")
    start_concept_id = state.get("concept_id", 0) if state else 0

    progress_bar = tqdm(
        range(start_concept_id, len(metadata)),
        initial=start_concept_id,
        total=len(metadata),
        desc="Processing concepts in multi-GPU steering"
    )

    for concept_id in progress_bar:
        # Create data for current concept
        current_df, (_, sae_link, sae_id) = create_data_steering(
            dataset_factory, metadata, concept_id,
            num_of_examples,
            n_steering_factors,
            steering_datasets, args)

        # Distribute task to all workers
        for _ in range(num_workers):
            task_queue.put((concept_id, current_df, sae_link, sae_id))

        # Collect results from all workers for this concept
        results_for_concept = {}
        remaining_workers = num_workers
        while remaining_workers > 0:
            try:
                model_name, c_id, results = result_queue.get(timeout=30)
                if results is not None:
                    results_for_concept[model_name] = results
                    remaining_workers -= 1
            except Empty:
                # Check if any worker died
                if any(not p.is_alive() for p in processes):
                    logger.error("One or more workers failed, terminating all processes")
                    # Terminate all processes
                    for p in processes:
                        if p.is_alive():
                            p.terminate()
                    # Wait for all processes to finish
                    for p in processes:
                        p.join()
                    # Clean up GPU memory
                    torch.cuda.empty_cache()
                    # Exit the program
                    sys.exit(1)
                # All workers still alive, continue waiting
                continue

        # Update DataFrame with results
        for model_name, results in results_for_concept.items():
            for k, v in results.items():
                current_df[f"{model_name}_{k}"] = v

        # Save results
        save(dump_dir, {"concept_id": concept_id + 1}, concept_id, "steering",
                current_df, rotation_freq)

    # Send exit signal to all workers
    for _ in range(num_workers):
        task_queue.put(None)

    # Clean up workers
    for p in processes:
        p.join()
    torch.cuda.empty_cache()


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
    args = DatasetArgs(custom_args=custom_args)
    if not args.steering_model_name:
        args.steering_model_name = args.model_name

    logger.warning("Inferencing with following configuration:")
    logger.warning(args)
    set_seed(args.seed)

    def check_latent_eval_done(args):
        # Check if at least one latent eval fragment exists.
        if os.path.exists(os.path.join(
            args.dump_dir, "inference", "latent_data_fragment_0.parquet")):
            return True
        return False

    if args.mode == "latent":
        infer_latent(args)
    elif args.mode == "steering":
        # steering eval must be done after latent eval.
        if not check_latent_eval_done(args):
            raise ValueError("Latent eval must be done before steering eval.")
        if args.multi_gpu:
            mp.set_start_method('spawn', force=True)
            infer_steering_multi_gpu(args)
        else:
            infer_steering(args)

if __name__ == "__main__":
    main()
