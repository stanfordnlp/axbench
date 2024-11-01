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


import os, argparse, yaml, json, glob, pickle, time
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

from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_RETRIES = 5
RETRY_DELAY = 1  # in seconds
STATE_FILE = "inference_state.pkl"
CONFIG_FILE = "config.json"
METADATA_FILE = "metadata.jsonl"
DEFAULT_ROTATION_FREQ = 1000
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

def create_tokenizer(model_name):
    """Create a new tokenizer instance."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    return tokenizer

def parallel_model_evaluation(gpu_to_models, eval_func, model_name, *args):
    """Generic parallel evaluation function for models.
    
    Args:
        gpu_to_models: Dict mapping GPU IDs to lists of models
        eval_func: Evaluation function to run
        model_name: Name of the model for creating tokenizer
        *args: Additional arguments to pass to eval_func
    """
    all_results = {}
    
    # Create a thread pool for each GPU
    with ThreadPoolExecutor(max_workers=len(gpu_to_models)) as executor:
        future_to_gpu = {}
        
        # Create tokenizer for each thread
        tokenizers = {
            gpu_id: create_tokenizer(model_name) 
            for gpu_id in gpu_to_models.keys()
        }
        
        # Submit evaluation tasks for each GPU
        for gpu_id, models in gpu_to_models.items():
            for model in models:
                # Pass thread-local tokenizer to the model
                model.tokenizer = tokenizers[gpu_id]
                future = executor.submit(eval_func, model, *args)
                future_to_gpu[future] = (gpu_id, model.__class__.__name__)
        
        # Collect results
        for future in as_completed(future_to_gpu):
            gpu_id, model_name = future_to_gpu[future]
            try:
                model_name, results = future.result()
                all_results[model_name] = results
            except Exception as e:
                logger.error(f"GPU {gpu_id} model {model_name} failed: {str(e)}")
                
    return all_results

def evaluate_latent_on_gpu(model, df):
    """Single model latent evaluation function."""
    results = model.predict_latent(df)
    return model.__class__.__name__, results

def evaluate_steering_on_gpu(model, df, concept_id, sae_link, sae_id, args):
    """Single model steering evaluation function."""
    results = model.predict_steer(
        df, concept_id=concept_id, sae_link=sae_link, sae_id=sae_id,
        batch_size=args.steering_batch_size, 
        eval_output_length=args.steering_output_length
    )
    return model.__class__.__name__, results

def clear_gpu_cache(gpu_id):
    """Clear the cache of a specified GPU."""
    with torch.cuda.device(f'cuda:{gpu_id}'):
        torch.cuda.empty_cache()

def manage_gpu_memory(gpu_to_models):
    """Manage the memory of all GPUs."""
    for gpu_id in gpu_to_models.keys():
        clear_gpu_cache(gpu_id)

def create_base_model(model_name, device):
    """Create and initialize a base model on specified device."""
    logger.warning(f"Loading base model to {device}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.steering_model_name if args.steering_model_name else args.model_name, 
        device_map=device
    )
    base_model.config.use_cache = False
    return base_model.eval()

def create_benchmark_model(model_name, base_model, tokenizer, layer, metadata, 
                         device, train_dir, dump_dir, master_data_dir=None, mode="latent"):
    """Create and initialize a benchmark model with all necessary setup."""
    model_class = getattr(axbench, model_name)
    logger.warning(f"Loading {model_class} to {device} for inference.\n")
    
    # Initialize model
    benchmark_model = model_class(
        base_model, tokenizer, layer=layer,
        low_rank_dimension=len(metadata)
    ).to(device)
    
    # Load weights
    load_kwargs = {"mode": mode} if mode == "steering" else {}
    benchmark_model.load(
        dump_dir=train_dir, 
        sae_path=metadata[0]["ref"],
        **load_kwargs
    )
    
    # Additional setup for steering mode
    if mode == "steering":
        benchmark_model.pre_compute_mean_activations(
            os.path.join(dump_dir, "inference"), 
            master_data_dir=master_data_dir
        )
    
    return benchmark_model


def create_and_distribute_models(model_names, args, tokenizer, layer, metadata, mode="latent"):
    """
    Create base model on device 0 for ReAXFactory and distribute other models to GPUs.
    Returns both the device 0 model and distributed models dict.
    """
    num_gpus = torch.cuda.device_count()
    has_gpu = num_gpus > 0
    device_0 = "cuda:0" if has_gpu else "cpu"
    
    # First create base model on device 0 for ReAXFactory
    base_model_device0 = create_base_model(
        args.steering_model_name if args.steering_model_name else args.model_name,
        device_0
    )
    
    if not args.multi_gpu:
        # Single GPU/CPU mode
        gpu_to_models = {0 if has_gpu else -1: []}
        
        # Create all benchmark models using device 0 model
        for model_name in model_names:
            if model_name in STEERING_EXCLUDE_MODELS:
                continue

            print(f"Creating benchmark model for {model_name} on {device_0}")
            benchmark_model = create_benchmark_model(
                model_name, base_model_device0, tokenizer, layer, metadata,
                device_0, args.train_dir, args.dump_dir, args.master_data_dir, mode
            )
            gpu_to_models[0 if has_gpu else -1].append(benchmark_model)
    
    else:
        # Multi-GPU mode
        gpu_to_models = {i: [] for i in range(num_gpus)} if has_gpu else {-1: []}
        base_models = {0: base_model_device0}  # Start with device 0 model
        
        # Distribute models across GPUs
        for idx, model_name in enumerate(model_names):
            gpu_id = idx % num_gpus if has_gpu else -1
            device = f"cuda:{gpu_id}" if has_gpu else "cpu"
            print(f"Creating base model for {model_name} on {device}")
            
            # Create base model for other GPUs if needed
            if gpu_id not in base_models and gpu_id != 0:
                base_models[gpu_id] = create_base_model(args.model_name, device)
            
            # Use appropriate base model
            current_base_model = base_models[gpu_id]
            
            benchmark_model = create_benchmark_model(
                model_name, current_base_model, tokenizer, layer, metadata,
                device, args.train_dir, args.dump_dir, args.master_data_dir, mode
            )
            
            gpu_to_models[gpu_id].append(benchmark_model)
    
    return base_model_device0, gpu_to_models

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
    
    factory_tokenizer = create_tokenizer(args.steering_model_name)

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
    progress_bar = tqdm(range(start_concept_id, len(metadata)), desc="Inferencing with concepts")
    # Get base model on device 0 and distributed models
    base_model_device0, gpu_to_models = create_and_distribute_models(
        args.models, args, factory_tokenizer, layer, metadata, mode="steering"
    )
    
    # Create dataset factory with device 0 model and its own tokenizer
    
    dataset_factory = SteeringDatasetFactory(
        base_model_device0, factory_tokenizer, dump_dir, 
        master_data_dir=args.master_data_dir, lm_client=lm_client,
        lm_model=args.lm_model
    )
    
    torch.cuda.empty_cache()
    for concept_id in progress_bar:
        # Create.
        current_df, (_, sae_link, sae_id) = create_data_steering(
            dataset_factory, metadata, concept_id, num_of_examples, 
            n_steering_factors, steering_datasets, args)

        print(f"Created {len(current_df)} examples for concept {concept_id} for steering eval.")

        # Pass model_name to parallel_model_evaluation
        all_results = parallel_model_evaluation(
            gpu_to_models, evaluate_steering_on_gpu, args.model_name,
            current_df, concept_id, sae_link, sae_id, args)
        
        # Update DataFrame.
        for model_name, results in all_results.items():
            for k, v in results.items():
                current_df[f"{model_name}_{k}"] = v
        
        manage_gpu_memory(gpu_to_models)

        # Save.
        save(dump_dir, {"concept_id": concept_id + 1}, concept_id, "steering",
            current_df, rotation_freq)

def infer_latent(args):
    data_dir = args.data_dir
    train_dir = args.train_dir
    dump_dir = args.dump_dir
    num_of_examples = args.latent_num_of_examples
    rotation_freq = args.rotation_freq
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
    factory_tokenizer = create_tokenizer(args.model_name)

    # Get base model on device 0 and distributed models
    base_model_device0, gpu_to_models = create_and_distribute_models(
        args.models, args, factory_tokenizer, layer, metadata, mode="latent"
    )
    
    # Create dataset factory with device 0 model and its own tokenizer
    dataset_factory = ReAXFactory(
        model, client, tokenizer, dump_dir,
        use_cache=True, master_data_dir=args.master_data_dir, 
        lm_model=args.lm_model
    )
    atexit.register(dataset_factory.save_cache)
    atexit.register(dataset_factory.reset_stats)    
    state = load_state(args.dump_dir, "latent")
    start_concept_id = state.get("concept_id", 0) if state else 0
    logger.warning(f"Starting concept index: {start_concept_id}")
    progress_bar = tqdm(range(start_concept_id, len(metadata)), desc="Inferencing with concepts")
    
    torch.cuda.empty_cache()
    for concept_id in progress_bar:
        # Create.
        current_df = create_data_latent(
            dataset_factory, metadata, concept_id, num_of_examples, args)

        # Pass model_name to parallel_model_evaluation for different tokenizers.
        all_results = parallel_model_evaluation(
            gpu_to_models, evaluate_latent_on_gpu, args.model_name, current_df)
        
        # Update DataFrame
        for model_name, results in all_results.items():
            for k, v in results.items():
                current_df[f"{model_name}_{k}"] = v
        
        manage_gpu_memory(gpu_to_models)
        
        # Save.
        save(dump_dir, {"concept_id": concept_id + 1}, concept_id, "latent",
            current_df, rotation_freq)

def main():
    custom_args = [
        {
            'args': ['--mode'],
            'kwargs': {
                'type': str,
                'default': "latent",
                'help': 'The inference mode.'
            }
        },
        {
            'args': ['--multi_gpu'],
            'kwargs': {
                'action': 'store_true',
                'help': 'Enable multi-GPU distribution. If false, all models will be on cuda:0 or cpu.'
            }
        }
    ]
    args = DatasetArgs(custom_args=custom_args)
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
        infer_steering(args)


if __name__ == "__main__":
    main()

