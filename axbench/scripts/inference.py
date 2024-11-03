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

from pyreax import (
    EXAMPLE_TAG, 
    ReAXFactory
)
from args.dataset_args import DatasetArgs

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

from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

MAX_RETRIES = 5
RETRY_DELAY = 1  # in seconds
STATE_FILE = "inference_state.pkl"
CONFIG_FILE = "config.json"
METADATA_FILE = "metadata.jsonl"
DEFAULT_ROTATION_FREQ = 1000


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
            contrast_concepts_map=contrast_concepts_map)
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
        [concept], num_of_examples, n_steering_factors, steering_datasets
    )
    current_df["concept_id"] = concept_id
    current_df["sae_link"] = sae_link
    current_df["sae_id"] = sae_id

    return current_df, (concept_id, sae_link, sae_id)

def create_tokenizer(model_name):
    """Create a new tokenizer instance."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    tokenizer.padding_side = "right"
    return tokenizer

def create_base_model(model_name, device):
    """Create and initialize a base model on specified device."""
    logger.warning(f"Loading base model to {device}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device
    )
    base_model.config.use_cache = False
    return base_model.eval()

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)  # Set spawn method

def setup_benchmark_model(model_class, base_model, tokenizer, layer, metadata,
                         train_dir, dump_dir, mode, master_data_dir=None,
                         model_name=None, gpu_id=None):
    """Setup and initialize a benchmark model with proper configurations."""
    # Create and move model to device
    model = model_class(
        base_model, tokenizer, layer=layer,
        low_rank_dimension=len(metadata)
    ).to(base_model.device)
    
    # Load weights with mode-specific kwargs
    load_kwargs = {"mode": mode} if mode == "steering" else {}
    model.load(
        dump_dir=train_dir,
        sae_path=metadata[0]["ref"],
        **load_kwargs
    )
    
    # Steering mode extra settings
    if mode == "steering":
        model.pre_compute_mean_activations(
            os.path.join(dump_dir, "inference"),
            master_data_dir=master_data_dir
        )
    
    # Log initialization if model_name and gpu_id are provided
    if model_name and gpu_id is not None:
        logger.info(f"Model {model_name} initialized on GPU {gpu_id} in {mode} mode")
        
    return model

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)  # Set spawn method

class SteeringModelWorker(mp.Process):
    """Model worker process that continuously processes tasks."""
    def __init__(self, model_name, gpu_id, args, metadata, layer, train_dir, dump_dir,
                 task_queue, result_queue, mode, base_model=None):
        super().__init__()
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.args = args
        self.metadata = metadata
        self.layer = layer
        self.train_dir = train_dir
        self.dump_dir = dump_dir
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.mode = mode
        self.base_model = base_model

    def run(self):
        try:
            # Set GPU environment
            torch.cuda.set_device(self.gpu_id)
            device = f'cuda:{self.gpu_id}'

            # Create model components
            tokenizer = create_tokenizer(self.args.model_name)

            # If base_model is not provided, create a new one
            if self.base_model is None:
                base_model = create_base_model(self.args.model_name, device)
            else:
                base_model = self.base_model
                logger.info(f"Using shared base_model for {self.model_name}")

            # Create benchmark model
            model_class = getattr(axbench, self.model_name)
            benchmark_model = setup_benchmark_model(
                model_class, base_model, tokenizer, self.layer,
                self.metadata, self.train_dir, self.dump_dir,
                self.mode, self.args.master_data_dir,
                model_name=self.model_name, gpu_id=self.gpu_id
            )

            logger.info(f"Model {self.model_name} initialized on GPU {self.gpu_id} in {self.mode} mode")

            # Process tasks continuously
            while True:
                task = self.task_queue.get()
                if task is None:  # Exit signal
                    break
                df_data, concept_id, sae_link, sae_id = task
                try:
                    # Run evaluation
                    if self.mode == "steering":
                        results = benchmark_model.predict_steer(
                            df_data, concept_id=concept_id,
                            sae_link=sae_link, sae_id=sae_id,
                            batch_size=self.args.steering_batch_size,
                            eval_output_length=self.args.steering_output_length
                        )
                    else:
                        results = benchmark_model.predict_latent(df_data)
                    self.result_queue.put((self.model_name, concept_id, results))
                except Exception as e:
                    logger.error(f"Error processing task on {self.model_name}: {str(e)}")
                    self.result_queue.put((self.model_name, concept_id, None))
        except Exception as e:
            logger.error(f"Worker process {self.model_name} failed: {str(e)}")
        finally:
            torch.cuda.empty_cache()

class ModelExecutor:
    def __init__(self, args, model_names, metadata, layer, train_dir, dump_dir, mode, factory_tokenizer, client=None):
        self.args = args
        self.model_names = model_names
        self.metadata = metadata
        self.layer = layer
        self.train_dir = train_dir
        self.dump_dir = dump_dir
        self.mode = mode
        self.factory_tokenizer = factory_tokenizer
        self.client = client  # For latent mode

        self.dataset_factory = None
        self.base_model = None
        self.models_or_workers = None

        self.setup()

    def setup(self):
        pass  # To be implemented by subclasses

    def process_concept(self, concept_id, current_df):
        pass  # To be implemented by subclasses

    def cleanup(self):
        pass  # To be implemented by subclasses

class SingleGPUModelExecutor(ModelExecutor):
    def setup(self):
        # Single-GPU setup
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.base_model = create_base_model(self.args.model_name, device)
        tokenizer = create_tokenizer(self.args.model_name)

        # Create all benchmark models
        self.models_or_workers = {}  # benchmark_models
        for model_name in self.model_names:
            model_class = getattr(axbench, model_name)
            model = setup_benchmark_model(
                model_class, self.base_model, tokenizer, self.layer,
                self.metadata, self.train_dir, self.dump_dir,
                self.mode, self.args.master_data_dir
            )
            self.models_or_workers[model_name] = model

        # Create dataset factory
        if self.mode == "steering":
            self.dataset_factory = SteeringDatasetFactory(self.base_model, self.factory_tokenizer, self.dump_dir)
        else:
            self.dataset_factory = ReAXFactory(self.base_model, self.client, self.factory_tokenizer, self.dump_dir)

    def process_concept(self, concept_id, current_df):
        all_results = {}
        for model_name, model in self.models_or_workers.items():
            try:
                if self.mode == "steering":
                    results = model.predict_steer(
                        current_df,
                        concept_id=concept_id,
                        sae_link=current_df["sae_link"].iloc[0],
                        sae_id=current_df["sae_id"].iloc[0],
                        batch_size=self.args.steering_batch_size,
                        eval_output_length=self.args.steering_output_length
                    )
                else:
                    results = model.predict_latent(current_df)
                all_results[model_name] = results
            except Exception as e:
                logger.error(f"Error processing task on {model_name}: {str(e)}")
        return all_results

    def cleanup(self):
        # Clean up GPU memory
        torch.cuda.empty_cache()

class MultiGPUModelExecutor(ModelExecutor):
    def setup(self):
        assert self.mode == "steering", "MultiGPUModelExecutor only supports steering mode"
        # Multi-GPU setup
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.models_or_workers = []  # Workers

        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        factory_base_model = None
        for idx, model_name in enumerate(self.model_names):
            gpu_id = idx % num_gpus if torch.cuda.is_available() else -1

            # If the worker is on device0, create a shared base model
            if gpu_id == 0 and factory_base_model is None:
                factory_base_model = create_base_model(self.args.model_name, "cuda:0")
                self.base_model = factory_base_model
                logger.info("Created shared base_model on cuda:0 for worker0 and factory")

            worker = SteeringModelWorker(
                model_name, gpu_id, self.args, self.metadata, self.layer, self.train_dir, self.dump_dir,
                self.task_queue, self.result_queue, self.mode,
                base_model=factory_base_model if gpu_id == 0 else None
            )
            worker.start()
            self.models_or_workers.append(worker)

        # Create dataset factory
        self.dataset_factory = SteeringDatasetFactory(self.base_model, self.factory_tokenizer, self.dump_dir)

    def process_concept(self, concept_id, current_df):
        # Distribute tasks to workers
        for _ in range(len(self.models_or_workers)):
            sae_link = current_df["sae_link"].iloc[0]
            sae_id = current_df["sae_id"].iloc[0]
            self.task_queue.put((current_df, concept_id, sae_link, sae_id))

        # Collect results
        all_results = {}
        for _ in range(len(self.models_or_workers)):
            model_name, concept_id_ret, results = self.result_queue.get()
            if results is not None:
                all_results[model_name] = results
        return all_results

    def cleanup(self):
        # Clean up workers
        for _ in self.models_or_workers:
            self.task_queue.put(None)
        for worker in self.models_or_workers:
            worker.join()

def infer_latent(args):
    # Common initialization logic
    data_dir = args.data_dir
    train_dir = args.train_dir
    dump_dir = args.dump_dir
    num_of_examples = args.latent_num_of_examples
    rotation_freq = args.rotation_freq
    config = load_config(train_dir)
    metadata = load_metadata_flatten(data_dir)
    layer = config["layer"]

    # Create OpenAI client and tokenizer
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

    # Latent mode only uses SingleGPUModelExecutor
    executor = SingleGPUModelExecutor(args, args.models, metadata, layer, train_dir, dump_dir, "latent", factory_tokenizer, client=client)

    try:
        state = load_state(args.dump_dir, "latent")
        start_concept_id = state.get("concept_id", 0) if state else 0

        progress_bar = tqdm(
            range(start_concept_id, len(metadata)),
            initial=start_concept_id,
            total=len(metadata),
            desc="Processing concepts"
        )

        for concept_id in progress_bar:
            # Create data for current concept
            current_df = create_data_latent(
                executor.dataset_factory, metadata, concept_id, num_of_examples, args)

            # Process concept
            all_results = executor.process_concept(concept_id, current_df)

            # Update DataFrame with results
            for model_name, results in all_results.items():
                for k, v in results.items():
                    current_df[f"{model_name}_{k}"] = v

            # Save results
            save(dump_dir, {"concept_id": concept_id + 1}, concept_id, "latent",
                 current_df, rotation_freq)

    finally:
        executor.cleanup()

def infer_steering(args):
    # Common initialization logic
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

    factory_tokenizer = create_tokenizer(args.model_name)

    # Choose executor based on multi_gpu flag
    if args.multi_gpu:
        executor = MultiGPUModelExecutor(args, args.models, metadata, layer, train_dir, dump_dir, "steering", factory_tokenizer)
    else:
        executor = SingleGPUModelExecutor(args, args.models, metadata, layer, train_dir, dump_dir, "steering", factory_tokenizer)

    try:
        state = load_state(args.dump_dir, "steering")
        start_concept_id = state.get("concept_id", 0) if state else 0

        progress_bar = tqdm(
            range(start_concept_id, len(metadata)),
            initial=start_concept_id,
            total=len(metadata),
            desc="Processing concepts"
        )

        for concept_id in progress_bar:
            # Create data for current concept
            current_df, (_, sae_link, sae_id) = create_data_steering(
                executor.dataset_factory, metadata, concept_id,
                num_of_examples,
                n_steering_factors,
                steering_datasets, args)

            # Process concept
            all_results = executor.process_concept(concept_id, current_df)

            # Update DataFrame with results
            for model_name, results in all_results.items():
                for k, v in results.items():
                    current_df[f"{model_name}_{k}"] = v

            # Save results
            save(dump_dir, {"concept_id": concept_id + 1}, concept_id, "steering",
                 current_df, rotation_freq)

    finally:
        executor.cleanup()

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
