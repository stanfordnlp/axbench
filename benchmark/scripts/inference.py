# inference with existing subspaces.
#
# example launch command:
#     python benchmark/scripts/inference.py --config benchmark/demo/sweep/inference.yaml --mode latent

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
import benchmark


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


def load_config(config_path):
    """
    Load metadata from a JSON lines file.
    """
    with open(Path(config_path) / CONFIG_FILE) as f:
        d = json.load(f)
    return d


def load_state(dump_dir):
    """
    Load the state from a file if it exists.
    
    Args:
        dump_dir (str): The directory to load the state file from.
    
    Returns:
        dict: The loaded state dictionary, or None if no state file exists.
    """
    state_path = os.path.join(f"{dump_dir}/inference", STATE_FILE)
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            return pickle.load(f)
    return None


def load_metadata_flatten(metadata_path):
    """
    Load flatten metadata from a JSON lines file.
    """
    metadata = []
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
                }
                metadata += [flatten_data]  # Return the metadata as is
    return metadata


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


def save(
    dump_dir, state, concept_id, partition,
    current_df, rotation_freq):
    """
    Save the current state, metadata, and DataFrame.
    """
    dump_dir = Path(dump_dir) / "inference"
    dump_dir.mkdir(parents=True, exist_ok=True)
    
    state_path = os.path.join(dump_dir, STATE_FILE)
    with open(state_path, "wb") as f:
        pickle.dump(state, f)
    
    fragment_index = concept_id // rotation_freq
    df_path = os.path.join(dump_dir, f"{partition}_data_fragment_{fragment_index}.csv")
    if os.path.exists(df_path):
        existing_df = pd.read_csv(df_path)
        combined_df = pd.concat([existing_df, current_df], ignore_index=True)
    else:
        combined_df = current_df
    combined_df.to_csv(df_path, index=False)


def create_data_latent(dataset_factory, metadata, concept_id, num_of_examples, args):
    # prepare concept related data.
    concept = metadata[concept_id]["concept"]
    sae_link = metadata[concept_id]["ref"]
    sae_id = int(sae_link.split("/")[-1]) 
    concept_genres_map = metadata[concept_id]["concept_genres_map"]
    contrast_concepts_map = metadata[concept_id]["contrast_concepts_map"]
    _, eval_contrast_concepts_map = \
        dataset_factory.prepare_concepts(
            [concept], 
            concept_genres_map=concept_genres_map,
            contrast_concepts_map=contrast_concepts_map)

    try:
        current_df = retry_with_backoff(
            dataset_factory.create_eval_df,
            [concept], num_of_examples, concept_genres_map, contrast_concepts_map,
            eval_contrast_concepts_map, input_length=args.input_length, output_length=args.output_length
        )
        current_df["concept_id"] = concept_id
        current_df["sae_link"] = sae_link
        current_df["sae_id"] = sae_id
    except Exception as e:
        logger.warning(f"Failed to create evaluation data for group {concept_id}: {e}")
        return

    return current_df


def infer_steering(args):

    raise NotImplementedError("Steering inference is not implemented yet.")


def infer_latent(args):
    
    data_dir = args.data_dir
    train_dir = args.train_dir
    dump_dir = args.dump_dir
    num_of_examples = args.num_of_examples
    rotation_freq = args.rotation_freq
    config = load_config(train_dir)
    metadata = load_metadata_flatten(data_dir)
    layer = config["layer"]
    
    # Load lm.
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], device_map="cpu")
    model.config.use_cache = False
    model = model.cuda()    
    model = model.eval()
    tokenizer =  AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.padding_side = "right"

    # Load dataset factory for evals.
    dataset_factory = ReAXFactory(model, tokenizer, dump_dir)

    # Pre-load inference models.
    benchmark_models = []
    for model_name in args.models:
        model_class = getattr(benchmark, model_name)
        logger.warning(f"Loading {model_class} from disk for inference.\n")
        benchmark_model = model_class(
            model, tokenizer, layer=layer, 
            low_rank_dimension=len(metadata))
        benchmark_model.load(
            dump_dir=train_dir, sae_path=metadata[0]["ref"])
        benchmark_models += [benchmark_model]

    state = load_state(args.dump_dir)
    start_concept_id = state.get("concept_id", 0) if state else 0
    logger.warning(f"Starting concept index: {start_concept_id}")
    progress_bar = tqdm(range(start_concept_id, len(metadata)), desc="Processing concepts")
    
    torch.cuda.empty_cache()
    with torch.no_grad():
        for concept_id in progress_bar:
            # Create.
            current_df = create_data_latent(
                dataset_factory, metadata, concept_id, num_of_examples, args)

            # Evaluate.
            for model_idx, model_name in enumerate(args.models):
                results = benchmark_models[model_idx].predict_latent(current_df)
                for k, v in results.items():
                    current_df[f"{model_name}_{k}"] = v
            
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
        }
    ]
    args = DatasetArgs(custom_args=custom_args)
    logger.warning("Inferencing with following configuration:")
    logger.warning(args)
    
    if args.mode == "latent":
        infer_latent(args)
    elif args.mode == "steering":
        infer_steering(args)


if __name__ == "__main__":
    main()

