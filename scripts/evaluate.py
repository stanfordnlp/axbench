# evaluate existing subspaces.
#
# example launch command:
#     python scripts/evaluate.py --config demo/sweep/evaluate.yaml --mode latent

try:
    # This library is our indicator that the required installs
    # need to be done.
    import pyreax

except ModuleNotFoundError:
    # relative import; better to pip install subctrl
    import sys
    sys.path.append("../../pyreax")
    import pyreax


import os, argparse, yaml, json, glob, pickle, requests
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download

from pyvene import IntervenableModel
from pyreax import (
    EXAMPLE_TAG, 
    ReAXFactory, 
    MaxReLUIntervention, 
    JumpReLUSAECollectIntervention,
    DatasetArgs
)
from pyreax import (
    gather_residual_activations, 
)

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_DELAY = 1  # in seconds
STATE_FILE = "evaluate_state.pkl"
CONFIG_FILE = "config.json"
METADATA_FILE = "metadata.jsonl"
DEFAULT_ROTATION_FREQ = 1000


def load_sae(ref):
    """load the sae metadata (e.g., column index) and weights"""
    sae_path = ref.split("https://www.neuronpedia.org/")[-1]
    sae_url = f"https://www.neuronpedia.org/api/feature/{sae_path}"
    
    headers = {"X-Api-Key": os.environ.get("NP_API_KEY")}
    response = requests.get(sae_url, headers=headers).json()
    hf_repo = response["source"]["hfRepoId"]
    hf_folder = response["source"]["hfFolderId"]
    path_to_params = hf_hub_download(
        repo_id=hf_repo,
        filename=f"{hf_folder}/params.npz",
        force_download=False,
    )
    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}

    return pt_params


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
    state_path = os.path.join(f"{dump_dir}/evaluate", STATE_FILE)
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            return pickle.load(f)
    return None


def load_reax(train_dir, rotation_index=0):
    """
    Load the state from a file if it exists.
    
    Args:
        dump_dir (str): The directory to load the state file from.
    
    Returns:
        dict: The loaded state dictionary, or None if no state file exists.
    """
    weight_dict = torch.load(
        f"{train_dir}/weight_fragment_{rotation_index}.pt"
    )
    bias_dict = torch.load(
        f"{train_dir}/bias_fragment_{rotation_index}.pt"
    )
    return weight_dict, bias_dict


def load_metadata(metadata_path):
    """
    Load metadata from a JSON lines file.
    """
    metadata = []
    with open(Path(metadata_path) / METADATA_FILE, 'r') as f:
        for line in f:
            data = json.loads(line)
            metadata += [data]  # Return the metadata as is
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


def eval_steering(args):

    raise NotImplementedError("Steering evaluation is not implemented yet.")


def eval_latent(args):
    
    data_dir = args.data_dir
    train_dir = args.train_dir
    dump_dir = args.dump_dir
    num_of_examples = args.num_of_examples
    rotation_freq = args.rotation_freq
    
    config = load_config(train_dir)
    metadata = load_metadata(data_dir)
    layer = config["layer"]
    
    # Load lm.
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], device_map="cpu")
    model.config.use_cache = False
    model = model.cuda()    
    model = model.eval()
    tokenizer =  AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.padding_side = "right"

    # Load and mount sae.
    sae_example_link = metadata[0]["refs"][0]
    sae_weights = load_sae(sae_example_link)
    sae_intervention = JumpReLUSAECollectIntervention(
        embed_dim=sae_weights['W_enc'].shape[0],
        low_rank_dimension=sae_weights['W_enc'].shape[1]
    )
    sae_intervention.load_state_dict(sae_weights, strict=False)
    _ = sae_intervention.cuda()
    pv_sae_model = IntervenableModel({
       "component": f"model.layers[{layer}].output",
       "intervention": sae_intervention}, model=model)
    
    # Init reax and whole weight loading (but we dont load weights).
    reax_intervention = MaxReLUIntervention(
        embed_dim=model.config.hidden_size, 
        low_rank_dimension=len(metadata[0]["concepts"]), # usually it's 2.
    )
    _ = reax_intervention.cuda()
    current_frag_id = 0
    reax_weights_dict, reax_bias_dict = load_reax(train_dir, current_frag_id)

    state = load_state(args.dump_dir)
    start_group_id = state.get("group_id", 0) if state else 0
    logger.warning(f"Starting group index: {start_group_id}")

    dataset_factory = ReAXFactory(model, tokenizer, dump_dir)
    progress_bar = tqdm(range(start_group_id, len(metadata)), desc="Processing concept groups")
    for group_id in progress_bar:
        # reload if needed.
        if group_id // rotation_freq > current_frag_id:
            current_frag_id = group_id // rotation_freq
            reax_weights_dict, reax_bias_dict = load_reax(train_dir, current_frag_id)

        group_metadata = metadata[group_id]
        concepts = group_metadata["concepts"]
        refs = group_metadata["refs"]
        concept_genres_map = group_metadata["concept_genres_map"]
        contrast_concepts_map = group_metadata["contrast_concepts_map"]
        
        # prepare concept related data.
        _, eval_contrast_concepts_map = \
            dataset_factory.prepare_concepts(
                concepts, 
                concept_genres_map=concept_genres_map,
                contrast_concepts_map=contrast_concepts_map)
        
        # generate with retry mechanism.
        try:
            current_df = retry_with_backoff(
                dataset_factory.create_eval_df,
                concepts, num_of_examples, concept_genres_map, contrast_concepts_map,
                eval_contrast_concepts_map, input_length=args.input_length, output_length=args.output_length
            )
        except Exception as e:
            logger.warning(f"Failed to create evaluation data for group {group_id}: {e}")
            return

        # load reax weight in.
        reax_intervention.proj.weight.data = reax_weights_dict[group_id].cuda()
        reax_intervention.proj.bias.data = reax_bias_dict[group_id].cuda()

        # inference and save data.
        print(current_df)



def main():
    custom_args = [
        {
            'args': ['--mode'],
            'kwargs': {
                'type': str,
                'default': "latent",
                'help': 'The evaluation mode.'
            }
        }
    ]
    args = DatasetArgs(custom_args=custom_args)
    logger.warning("Generating datasets with the following configuration:")
    logger.warning(args)
    
    if args.mode == "latent":
        eval_latent(args)
    elif args.mode == "steering":
        eval_steering(args)


if __name__ == "__main__":
    main()

