# score evaluation results.
# 
# example launch command:
#     python axbench/scripts/evaluate.py --config axbench/demo/sweep/evaluate.yaml --mode latent


try:
    # This library is our indicator that the required installs
    # need to be done.
    import pyreax

except ModuleNotFoundError:
    # relative import; better to pip install subctrl
    import sys
    sys.path.append("../../pyreax")
    import pyreax

import os, argparse, yaml, json, glob, pickle
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch
from pathlib import Path
import numpy as np

import axbench
from axbench import (
    plot_aggregated_roc, 
    plot_perplexity,
    plot_strength,
    plot_lm_judge_rating
)
from args.eval_args import EvalArgs

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)

STATE_FILE = "evaluate_state.pkl"


def data_generator(data_dir, mode):
    """
    Generator function to read data files and yield data subsets by group_id.

    Args:
        data_dir (str): Path to the data directory.
        mode (str): Mode of operation ('latent' or 'steering').

    Yields:
        (group_id, df_subset): A tuple containing the group_id and subset DataFrame.
    """
    # Get list of files sorted by index
    file_list = sorted(glob.glob(os.path.join(data_dir, f'{mode}_data_fragment_*.parquet')))
    for file_path in file_list:
        df = pd.read_parquet(file_path)
        concept_ids = df['concept_id'].unique()
        concept_ids.sort()
        for concept_id in concept_ids:
            df_subset = df[df['concept_id'] == concept_id]
            yield (concept_id, df_subset)


def save_results(dump_dir, state, concept_id, partition, eval_results, rotation_freq):
    """
    Save the results dictionary to a .jsonl file.
    Each line in the file represents one concept_id's evaluation results.
    """
    # handle training df first
    dump_dir = Path(dump_dir) / "evaluate"
    dump_dir.mkdir(parents=True, exist_ok=True)
    
    # Save state
    state_path = os.path.join(dump_dir, f"{partition}_{STATE_FILE}")
    with open(state_path, "wb") as f:
        pickle.dump(state, f)
    
    # Define the output file path for JSON Lines
    fragment_index = concept_id // rotation_freq
    result_path = Path(dump_dir) / f"{partition}_fragment_{fragment_index}.jsonl"
    result_entry = {
        "concept_id": int(concept_id),
        "results": eval_results
    }
    with open(result_path, "a") as f:
        f.write(json.dumps(result_entry) + "\n")


def load_state(dump_dir, mode):
    """
    Load the state from a file if it exists.
    
    Args:
        dump_dir (str): The directory to load the state file from.
    
    Returns:
        dict: The loaded state dictionary, or None if no state file exists.
    """
    state_path = os.path.join(f"{dump_dir}/evaluate", f"{mode}_{STATE_FILE}")
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            return pickle.load(f)
    return None


def plot_steering(dump_dir):
    dump_dir = Path(dump_dir) / "evaluate"
    # aggregate all results
    file_list = sorted(glob.glob(os.path.join(dump_dir, 'steering_fragment_*.jsonl')))
    aggregated_results = []
    for file_path in file_list:
        aggregated_results += load_jsonl(file_path)

    # other plot goes here
    plot_perplexity(aggregated_results, write_to_path=dump_dir)
    plot_strength(aggregated_results, write_to_path=dump_dir)
    # plot_lm_judge_rating(aggregated_results, write_to_path=dump_dir)


def eval_steering(args):

    data_dir = args.data_dir
    dump_dir = args.dump_dir
    rotation_freq = args.rotation_freq
    df_generator = data_generator(args.data_dir, mode="steering")

    state = load_state(args.dump_dir, mode="steering")
    start_concept_id = state.get("concept_id", 0) if state else 0
    logger.warning(f"Starting concept_id: {start_concept_id}")

    for concept_id, current_df in df_generator:
        if concept_id < start_concept_id:
            continue
        logger.warning(f"Evaluating concept_id: {concept_id}")
        
        # Initialize a dictionary for storing evaluation results for this `concept_id`
        eval_results = {}
        for model_name in args.models:
            for evaluator_name in args.steering_evaluators:
                evaluator_class = getattr(axbench, evaluator_name)
                evaluator = evaluator_class(model_name, dump_dir=dump_dir)
                # Call each evaluator and store results
                eval_result = evaluator.compute_metrics(current_df)
                if evaluator.__str__() not in eval_results:
                    eval_results[evaluator.__str__()] = {}
                eval_results[evaluator.__str__()][model_name.__str__()] = eval_result
        save_results(
            dump_dir, {"concept_id": concept_id + 1}, 
            concept_id, 'steering', eval_results, rotation_freq)

    # final plot
    plot_steering(dump_dir)


def load_jsonl(jsonl_path):
    """
    Load data from a JSON lines file.
    """
    jsonl_data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            jsonl_data += [data]
    return jsonl_data
    

def plot_latent(dump_dir):
    dump_dir = Path(dump_dir) / "evaluate"
    # aggregate all results
    file_list = sorted(glob.glob(os.path.join(dump_dir, 'latent_fragment_*.jsonl')))
    aggregated_results = []
    for file_path in file_list:
        aggregated_results += load_jsonl(file_path)

    # roc plot
    roc_results = [aggregated_result["results"]["AUCROCEvaluator"] 
                   for aggregated_result in aggregated_results]
    plot_aggregated_roc(roc_results, write_to_path=dump_dir)

    # other plot goes here


def eval_latent(args):

    data_dir = args.data_dir
    dump_dir = args.dump_dir
    rotation_freq = args.rotation_freq
    df_generator = data_generator(args.data_dir, mode="latent")

    state = load_state(args.dump_dir, mode="latent")
    start_concept_id = state.get("concept_id", 0) if state else 0
    logger.warning(f"Starting concept_id: {start_concept_id}")

    for concept_id, current_df in df_generator:
        if concept_id < start_concept_id:
            continue
        logger.warning(f"Evaluating concept_id: {concept_id}")
        
        # Initialize a dictionary for storing evaluation results for this `concept_id`
        eval_results = {}
        for model_name in args.models:
            for evaluator_name in args.latent_evaluators:
                evaluator_class = getattr(axbench, evaluator_name)
                evaluator = evaluator_class(model_name)
                # Call each evaluator and store results
                eval_result = evaluator.compute_metrics(current_df)
                if evaluator.__str__() not in eval_results:
                    eval_results[evaluator.__str__()] = {}
                eval_results[evaluator.__str__()][model_name.__str__()] = eval_result
        save_results(
            dump_dir, {"concept_id": concept_id + 1}, 
            concept_id, 'latent', eval_results, rotation_freq)

    # final plot
    plot_latent(dump_dir)


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
    args = EvalArgs(custom_args=custom_args)
    logger.warning("Evaluating generations with the following configuration:")
    logger.warning(args)
    
    if args.mode == "latent":
        eval_latent(args)
    elif args.mode == "steering":
        eval_steering(args)


if __name__ == "__main__":
    main()

