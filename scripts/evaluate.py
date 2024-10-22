# score evaluation results.
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

import os, argparse, yaml, json, glob, pickle
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch
from pathlib import Path
import numpy as np

import reax_benchmark
from reax_benchmark import AUCROCEvaluator, plot_aggregated_roc
from args.eval_args import EvalArgs

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)

STATE_FILE = "evaluate_state.pkl"


def data_generator(data_dir):
    """
    Generator function to read data files and yield data subsets by group_id.

    Args:
        data_dir (str): Path to the data directory.

    Yields:
        (group_id, df_subset): A tuple containing the group_id and subset DataFrame.
    """
    # Get list of files sorted by index
    file_list = sorted(glob.glob(os.path.join(data_dir, 'latent_data_fragment_*.csv')))
    for file_path in file_list:
        df = pd.read_csv(file_path)
        reax_ids = df['reax_id'].unique()
        reax_ids.sort()
        for reax_id in reax_ids:
            df_subset = df[df['reax_id'] == reax_id]
            yield (reax_id, df_subset)


def load_evaluators(args):
    """
    Load evaluators based on provided arguments.

    Args:
        args: Command line arguments containing evaluator information.

    Returns:
        list: List of initialized evaluator objects.
    """
    evaluators = []
    if args.mode == "latent":
        for evaluator_name in args.latent_evaluators:
            # Dynamically load the evaluator class if it exists in pyreax
            if hasattr(reax_benchmark, evaluator_name):
                evaluator_class = getattr(reax_benchmark, evaluator_name)
                evaluators.append(evaluator_class())
            else:
                logger.warning(f"Evaluator {evaluator_name} not found in pyreax.")
    elif args.mode == "steer":
        logger.warning("Steering evaluators not implemented yet.")
    return evaluators


def save_results(dump_dir, state, reax_id, partition, eval_results, rotation_freq):
    """
    Save the results dictionary to a .jsonl file.
    Each line in the file represents one reax_id's evaluation results.
    """
    # handle training df first
    dump_dir = Path(dump_dir) / "evaluate"
    dump_dir.mkdir(parents=True, exist_ok=True)
    
    # Save state
    state_path = os.path.join(dump_dir, STATE_FILE)
    with open(state_path, "wb") as f:
        pickle.dump(state, f)
    
    # Define the output file path for JSON Lines
    fragment_index = reax_id // rotation_freq
    result_path = Path(dump_dir) / f"{partition}_fragment_{fragment_index}.jsonl"
    result_entry = {
        "reax_id": int(reax_id),
        "results": eval_results
    }
    with open(result_path, "a") as f:
        f.write(json.dumps(result_entry) + "\n")


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


def eval_steering(args):

    raise NotImplementedError("Steering inference is not implemented yet.")


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
    evaluators = load_evaluators(args)
    df_generator = data_generator(args.data_dir)

    state = load_state(args.dump_dir)
    start_reax_id = state.get("reax_id", 0) if state else 0
    logger.warning(f"Starting reax index: {start_reax_id}")

    for reax_id, group_df in df_generator:
        if reax_id < start_reax_id:
            continue
        logger.warning(f"Evaluating reax_id: {reax_id}")
        
        # Initialize a dictionary for storing evaluation results for this `reax_id`
        eval_results = {}
        for evaluator in evaluators:
            # Call each evaluator and store results
            eval_result = evaluator.compute_metrics(group_df)
            eval_results[evaluator.__str__()] = eval_result
        save_results(dump_dir, {"reax_id": reax_id + 1}, reax_id, 'latent', eval_results, rotation_freq)

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

