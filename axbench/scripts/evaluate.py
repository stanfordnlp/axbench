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

from pyreax import (
    LanguageModel
)

import os, argparse, yaml, json, glob, pickle, tempfile
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch
from pathlib import Path
import numpy as np
from openai import AsyncOpenAI
import httpx, asyncio

import axbench
from axbench import (
    plot_aggregated_roc, 
    plot_metrics,
    plot_accuracy_bars,
    plot_win_rates
)
from args.eval_args import EvalArgs
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing


import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)

STATE_FILE = "evaluate_state.pkl"
STEERING_EXCLUDE_MODELS = {}
LATENT_EXCLUDE_MODELS = {"PromptSteering"}


def data_generator(data_dir, mode):
    """
    Generator function to read data files and yield data subsets by group_id.
    Pre-loads data in chunks to reduce I/O bottlenecks.

    Args:
        data_dir (str): Path to the data directory.
        mode (str): Mode of operation ('latent' or 'steering').

    Yields:
        (group_id, df_subset): A tuple containing the group_id and subset DataFrame.
    """
    # Get list of files sorted by index
    file_list = sorted(glob.glob(os.path.join(data_dir, f'{mode}_data_fragment_*.parquet')))
    
    # Pre-load and organize data by concept_id
    concept_data = {}
    for file_path in tqdm(file_list, desc="Loading data"):
        df = pd.read_parquet(file_path)
        # Group by concept_id and store in dictionary
        for concept_id, group in df.groupby('concept_id'):
            if concept_id not in concept_data:
                concept_data[concept_id] = []
            concept_data[concept_id].append(group)
    
    # Yield concatenated data for each concept_id
    for concept_id in sorted(concept_data.keys()):
        if len(concept_data[concept_id]) > 1:
            df_subset = pd.concat(concept_data[concept_id])
        else:
            df_subset = concept_data[concept_id][0]
        yield (concept_id, df_subset)


def winrate_data_generator(data_dir, aggregated_results):
    best_factors = {}
    for result in aggregated_results:
        best_factors[result["concept_id"]] = {}
        # pick the best factor each method
        for method, scores in result["results"]["LMJudgeConceptFollowingEvaluator"].items():
            best_factor = scores["factor"][np.argmax(scores["lm_judge_rating"])]
            # one caveat here is the best factor for prompt steering is the 0th element
            best_factors[result["concept_id"]][method] = best_factor
    
    df_generator = data_generator(data_dir, mode="steering")
    best_dfs = []
    for concept_id, current_df in df_generator:
        # if concept_id >= start_concept_id:
        concept_best_dfs = {}
        for method, factor in best_factors[concept_id].items():
            include_columns = ["concept_id", "input_concept", "input_id", "original_prompt", "factor",f"{method}_steered_generation"]
            method_df = current_df[include_columns]
            method_best_df = method_df[method_df["factor"]==factor]
            concept_best_dfs[method] = method_best_df.copy()
            concept_best_df = method_best_df[["concept_id", "input_concept", "input_id", "original_prompt"]].copy()
        for method in best_factors[concept_id].keys():
            # Use merge instead of direct assignment to ensure proper alignment
            concept_best_df = concept_best_df.merge(
                concept_best_dfs[method][['concept_id', 'input_concept', 'input_id', f"{method}_steered_generation"]],
                on=['concept_id', 'input_concept', 'input_id'],
                how='left')
        yield (concept_id, concept_best_df)


def save_results(dump_dir, state, concept_id, partition, eval_results):
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
    result_path = Path(dump_dir) / f"{partition}.jsonl"
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


def combine_scores(concept_data):
    """Combine scores from concept and following evaluators for each method."""
    concept_scores = concept_data["results"]["LMJudgeConceptEvaluator"]
    following_scores = concept_data["results"]["LMJudgeFollowingEvaluator"]
    combined_evaluator = {}
    # For each method (L1LinearProbe, ReAX, etc.)
    for method in concept_scores.keys():
        concept_ratings = concept_scores[method]["lm_judge_rating"]
        following_ratings = following_scores[method]["lm_judge_rating"]
        factors = concept_scores[method]["factor"]  # factors are same in both
        # Multiply corresponding scores
        combined_ratings = [(c * f) for c, f in zip(concept_ratings, following_ratings)]
        combined_evaluator[method] = {
            "lm_judge_rating": combined_ratings,
            "factor": factors
        }
    return combined_evaluator


def process_jsonl_file(jsonl_lines):
    for data in jsonl_lines:
        data["results"]["LMJudgeConceptFollowingEvaluator"] = \
            combine_scores(data)
    return jsonl_lines


def plot_steering(aggregated_results, dump_dir):
    try:
        configs = [
            {
                'evaluator_name': 'PerplexityEvaluator',
                'metric_name': 'perplexity',
                'y_label': 'Perplexity',
                'use_log_scale': True
            },
            {
                'evaluator_name': 'PerplexityEvaluator',
                'metric_name': 'strength',
                'y_label': 'Strength',
                'use_log_scale': False
            },
            {
                'evaluator_name': 'LMJudgeConceptEvaluator',
                'metric_name': 'lm_judge_rating',
                'y_label': 'Steering',
                'use_log_scale': False
            },
            {
                'evaluator_name': 'LMJudgeFollowingEvaluator',
                'metric_name': 'lm_judge_rating',
                'y_label': 'Relevance',
                'use_log_scale': False
            },
            {
                'evaluator_name': 'LMJudgeConceptFollowingEvaluator',
                'metric_name': 'lm_judge_rating',
                'y_label': 'Steering*Relevance',
                'use_log_scale': False
            }
        ]
        plot_metrics(
            jsonl_data=aggregated_results,
            configs=configs,
            write_to_path=dump_dir
        )
        plot_win_rates(
            jsonl_data=aggregated_results,
            write_to_path=dump_dir  # Replace with your desired output directory
        )
    except Exception as e:
        logger.warning(f"Failed to plot: {e}")


def eval_steering_single_task(args_tuple):
    """Helper function to evaluate a single concept-model-evaluator combination"""
    concept_id, current_df, evaluator_name, model_name, dump_dir, lm_model, winrate_baseline = args_tuple
    
    # Create LanguageModel instance within the worker process
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
    lm_model = LanguageModel(
        lm_model,
        client,
        dump_dir=dump_dir,
        use_cache=False
    )
    
    try:
        evaluator_class = getattr(axbench, evaluator_name)
        evaluator = evaluator_class(
            model_name, dump_dir=dump_dir, 
            concept_id=concept_id, lm_model=lm_model, winrate_baseline=winrate_baseline)
        eval_result = evaluator.compute_metrics(current_df)
        return (concept_id, evaluator.__str__(), model_name.__str__(), eval_result, lm_model.stats.get_report(), current_df)
    finally:
        # Properly close both the HTTP client and async client
        async def cleanup():
            await client.close()
        asyncio.run(cleanup())


def eval_steering(args):
    """
    Evaluate steering performance using multi-processing for all tasks
    """
    data_dir = args.data_dir
    dump_dir = args.dump_dir
    rotation_freq = args.rotation_freq

    # Initialize data generator
    df_generator = data_generator(args.data_dir, mode="steering")

    # Load previous state if exists
    state = load_state(args.dump_dir, mode="steering")
    start_concept_id = state.get("concept_id", 0) if state else 0
    logger.warning(f"Starting concept_id: {start_concept_id}")

    # Create all evaluation tasks - flattened for maximum parallelization
    all_tasks = [
        (concept_id, current_df, evaluator_name, model_name, args.dump_dir, args.lm_model, args.winrate_baseline)
        for concept_id, current_df in df_generator
        if concept_id >= start_concept_id
        for evaluator_name in args.steering_evaluators
        for model_name in args.models
        if model_name not in STEERING_EXCLUDE_MODELS
    ]

    # Group results by concept_id
    all_results = {}
    
    # Run all evaluations with process pool
    logger.warning(f"Number of workers: {args.num_of_workers}; Number of CPUs: {multiprocessing.cpu_count()}")
    if not hasattr(args, 'num_of_workers') or args.num_of_workers is None:
        args.num_of_workers = max(1, multiprocessing.cpu_count() - 1)
    lm_reports = []
    all_dfs = []
    with ProcessPoolExecutor(max_workers=args.num_of_workers) as executor:
        for concept_id, evaluator_str, model_str, result, lm_report, current_df in executor.map(
            eval_steering_single_task, all_tasks):
            if concept_id not in all_results:
                all_results[concept_id] = {}
            if evaluator_str not in all_results[concept_id]:
                all_results[concept_id][evaluator_str] = {}
            all_results[concept_id][evaluator_str][model_str] = result
            lm_reports += [lm_report]
            all_dfs += [current_df]
            logger.warning(f"Completed task for concept_id: {concept_id}, model: {model_str}, evaluator: {evaluator_str}")
    
    # Batch save all results
    for concept_id, eval_results in sorted(all_results.items()):
        save_results(
            dump_dir, 
            {"concept_id": concept_id + 1}, 
            concept_id, 
            'steering', 
            eval_results,
        )

    # Reload for plotting and optional winrate
    aggregated_results = process_jsonl_file(
            load_jsonl(os.path.join(Path(dump_dir) / "evaluate" / 'steering.jsonl')))

    if args.run_winrate:
        winrate_results = {}
        winrate_df_generator = winrate_data_generator(data_dir, aggregated_results)
        # Create all winrate evaluation tasks - flattened for maximum parallelization
        winrate_tasks = [
            (concept_id, current_df, "WinRateEvaluator", model_name, args.dump_dir, args.lm_model, args.winrate_baseline)
            for concept_id, current_df in winrate_df_generator
            if concept_id >= start_concept_id
            for model_name in args.models
            if model_name != args.winrate_baseline
        ]

        winrate_dfs = []
        with ProcessPoolExecutor(max_workers=args.num_of_workers) as executor:
            for concept_id, __builtins__, model_str, result, lm_report, current_df in executor.map(
                eval_steering_single_task, winrate_tasks):
                if concept_id not in winrate_results:
                    winrate_results[concept_id] = {}
                winrate_results[concept_id][model_str] = result
                lm_reports += [lm_report]
                winrate_dfs += [current_df]
                logger.warning(f"Completed winrate task for concept_id: {concept_id}, model: {model_str}")

        for concept_id, winrate_result in winrate_results.items():
            aggregated_results[concept_id]["results"]["WinRateEvaluator"] = winrate_result

        # update the winrate jsonl file
        result_path = Path(dump_dir) / "evaluate" / f"steering.jsonl"
        with open(result_path, "w") as f:
            for result_entry in aggregated_results:
                f.write(json.dumps(result_entry) + "\n")

    # Aggregate LM reports
    aggregated_lm_report = {
        "total_calls": sum([report["total_calls"] for report in lm_reports]),
        "total_cache_hits": sum([report["total_cache_hits"] for report in lm_reports]),
        "total_price": sum([report["total_price"] for report in lm_reports])
    }
    logger.warning("="*20)  
    logger.warning(f"Total calls: {aggregated_lm_report['total_calls']}, "
                   f"Total cache hits: {aggregated_lm_report['total_cache_hits']}")
    logger.warning(f"Total price: ${aggregated_lm_report['total_price']}")
    logger.warning("="*20)

    # Generate final plot
    logger.warning("Generating final plot...")
    plot_steering(aggregated_results, Path(dump_dir) / "evaluate")
    logger.warning("Evaluation completed!")


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
    aggregated_results = load_jsonl(os.path.join(dump_dir, 'latent.jsonl'))
    try:
        plot_aggregated_roc(aggregated_results, write_to_path=dump_dir)
        plot_accuracy_bars(aggregated_results, "HardNegativeEvaluator", write_to_path=dump_dir)
    except Exception as e:
        logger.warning(f"Failed to plot: {e}")


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
            if model_name in LATENT_EXCLUDE_MODELS:
                continue
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
            concept_id, 'latent', eval_results)

    # Generate final plot
    logger.warning("Generating final plot...")
    plot_latent(dump_dir)
    logger.warning("Evaluation completed!")

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

