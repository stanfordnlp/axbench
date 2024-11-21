# score evaluation results.
# 
# example launch command:
#     python axbench/scripts/evaluate.py --config axbench/demo/sweep/evaluate.yaml --mode latent

import shutil
from ..models.language_models import (
    LanguageModel
)

import os, argparse, yaml, json, glob, pickle, tempfile, copy
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch
from pathlib import Path
import numpy as np
from openai import AsyncOpenAI
import httpx, asyncio
import datetime
import yaml

import axbench
from ..utils.plot_utils import (
    plot_aggregated_roc, 
    plot_metrics,
    plot_accuracy_bars,
    plot_win_rates,
)
from ..utils.prompt_utils import (
    generate_html_with_highlight_text,
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
LATENT_EXCLUDE_MODELS = {"PromptSteering", "PromptBaseline"}


def data_generator(data_dir, mode, winrate_split_ratio=None):
    """
    Generator function to read data files and yield data subsets by group_id.
    Pre-loads data in chunks to reduce I/O bottlenecks.

    Args:
        data_dir (str): Path to the data directory.
        mode (str): Mode of operation ('latent' or 'steering').

    Yields:
        (group_id, df_subset): A tuple containing the group_id and subset DataFrame.
    """
    # Pre-load and organize data by concept_id
    concept_data = {}
    if mode == "latent":
        df = pd.read_parquet(os.path.join(data_dir, f'latent_data.parquet'))
    elif mode == "steering" or mode == "winrate":
        df = pd.read_parquet(os.path.join(data_dir, f'steering_data.parquet'))
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
        if winrate_split_ratio is not None:
            n_input_ids = df_subset["input_id"].max()+1
            n_steering_ids = n_input_ids - round(n_input_ids * winrate_split_ratio)
            if mode == "steering":
                df_subset = df_subset[df_subset["input_id"] < n_steering_ids]
            elif mode == "winrate":
                df_subset = df_subset[df_subset["input_id"] >= n_steering_ids]
        yield (concept_id, df_subset)


def get_best_factors(aggregated_results):
    best_factors = {}
    for result in aggregated_results:
        best_factors[result["concept_id"]] = {}
        for method, scores in result["results"]["LMJudgeEvaluator"].items():
            best_factors[result["concept_id"]][method] = scores["factor"][np.argmax(scores["lm_judge_rating"])]
    return best_factors


def winrate_data_generator(data_dir, aggregated_results, winrate_split_ratio):
    best_factors = get_best_factors(aggregated_results)
    df_generator = data_generator(data_dir, mode="winrate", winrate_split_ratio=winrate_split_ratio)
    for concept_id, current_df in df_generator:
        # if concept_id >= start_concept_id: # TODO: uncomment this when we fix our pipeline
        concept_best_dfs = {}
        for method, factor in best_factors[concept_id].items():
            include_columns = ["concept_id", "input_concept", "input_id", "original_prompt", "steered_input", "factor", f"{method}_steered_generation"]
            method_df = current_df[include_columns]
            method_best_df = method_df[method_df["factor"]==factor]
            concept_best_dfs[method] = method_best_df.copy()
            concept_best_df = method_best_df[["concept_id", "input_concept", "input_id", "original_prompt", "steered_input"]].copy()
        for method in best_factors[concept_id].keys():
            # Use merge instead of direct assignment to ensure proper alignment
            concept_best_df = concept_best_df.merge(
                concept_best_dfs[method][['concept_id', 'input_concept', 'input_id', f"{method}_steered_generation"]],
                on=['concept_id', 'input_concept', 'input_id'],
                how='left')
        yield (concept_id, concept_best_df)


def save_results(dump_dir, state, concept_id, partition, eval_results, eval_df=None):
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

    # save the steering ratings for each example
    if eval_df is not None:
        sorted_evaluator_names = sorted(list(eval_df.keys()))
        sorted_model_names = sorted(list(eval_df[sorted_evaluator_names[0]].keys()))
        if len(sorted_evaluator_names) == 0:
            return
        current_df = eval_df[sorted_evaluator_names[0]][sorted_model_names[0]].copy()
        for evaluator_name in sorted_evaluator_names:
            if evaluator_name == "PerplexityEvaluator":
                continue
            for model_name in sorted_model_names:
                if evaluator_name == sorted_evaluator_names[0] and model_name == sorted_model_names[0]:
                    continue
                current_df[f"{model_name}_{evaluator_name}"] = eval_df[evaluator_name][model_name][f"{model_name}_{evaluator_name}"]

                current_df[f"{model_name}_{evaluator_name}_relevance_concept_ratings"] = \
                    eval_df[evaluator_name][model_name][f"{model_name}_{evaluator_name}_relevance_concept_ratings"]
                current_df[f"{model_name}_{evaluator_name}_relevance_concept_completions"] = \
                    eval_df[evaluator_name][model_name][f"{model_name}_{evaluator_name}_relevance_concept_completions"]
                
                current_df[f"{model_name}_{evaluator_name}_relevance_instruction_ratings"] = \
                    eval_df[evaluator_name][model_name][f"{model_name}_{evaluator_name}_relevance_instruction_ratings"]
                current_df[f"{model_name}_{evaluator_name}_relevance_instruction_completions"] = \
                    eval_df[evaluator_name][model_name][f"{model_name}_{evaluator_name}_relevance_instruction_completions"]
                
                current_df[f"{model_name}_{evaluator_name}_fluency_ratings"] = \
                    eval_df[evaluator_name][model_name][f"{model_name}_{evaluator_name}_fluency_ratings"]
                current_df[f"{model_name}_{evaluator_name}_fluency_completions"] = \
                    eval_df[evaluator_name][model_name][f"{model_name}_{evaluator_name}_fluency_completions"]
                
        df_path = os.path.join(dump_dir, f"{partition}_data.parquet")
        if os.path.exists(df_path):
            existing_df = pd.read_parquet(df_path)
            combined_df = pd.concat([existing_df, current_df], ignore_index=True)
        else:
            combined_df = current_df
        combined_df.to_parquet(df_path, index=False)


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


def combine_scores_per_concept(concept_data):
    """Combine scores from concept and following evaluators for each method."""
    return concept_data["results"]["LMJudgeEvaluator"]


def process_jsonl_file(jsonl_lines):
    for data in jsonl_lines:
        data["results"]["LMJudgeEvaluator"] = \
            combine_scores_per_concept(data)
    return jsonl_lines


def plot_steering(aggregated_results, dump_dir, report_to=[], wandb_name=None):
    try:
        configs = [
            {
                'evaluator_name': 'PerplexityEvaluator',
                'metric_name': 'perplexity',
                'y_label': 'Perplexity',
                'use_log_scale': False
            },
            {
                'evaluator_name': 'LMJudgeEvaluator',
                'metric_name': 'relevance_concept_ratings',
                'y_label': 'Concept',
                'use_log_scale': False
            },
            {
                'evaluator_name': 'LMJudgeEvaluator',
                'metric_name': 'relevance_instruction_ratings',
                'y_label': 'Instruct',
                'use_log_scale': False
            },
            {
                'evaluator_name': 'LMJudgeEvaluator',
                'metric_name': 'fluency_ratings',
                'y_label': 'Fluency',
                'use_log_scale': False
            },
            {
                'evaluator_name': 'LMJudgeEvaluator',
                'metric_name': 'lm_judge_rating',
                'y_label': 'Aggregated',
                'use_log_scale': False
            },
            {
                'evaluator_name': 'PerplexityEvaluator',
                'metric_name': 'strength',
                'y_label': 'Strength',
                'use_log_scale': False
            },
        ]
        plot_metrics(
            jsonl_data=aggregated_results,
            configs=configs,
            write_to_path=dump_dir, 
            report_to=report_to,
            wandb_name=wandb_name
        )
        plot_win_rates(
            jsonl_data=aggregated_results,
            write_to_path=dump_dir,  # Replace with your desired output directory
            report_to=report_to,
            wandb_name=wandb_name
        )
    except Exception as e:
        logger.warning(f"Failed to plot: {e}")


def eval_steering_single_task(args_tuple):
    """Helper function to evaluate a single concept-model-evaluator combination"""
    concept_id, current_df, evaluator_name, model_name, dump_dir, \
        lm_model, winrate_baseline, lm_caches = args_tuple
    
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
        use_cache=True,
        cache_level="prompt",
        cache_tag="evaluate",
        master_data_dir="axbench/data",
        temperature=0.7
    )
    # overwrite cache if any.
    if bool(lm_caches):
        lm_model.cache_in_mem = lm_caches
    
    try:
        evaluator_class = getattr(axbench, evaluator_name)
        evaluator = evaluator_class(
            model_name, dump_dir=dump_dir, 
            concept_id=concept_id, lm_model=lm_model, winrate_baseline=winrate_baseline)
        eval_result = evaluator.compute_metrics(current_df)
        return (concept_id, evaluator.__str__(), model_name.__str__(), eval_result, \
                lm_model.stats.get_report(), None if bool(lm_caches) else lm_model.cache_in_mem, current_df)
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

    # Initialize data generator
    df_generator = data_generator(
        args.data_dir, mode="steering", 
        winrate_split_ratio=args.winrate_split_ratio)

    # Load previous state if exists
    state = load_state(args.dump_dir, mode="steering")
    start_concept_id = state.get("concept_id", 0) if state else 0
    logger.warning(f"Starting concept_id: {start_concept_id}")

    # Create all evaluation tasks - flattened for maximum parallelization
    all_tasks = [
        (concept_id, current_df, evaluator_name, model_name, args.dump_dir, \
         args.lm_model, args.winrate_baseline, {})
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
    eval_dfs = {}
    lm_caches = {}
    with ProcessPoolExecutor(max_workers=args.num_of_workers) as executor:
        for concept_id, evaluator_str, model_str, result, lm_report, lm_cache, current_df in executor.map(
            eval_steering_single_task, all_tasks):
            if concept_id not in all_results:
                all_results[concept_id] = {}
                eval_dfs[concept_id] = {}
            if evaluator_str not in all_results[concept_id]:
                all_results[concept_id][evaluator_str] = {}
                eval_dfs[concept_id][evaluator_str] = {}
            all_results[concept_id][evaluator_str][model_str] = result
            if "raw_relevance_concept_ratings" in result or \
                "raw_relevance_instruction_ratings" in result or \
                "raw_fluency_ratings" in result or \
                "raw_aggregated_ratings" in result:
                current_df[f"{model_str}_{evaluator_str}_relevance_concept_ratings"] = result["raw_relevance_concept_ratings"]
                current_df[f"{model_str}_{evaluator_str}_relevance_instruction_ratings"] = result["raw_relevance_instruction_ratings"]
                current_df[f"{model_str}_{evaluator_str}_fluency_ratings"] = result["raw_fluency_ratings"]
                current_df[f"{model_str}_{evaluator_str}"] = result["raw_aggregated_ratings"]
                current_df[f"{model_str}_{evaluator_str}_relevance_concept_completions"] = result["relevance_concept_completions"]
                current_df[f"{model_str}_{evaluator_str}_relevance_instruction_completions"] = result["relevance_instruction_completions"]
                current_df[f"{model_str}_{evaluator_str}_fluency_completions"] = result["fluency_completions"]
                eval_dfs[concept_id][evaluator_str][model_str] = current_df.copy()
            lm_reports += [lm_report]
            lm_caches.update(lm_cache)
            logger.warning(f"Completed task for concept_id: {concept_id}, model: {model_str}, evaluator: {evaluator_str}")

    # Batch save all results
    for concept_id, eval_results in sorted(all_results.items()):
        save_results(
            dump_dir, 
            {"concept_id": concept_id + 1}, 
            concept_id, 
            'steering', 
            eval_results, 
            eval_dfs[concept_id]
        )
        
    # Reload for plotting and optional winrate
    aggregated_results = process_jsonl_file(
        load_jsonl(os.path.join(Path(dump_dir) / "evaluate" / 'steering.jsonl')))

    if args.run_winrate:
        winrate_results = {}
        winrate_df_generator = winrate_data_generator(data_dir, aggregated_results, args.winrate_split_ratio)
        # Create all winrate evaluation tasks - flattened for maximum parallelization
        winrate_tasks = [
            (concept_id, current_df, "WinRateEvaluator", model_name, args.dump_dir, \
             args.lm_model, args.winrate_baseline, copy.deepcopy(lm_caches))
            for concept_id, current_df in winrate_df_generator
            # if concept_id >= start_concept_id # TODO: uncomment this when we fix our pipeline
            for model_name in args.models
            if model_name != args.winrate_baseline
        ]

        winrate_dfs = {}
        model_strs = set()
        with ProcessPoolExecutor(max_workers=args.num_of_workers) as executor:
            for concept_id, _, model_str, result, lm_report, _, current_df in executor.map(
                eval_steering_single_task, winrate_tasks):
                if concept_id not in winrate_results:
                    winrate_results[concept_id] = {}
                    winrate_dfs[concept_id] = {}
                winrate_results[concept_id][model_str] = result
                lm_reports += [lm_report]
                winrate_dfs[concept_id][model_str] = current_df
                model_strs.add(model_str)
                logger.warning(f"Completed winrate task for concept_id: {concept_id}, model: {model_str}")
        model_strs = list(model_strs)
        if winrate_dfs:
            vertical_concat = []
            for concept_id, winrate_df in winrate_dfs.items():
                # Take the first DataFrame as base and only keep unique columns from others
                base_df = winrate_dfs[concept_id][model_strs[0]]
                for model_str in model_strs[1:]:
                    # Only keep the win_result column from other DataFrames
                    win_col = f"{model_str}_win_result"
                    if win_col in winrate_dfs[concept_id][model_str].columns:
                        base_df[win_col] = winrate_dfs[concept_id][model_str][win_col]
                vertical_concat.append(base_df)
            winrate_df = pd.concat(vertical_concat, axis=0)
            winrate_df.to_parquet(Path(dump_dir) / "evaluate" / f"winrate.parquet", engine='pyarrow')

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
    plot_steering(aggregated_results, Path(dump_dir) / "evaluate", args.report_to, args.wandb_name)
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
    

def plot_latent(dump_dir, report_to=[], wandb_name=None):
    dump_dir = Path(dump_dir) / "evaluate"
    # aggregate all results
    aggregated_results = load_jsonl(os.path.join(dump_dir, 'latent.jsonl'))
    try:
        plot_aggregated_roc(
            aggregated_results, write_to_path=dump_dir, report_to=report_to, wandb_name=wandb_name)
        plot_accuracy_bars(
            aggregated_results, "HardNegativeEvaluator", write_to_path=dump_dir, 
            report_to=report_to, wandb_name=wandb_name)
    except Exception as e:
        logger.warning(f"Failed to plot: {e}")


def eval_latent(args):

    data_dir = args.data_dir
    dump_dir = args.dump_dir
    latent_data_path = os.path.join(data_dir, f'latent_data.parquet')
    if not os.path.exists(latent_data_path):
        logger.warning(f"Latent data not found at {latent_data_path}")
        return
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
            concept_id, 'latent', eval_results, None)

    # Generate final plot
    logger.warning("Generating final plot...")
    plot_latent(dump_dir, args.report_to, args.wandb_name)
    logger.warning("Evaluation completed!")


def main():
    custom_args = [
        {
            'args': ['--mode'],
            'kwargs': {
                'type': str,
                'default': "all",
                'help': 'The evaluation mode.'
            }
        }
    ]
    args = EvalArgs(custom_args=custom_args, section="evaluate")
    args.data_dir = f"{args.dump_dir}/inference"
    logger.warning("Evaluating generations with the following configuration:")
    logger.warning(args)
    
    dump_dir = Path(args.dump_dir) / "evaluate"
    dump_dir.mkdir(parents=True, exist_ok=True)

    # now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    # start wandb logging
    if args.report_to is not None and "wandb" in args.report_to:
        import wandb
        wandb_name = f"{args.dump_dir.split('/')[-1]}"
        run = wandb.init(
            project="AxBench", 
            entity=f"{args.wandb_entity}",
            name=f"{wandb_name}_{args.mode}" if args.run_name is None else f"{args.run_name}_{wandb_name}_{args.mode}",
        )
        
        with open(args.config_file, 'r') as file:
            additional_args = yaml.safe_load(file)
            run.summary.update(additional_args)

    if args.mode == "latent":
        eval_latent(args)
    elif args.mode == "steering":
        eval_steering(args)
    elif args.mode == "all":
        eval_latent(args)
        eval_steering(args)

    if args.report_to is not None and "wandb" in args.report_to:
        # log more metadata into wandb for visualization
        if (Path(args.dump_dir) / "evaluate" / "steering.jsonl").is_file() or \
            (Path(args.dump_dir) / "evaluate" / "latent.jsonl").is_file():
            metadata_path = Path(args.dump_dir) / "generate" / "metadata.jsonl"
            metadata = load_jsonl(metadata_path)

        concepts = []
        if (Path(args.dump_dir) / "evaluate" / "latent.jsonl").is_file():
            latent_path = Path(args.dump_dir) / "evaluate" / "latent.jsonl"
            latent_results = load_jsonl(latent_path)
            top_logits_path = Path(args.dump_dir) / "inference" / "top_logits.jsonl"
            top_logits_results = load_jsonl(top_logits_path)

            idx = 0
            for metadata_entry in metadata:
                for concept_idx, concept in enumerate(metadata_entry["concepts"]):
                    sae_link = metadata_entry["refs"][concept_idx]
                    auc = latent_results[idx]["results"]["AUCROCEvaluator"]["ReFT"]["roc_auc"]
                    max_act = latent_results[idx]["results"]["AUCROCEvaluator"]["ReFT"]["max_act"]
                    top_logits = top_logits_results[idx]["results"]["ReFT"]["top_logits"][0]
                    neg_logits = top_logits_results[idx]["results"]["ReFT"]["neg_logits"][0]
                    concepts += [[
                        idx, concept, None, auc, max_act, None, sae_link
                    ]]
                    top_table = wandb.Table(data=[(t[1], t[0] )for t in top_logits], columns=["logits", "token", ])
                    neg_table = wandb.Table(data=[(t[1], t[0] )for t in neg_logits], columns=["logits", "token", ])
                    wandb.log({f"positive_logits/{idx}": wandb.plot.bar(top_table, "token", "logits",
                                                    title=f"{concept} ({idx})")})
                    wandb.log({f"negative_logits/{idx}": wandb.plot.bar(neg_table, "token", "logits",
                                                    title=f"{concept} ({idx})")})
                    idx += 1
            
            # log token level heatmaps
            inference_path = Path(args.dump_dir) / "inference" / "latent_data.parquet"
            inference_df = pd.read_parquet(inference_path)
            heatmap_html = generate_html_with_highlight_text(inference_df)
            wandb.log({"latent/token_heatmap": wandb.Html(heatmap_html)})

        if (Path(args.dump_dir) / "evaluate" / "steering.jsonl").is_file():
            steering_path = Path(args.dump_dir) / "evaluate" / "steering.jsonl"
            steering_results = load_jsonl(steering_path)
            best_factors = get_best_factors(steering_results)

            idx = 0
            for metadata_entry in metadata:
                for concept_idx, concept in enumerate(metadata_entry["concepts"]):
                    sae_link = metadata_entry["refs"][concept_idx]
                    winrate = steering_results[idx]["results"]["WinRateEvaluator"]["ReFT"]["win_rate"]
                    best_factor = best_factors[idx]["ReFT"]
                    if len(concepts) <= idx:
                        concepts += [[
                            idx, concept, winrate, None, None, None, None
                        ]]
                    else:
                        concepts[idx][2] = winrate
                        concepts[idx][5] = best_factor
                    idx += 1

            # win-rate table logging
            steering_path = Path(args.dump_dir) / "evaluate" / "winrate.parquet"
            winrate_df = pd.read_parquet(steering_path)
            wandb.log({
                "steering/winrate": wandb.Table(dataframe=winrate_df)})
        
        if concepts is not None:
            wandb.log({
                "concept_table":  wandb.Table(
                    columns=[
                        "concept_id", "concept", "winrate", "auc", 
                        "max_act", "best_factor", "sae_link"], data=concepts)})

        run.finish()


if __name__ == "__main__":
    main()

