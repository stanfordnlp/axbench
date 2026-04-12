# score evaluation results.
# 
# example launch command:
#     python axbench/scripts/evaluate.py --config axbench/demo/sweep/evaluate.yaml --mode latent

import shutil
from axbench.models.language_models import (
    LanguageModel
)
import ast
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
from axbench.scripts.inference import LATENT_EXCLUDE_MODELS, STEERING_EXCLUDE_MODELS
import axbench
from axbench.utils.plot_utils import (
    plot_aggregated_roc, 
    plot_metrics,
    plot_accuracy_bars,
    plot_win_rates,
    plot_metrics_multiple_datasets,
)
from axbench.templates.html_templates import (
    generate_html_with_highlight_text,
)
from axbench.scripts.args.eval_args import EvalArgs
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
#multiprocessing.set_start_method("spawn", force=True)
from axbench.utils.constants import *
import stanza

# Load the English pipeline
nlp = stanza.Pipeline('en', processors='tokenize,pos', device = "cpu")

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)

STATE_FILE = "evaluate_state.pkl"

def harmonic_mean(scores):
    # Return 0 if any score is 0 to maintain strict evaluation
    if 0 in scores:
        return 0
    return len(scores) / sum(1/s for s in scores)

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
    elif "steering" in mode or mode == "winrate":
        df = pd.read_parquet(os.path.join(data_dir, f'steering_data.parquet'))
    elif mode == "train_data":
        df = pd.read_parquet(os.path.join(data_dir, f'dpo_train_data.parquet'))
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
        if winrate_split_ratio is not None and float(winrate_split_ratio) > 0:
            n_input_ids = df_subset["input_id"].max()+1
            n_steering_ids = n_input_ids - round(n_input_ids * winrate_split_ratio)
            if mode == "steering":
                df_subset = df_subset[df_subset["input_id"] < n_steering_ids]
            elif mode == "steering_test" or mode == "winrate":
                df_subset = df_subset[df_subset["input_id"] >= n_steering_ids]
        yield (concept_id, df_subset)


def get_best_factors(aggregated_results):
    best_factors = {}
    for result in aggregated_results:
        best_factors[result["concept_id"]] = {}
        for method, scores in result["results"]["LMJudgeEvaluator"].items():
            best_factors[result["concept_id"]][method] = scores["factor"][np.argmax(scores["lm_judge_rating"])]
    return best_factors

def get_best_factors_rule(steered_data):
    # Store best scores for each concept
    concepts = steered_data['concept_id'].unique()
    best_scores = []    
    # For each concept, split data and find best factor
    for concept in concepts:
        concept_data = steered_data[steered_data['concept_id'] == concept]     
        # Get indices for this concept's data
        indices = concept_data.index.values   
        # Randomly split indices into train and test
        np.random.seed(42)  # for reproducibility
        train_indices = np.random.choice(indices, size=len(indices)//2, replace=False)
        test_indices = np.array([idx for idx in indices if idx not in train_indices])        
        # Split data
        train_data = concept_data.loc[train_indices]
        test_data = concept_data.loc[test_indices]       
        # Find the factor that gives max RuleEvaluator score on train data
        train_rule_scores = train_data['PreferenceVector_RuleEvaluator']
        best_factor = train_data.loc[train_rule_scores.idxmax(), 'factor']
        
        # Get scores for the best factor using test data
        test_factor_data = test_data[test_data['factor'] == best_factor]
        
        if len(test_factor_data) > 0:  # Only add if we have test data for this factor
            # Get all metrics for this best factor using mean of test data
            metrics_data = {
                'Concept': f'Concept {concept}',
                'Factor': best_factor,
                'Overall': test_factor_data['PreferenceVector_RuleEvaluator'].mean(),
                'Rule Following':  test_factor_data['PreferenceVector_RuleEvaluator_rule_following'].mean(),
                'Relevance': test_factor_data['PreferenceVector_LMJudgeEvaluator_relevance_instruction_ratings'].mean(),
                'Fluency': test_factor_data['PreferenceVector_LMJudgeEvaluator_fluency_ratings'].mean()
            }
        best_scores.append(metrics_data['Factor'])

    return best_scores

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
        # print(sorted_evaluator_names)
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
                if evaluator_name == "LMJudgeEvaluator":
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
                else:
                    try:
                        current_df[f"{model_name}_{evaluator_name}_rule_following"] = \
                            eval_df[evaluator_name][model_name][f"{model_name}_{evaluator_name}_rule_following"]

                        
                    except:
                        current_df[f"{model_name}_{evaluator_name}_rule_following_winning"] = \
                            eval_df[evaluator_name][model_name][f"{evaluator_name}_rule_following_winning"]
                        current_df[f"{model_name}_{evaluator_name}_rule_following_losing"] = \
                            eval_df[evaluator_name][model_name][f"{evaluator_name}_rule_following_losing"]
                    
        df_path = os.path.join(dump_dir, f"{partition}_data.parquet")
        for model_name in sorted_model_names:
            if "RuleEvaluator" in sorted_evaluator_names:

                col1 = f"{model_name}_LMJudgeEvaluator_relevance_instruction_ratings"
                col2 = f"{model_name}_LMJudgeEvaluator_fluency_ratings"
                col3 = f"{model_name}_RuleEvaluator_rule_following"
                
                def safe_hmean(row):
                    values = [row.get(col1, None), row.get(col2, None), row.get(col3, None)]
                    print("out", values)
                    dataset_name = current_df["dataset_name"].iloc[0] if "dataset_name" in current_df.columns else ""
                    print(dataset_name)
                    if "Suppress" in dataset_name or "Attack" in dataset_name:
                        values_ = [values[0], values[1], 2 - values[2]]
                        print("in", values_)
                    else:
                        values_ = values
                    return harmonic_mean(values_)
                
                current_df[f"{model_name}_{evaluator_name}"] = current_df.apply(safe_hmean, axis=1)

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
    state_path = os.path.join(f"{dump_dir}", f"{mode}_{STATE_FILE}")
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


def plot_steering(aggregated_results, dump_dir, report_to=[], wandb_name=None, mode=None, rule = False):
    try:
        configs = [
            # {
            #     'evaluator_name': 'PerplexityEvaluator',
            #     'metric_name': 'perplexity',
            #     'y_label': 'Perplexity',
            #     'use_log_scale': False
            # },
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
            {
                'evaluator_name': 'RuleEvaluator',
                'metric_name': 'rule_following',
                'y_label': 'Rule',
                'use_log_scale': False
            }
        ]
        plot_metrics(
            jsonl_data=aggregated_results,
            configs=configs,
            write_to_path=dump_dir, 
            report_to=report_to,
            wandb_name=wandb_name,
            mode=mode
        )

    except Exception as e:
        logger.warning(f"Failed to plot: {e}")


def eval_steering_single_task(args_tuple):
    """Helper function to evaluate a single concept-model-evaluator combination"""
    concept_id, current_df, evaluator_name, model_name, dump_dir, \
        lm_model, winrate_baseline, lm_caches, steer_dataset_type = args_tuple
    
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
        temperature=0.0
    )
    # overwrite cache if any.
    if bool(lm_caches):
        lm_model.cache_in_mem = lm_caches
    
    try:
        evaluator_class = getattr(axbench, evaluator_name)
        ### accomodate for rule special cases that need stanza
        if "Rule" in evaluator_name and current_df['input_concept'].iloc[0] in NEED_STANZA:
            evaluator = evaluator_class(
            model_name, dump_dir=dump_dir, 
            concept_id=concept_id, lm_model=lm_model, winrate_baseline=winrate_baseline, steer_dataset_type=steer_dataset_type, nlp=nlp)
        else:
            evaluator = evaluator_class(
            model_name, dump_dir=dump_dir, 
            concept_id=concept_id, lm_model=lm_model, winrate_baseline=winrate_baseline, steer_dataset_type=steer_dataset_type)

        if "Rule" in evaluator_name and "winning_output" not in current_df.columns:
            eval_result = evaluator.compute_metrics(current_df, rule_type = CONCEPT_TO_RULE[current_df['input_concept'].iloc[0]])
        elif "Rule" in evaluator_name and "winning_output" in current_df.columns:
            eval_result = evaluator.compute_metrics_train(current_df, rule_type = CONCEPT_TO_RULE[current_df['output_concept'].iloc[0]])
        else:
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
        args.data_dir, mode=args.mode, 
        winrate_split_ratio=args.winrate_split_ratio)

    # Load previous state if exists
    state = load_state(args.dump_dir, mode=args.mode)
    start_concept_id = state.get("concept_id", 0) if state else 0
    finish_stanza = state.get("stanza", False) if state else False
    logger.warning(f"Starting concept_id: {start_concept_id}")

    if "PromptSteering" in args.models and args.defense is not None:
        # Remove PromptSteering from the models list
        args.models = [model for model in args.models if model != "PromptSteering"]
        # Add PromptSteering_d for each defense method
        if isinstance(args.defense, str):
            defense_list = ast.literal_eval(args.defense)
        else:
            defense_list = args.defense
        args.models.extend([f"PromptSteering_{d}" for d in defense_list])
        print(args.models)
        print("_"*100)


    all_tasks = [
        (concept_id, current_df, evaluator_name, model_name, args.dump_dir, \
         args.lm_model, args.winrate_baseline, { }, args.steer_data_type)
        for concept_id, current_df in df_generator
        if concept_id >= start_concept_id
        for evaluator_name in args.steering_evaluators
        for model_name in args.models
        if model_name not in STEERING_EXCLUDE_MODELS
        if not("Rule" in evaluator_name and current_df['input_concept'].iloc[0] in NEED_STANZA)
    ]

    # Group results by concept_id

    

    # Run all evaluations with process pool
    logger.warning(f"Number of workers: {args.num_of_workers}; Number of CPUs: {multiprocessing.cpu_count()}")
    if not hasattr(args, 'num_of_workers') or args.num_of_workers is None:
        args.num_of_workers = max(1, multiprocessing.cpu_count() - 1)

    lm_reports = []
    eval_dfs = {}
    all_results = {}
    lm_caches = {}
    ### accomodate for rule special cases that need stanza
    temp_results_path = os.path.join(dump_dir, f"temp_all_results.pkl")
    temp_dfs_path = os.path.join(dump_dir, f"temp_eval_dfs.pkl")
    print("finish stanza", finish_stanza)
    if not finish_stanza:
        # Create all evaluation tasks - flattened for maximum parallelization
        print("here")

        df_generator = data_generator(
            args.data_dir, mode=args.mode, 
            winrate_split_ratio=args.winrate_split_ratio)

        all_tasks_rule_special = [
            (concept_id, current_df, evaluator_name, model_name, args.dump_dir, \
            args.lm_model, args.winrate_baseline, { }, args.steer_data_type)
            for concept_id, current_df in df_generator
            if concept_id >= start_concept_id
            for evaluator_name in args.steering_evaluators
            for model_name in args.models
            if model_name not in STEERING_EXCLUDE_MODELS
            if ("Rule" in evaluator_name and current_df['input_concept'].iloc[0] in NEED_STANZA)
        ]

        for task in all_tasks_rule_special:
            concept_id, evaluator_str, model_str, result, lm_report, lm_cache, current_df = eval_steering_single_task(task)

            if concept_id not in all_results:
                all_results[concept_id] = {}
                eval_dfs[concept_id] = {}

            if evaluator_str not in all_results[concept_id]:
                all_results[concept_id][evaluator_str] = {}
                eval_dfs[concept_id][evaluator_str] = {}

            all_results[concept_id][evaluator_str][model_str] = result

            if ("raw_relevance_concept_ratings" in result or
                "raw_relevance_instruction_ratings" in result or
                "raw_fluency_ratings" in result or
                "raw_aggregated_ratings" in result) and evaluator_str == "LMJudgeEvaluator":
                current_df[f"{model_str}_{evaluator_str}_relevance_concept_ratings"] = result["raw_relevance_concept_ratings"]
                current_df[f"{model_str}_{evaluator_str}_relevance_instruction_ratings"] = result["raw_relevance_instruction_ratings"]
                current_df[f"{model_str}_{evaluator_str}_fluency_ratings"] = result["raw_fluency_ratings"]
                current_df[f"{model_str}_{evaluator_str}"] = result["raw_aggregated_ratings"]
                current_df[f"{model_str}_{evaluator_str}_relevance_concept_completions"] = result["relevance_concept_completions"]
                current_df[f"{model_str}_{evaluator_str}_relevance_instruction_completions"] = result["relevance_instruction_completions"]
                current_df[f"{model_str}_{evaluator_str}_fluency_completions"] = result["fluency_completions"]

                eval_dfs[concept_id][evaluator_str][model_str] = current_df.copy()

            if "rule_following" in result and evaluator_str == "RuleEvaluator" and args.mode != "train_data":
                current_df[f"{model_str}_{evaluator_str}_rule_following"] = result["raw_rule_following"]
                eval_dfs[concept_id][evaluator_str][model_str] = current_df.copy()
            
            elif "rule_following" in result and evaluator_str == "RuleEvaluator" and args.mode == "train_data":
                current_df[f"{evaluator_str}_rule_following_winning"] = result["raw_rule_following_winning"]
                current_df[f"{evaluator_str}_rule_following_losing"] = result["raw_rule_following_losing"]
                eval_dfs[concept_id][evaluator_str][model_str] = current_df.copy()

            lm_reports.append(lm_report)
            lm_caches.update(lm_cache)
            
            logger.warning(f"Completed task for concept_id: {concept_id}, model: {model_str}, evaluator: {evaluator_str}")
        

            # Save all_results and eval_dfs as temporary files

        
        with open(temp_results_path, "wb") as f:
            pickle.dump(all_results, f)
        with open(temp_dfs_path, "wb") as f:
            pickle.dump(eval_dfs, f)

        # Save state
        state_path = os.path.join(args.dump_dir, f"{args.mode}_{STATE_FILE}") 
        with open(state_path, "wb") as f:
            pickle.dump({"concept_id": 0, "stanza": True}, f)


    if os.path.exists(temp_results_path):
        with open(temp_results_path, "rb") as f:
            print("loading temp results")
            all_results = pickle.load(f)
            
    if os.path.exists(temp_dfs_path):
        with open(temp_dfs_path, "rb") as f:
            print("loading temp dfs")
            eval_dfs = pickle.load(f)

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
            if ("raw_relevance_concept_ratings" in result or \
                "raw_relevance_instruction_ratings" in result or \
                "raw_fluency_ratings" in result or \
                "raw_aggregated_ratings" in result) and evaluator_str == "LMJudgeEvaluator":
                current_df[f"{model_str}_{evaluator_str}_relevance_concept_ratings"] = result["raw_relevance_concept_ratings"]
                current_df[f"{model_str}_{evaluator_str}_relevance_instruction_ratings"] = result["raw_relevance_instruction_ratings"]
                current_df[f"{model_str}_{evaluator_str}_fluency_ratings"] = result["raw_fluency_ratings"]
                current_df[f"{model_str}_{evaluator_str}"] = result["raw_aggregated_ratings"]
                current_df[f"{model_str}_{evaluator_str}_relevance_concept_completions"] = result["relevance_concept_completions"]
                current_df[f"{model_str}_{evaluator_str}_relevance_instruction_completions"] = result["relevance_instruction_completions"]
                current_df[f"{model_str}_{evaluator_str}_fluency_completions"] = result["fluency_completions"]              
                eval_dfs[concept_id][evaluator_str][model_str] = current_df.copy()

            if "rule_following" in result and evaluator_str == "RuleEvaluator" and args.mode != "train_data":
                current_df[f"{model_str}_{evaluator_str}_rule_following"] = result["raw_rule_following"]
                eval_dfs[concept_id][evaluator_str][model_str] = current_df.copy()

            elif "rule_following" in result and evaluator_str == "RuleEvaluator" and args.mode == "train_data":
                current_df[f"{evaluator_str}_rule_following_winning"] = result["raw_rule_following_winning"]
                current_df[f"{evaluator_str}_rule_following_losing"] = result["raw_rule_following_losing"]
                eval_dfs[concept_id][evaluator_str][model_str] = current_df.copy()
            
            lm_reports += [lm_report]
            lm_caches.update(lm_cache)
            logger.warning(f"Completed task for concept_id: {concept_id}, model: {model_str}, evaluator: {evaluator_str}")

        
    for concept_id, eval_results in sorted(all_results.items()):
        save_results(
            dump_dir, 
            {"concept_id": concept_id+1, "stanza": True}, 
            concept_id, 
            args.mode, 
            eval_results, 
            eval_dfs[concept_id]
        )
        

    if args.mode == "train_data":
        return
    # Reload for plotting and optional winrate
    
    
    aggregated_results = process_jsonl_file(
        load_jsonl(os.path.join(dump_dir / f'{args.mode}.jsonl')))
   
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
    plot_steering(aggregated_results, dump_dir, args.report_to, args.wandb_name, args.mode)
    plot_metrics_multiple_datasets(os.path.join(dump_dir, "steering_data.parquet"), dump_dir, args.report_to, args.wandb_name, args.mode, rule = args.steer_data_type=="rule")
    #plot_metrics_multiple_datasets(os.path.join(dump_dir, "steering_data.parquet"), dump_dir, args.report_to, args.wandb_name, args.mode, rule = args.steer_data_type=="rule")
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
    # aggregate all results
    aggregated_results = load_jsonl(os.path.join(dump_dir, 'latent.jsonl'))
    plot_aggregated_roc(
        aggregated_results, write_to_path=dump_dir, report_to=report_to, wandb_name=wandb_name)
    plot_accuracy_bars(
        aggregated_results, "HardNegativeEvaluator", write_to_path=dump_dir, 
        report_to=report_to, wandb_name=wandb_name)


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
        for evaluator_name in args.latent_evaluators:
            for model_name in args.models:
                if model_name in LATENT_EXCLUDE_MODELS:
                    continue
                evaluator_class = getattr(axbench, evaluator_name)
                evaluator = evaluator_class(model_name)
                result = evaluator.compute_metrics(current_df)
                if evaluator_name not in eval_results:
                    eval_results[evaluator_name] = {}
                eval_results[evaluator_name][model_name] = result

        save_results(
            dump_dir,
            {"concept_id": concept_id + 1},
            concept_id,
            "latent",
            eval_results,
        )

    # Generate plots
    plot_latent(dump_dir, args.report_to if hasattr(args, 'report_to') else [],
                args.wandb_name if hasattr(args, 'wandb_name') else None)


def main():
    custom_args = [
        {
            'args': ['--mode'],
            'kwargs': {
                'type': str,
                'default': "all",
                'help': 'The evaluation mode.'
            }
        },
    ]
    args = EvalArgs(custom_args=custom_args, section="evaluate", ignore_unknown=True)
    if args.mode == "train_data":
        args.data_dir = f"{args.dump_dir}/generate" if args.overwrite_inference_dump_dir is None else Path(args.overwrite_inference_dump_dir)
    else:
        args.data_dir = f"{args.dump_dir}/inference" if args.overwrite_inference_dump_dir is None else Path(args.overwrite_inference_dump_dir)
    logger.warning("Evaluating generations with the following configuration:")
    logger.warning(args)

    dump_dir = Path(args.dump_dir) / "evaluate" if args.overwrite_evaluate_dump_dir is None else Path(args.overwrite_evaluate_dump_dir)

    dump_dir.mkdir(parents=True, exist_ok=True)
    args.dump_dir = dump_dir

    if args.mode == "latent":
        eval_latent(args)
    elif "steering" in args.mode: # steering or steering_test
        eval_steering(args)
    elif args.mode == "train_data":
        eval_steering(args)
    elif args.mode == "all":
        eval_latent(args)
        eval_steering(args)

if __name__ == "__main__":
    main()