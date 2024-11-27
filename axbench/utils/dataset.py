import pandas as pd
import asyncio
import os, random, json, time, requests, copy, asyncio, csv, math
import torch, transformers, datasets
from datasets import load_from_disk
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union, List, Any
from pathlib import Path
import pandas as pd
from collections import namedtuple

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


from ..models.language_models import LanguageModel
from ..utils.prompt_utils import (
    get_contrast_concepts,
    get_concept_genres,
    get_random_content,
    modify_content_with_concept,
    modify_content_with_polysemantic_concepts,
    sample_index_exclude,
    continue_with_concept,
    response_with_concept,
    continue_without_concept,
    response_without_concept,
    continue_with_polysemantic_concepts,
    response_with_polysemantic_concepts,
    continue_with,
    response_with,
)
from ..utils.constants import EXAMPLE_TAG, EMPTY_CONCEPT
from ..utils.model_utils import get_model_continues


T_PROMPT_STEERING = """You must answer the question with content \
related to %s even if it is not related to the question or it does not make sense."""


T_GENERATE_STEERING_PROMPT = """Generate a prompt to guide a language \
model in producing responses. 

Objective: 
Direct the model to include content related to %s (the concept) in its responses. 
Ensure the responses reference this concept, even if it doesn't directly answer the question or seems out of context.
Optionally, provide in-context examples to reinforce this behavior.
        
Return only the final prompt without any additional text."""


# special types for dataset
Prompt = namedtuple("Prompt", ["concept", "tag", "content"])


async def run_tasks(tasks):
    # Gather and run all provided tasks concurrently, and collect their results
    results = await asyncio.gather(*tasks)
    return results


class DatasetFactory(object):
    """Main class of async generating training pairs for two subspaces"""

    def __init__(
        self, model, client, tokenizer, dataset_category, num_of_examples, output_length, dump_dir, 
        use_cache=True, master_data_dir=None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer

        # prepare lm model
        lm_model = kwargs.get("lm_model", "gpt-4o-mini")
        self.lm_model = LanguageModel(
            lm_model, client, dump_dir, 
            use_cache=use_cache, master_data_dir=master_data_dir
        )
        self.seed = kwargs.get("seed", 42)
        self.logger = kwargs.get("logger", logger)

        # load seed sentences
        self.seed_sentences = load_from_disk(os.path.join(master_data_dir, "seed_sentences"))
        self.seed_instructions = load_from_disk(os.path.join(master_data_dir, "seed_instructions"))
        self.dataset_category = dataset_category

        # load negative examples all at once
        if not kwargs.get("is_inference", False):
            start = time.time()
            self.logger.warning("[Traing only] Creating random examples as negative examples for all concepts.")
            functor = continue_with if self.dataset_category == "continuation" else response_with
            random_examples = []
            random_content = get_random_content(
                self.seed_sentences if self.dataset_category == "continuation" else self.seed_instructions, 
                tokenizer=tokenizer, count=num_of_examples, 
                genres={"random": ["text"]}, 
                concepts=["random"], length=None, split="test"
            )
            continue_task = functor(self.lm_model, self.tokenizer, content=random_content["random"], length=output_length)
            concept_outputs = asyncio.run(run_tasks([continue_task]))[0]
            for i, (prompt, output) in enumerate(zip(random_content["random"], concept_outputs)):
                random_examples += [[
                    prompt, output, EMPTY_CONCEPT, self.dataset_category
                ]]
            self.negative_df = pd.DataFrame(
                random_examples, 
                columns = ['input', 'output', 'output_concept', 'category'])
            self.negative_df["concept_id"] = -1
            self.logger.warning(f"Finished creating negative examples in {round(time.time() - start, 3)} sec.")
    
    def save_cache(self):
        """Save the language model cache before exiting"""
        self.lm_model.save_cache()

    def reset_stats(self):
        """Reset API costs"""
        self.lm_model.dump()
        self.lm_model.stats.print_report()
        self.lm_model.stats.reset()

    def prepare_genre_concepts(self, concepts, **kwargs):
        start = time.time()
        tasks = []

        # prepare genres if needed
        concept_genres_map = kwargs.get("concept_genres_map", None)
        if concept_genres_map is None:
            logger.warning("Creating genre for the inputs (not provided).")
            genre_task = get_concept_genres(
                self.lm_model, concepts, 
                api_tag=kwargs.get("api_tag", "")
            )
            tasks.append(genre_task)
        
        # run tasks
        res = asyncio.run(run_tasks(tasks))
        concept_genres_map = res[0]

        # log
        logger.warning(f"Init finished in {round(time.time() - start, 3)} sec.")
        return concept_genres_map

    def prepare_concepts(self, concepts, **kwargs):
        start = time.time()
        tasks = []
        
        # contrast concepts
        logger.warning("Creating contrast concepts for the inputs.")
        contrast_task = get_contrast_concepts(
            self.lm_model, concepts, kwargs.get("contrast_concepts_map", None), 
            api_tag=kwargs.get("api_tag", ""))
        tasks.append(contrast_task)

        # prepare genres if needed
        concept_genres_map = kwargs.get("concept_genres_map", None)
        if concept_genres_map is None:
            logger.warning("Creating genre for the inputs (not provided).")
            genre_task = get_concept_genres(
                self.lm_model, concepts, 
                api_tag=kwargs.get("api_tag", "")
            )
            tasks.append(genre_task)
        
        # run tasks
        res = asyncio.run(run_tasks(tasks))
        contrast_concepts_map = res[0]
        if len(res) > 1:
            concept_genres_map = res[1]

        # log
        for concept in concepts:
            logger.warning(f"Found {len(contrast_concepts_map[concept])} contrast concept(s) for concept: {concept}.")
        logger.warning(f"Init finished in {round(time.time() - start, 3)} sec.")
        return concept_genres_map, contrast_concepts_map

    def create_eval_df(
        self, concepts, subset_n, concept_genres_map, 
        train_contrast_concepts_map, eval_contrast_concepts_map, **kwargs):
        """category: positive, negative, hard negative"""
               
        # start logging
        start = time.time()
        self.logger.warning("Creating dataframe.")
        
        # init vars
        lm_model, model, tokenizer = self.lm_model, self.model, self.tokenizer 
        output_length = kwargs.get("output_length", 32)

        all_examples = []
        concepts_random_content = get_random_content(
            self.seed_sentences if self.dataset_category == "continuation" else self.seed_instructions, 
            tokenizer=tokenizer, count=subset_n*3, 
            genres=concept_genres_map, concepts=concepts, length=None, split="test"
        )

        if self.dataset_category == "continuation":
            functors = [continue_with_concept, continue_without_concept, continue_with_polysemantic_concepts]
        elif self.dataset_category == "instruction":
            functors = [response_with_concept, response_without_concept, response_with_polysemantic_concepts]
        else:
            raise ValueError(f"Unknown dataset category: {self.dataset_category}")

        for idx, concept in enumerate(concepts):
            # positive continuation / instruction
            continue_task = functors[0](
                self.lm_model, self.tokenizer, 
                concepts=[concept]*len(concepts_random_content[concept][:subset_n]), 
                content=concepts_random_content[concept][:subset_n], length=output_length)
            concept_outputs = asyncio.run(run_tasks([continue_task]))[0]
            for i, (prompt, output) in enumerate(zip(concepts_random_content[concept][:subset_n], concept_outputs)):
                all_examples += [[
                    prompt, output, concept, "positive"
                ]]

            # negative continuation / instruction
            continue_task = functors[1](
                self.lm_model, self.tokenizer, 
                content=concepts_random_content[concept][subset_n:2*subset_n], 
                concepts=[concept]*len(concepts_random_content[concept][subset_n:2*subset_n]), 
                length=output_length)
            concept_outputs = asyncio.run(run_tasks([continue_task]))[0]
            for i, (prompt, output) in enumerate(zip(concepts_random_content[concept][subset_n:2*subset_n], concept_outputs)):
                all_examples += [[
                    prompt, output, concept, "negative"
                ]]

            # hard negative continuation / instruction
            splits = [
                ("hard negative", eval_contrast_concepts_map[concept]),
            ]
            eval_tasks = []
            tags = []
            for (label, polysemantic_meanings) in splits:
                if len(polysemantic_meanings) != 0:
                    polysemantic_random_content = \
                        concepts_random_content[concept][2*subset_n:2*subset_n+len(polysemantic_meanings)]
                    eval_tasks.append(functors[2](
                        client=lm_model, tokenizer=tokenizer,
                        polysemantic_concepts=polysemantic_meanings,
                        concept=concept, content=polysemantic_random_content,
                        length=output_length
                    ))
                    tags.append((label, concept, idx))
            hard_negative_eval_content = asyncio.run(run_tasks(eval_tasks))
            for (tag, concept, idx), eval_content in zip(tags, hard_negative_eval_content):
                all_examples += [[content[0], content[2], "//".join(content[1]), tag] for content in eval_content[1]]

        # update the column definitions of the DataFrame
        df = pd.DataFrame(
            all_examples, 
            columns = [
                'input', 'output', 'output_concept', 'category'
            ])
        self.logger.warning(f"Finished creating current dataframe in {round(time.time() - start, 3)} sec.")
        return df
    
    def create_train_df(self, concept, n, concept_genres_map, **kwargs):
        lm_model, model, tokenizer = self.lm_model, self.model, self.tokenizer
        
        start = time.time()
        self.logger.warning("Creating dataframe.")
        all_examples = []

        output_length = kwargs.get("output_length", 32)

        functors = []
        if self.dataset_category == "continuation":
            functors = [continue_with_concept, continue_without_concept]
        else:
            functors = [response_with_concept, response_without_concept]
        
        # random sentence or instruction
        concepts_random_content = get_random_content(
            self.seed_sentences if self.dataset_category == "continuation" else self.seed_instructions, 
            tokenizer=tokenizer, count=n*2, 
            genres=concept_genres_map, concepts=[concept], length=None, split="train"
        )

        # positive continuation / instruction
        continue_task = functors[0](
            self.lm_model, self.tokenizer, 
            concepts=[concept]*len(concepts_random_content[concept][:n]), 
            content=concepts_random_content[concept][:n], length=output_length)
        concept_outputs = asyncio.run(run_tasks([continue_task]))[0]
        for i, (prompt, output) in enumerate(zip(concepts_random_content[concept][:n], concept_outputs)):
            all_examples += [[
                prompt, output, concept, self.dataset_category
            ]]

        # update the column definitions of the DataFrame
        df = pd.DataFrame(
            all_examples, 
            columns = [
                'input', 'output', 'output_concept', 'category'
            ])
        self.logger.warning(f"Finished creating current dataframe in {round(time.time() - start, 3)} sec.")
        return df
        

async def get_steering_prompts(client, concepts):
    prompts = []
    for concept in concepts:
        prompts += [T_GENERATE_STEERING_PROMPT % (concept)]
    responses = await client.chat_completions("get_steering_prompts", prompts)
    return responses


class SteeringDatasetFactory(object):
    def __init__(
        self, tokenizer, dump_dir, **kwargs):
        self.tokenizer = tokenizer
        self.master_data_dir = kwargs.get("master_data_dir", None)
        if kwargs.get("lm_client", None):
            self.lm_model = LanguageModel(
                kwargs.get("lm_model", "gpt-4o-mini"), kwargs["lm_client"], dump_dir, 
                use_cache=True, master_data_dir=self.master_data_dir
            )

    def create_eval_df(self, concepts, subset_n, steering_factors, steering_datasets):
        for dataset_name in steering_datasets:
            if dataset_name == "OUATPrefix":
                # we generate subset_n * n_steering_factors examples for OUATPrefix.
                # OUATPrefix is basically a prefix dataset.
                # "Once upon a time, " is the prefix, and there is no other labels.
                # we also need to label these in groups:
                # each one of subset_n group has the same group id.
                all_examples = []
                for idx, concept in enumerate(concepts):
                    for i in range(subset_n):
                        for factor in steering_factors:
                            all_examples += [
                                [dataset_name, idx, concept, i, factor, "Once upon a time, there was a ", ]
                            ]
                df = pd.DataFrame(
                    all_examples, 
                    columns = [
                        'dataset_name', 'concept_id', 'input_concept', 'input_id', 'factor', 'input'])
                return df
            elif dataset_name == "AlpacaEval":
                # load alpaca eval dataset.
                assert self.master_data_dir is not None, "Master data dir is required for AlpacaEval."
                alpaca_eval_path = os.path.join(self.master_data_dir, "alpaca_eval.json")
                alpaca_eval_df = pd.read_json(alpaca_eval_path)

                # get gpt-4o boosted steering prompts.
                steering_prompts = asyncio.run(get_steering_prompts(self.lm_model, concepts))
                steering_prompts = [prompt.strip() for prompt in steering_prompts]
                all_examples = []
                for idx, concept in enumerate(concepts):
                    # sample a random example from alpaca eval dataset.
                    sampled_prompts = alpaca_eval_df.sample(subset_n)["instruction"].tolist()
                    for i in range(subset_n):
                        sampled_prompt = sampled_prompts[i]
                        # for prompt-based steering ONLY.
                        steering_prompt = steering_prompts[idx] \
                            if steering_prompts[idx] != "" else T_PROMPT_STEERING % (concept)
                        steered_prompt = f" {steering_prompt}\n\nQuestion: {sampled_prompt}"
                        formatted_steered_prompt = self.tokenizer.apply_chat_template(
                            [{"role": "user", "content": steered_prompt}], 
                            tokenize=True, add_generation_prompt=True)[1:] # get rid of bos token
                        formatted_steered_prompt = self.tokenizer.decode(formatted_steered_prompt)
                        # apply the tokenizer chat format to the prompt.
                        formatted_prompt = self.tokenizer.apply_chat_template(
                            [{"role": "user", "content": sampled_prompt}], 
                            tokenize=True, add_generation_prompt=True)[1:] # get rid of bos token
                        formatted_prompt = self.tokenizer.decode(formatted_prompt)
                        for factor in steering_factors:
                            all_examples += [[
                                dataset_name, idx, concept, i, factor, 
                                sampled_prompt, formatted_steered_prompt, formatted_prompt
                            ]]
                df = pd.DataFrame(
                    all_examples, 
                    columns = [
                        'dataset_name', 'concept_id', 'input_concept', 
                        'input_id', 'factor', 'original_prompt', 'steered_input', 'input'])
                return df
            elif dataset_name == "AlpacaEval_Suppress" or dataset_name == "AlpacaEval_Synergy":
                # load alpaca eval dataset.
                assert self.master_data_dir is not None, "Master data dir is required for AlpacaEval."
                alpaca_eval_path = os.path.join(self.master_data_dir, "alpaca_eval.json")
                alpaca_eval_df = pd.read_json(alpaca_eval_path)
                common_steering_factors = steering_factors
                if dataset_name == "AlpacaEval_Suppress":
                    common_steering_factors = [f*-1.0 for f in common_steering_factors]
                # get gpt-4o boosted steering prompts.
                steering_prompts = asyncio.run(get_steering_prompts(self.lm_model, concepts))
                steering_prompts = [prompt.strip() for prompt in steering_prompts]
                all_examples = []
                for idx, concept in enumerate(concepts):
                    for i in range(subset_n):
                        # sample a random example from alpaca eval dataset.
                        sampled_prompt = alpaca_eval_df.sample(1)["instruction"].tolist()[0]
                        # for prompt-based steering ONLY.
                        steering_prompt = steering_prompts[idx] \
                            if steering_prompts[idx] != "" else T_PROMPT_STEERING % (concept)
                        steered_prompt = f" {steering_prompt}\n\nQuestion: {sampled_prompt}"
                        formatted_steered_prompt = self.tokenizer.apply_chat_template(
                            [{"role": "user", "content": steered_prompt}], 
                            tokenize=False, add_generation_prompt=True)
                        for factor in common_steering_factors:
                            all_examples += [[
                                dataset_name, idx, concept, i, factor, 
                                sampled_prompt, formatted_steered_prompt,
                            ]]
                df = pd.DataFrame(
                    all_examples, 
                    columns = [
                        'dataset_name', 'concept_id', 'input_concept', 
                        'input_id', 'factor', 'original_prompt', 'input'])
                return df
            else:
                # not implemented yet.
                raise NotImplementedError(f"Steering dataset {dataset_name} not implemented.")