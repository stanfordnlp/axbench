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
)
from ..utils.constants import EXAMPLE_TAG
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
        self, model, client, tokenizer, dump_dir, 
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

    def create_eval_df(self, concepts, subset_n, concept_genres_map, train_contrast_concepts_map, eval_contrast_concepts_map, **kwargs):
        """category: positive, negative, hard negative"""
               
        # start logging
        start = time.time()
        self.logger.warning("Creating dataframe.")
        
        # init vars
        lm_model, model, tokenizer = self.lm_model, self.model, self.tokenizer 
        all_examples = []
        input_length = kwargs.get("input_length", 32)
        output_length = kwargs.get("output_length", 10)

        concepts_random_content = get_random_content(
            self.seed_sentences, tokenizer=tokenizer, count=subset_n*3, 
            genres=concept_genres_map, concepts=concepts, length=input_length, split="test"
        )

        tags = []
        eval_tasks = []
        negative_examples = []
        for idx, concept in enumerate(concepts):

            # positive
            eval_tasks.append(modify_content_with_concept(
                client=lm_model, tokenizer=tokenizer,
                content=[(concept, "", content) for content in concepts_random_content[concept][:subset_n]],  # keep the same interface
                length=input_length
            ))
            tags.append(("positive", concept, idx))

            # negative
            negative_examples += [[content, concept, "negative"] for content in concepts_random_content[concept][subset_n:2*subset_n]]

            # hard negative seen + unseen
            splits = [
                ("hard negative seen", train_contrast_concepts_map[concept]),
                ("hard negative unseen", eval_contrast_concepts_map[concept]),
            ]
            for (label, polysemantic_meanings) in splits:
                if len(polysemantic_meanings) != 0:
                    polysemantic_random_content = \
                        concepts_random_content[concept][2*subset_n:2*subset_n+len(polysemantic_meanings)]
                    eval_tasks.append(modify_content_with_polysemantic_concepts(
                        client=lm_model, tokenizer=tokenizer,
                        polysemantic_concepts=polysemantic_meanings,
                        concept=concept, content=polysemantic_random_content,
                        length=input_length
                    ))
                    tags.append((label, concept, idx))

        # run tasks
        all_eval_content = asyncio.run(run_tasks(eval_tasks))

        # make dataset
        hard_negative_examples = []
        for (tag, concept, idx), eval_content in zip(tags, all_eval_content):
            if tag in {"positive"}:
                all_examples += [[content, concept, tag] for content in eval_content]
            elif tag in {"hard negative seen", "hard negative unseen"}:
                hard_negative_examples += [[content[1], "//".join(content[0]), tag] for content in eval_content[1]]
        all_examples += negative_examples + hard_negative_examples
        
        # make df
        df = pd.DataFrame(all_examples, columns=['input', 'input_concept', 'category'])
        df = df[df["input"].str.strip() != ""]
        self.logger.warning(f"Finished creating current dataframe in {round(time.time() - start, 3)} sec.")
        return df
    
    def create_train_df(self, concept, n, concept_genres_map, **kwargs):
        lm_model, model, tokenizer = self.lm_model, self.model, self.tokenizer
        
        start = time.time()
        self.logger.warning("Creating dataframe.")
        all_examples = []

        input_length = kwargs.get("input_length", 32)
        output_length = kwargs.get("output_length", 10)

        # for each concept, we create a set of seed random content.
        concepts_random_content = get_random_content(
            self.seed_sentences, tokenizer=tokenizer, count=n, 
            genres=concept_genres_map, concepts=[concept], length=input_length, split="train"
        )

        continue_task = continue_with_concept(
            self.lm_model, self.tokenizer, 
            concepts=[concept]*len(concepts_random_content[concept]), 
            content=concepts_random_content[concept], length=output_length)
        concept_outputs = asyncio.run(run_tasks([continue_task]))[0]

        for i, (prompt, output) in enumerate(zip(concepts_random_content[concept], concept_outputs)):
            all_examples += [[
                prompt, output, 0, 0, "empty", concept
            ]]

        # update the column definitions of the DataFrame
        df = pd.DataFrame(
            all_examples, 
            columns = [
                'input', 'output', 'input_subspace', 'output_subspace', 'input_concept', 'output_concept',
            ])
        self.logger.warning(f"Finished creating current dataframe in {round(time.time() - start, 3)} sec.")
        return df

    # Unused for now.
    def create_paired_train_df(self, concepts, n, concept_genres_map, contrast_concepts_map, **kwargs):
        """This is delayed for now :)"""
        concept2id = {concept: i for i, concept in enumerate(concepts)}
        lm_model, model, tokenizer = self.lm_model, self.model, self.tokenizer
        
        start = time.time()
        self.logger.warning("Creating dataframe.")
        n_per_concept = n // (len(concepts) + 1)
        all_examples = []

        input_length = kwargs.get("input_length", 32)
        output_length = kwargs.get("output_length", 10)
        
        # for each concept, we create a set of seed random content.
        concepts_random_content = get_random_content(
            self.seed_sentences, tokenizer=tokenizer, count=3*n, 
            genres=concept_genres_map, concepts=concepts, length=input_length, split="train"
        )
        
        # for concepts with polysemantic senses, we create additional examples.
        polysemantic_tasks = []
        for concept in concepts:
            if len(contrast_concepts_map[concept]) != 0:
                count = n_per_concept // (len(concepts)*2)
                polysemantic_concepts = [random.choice(contrast_concepts_map[concept]) for _ in range(count)]
                polysemantic_tasks.append(modify_content_with_polysemantic_concepts(
                    client=lm_model, tokenizer=tokenizer,
                    polysemantic_concepts=polysemantic_concepts,
                    concept=concept, content=concepts_random_content[concept][:n],
                    length=input_length
                ))
        polysemantic_content = asyncio.run(run_tasks(polysemantic_tasks))        
        polysemantic_content = {content[0]: content[1] for content in polysemantic_content}
        
        # aggregate these null examples.
        null_prompts = []
        for concept in concepts:
            n_random = (n_per_concept // len(concepts)) if len(contrast_concepts_map[concept]) == 0 else n_per_concept // (len(concepts)*2)
            for content in concepts_random_content[concept][n:n+n_random]:
                null_prompts.append(
                    Prompt(concept=concept, tag="empty", content=content))
            if len(contrast_concepts_map[concept]) != 0:
                for content in polysemantic_content[concept]:
                    null_prompts.append(
                        Prompt(concept=concept, tag=f"{content[0][0]}//{content[0][1]}",
                               content=content[1]))

        # get continuations from STEERED MODEL (not datagen model)
        null_outputs = get_model_continues(
            model=model, tokenizer=tokenizer, prompts=[p.content for p in null_prompts],
            max_new_tokens=int(output_length*1.5)
        )

        # Save control examples
        for prompt, output in zip(null_prompts, null_outputs):
            in_idx = concept2id[prompt.concept]
            out_idx = sample_index_exclude(len(concepts), in_idx)
            all_examples += [[
                prompt.content, output, EXAMPLE_TAG.CONTROL.value,
                in_idx, out_idx, prompt.tag, "empty",
            ]]
            
        # modify exist content to have desired concepts.
        modify_prompts = []
        for concept in concepts:
            for prompt in concepts_random_content[concept][2*n:2*n+len(null_prompts)]:
                modify_prompts.append(
                    Prompt(concept=concept, tag="empty", content=prompt))  # include source content ID
        modify_task = modify_content_with_concept(
            client=lm_model, tokenizer=tokenizer,
            content=[(p.concept, p.tag, p.content) for p in modify_prompts],  # keep the same interface
            length=input_length
        )
        concept_prompts = asyncio.run(run_tasks([modify_task]))[0]

        # process experiment examples with content tracking
        inverse_concepts = [concepts[sample_index_exclude(len(concepts), concept2id[prompt[0]])]
            for prompt in modify_prompts]
        continue_task = continue_with_concept(
            self.lm_model, self.tokenizer, 
            concepts=inverse_concepts, content=concept_prompts, length=output_length)
        concept_outputs = asyncio.run(run_tasks([continue_task]))[0]

        for i, (prompt, output) in enumerate(zip(concept_prompts, concept_outputs)):
            in_idx = concept2id[modify_prompts[i][0]]
            out_idx = concept2id[inverse_concepts[i]]
            all_examples += [[
                prompt, output, EXAMPLE_TAG.EXPERIMENT.value,
                in_idx, out_idx, modify_prompts[i][0], inverse_concepts[i],
            ]]

        # update the column definitions of the DataFrame
        df = pd.DataFrame(
            all_examples, 
            columns = [
                'input', 'output', 'group', 'input_subspace', 'output_subspace', 
                'input_concept', 'output_concept',
            ])
        self.logger.warning(f"Finished creating current dataframe in {round(time.time() - start, 3)} sec.")
        return df
        

@dataclass
class ReftDataCollator(object):
    """Collate examples for ReFT."""
    
    tokenizer: transformers.AutoTokenizer
    data_collator: transformers.DataCollator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        max_intervention_len = max([len(inst["intervention_locations"][0]) for inst in instances])
        max_seq_len = max([len(inst["input_ids"]) for inst in instances])
        
        for inst in instances:
            non_pad_len = len(inst["input_ids"])

            _intervention_mask = torch.ones_like(inst["intervention_locations"][0])
            _intervention_location_paddings = torch.tensor(
                [[len(inst["input_ids"]) for _ in range(max_intervention_len - len(inst["intervention_locations"][0]))]])
            _intervention_mask_paddings = torch.tensor(
                [0 for _ in range(max_intervention_len - len(inst["intervention_locations"][0]))])
            inst["intervention_locations"] = torch.cat([inst["intervention_locations"], _intervention_location_paddings], dim=-1).int()
            inst["intervention_masks"] = torch.cat([_intervention_mask, _intervention_mask_paddings], dim=-1).int()
            inst["input_subspaces"] = inst["input_subspaces"].int()
            inst["output_subspaces"] = inst["output_subspaces"].int()
            _input_id_paddings = torch.tensor(
                [self.tokenizer.pad_token_id for _ in range(max_seq_len - non_pad_len)])
            # intervention sink token
            inst["input_ids"] = torch.cat((inst["input_ids"], torch.tensor([self.tokenizer.pad_token_id]), _input_id_paddings)).int()

            _label_paddings = torch.tensor([-100 for _ in range(max_seq_len - non_pad_len + 1)])
            inst["labels"] = torch.cat((inst["labels"], _label_paddings))
            
            inst["attention_mask"] = (inst["input_ids"] != self.tokenizer.pad_token_id).int()

        batch_inputs = self.data_collator(instances)
        return batch_inputs



def make_data_module(
    tokenizer: transformers.PreTrainedTokenizer, df,
):
    """Make dataset and collator for supervised fine-tuning with kl div loss."""
    
    all_base_input_ids, all_intervention_locations, all_output_ids, \
        all_input_subspaces, all_output_subspaces = [], [], [], [], []
    all_prompt_lengths = []

    for _, row in df.iterrows():
        _input, _output, _input_subspace, _output_subspace = row["input"], row["output"], \
            int(row["input_subspace"]), int(row["output_subspace"])
        # prepare input ids
        base_prompt = _input
        if isinstance(_output, float):
            _output = tokenizer.eos_token
        base_input = base_prompt + _output
        base_prompt_ids = tokenizer(
            base_prompt, max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]
        base_input_ids = tokenizer(
            base_input, max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]
        base_prompt_length = len(base_prompt_ids)
        base_length = len(base_input_ids)

        # output ids with prompt token mask
        output_ids = base_input_ids.clone()
        output_ids[:base_prompt_length] = -100

        # self.logger.warning("tokens with lm loss:")
        # self.logger.warning(tokenizer.batch_decode(output_ids[output_ids!=-100].unsqueeze(dim=-1)))

        intervention_locations = torch.tensor([[i for i in range(1, base_prompt_length)]])
        all_intervention_locations.append(intervention_locations)
        all_base_input_ids.append(base_input_ids)
        all_output_ids.append(output_ids)
        all_input_subspaces.append(torch.tensor(_input_subspace))
        all_output_subspaces.append(torch.tensor(_output_subspace))
        all_prompt_lengths.append(torch.tensor(base_prompt_length - 1)) # exclude bos token
        
    train_dataset = datasets.Dataset.from_dict({
        "input_ids": all_base_input_ids,
        "intervention_locations": all_intervention_locations,
        "labels": all_output_ids,
        "input_subspaces": all_input_subspaces,
        "output_subspaces": all_output_subspaces,
        "prompt_lengths": all_prompt_lengths,
    })
    train_dataset.set_format(
        type='torch', columns=[
            'input_ids', 'intervention_locations', 
            'labels', 'input_subspaces', 'output_subspaces', 'prompt_lengths'])

    data_collator_fn = transformers.DefaultDataCollator(
        return_tensors="pt"
    )
    data_collator = ReftDataCollator(tokenizer=tokenizer, data_collator=data_collator_fn)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


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
                            tokenize=False, add_generation_prompt=True)
                        # apply the tokenizer chat format to the prompt.
                        formatted_prompt = self.tokenizer.apply_chat_template(
                            [{"role": "user", "content": sampled_prompt}], 
                            tokenize=False, add_generation_prompt=True)
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