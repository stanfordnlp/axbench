import os, random, json, time
import torch, transformers, datasets

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union, List, Any
from pathlib import Path
import pandas as pd
from transformers.utils import logging
logger = logging.get_logger("SubCTRLDataset")

from .constants import *
from .model_utils import *
from .prompt_utils import *
from .language_models import *


def sample_index_exclude(index_range, exclude_index):
    if exclude_index < 0 or exclude_index >= index_range:
        raise ValueError("exclude_index must be within the valid index range.")
    possible_indices = [i for i in range(index_range) if i != exclude_index]
    return random.choice(possible_indices)


class SubCTRLFactory(object):
    """Main class of generating training pairs for two subspaces"""

    def __init__(
        self, model, tokenizer, 
        concepts, dump_dir,
        skip_contrast_concept=False,
        **kwargs
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.concepts = concepts
        # we need at least two concepts to work.
        assert len(self.concepts) >= 2

        # prepare lm model
        lm_model = kwargs["lm_model"] if "lm_model" in kwargs else "gpt-4o"
        self.lm_model = LanguageModel(lm_model, dump_dir)

        # dump dir
        cur_save_dir = Path(dump_dir) / "datasets"
        cur_save_dir.mkdir(parents=True, exist_ok=True)
        self.dump_dir = cur_save_dir

        self.skip_contrast_concept = skip_contrast_concept
        self._prepare_contrast_concepts(**kwargs)

    def get_total_price(self):
        """Estimate API costs"""
        return round(self.lm_model.stats.get_total_price(), 3)
    
    def _get_config(self):
        """Config for the dataset"""
        pass
    
    def _prepare_contrast_concepts(self, **kwargs):
        """Cache a set of contrast concepts"""
        if "skip_contrast" in kwargs and kwargs["skip_contrast"]:
            self.contrast_concepts_map = {}
            return

        logger.warning("Prepare contrast concepts.")
        self.contrast_concepts_map = {}
        for concept in self.concepts:
            if self.skip_contrast_concept:
                logger.warning(f"Skipping contrast concept creation for {concept}.")
                # we don't do any contrast concept creation.
                self.contrast_concepts_map[concept] = []
                continue
            contrast_concepts = get_contrast_concepts(
                self.lm_model, concept,
            )
            filtered_contrast_concepts = []
            for c in contrast_concepts:
                if c.strip() != "":
                    filtered_contrast_concepts += [c]
            self.contrast_concepts_map[concept] = filtered_contrast_concepts
            logger.warning(
                f"Fectching {len(contrast_concepts)} contrast concepts for concept: {concept}")
    
    def create_df(self, n=10):
        
        start = time.time()
        logger.warning("Creating dataframe.")
        n_per_concept = n // (len(self.concepts) + 1)
        all_examples = []

        # null examples first
        for idx, concept in enumerate(self.concepts):
            # there is a fixed amount null examples for each concept
            n_null = n_per_concept // len(self.concepts)
            # control pairs: inputs are llm generated, outputs are model's generations.
            polysemantic_meanings = self.contrast_concepts_map[concept]
            polysemantic_prompt_concepts = []
            polysemantic_prompts = []
            if len(polysemantic_meanings) == 0:
                # if there is no polysemantic meanings, we sample random sentences.
                for _ in range(n_null):
                    polysemantic_prompt = get_n_random_sentence(
                        self.lm_model, self.concepts, exist_sentences=polysemantic_prompts)
                    polysemantic_prompt_concepts.append("null")
                    polysemantic_prompts += [polysemantic_prompt]
            else:
                # if there are polysemantic meanings, we iteratively create sentences.
                # note that we don't do it in batch to lower the risk.
                polysemantic_meanings = extend_list_with_random_elements(
                    polysemantic_meanings, n_null) # it must have n_null
                for polysemantic_meaning in polysemantic_meanings:
                    polysemantic_prompt = get_contrast_sentence(
                        self.lm_model, polysemantic_meaning, concept, 
                        exist_sentences=polysemantic_prompts)
                    polysemantic_prompts += [polysemantic_prompt]
                    polysemantic_prompt_concepts += [polysemantic_meaning]
                # supply with random draw of existing ones.
                n_random = n_null - len(polysemantic_meanings)
                for _ in range(n_random):
                    rand_idx = random.choice([i for i in range(len(polysemantic_prompts))])
                    polysemantic_prompts += [polysemantic_prompts[rand_idx]]
                    polysemantic_prompt_concepts += [polysemantic_prompt_concepts[rand_idx]]
            # cut off overflows.
            polysemantic_prompts = polysemantic_prompts[:n_null]
            polysemantic_outputs = get_model_continues(
                self.model, self.tokenizer, polysemantic_prompts, max_new_tokens=20)
            for i in range(len(polysemantic_prompts)):
                output_idx = sample_index_exclude(len(self.concepts), idx)
                all_examples += [[
                    polysemantic_prompts[i], polysemantic_outputs[i], 
                    EXAMPLE_TAG.CONTROL, idx, output_idx, polysemantic_prompt_concepts[i], "null"]]
        
        # regular examples
        for idx, concept in enumerate(self.concepts):
            logger.warning(f"Fectching data for {idx}/{len(self.concepts)} concept: {concept}")
            # experiment pairs: inputs and outputs are all llm generated.
            concept_prompts = []
            concept_outputs = []
            for _ in range(n_per_concept):
                concept_prompt = get_sentence_with_concept(
                    self.lm_model, concept=concept, exist_sentences=concept_prompts)
                output_idx = sample_index_exclude(len(self.concepts), idx)
                concept_output = get_continue_with_concept(
                    self.lm_model, concept=self.concepts[output_idx], 
                    sentence=concept_prompt, exist_continues="\n".join(concept_outputs)
                )
                concept_prompts += [concept_prompt]
                concept_outputs += [concept_output]
                all_examples += [[
                    concept_prompt, concept_output, EXAMPLE_TAG.EXPERIMENT, 
                    idx, output_idx, concept, self.concepts[output_idx]
                ]]

        df = pd.DataFrame(
            all_examples, 
            columns = [
                'input', 'output', 'group', 'input_subspace', 'output_subspace', 
                'input_concept', 'output_concept'])
        end = time.time()
        elapsed = round(end-start, 3)
        total_price = self.get_total_price()
        logger.warning(f"Finished creating dataframe in {elapsed} sec with ${total_price}.")
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
            
            _intervention_location_paddings = torch.tensor(
                [[len(inst["input_ids"]) for _ in range(max_intervention_len - len(inst["intervention_locations"][0]))]])
            inst["intervention_locations"] = torch.cat([inst["intervention_locations"], _intervention_location_paddings], dim=-1).int()

            inst["input_subspaces"] = inst["input_subspaces"].int()
            inst["output_subspaces"] = inst["output_subspaces"].int()
            inst["groups"] = inst["groups"].int()

            _input_id_paddings = torch.tensor(
                [self.tokenizer.pad_token_id for _ in range(max_seq_len - non_pad_len)])
            inst["input_ids"] = torch.cat((inst["input_ids"], torch.tensor([self.tokenizer.pad_token_id]), _input_id_paddings)).int()

            _label_paddings = torch.tensor([-100 for _ in range(max_seq_len - non_pad_len+1)])
            inst["labels"] = torch.cat((inst["labels"], _label_paddings))
            
            inst["attention_mask"] = (inst["input_ids"] != self.tokenizer.pad_token_id).int()
            
        batch_inputs = self.data_collator(instances)
        return batch_inputs


def make_data_module(
    tokenizer: transformers.PreTrainedTokenizer, model, df,
):
    """Make dataset and collator for supervised fine-tuning with kl div loss."""
    
    all_base_input_ids, all_intervention_locations, all_output_ids, \
        all_input_subspaces, all_output_subspaces, all_groups = [], [], [], [], [], []

    for _, row in df.iterrows():
        _input, _output, _input_subspace, _output_subspace = row["input"], row["output"], \
            int(row["input_subspace"]), int(row["output_subspace"])
        if str(row["group"]) == "EXAMPLE_TAG.CONTROL":
            _group = 0
        else:
            _group = 1

        # prepare input ids
        base_prompt = _input
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

        # print("tokens with lm loss:")
        # print(tokenizer.batch_decode(output_ids[output_ids!=-100].unsqueeze(dim=-1)))

        intervention_locations = torch.tensor([[i for i in range(1, base_prompt_length)]])
        all_intervention_locations.append(intervention_locations)
        all_base_input_ids.append(base_input_ids)
        all_output_ids.append(output_ids)
        all_input_subspaces.append(torch.tensor(_input_subspace))
        all_output_subspaces.append(torch.tensor(_output_subspace))
        all_groups.append(torch.tensor(_group))
        
    train_dataset = datasets.Dataset.from_dict({
        "input_ids": all_base_input_ids,
        "intervention_locations": all_intervention_locations,
        "labels": all_output_ids,
        "input_subspaces": all_input_subspaces,
        "output_subspaces": all_output_subspaces,
        "groups": all_groups,
    })
    train_dataset.set_format(
        type='torch', columns=[
            'input_ids', 'intervention_locations', 
            'labels', 'input_subspaces', 'output_subspaces', 'groups'])

    data_collator_fn = transformers.DefaultDataCollator(
        return_tensors="pt"
    )
    data_collator = ReftDataCollator(tokenizer=tokenizer, data_collator=data_collator_fn)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)