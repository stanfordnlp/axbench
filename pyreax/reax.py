import os, random, json, time, requests, copy
import torch, transformers, datasets

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union, List, Any
from pathlib import Path
import pandas as pd

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)

from .config import *
from .constants import *
from .model_utils import *
from .prompt_utils import *
from .language_models import *

import numpy as np
from huggingface_hub import hf_hub_download


def sample_index_exclude(index_range, exclude_index):
    if exclude_index < 0 or exclude_index >= index_range:
        raise ValueError("exclude_index must be within the valid index range.")
    possible_indices = [i for i in range(index_range) if i != exclude_index]
    return random.choice(possible_indices)


def load_concepts(dump_dir):
    sae_concepts = []
    if ".txt" in dump_dir:
        with open(dump_dir, 'r') as file:
            concepts = [line.strip() for line in file.readlines()]
        if concepts[0].startswith("http://") or concepts[0].startswith("https://"):
            logger.warning("Detect external links. Pull concept info from the link.")
            for concept in concepts:
                if "www.neuronpedia.org" not in concept:
                    raise ValueError(f"Pulling from {concept} is not supported.")
                sae_path = concept.split("https://www.neuronpedia.org/")[-1]
                sae_url = f"https://www.neuronpedia.org/api/feature/{sae_path}"
                headers = {"X-Api-Key": os.environ.get("NP_API_KEY")}
                response = requests.get(sae_url, headers=headers).json()
                explanation = response["explanations"][0]["description"]
                sae_concepts += [explanation.strip()]
            return sae_concepts, concepts
        return concepts, ["null"]*len(concepts)
    elif ".json" in dump_dir:
        concepts = []
        # this must be a neuropedia export.
        with open(dump_dir, 'r') as file:
            json_concepts = json.load(file)
        for concept in json_concepts:
            sae_concepts += [concept["description"].strip()]
            model = concept["modelId"]
            sae_model = concept["layer"]
            subspace_id = concept["index"]
            concepts += [f"https://www.neuronpedia.org/{model}/{sae_model}/{subspace_id}"]
        return sae_concepts, concepts
    else:
        raise ValueError(f"Unsupported file type: {dump_dir}.")
        


def load_sae(concept_metadata):
    """load the sae metadata (e.g., column index) and weights"""
    sae_path = json.loads(concept_metadata[0])["sae_concept"].split("https://www.neuronpedia.org/")[-1]
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


def load_reax(dump_dir):
    """load the saved subspaces for evaluations"""
    dump_dir = Path(dump_dir)
    config = load_config_from_json(dump_dir / 'config.json')
    training_df = pd.read_csv(dump_dir / "training_df.csv")
    with open(dump_dir / "concept_metadata.jsonl", 'r') as f:
        concept_metadata = list(f)
    weights = torch.load(dump_dir / "weights.pt")
    return config, training_df, concept_metadata, weights
    

def save_reax(dump_dir, config, training_df, reax_factory, sae_metadata, weights):
    """save training data, concept metadata, subspace artifacts"""

    concepts = reax_factory.concepts
    contrast_concepts_map = reax_factory.contrast_concepts_map
    
    reax_model_name = config.get_model_name()
    
    # handle training df first
    dump_dir = Path(dump_dir) / reax_model_name
    dump_dir.mkdir(parents=True, exist_ok=True)
    
    # handle config
    config_file = dump_dir / 'config.json'
    if config_file.exists():
        existing_config = load_config_from_json(config_file)
        try:
            compare_configs(config, existing_config)
            logger.warning("Passing the checked. The memory config and the loaded config are identical.")
        except ConfigMismatchError as e:
            logger.warning(str(e))
    else:
        with open(dump_dir / 'config.json', 'w') as f:
            f.write(repr(config))
    
    # handle training data
    training_file = dump_dir / "training_df.csv"
    if training_file.exists():
        training_df = pd.concat([pd.read_csv(training_file), training_df], axis=0)
    training_df.to_csv(training_file, index=False)

    # handle metadata
    concept_metadata_file = Path(dump_dir) / "concept_metadata.jsonl"
    if concept_metadata_file.exists():
        with open(concept_metadata_file, 'r') as f:
            saved_concept_metadata = list(f)
        curr_idx = len(saved_concept_metadata)
    else:
        curr_idx = 0
    with open(concept_metadata_file, 'a') as f:
        for i in range(len(concepts)):
            concept_metadata = {
                "_id": curr_idx,
                "concept": concepts[i],
                "sae_concept": sae_metadata[i],
                "contrast_concepts": contrast_concepts_map[concepts[i]]
            }
            f.write(json.dumps(concept_metadata) + '\n')
            curr_idx += 1

    # handle weights
    weights_file = Path(dump_dir) / "weights.pt"
    if weights_file.exists():
        weights = torch.cat([torch.load(weights_file), weights.data.cpu()], dim=0)
    torch.save(weights.data.cpu(), weights_file)


class ReAXFactory(object):
    """Main class of generating training pairs for two subspaces"""

    def __init__(
        self, model, tokenizer, 
        concepts,
        dump_dir,
        skip_contrast_concept=False,
        contrast_concepts=None,
        **kwargs
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.concepts = concepts
        # we need at least two concepts to work.
        if len(self.concepts) < 2:
            logger.warning(f"Less than 2 concepts are provided. Only eval mode is allowed.")

        # prepare lm model
        lm_model = kwargs["lm_model"] if "lm_model" in kwargs else "gpt-4o"
        self.lm_model = LanguageModel(lm_model, dump_dir)

        self.exist_contrast_concepts = contrast_concepts
        self.skip_contrast_concept = skip_contrast_concept
        self._prepare_contrast_concepts(contrast_concepts, **kwargs)

    def save(self, df, weights, metadata):
        pass
    
    def get_total_price(self):
        """Estimate API costs"""
        return round(self.lm_model.stats.get_total_price(), 3)

    def reset_stats(self):
        """Reset API costs"""
        self.lm_model.dump()
        self.lm_model.stats.reset()
    
    def _get_config(self):
        """Config for the dataset"""
        pass
    
    def _prepare_contrast_concepts(self, contrast_concepts, **kwargs):
        """Cache a set of contrast concepts"""
        if "skip_contrast" in kwargs and kwargs["skip_contrast"]:
            self.contrast_concepts_map = {}
            return

        start = time.time()
        logger.warning("Prepare contrast concepts.")
        self.contrast_concepts_map = {}
        for concept in self.concepts:
            if self.skip_contrast_concept:
                logger.warning(f"Skipping contrast concept creation for {concept}.")
                # we don't do any contrast concept creation.
                self.contrast_concepts_map[concept] = []
                continue
            filtered_concepts = get_contrast_concepts(
                self.lm_model, concept, contrast_concepts)
            self.contrast_concepts_map[concept] = filtered_concepts
            logger.warning(
                f"Fectching {len(filtered_concepts)} contrast concepts for concept: {concept}")
        end = time.time()
        elapsed = round(end-start, 3)
        total_price = self.get_total_price()
        logger.warning(f"Finished preparing contrast concepts in {elapsed} sec with ${total_price}.")

    def create_eval_df(self, n=10, category="positive"):
        """category: positive, negative, hard negative"""
        all_examples = []
        logger.warning("Creating dataframe.")
        n_per_concept = n // (len(self.concepts))
        if category == "positive":
            for idx, concept in enumerate(self.concepts):
                # experiment pairs: inputs and outputs are all llm generated.
                concept_prompts = []
                for _ in range(n_per_concept):
                    concept_prompt = get_simple_sentence_with_concept(
                        self.lm_model, concept=concept, exist_sentences=concept_prompts)
                    concept_prompts += [concept_prompt]
                    all_examples += [[
                        concept_prompt, concept, category
                    ]]
        elif category == "negative":
            for idx, concept in enumerate(self.concepts):
                random_prompts = []
                for _ in range(n_per_concept):
                    random_prompt = get_random_sentence(
                        self.lm_model, self.concepts, exist_sentences=random_prompts)
                    random_prompts += [random_prompt]
                    all_examples += [[
                        random_prompt, "null", category
                    ]]
        elif category == "hard negative":
            for idx, concept in enumerate(self.concepts):
                # unseen contrast concepts
                polysemantic_prompts = []
                polysemantic_meanings = self.contrast_concepts_map[concept]
                if len(polysemantic_meanings) != 0:
                    for polysemantic_meaning in polysemantic_meanings:
                        polysemantic_prompt = get_contrast_sentence(
                            self.lm_model, polysemantic_meaning, concept, 
                            exist_sentences=polysemantic_prompts)
                        polysemantic_prompts += [polysemantic_prompt]
                        all_examples += [[
                            polysemantic_prompt, "OOD/"+"/".join(polysemantic_meaning), category
                        ]]
                # seen contrast concepts
                if self.exist_contrast_concepts is not None:
                    polysemantic_prompts = []
                    polysemantic_meanings = self.exist_contrast_concepts[concept]
                    if len(polysemantic_meanings) != 0:
                        for polysemantic_meaning in polysemantic_meanings:
                            polysemantic_prompt = get_contrast_sentence(
                                self.lm_model, polysemantic_meaning, concept, 
                                exist_sentences=polysemantic_prompts)
                            polysemantic_prompts += [polysemantic_prompt]
                            all_examples += [[
                                polysemantic_prompt, "ID/"+"/".join(polysemantic_meaning), category
                            ]]
        df = pd.DataFrame(
            all_examples, 
            columns = [
                'input', 'input_concept', 'category'])
        return df

    def create_df(self, n=10):
        assert len(self.concepts) >= 2
        
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
                    polysemantic_prompt = get_random_sentence(
                        self.lm_model, self.concepts, exist_sentences=polysemantic_prompts)
                    polysemantic_prompt_concepts.append("null")
                    polysemantic_prompts += [polysemantic_prompt]
            else:
                _polysemantic_meanings = extend_list_with_random_elements(
                    copy.deepcopy(polysemantic_meanings), n_null//2)
                for polysemantic_meaning in _polysemantic_meanings:
                    polysemantic_prompt = get_contrast_sentence(
                        self.lm_model, polysemantic_meaning, concept, 
                        exist_sentences=polysemantic_prompts)
                    polysemantic_prompts += [polysemantic_prompt]
                    polysemantic_prompt_concepts += [concept+":"+polysemantic_meaning[0]+"/"+polysemantic_meaning[1]]
                # if there is no polysemantic meanings, we sample random sentences.
                n_random = n_null - len(_polysemantic_meanings)
                for _ in range(n_random):
                    polysemantic_prompt = get_random_sentence(
                        self.lm_model, self.concepts, exist_sentences=polysemantic_prompts)
                    polysemantic_prompt_concepts.append("null")
                    polysemantic_prompts += [polysemantic_prompt]
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
                    sentence=concept_prompt, exist_continues=concept_outputs
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
        logger.warning(f"Finished creating current dataframe in {elapsed} sec with ${total_price}.")
        # reset the cost.
        self.reset_stats()
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

        # logger.warning("tokens with lm loss:")
        # logger.warning(tokenizer.batch_decode(output_ids[output_ids!=-100].unsqueeze(dim=-1)))

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