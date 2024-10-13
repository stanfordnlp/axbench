import os, random, json, time, requests, copy, asyncio, csv, math
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

from .utils.config import *
from .utils.constants import *
from .utils.model_utils import *
from .utils.prompt_utils import *
from .language_models import *

import numpy as np
from huggingface_hub import hf_hub_download


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
    elif ".csv" in dump_dir:
        # for csv, then the format is <concept>,<url>
        # no http connection is needed
        concepts = []
        with open(dump_dir, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                sae_concepts += [row[0]]
                concepts += [row[1]]
        return sae_concepts, concepts
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
    weight = torch.load(dump_dir / "weight.pt")
    bias = torch.load(dump_dir / "bias.pt")
    return config, training_df, concept_metadata, weight, bias
    

def save_reax_df(dump_dir, df, filename):
    file = dump_dir / filename
    if file.exists():
        new_df = pd.concat([pd.read_csv(file), df], axis=0)
    new_df.to_csv(file, index=False)


def save_reax(dump_dir, config, training_df, reax_factory, sae_metadata, intervention):
    """save training data, concept metadata, subspace artifacts"""

    concepts = reax_factory.concepts
    contrast_concepts_map = reax_factory.contrast_concepts_map
    concept_genres_map = reax_factory.concept_genres_map
    
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
                "contrast_concepts": contrast_concepts_map[concepts[i]],
                "concept_genres": concept_genres_map[concepts[i]]
            }
            f.write(json.dumps(concept_metadata) + '\n')
            curr_idx += 1

    # handle weight and bias
    weight_file = Path(dump_dir) / "weight.pt"
    if weight_file.exists():
        weight = torch.cat([torch.load(weight_file), intervention.proj.weight.data.cpu()], dim=0)
    else:
        weight = intervention.proj.weight.data.cpu()
    torch.save(weight.data.cpu(), weight_file)
    
    bias_file = Path(dump_dir) / "bias.pt"
    if bias_file.exists():
        bias = torch.cat([torch.load(bias_file), intervention.proj.bias.data.cpu()], dim=0)
    else:
        bias = intervention.proj.bias.data.cpu()
    torch.save(bias.data.cpu(), bias_file)


async def run_tasks(tasks):
    # Gather and run all provided tasks concurrently, and collect their results
    results = await asyncio.gather(*tasks)
    return results


class ReAXFactory(object):
    """Main class of async generating training pairs for two subspaces"""

    def __init__(
        self, model, tokenizer, dump_dir, **kwargs):
        self.model = model
        self.tokenizer = tokenizer

        # prepare lm model
        lm_model = kwargs.get("lm_model", "gpt-4o")
        self.lm_model = LanguageModel(lm_model, dump_dir)

    def get_total_price(self):
        """Estimate API costs"""
        return round(self.lm_model.stats.get_total_price(), 3)

    def reset_stats(self):
        """Reset API costs"""
        self.lm_model.dump()
        self.lm_model.stats.reset()

    def prepare_concepts(self, concepts, **kwargs):
        start = time.time()
        logger.warning("Creating genre and contrast concepts for the inputs.")
        genre_task = get_concept_genres(self.lm_model, concepts)
        contrast_task = get_contrast_concepts(
            self.lm_model, concepts, kwargs.get("contrast_concepts", None))
        concept_genres_map, contrast_concepts_map = asyncio.run(
            run_tasks([genre_task, contrast_task]))

        end = time.time()
        elapsed = round(end - start, 3)
        total_price = self.get_total_price()
        logger.warning(
            f"Init finished in {elapsed} sec. (current cost: ${total_price})"
        )
        return concept_genres_map, contrast_concepts_map

    def create_eval_df(self, concepts, subset_n, concept_genres_map, contrast_concepts_map, **kwargs):
        """category: positive, negative, hard negative"""
        all_examples = []
        logger.warning("Creating dataframe.")
        n_per_concept = n // (len(self.concepts))

        input_length = kwargs["input_length"] if "input_length" in kwargs else 64
        output_length = kwargs["output_length"] if "output_length" in kwargs else 10

        eval_tasks = []
        tags = []
        for idx, concept in enumerate(self.concepts):
            # positive
            eval_tasks.append(get_content_with_concept(
                self.lm_model, self.tokenizer, subset_n, self.concept_genres_map[concept], 
                concept=concept, length=input_length))
            tags.append(("positive", concept))
            # negative
            eval_tasks.append(get_random_content(
                self.lm_model, tokenizer, subset_n, self.concept_genres_map[concept], [concept], 
                length=input_length))
            tags.append(("negative", concept))
            # hard negative seen
            exist_polysemantic_meanings = contrast_concepts_map[concept]
            if len(exist_polysemantic_meanings) != 0:
                eval_tasks.append(get_content_with_polysemantic_concepts(
                    self.lm_model, tokenizer, self.concept_genres_map[concept], 
                    polysemantic_meanings, concept, length=input_length))
                tags.append(("hard negative seen", concept))
            # hard negative unseen
            polysemantic_meanings = self.contrast_concepts_map[concept]
            if len(polysemantic_meanings) != 0:
                eval_tasks.append(get_content_with_polysemantic_concepts(
                    self.lm_model, tokenizer, self.concept_genres_map[concept], 
                    polysemantic_meanings, concept, length=input_length))
                tags.append(("hard negative unseen", concept))
        eval_content = asyncio.run(run_tasks(eval_tasks))

        for (tag, concept), eval_content in zip(tags, eval_content):
            if tag in {"positive", "negative"}:
                all_examples += [[content, concept, tag] for content in eval_content]
            elif tag == {"hard negative seen", "hard negative unseen"}:
                all_examples += [[content[0], "//".join(content[1]), tag] for content in eval_content[1]]

        df = pd.DataFrame(
            all_examples, 
            columns = [
                'input', 'input_concept', 'category'])
        return df

    def create_train_df(self, concepts, n, concept_genres_map, contrast_concepts_map, **kwargs):
        concept2id = {concept: i for i, concept in enumerate(concepts)}
        
        start = time.time()
        logger.warning("Creating dataframe.")
        n_per_concept = n // (len(concepts) + 1)
        all_examples = []

        input_length = kwargs.get("input_length", 64)
        output_length = kwargs.get("output_length", 10)
        
        # for each concept, we create a set of seed random content.
        random_content_task = get_random_content(
            self.lm_model, self.tokenizer, n_per_concept // len(concepts), 
            concept_genres_map, concepts, length=input_length)
        concepts_random_content = asyncio.run(run_tasks([random_content_task]))[0]
        
        # for concepts with polysemantic senses, we create additional examples.
        polysemantic_tasks = []
        for concept in concepts:
            if len(contrast_concepts_map[concept]) != 0:
                polysemantic_tasks.append(modify_content_with_polysemantic_concepts(
                    self.lm_model, self.tokenizer,
                    extend_list_with_random_elements(
                        copy.deepcopy(contrast_concepts_map[concept]), 
                        n_per_concept // (len(concepts)*2)), 
                    concept, concepts_random_content[concept],
                    length=input_length))
        polysemantic_content = asyncio.run(run_tasks(polysemantic_tasks))        
        polysemantic_content = {content[0]: content[1] for content in polysemantic_content}
        
        # aggregate these null examples.
        null_prompts = []
        for concept in concepts:
            if len(contrast_concepts_map[concept]) == 0:
                for content in concepts_random_content[concept]:
                    null_prompts += [(concept, "null", content)]
            else:
                n_random = n_per_concept // (len(concepts)*2)
                for content in concepts_random_content[concept][:n_random]:
                    null_prompts += [(concept, "null", content)]
                for content in polysemantic_content[concept]:
                    null_prompts += [(concept, f"{content[0][0]}//{content[0][1]}", content[1])]
        null_outputs = get_model_continues(
            self.model, self.tokenizer, [prompt[-1] for prompt in null_prompts], 
            max_new_tokens=int(output_length*1.5))
        for (concept, tag, prompt), output in zip(null_prompts, null_outputs):
            in_idx = concept2id[concept]
            out_idx = sample_index_exclude(len(concepts), in_idx)
            all_examples += [[prompt, output, EXAMPLE_TAG.CONTROL, in_idx, out_idx, tag, "null"]]
        
        # modify exist content to have desired concepts.
        modify_prompts = []
        for concept in concepts:
            for prompt in null_prompts:
                modify_prompts.append((concept, prompt[1], prompt[2]))
        modify_task = modify_content_with_concept(
            self.lm_model, self.tokenizer, content=modify_prompts, length=input_length)
        concept_prompts = asyncio.run(run_tasks([modify_task]))[0]
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
                prompt, output, EXAMPLE_TAG.EXPERIMENT, 
                in_idx, out_idx, modify_prompts[i][0], inverse_concepts[i]]]

        df = pd.DataFrame(
            all_examples, 
            columns = [
                'input', 'output', 'group', 'input_subspace', 'output_subspace', 
                'input_concept', 'output_concept'])
        end = time.time()
        elapsed = round(end-start, 3)
        total_price = self.get_total_price()
        logger.warning(f"Finished creating current dataframe in {elapsed} sec. (current cost: ${total_price})")
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