import os, random, json, time

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


class SubCTRLDataset(object):
    """Main class of generating training pairs for two subspaces"""

    def __init__(
        self, model, tokenizer, 
        concepts, dump_dir,
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
            contrast_concepts = get_contrast_concepts(
                self.lm_model, concept, model=self.lm_model,
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
        
        assert n % len(self.concepts) == 0, \
            f"Number of examples {n} is not devidable by {len(self.concepts)} concepts."
        n_per_concept = n // len(self.concepts)

        all_examples = []
        
        for idx, concept in enumerate(self.concepts):
            logger.warning(f"Fectching data for {idx}/{len(self.concepts)} concept: {concept}")
            n_null = n_per_concept // 2
            n_concept = n_per_concept - n_null
            
            # control pairs: inputs are llm generated, outputs are model's generations.
            polysemantic_meanings = self.contrast_concepts_map[concept]
            polysemantic_prompt_concepts = []
            polysemantic_prompts = []
            if len(polysemantic_meanings) == 0:
                # if there is no polysemantic meanings, we sample random sentences.
                polysemantic_prompts = get_n_random_sentences(
                    self.lm_model, N=n_null, model=self.lm_model)
                polysemantic_prompt_concepts = ["null"] * n_null
            else:
                # if there are polysemantic meanings, we iteratively create sentences.
                # note that we don't do it in batch to lower the risk.
                polysemantic_meanings = extend_list_with_random_elements(
                    polysemantic_meanings, n_null) # it must have n_null
                for polysemantic_meaning in polysemantic_meanings:
                    polysemantic_prompt = get_contrast_sentence(
                        self.lm_model, polysemantic_meaning, concept, 
                        exist_sentences="\n".join(polysemantic_prompts))
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
                self.model, self.tokenizer, polysemantic_prompts, max_new_tokens=10)
            
            for i in range(len(polysemantic_prompts)):
                output_idx = sample_index_exclude(len(self.concepts), idx)
                all_examples += [[
                    polysemantic_prompts[i], polysemantic_outputs[i], 
                    EXAMPLE_TAG.CONTROL, idx, output_idx, polysemantic_prompt_concepts[i], "null"]]

            # experiment pairs: inputs and outputs are all llm generated.
            concept_prompts = get_sentences_with_concept(
                self.lm_model, concept=concept, N=n_concept, model=self.lm_model)
            concept_outputs = []
            for i, concept_prompt in enumerate(concept_prompts):
                output_idx = sample_index_exclude(len(self.concepts), idx)
                concept_output = get_continue_with_concept(
                    self.lm_model, concept=self.concepts[output_idx], 
                    sentence=concept_prompt, exist_continues="\n".join(concept_outputs),
                    model=self.lm_model
                )
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

