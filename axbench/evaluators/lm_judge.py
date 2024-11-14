import pandas as pd
import numpy as np
from .evaluator import Evaluator
from .prompt_templates import *
import asyncio

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


class LMJudgeEvaluator(Evaluator):
    DEFAULT_RATING = 0.0
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.lm_model = kwargs.get("lm_model", None)
        self.concept_id = kwargs.get("concept_id", None)

    def __str__(self):
        return 'LMJudgeEvaluator'

    def _get_rating_from_completion(self, completion):
        if "Rating:" in completion:
            rating_text = completion.split("Rating:")[-1].strip()
            rating_text = rating_text.split('\n')[0].strip()
            rating_text = rating_text.replace('[', '').replace(']', '')
            rating_text = rating_text.rstrip('.').strip('"').strip("'").strip("*").strip()
            rating = float(rating_text)
        else:
            logger.warning(f"Cannot find rating value: {completion}")
            rating = self.DEFAULT_RATING
        return rating

    def _get_ratings_from_completions(self, completions, min_rating=0.0, max_rating=1.0):
        ratings = []
        for completion in completions:
            try:
                # Look for rating in various formats
                rating = self._get_rating_from_completion(completion)
                if rating is not None and min_rating <= rating <= max_rating:
                    ratings.append(rating)
                else:
                    logger.warning(f"Invalid rating value: {rating}")
                    ratings.append(self.DEFAULT_RATING)
            except Exception as e:
                logger.warning(f"Failed to parse rating:\n\n{completion}\nError: {str(e)}")
                ratings.append(self.DEFAULT_RATING)
        return ratings
    
    def _get_ratings_from_prompts(self, prompts, api_name, min_rating=0.0, max_rating=1.0):
        async def process_batch():
            return await self.lm_model.chat_completions(
                f"{api_name}_{self.model_name}_LMJudgeEvaluator", prompts, batch_size=32
            )

        # If we're already in an event loop, use that
        completions = asyncio.run(process_batch())
        return self._get_ratings_from_completions(completions, min_rating, max_rating)

    def _get_all_ratings_from_data(self, data, column_name):
        model_relevance_concept_prompts = []
        model_relevance_instruction_prompts = []
        model_fluency_prompts = []
        # This is a generation dataset.
        for idx, row in data.iterrows():
            input_concept = row["input_concept"]
            original_prompt = row["original_prompt"]
            generation = row[f"{column_name}_steered_generation"]
            model_relevance_concept_prompts += [UNIDIRECTIONAL_PAIRWISE_EVALUATION_CONCEPT_RELEVANCE_TEMPLATE.format(
                concept=input_concept,
                sentence=generation
            )]
            model_relevance_instruction_prompts += [UNIDIRECTIONAL_PAIRWISE_EVALUATION_INSTRUCTION_RELEVANCE_TEMPLATE.format(
                instruction=original_prompt,
                sentence=generation
            )]
            model_fluency_prompts += [UNIDIRECTIONAL_PAIRWISE_EVALUATION_FLUENCY_TEMPLATE.format(
                sentence=generation
            )]
        model_relevance_concept_ratings = self._get_ratings_from_prompts(model_relevance_concept_prompts, f"{column_name}_concept")
        model_relevance_instruction_ratings = self._get_ratings_from_prompts(model_relevance_instruction_prompts, f"{column_name}_instruction")
        model_fluency_ratings = self._get_ratings_from_prompts(model_fluency_prompts, f"{column_name}_fluency", max_rating=2.0)
        return list(zip(model_relevance_concept_prompts, model_relevance_concept_ratings)), \
               list(zip(model_relevance_instruction_prompts, model_relevance_instruction_ratings)), \
               list(zip(model_fluency_prompts, model_fluency_ratings))

    def compute_metrics(self, data, write_to_dir=None):
        """
        We record three scores separately:
        1. Check concept relevance [score: 0-1]
        2. Check instruction relevance [score: 0-1]
        3. Check fluency [score: 0-2]

        We then aggregate these scores with these rules:
        - If the answer gets 1 for the first two checks, it gets a score of 1.
        - We then add the fluency score to get the final score.
        """
        logger.warning(
            f"Starting task for concept_id: {self.concept_id}, "
            f"model: {self.model_name}, evaluator: {self.__str__()}")
        data_copy = data.copy()
        
        model_relevance_concept_ratings, model_relevance_instruction_ratings, model_fluency_ratings = \
            self._get_all_ratings_from_data(data_copy, self.model_name)
        
        all_relevance_concept_ratings = []
        all_relevance_instruction_ratings = []
        all_fluency_ratings = []
        all_aggregated_ratings = []

        for i in range(len(model_relevance_concept_ratings)):
            all_relevance_concept_ratings += [model_relevance_concept_ratings[i][-1]]
            all_relevance_instruction_ratings += [model_relevance_instruction_ratings[i][-1]]
            all_fluency_ratings += [model_fluency_ratings[i][-1]]

            if model_relevance_concept_ratings[i][-1] == 1 and model_relevance_instruction_ratings[i][-1] == 1:
                all_aggregated_ratings += [1 + model_fluency_ratings[i][-1]]
            else:
                all_aggregated_ratings += [0]

        metrics = {
            "lm_judge_rating": [],
            "relevance_concept_ratings": [],
            "relevance_instruction_ratings": [],
            "fluency_ratings": [],
            "factor": [],
            "raw_relevance_concept_ratings": all_relevance_concept_ratings,
            "raw_relevance_instruction_ratings": all_relevance_instruction_ratings,
            "raw_fluency_ratings": all_fluency_ratings,
            "raw_aggregated_ratings": all_aggregated_ratings
        }
        data_copy[f"{self.model_name}_lm_judge_rating"] = all_aggregated_ratings
        data_copy[f"{self.model_name}_relevance_concept_ratings"] = all_relevance_concept_ratings
        data_copy[f"{self.model_name}_relevance_instruction_ratings"] = all_relevance_instruction_ratings
        data_copy[f"{self.model_name}_fluency_ratings"] = all_fluency_ratings

        # group by factor only and compute means
        grouped = data_copy.groupby("factor")
        for factor, group in grouped:
            metrics["lm_judge_rating"].append(group[f"{self.model_name}_lm_judge_rating"].mean())
            metrics["relevance_concept_ratings"].append(group[f"{self.model_name}_relevance_concept_ratings"].mean())
            metrics["relevance_instruction_ratings"].append(group[f"{self.model_name}_relevance_instruction_ratings"].mean())
            metrics["fluency_ratings"].append(group[f"{self.model_name}_fluency_ratings"].mean())
            metrics["factor"].append(factor)

        return metrics



