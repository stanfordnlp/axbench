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
    MIN_RATING = 1
    MAX_RATING = 10
    DEFAULT_RATING = 5.0
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.lm_model = kwargs.get("lm_model", None)
        self.concept_id = kwargs.get("concept_id", None)

    def _get_rating_from_completion(self, completion):
        # Try to find "Rating: X" or "Rating: [[X]]" format
        if "Rating:" in completion:
            rating_text = completion.split("Rating:")[-1].strip()
            # Remove any trailing text after the rating
            rating_text = rating_text.split('\n')[0].strip()
            # Remove brackets if present
            rating_text = rating_text.replace('[', '').replace(']', '')
            # Remove any trailing period
            rating_text = rating_text.rstrip('.').strip('"').strip("'").strip("*").strip()
            # Convert to float
            rating = float(rating_text)
        
        # Try to find "**Rating: X**" format (markdown)
        elif "**Rating:" in completion:
            rating_text = completion.split("**Rating:")[-1].split("**")[0].strip()
            rating = float(rating_text)
        return rating

    def _get_ratings_from_completions(self, completions):
        ratings = []
        for completion in completions:
            try:
                # Look for rating in various formats
                rating = self._get_rating_from_completion(completion)
                if rating is not None and self.MIN_RATING <= rating <= self.MAX_RATING:
                    ratings.append(rating)
                else:
                    logger.warning(f"Invalid rating value: {rating}")
                    ratings.append(self.DEFAULT_RATING)
                    
            except Exception as e:
                logger.warning(f"Failed to parse rating:\n\n{completion}\nError: {str(e)}")
                ratings.append(self.DEFAULT_RATING)
        return ratings
    
    def compute_metrics(self, data, write_to_dir=None):
        logger.warning(
            f"Starting task for concept_id: {self.concept_id}, "
            f"model: {self.model_name}, evaluator: {self.__str__()}")

        data_copy = data.copy()
        
        # Using OpenAI API to judge the quality of the generated data
        prompts = []
        # This is a generation dataset.
        for _, row in data_copy.iterrows():
            if self.template == UNIDIRECTIONAL_RATING_TEMPLATE:
                prompts += [self.template % (
                    row["original_prompt"], row["input_concept"], row[f"{self.model_name}_steered_generation"])]

        async def process_batch():
            try:
                return await self.lm_model.chat_completions(
                    f"{self.concept_id}_{self.model_name}_LMJudgeEvaluator", 
                    prompts, batch_size=32
                )
            finally:
                await self.lm_model.close()

        # If we're already in an event loop, use that
        try:
            loop = asyncio.get_running_loop()
            completions = loop.run_until_complete(process_batch())
        except RuntimeError:
            # If no event loop exists, create one
            completions = asyncio.run(process_batch())

        ratings = self._get_ratings_from_completions(completions)
        data_copy[f"{self.model_name}_lm_judge_rating"] = ratings
        metrics = {
            "lm_judge_rating": [],
            "factor": [],
            "raw_lm_judge_rating": [],
            "raw_rating_flattened": ratings
        }

        # group by factor only and compute means
        grouped = data_copy.groupby("factor")
        for factor, group in grouped:
            lm_judge_rating = group[f"{self.model_name}_lm_judge_rating"].mean()
            metrics["lm_judge_rating"].append(lm_judge_rating)
            metrics["factor"].append(factor)
            metrics["raw_lm_judge_rating"].append(list(group[f"{self.model_name}_lm_judge_rating"].values))

        return metrics


class LMJudgeConceptFollowingEvaluator(LMJudgeEvaluator):
    template = UNIDIRECTIONAL_RATING_TEMPLATE
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)
        self.use_icl = kwargs.get("use_icl", False)
        self.template = UNIDIRECTIONAL_RATING_NO_ICL_TEMPLATE if not self.use_icl else self.template
        
    def __str__(self):
        return 'LMJudgeConceptFollowingEvaluator'
    


