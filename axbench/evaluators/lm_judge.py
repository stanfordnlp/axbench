import pandas as pd
import numpy as np
from .evaluator import Evaluator
import asyncio

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


INSTRUCTION_CONCEPT_BASED_RATING_TEMPLATE = """[System]
You are an impartial evaluator. Your task is to determine if the response fragment both:
1. Contains clear references or terms related to the specified concept (not just superficial mentions)
2. Presents these references within a grammatically coherent structure (not just random related words)

Provide a rating of 1 if the response meets ALL of these conditions:
- The concept must be clearly and meaningfully incorporated
- The response fragment must form a grammatically sensible phrase or sentence

Provide a rating of 0 if ANY of these conditions are met:
- The concept is missing or only superficially mentioned in the response fragment
- The words are jumbled or lack basic grammatical structure in the response fragment

First provide a brief explanation (1-2 sentences) of whether and how the concept appears in a coherent way. \
Then provide your rating in exactly this format: "Rating: [[score]]"

[Concept Start]
%s
[Concept End]

[Response Start]
%s
[Response End]
"""


INSTRUCTION_RELATEDNESS_BASED_RATING_TEMPLATE = """[System]
You are an impartial evaluator. Your task is to assess whether the response fragment is \
meaningfully related to the given instruction.

Provide a rating of 1 if the response meets ALL of these conditions:
- The response fragment is related to the instruction in a meaningful way
- The response fragment must form a grammatically sensible phrase or sentence

Provide a rating of 0 if ANY of these conditions are met:
- The response fragment is not related to the instruction at all
- The words are jumbled or lack basic grammatical structure in the response fragment

Begin with a brief explanation (1-2 sentences). Then provide your rating in this format: "Rating: [[score]]".

[Instruction Start]
%s
[Instruction End]

[Response Start]
%s
[Response End]
"""


GENERATION_RATING_TEMPLATE = """[System]
You are an impartial evaluator. Your task is to determine if the sentence fragment both:
1. Contains clear references or terms related to the specified concept (not just superficial mentions)
2. Presents these references within a grammatically coherent structure (not just random related words)

Rate 1 if the fragment satisfies BOTH criteria:
- The concept must be clearly and meaningfully incorporated
- The fragment must form a grammatically sensible phrase or sentence

Rate 0 if either:
- The concept is missing or only superficially mentioned
- The words are jumbled or lack basic grammatical structure

First provide a brief explanation (1-2 sentences) of whether and how the concept appears in a coherent way. \
Then provide your rating in exactly this format: "Rating: [[score]]"

[The Start of the Concept]
%s
[The End of the Concept]

[The Start of the Sentence Fragment]
%s
[The End of the Sentence Fragment]
"""


class LMJudgeEvaluator(Evaluator):
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
                if rating is not None and 0 <= rating <= 1:
                    ratings.append(rating)
                else:
                    logger.warning(f"Invalid rating value: {rating}")
                    ratings.append(0.0)
                    
            except Exception as e:
                logger.warning(f"Failed to parse rating:\n\n{completion}\nError: {str(e)}")
                ratings.append(0.0)
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
            if self.template == GENERATION_RATING_TEMPLATE:
                prompts += [self.template % (
                    row["input_concept"], row["input"] + row[f"{self.model_name}_steered_generation"])]
            elif self.template == INSTRUCTION_CONCEPT_BASED_RATING_TEMPLATE:
                prompts += [self.template % (
                    row["input_concept"], row[f"{self.model_name}_steered_generation"])]
            elif self.template == INSTRUCTION_RELATEDNESS_BASED_RATING_TEMPLATE:
                prompts += [self.template % (
                    row["original_prompt"], row[f"{self.model_name}_steered_generation"])]

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
        # overwrite the original data to add new ratings.
        data[f"{self.model_name}_lm_judge_rating"] = ratings
        metrics = {
            "lm_judge_rating": [],
            "factor": []
        }
        
        # group by factor only and compute means
        grouped = data_copy.groupby("factor")
        for factor, group in grouped:
            lm_judge_rating = group[f"{self.model_name}_lm_judge_rating"].mean()
            metrics["lm_judge_rating"].append(lm_judge_rating)
            metrics["factor"].append(factor)

        return metrics


class LMJudgeConceptEvaluator(LMJudgeEvaluator):
    template = INSTRUCTION_CONCEPT_BASED_RATING_TEMPLATE
    def __str__(self):
        return 'LMJudgeConceptEvaluator'


class LMJudgeFollowingEvaluator(LMJudgeEvaluator):
    template = INSTRUCTION_RELATEDNESS_BASED_RATING_TEMPLATE
    def __str__(self):
        return 'LMJudgeFollowingEvaluator'


class LMJudgeContinuationEvaluator(LMJudgeEvaluator):
    template = GENERATION_RATING_TEMPLATE
    def __str__(self):
        return 'LMJudgeContinuationEvaluator'

