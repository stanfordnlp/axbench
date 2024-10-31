try:
    # This library is our indicator that the required installs
    # need to be done.
    import pyreax

except ModuleNotFoundError:
    # relative import; better to pip install subctrl
    import sys
    sys.path.append("../../pyreax")
    import pyreax

from pyreax import (
    LanguageModel
)

import pandas as pd
import numpy as np
from .evaluator import Evaluator
import asyncio

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


RATING_TEMPLATE = """[System]
Please act as an impartial evaluator and assess the fluency of the provided sentence fragment \
and determine if it effectively contains terms related to the specified concept. Note that sentences may be \
incomplete or fragmented; please rate what is provided. Your evaluation should focus on two \
factors: (1) Fluency: whether the fragment is grammatically coherent, clear, and natural; \
(2) Concept Presence: whether the fragment clearly has terms related to the user-specified concept. \
Rate between 0 and 10 based on fluency and effective incorporation of the concept. Provide a \
brief explanation (1-2 sentences) of your rating to support your decision. After your explanation, \
give the rating in this exact format: "Rating: [[score]]".

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
        lm_model = kwargs.get("lm_model", "gpt-4o-mini") # gpt-4o-mini is cheaper for large-scale eval.
        self.lm_model = LanguageModel(lm_model, kwargs.get("client", None), dump_dir=kwargs.get("dump_dir", None))
        self.concept_id = kwargs.get("concept_id", None)

    def __str__(self):
        return 'LMJudgeEvaluator'
    
    def compute_metrics(self, data, write_to_dir=None):

        logger.warning(
            f"Starting task for concept_id: {self.concept_id}, "
            f"model: {self.model_name}, evaluator: {self.__str__()}")

        data = data.copy()
        
        # Using OpenAI API to judge the quality of the generated data
        prompts = []
        for _, row in data.iterrows():
            prompts += [RATING_TEMPLATE % (
                row["input_concept"], row["input"] + row[f"{self.model_name}_steered_generation"])]

        async def process_batch():
            try:
                return await self.lm_model.chat_completions(
                    f"{self.concept_id}_{self.model_name}_{self.__str__()}_LMJudgeEvaluator", 
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
        
        price = round(self.lm_model.stats.get_total_price(), 3)
        self.lm_model.dump()
        logger.warning(f"LMJudgeEvaluator API costs: ${price}")
        ratings = []
        for completion in completions:
            try:
                # Look for rating in various formats
                rating = None
                
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
                    rating = float(rating_text)/10.0
                
                # Try to find "**Rating: X**" format (markdown)
                elif "**Rating:" in completion:
                    rating_text = completion.split("**Rating:")[-1].split("**")[0].strip()
                    rating = float(rating_text)/10.0
                
                if rating is not None and 0 <= rating <= 1:
                    ratings.append(rating)
                else:
                    logger.warning(f"Invalid rating value: {rating}")
                    ratings.append(0.0)
                    
            except Exception as e:
                logger.warning(f"Failed to parse rating:\n\n{completion}\nError: {str(e)}")
                ratings.append(0.0)
        data[f"{self.model_name}_lm_judge_rating"] = ratings

        metrics = {
            "lm_judge_rating": [],
            "factor": []
        }
        
        # group by factor only and compute means
        grouped = data.groupby("factor")
        for factor, group in grouped:
            lm_judge_rating = group[f"{self.model_name}_lm_judge_rating"].mean()
            metrics["lm_judge_rating"].append(lm_judge_rating)
            metrics["factor"].append(factor)

        return metrics

