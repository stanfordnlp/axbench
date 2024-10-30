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
and determine if it effectively contains the specified concept. Note that sentences may be \
incomplete or fragmented; please rate what is provided. Your evaluation should focus on two \
factors: (1) Fluency: whether the fragment is grammatically coherent, clear, and natural; \
(2) Concept Presence: whether the fragment clearly incorporates the user-specified concept. \
If the concept is missing, rate the fragment with a score of 0. If the concept is present, \
rate between 1 and 10 based on fluency and effective incorporation of the concept. Provide a \
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
        lm_model = kwargs.get("lm_model", "gpt-4o")
        self.lm_model = LanguageModel(lm_model, dump_dir=kwargs.get("dump_dir", None))

    def __str__(self):
        return 'LMJudgeEvaluator'
    
    def compute_metrics(self, data, write_to_dir=None):
        data = data.copy()
        
        # Using OpenAI API to judge the quality of the generated data
        prompts = []
        for _, row in data.iterrows():
            prompts += [RATING_TEMPLATE % (
                row["input_concept"], row["input"] + row[f"{self.model_name}_steered_generation"])]
        completions = asyncio.run(self.lm_model.chat_completions(f"{self.model_name}_LMJudgeEvaluator", prompts, batch_size=128))
        price = round(self.lm_model.stats.get_total_price(), 3)
        self.lm_model.dump()
        logger.warning(f"LMJudgeEvaluator API costs: ${price}")
        ratings = []
        for completion in completions:
            try:
                if "[[" in completion:
                    rating = (int(completion.split("[[")[-1].strip("]]").strip()))/10.0
                    ratings.append(rating)
                else:
                    rating = float(completion.split("Rating: ")[-1].strip())
                    ratings.append(rating)
            except:
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

