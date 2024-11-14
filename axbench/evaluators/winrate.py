from .evaluator import Evaluator
from .prompt_templates import *
import asyncio, random, re
from collections import Counter
import pickle, os

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


class WinRateEvaluator(Evaluator):
    DEFAULT_RATING = 0.0
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.lm_model = kwargs.get("lm_model", None)
        self.winrate_baseline = kwargs.get(
            "winrate_baseline", "PromptSteering")
        self.dump_dir = kwargs.get("dump_dir", None)

    def __str__(self):
        return 'WinRateEvaluator'

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
                f"{api_name}_{self.winrate_baseline}_WinRateEvaluator", prompts, batch_size=32
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

    def compute_metrics(self, data):
        """
        This is a three-stage pipeline:
        1. Check concept relevance [score: 0-1]
        2. Check instruction relevance [score: 0-1]
        3. Check fluency [score: 0-2]

        Winning conditions:
        - A winning answer must get 1 for first two checks.
        - If no answer gets 1 for the first two checks, declare a tie.
        - If both answers get 1 for the first two checks, the answer that scores higher in the
          fluency check wins. If both answers score the same in the fluency check, declare a tie.
        """
        data_copy = data.copy()
        data_copy = data_copy.reset_index(drop=True)

        baseline_relevance_concept_ratings, baseline_relevance_instruction_ratings, baseline_fluency_ratings = \
            self._get_all_ratings_from_data(data_copy, self.winrate_baseline)
        model_relevance_concept_ratings, model_relevance_instruction_ratings, model_fluency_ratings = \
            self._get_all_ratings_from_data(data_copy, self.model_name)
        
        # calculate win rate.
        winning_results = []
        for i in range(len(baseline_relevance_concept_ratings)):
            # based?
            is_baseline_based = baseline_relevance_concept_ratings[i][-1] == 1 and \
                baseline_relevance_instruction_ratings[i][-1] == 1
            is_model_based = model_relevance_concept_ratings[i][-1] == 1 and \
                model_relevance_instruction_ratings[i][-1] == 1

            # if both fail the first two checks, it is a tie.
            if not is_baseline_based and not is_model_based:
                # no one wins.
                winning_results += ["tie"]
                continue
            
            # if both pass the first two checks, check fluency.
            if is_baseline_based and is_model_based:
                if baseline_fluency_ratings[i][-1] == 0 and model_fluency_ratings[i][-1] == 0:
                    # no one wins.
                    winning_results += ["tie"]
                    continue
                if baseline_fluency_ratings[i][-1] > model_fluency_ratings[i][-1]:
                    winning_results += ["baseline"]
                elif baseline_fluency_ratings[i][-1] < model_fluency_ratings[i][-1]:
                    winning_results += ["model"]
                else:
                    winning_results += ["tie"]
                continue
            
            # only one passes the first two checks, then check the one that passes.
            if is_baseline_based:
                if baseline_fluency_ratings[i][-1] == 0:
                    winning_results += ["tie"]
                else:
                    winning_results += ["baseline"]
            elif is_model_based:
                if model_fluency_ratings[i][-1] == 0:
                    winning_results += ["tie"]
                else:
                    winning_results += ["model"]

        data[f"{self.model_name}_win_result"] = winning_results
        
        counter = Counter(winning_results)
        win_count = counter["model"]
        loss_count = counter["baseline"]
        tie_count = counter["tie"]
        total_samples = len(winning_results)

        metrics = {
            "win_rate": float(win_count / total_samples),
            "loss_rate": float(loss_count / total_samples),
            "tie_rate": float(tie_count / total_samples),
            "baseline_model": self.winrate_baseline,
        }

        return metrics
            
