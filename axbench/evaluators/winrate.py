from .evaluator import Evaluator
import asyncio, random, re

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)

################################################################################
# WARNING:
# We alternate baseline and model for answer A and B; but we will convert
# to a system where A means the baseline wins, B means the model wins, and C means a tie.
################################################################################


PAIRWISE_EVALUATION_TEMPLATE = """[System]
You are an impartial evaluator tasked with assessing the quality of response fragments provided by two AI assistants in answer to the user question below. Your evaluation must follow these specific criteria:

1. The primary criterion is whether the response fragment contains clear references or terms related to the specified concept. Responses lacking concept-related content should be disqualified unless both responses lack the concept.
2. If both responses contain the concept, evaluate based on how well each response follows the user's instructions.
3. If only one response contains the concept, check its alignment with the instruction:
    - If it aligns well, select it as the better response.
    - If it does not align well, declare a tie.
4. If neither response contains the concept, declare a tie.
5. Avoid allowing factors such as response length, assistant names, or response order to influence your judgment.

Begin your evaluation by comparing the two response fragments and provide a short explanation. Avoid any position biases and ensure that the order in which the response fragments were presented does not influence your decision. Do not allow the length of the response fragments to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.

[User Question]
{question}

[Concept]
{concept}

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""


class WinRateEvaluator(Evaluator):
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.lm_model = kwargs.get("lm_model", None)
        self.winrate_baseline = kwargs.get(
            "winrate_baseline", "PromptSteering")

    def __str__(self):
        return 'WinRateEvaluator'

    def _get_final_verdict(self, text):
        # Try to find text inside either [[...]] or [...]
        verdict = re.search(r'\[\[(.*?)\]\]$', text.strip())
        try:
            if verdict:
                return verdict.group(1)
            else:
                # If no verdict in brackets, return the last letter of the text
                pred_answers = re.findall(r'A|B|C', text.strip())
                if pred_answers:
                    return pred_answers[-1]
                else:
                    return "C" # default to a tie
        except Exception as e:
            logger.warning(f"Error getting final verdict: {e}")
            return "C" # default to a tie

    def compute_metrics(self, data):
        data_copy = data.copy()
        data_copy = data_copy.reset_index(drop=True)

        tags = []
        prompts = []
        # This is a generation dataset.
        for idx, row in data_copy.iterrows():
            input_concept = row["input_concept"]
            original_prompt = row["original_prompt"]
            baseline_generation = row[f"{self.winrate_baseline}_steered_generation"]
            model_generation = row[f"{self.model_name}_steered_generation"]
            is_baseline_a = random.random() < 0.5
            prompt = PAIRWISE_EVALUATION_TEMPLATE.format(
                question=original_prompt,
                concept=input_concept,
                answer_a=baseline_generation if is_baseline_a else model_generation,
                answer_b=model_generation if is_baseline_a else baseline_generation
            )
            tags += [(idx, self.model_name, is_baseline_a)]
            prompts += [prompt]

        async def process_batch():
            try:
                return await self.lm_model.chat_completions(
                    f"{self.winrate_baseline}_WinRateEvaluator", prompts, batch_size=32
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

        results_winrate = []
        for i, completion in enumerate(completions):
            row_idx, model, is_baseline_a = tags[i]
            verdict = self._get_final_verdict(completion)
            prompt = prompts[i]
            if verdict == "A" and is_baseline_a:
                who_wins = "A"
            elif verdict == "A" and not is_baseline_a:
                who_wins = "B"
            elif verdict == "B" and not is_baseline_a:
                who_wins = "A"
            elif verdict == "B" and is_baseline_a:
                who_wins = "B"
            elif verdict == "C":
                who_wins = "C"
            else:
                assert False, f"Unknown verdict: {verdict}"
            results_winrate += [who_wins]
        data[f"{self.model_name}_win_result"] = results_winrate

        total_samples = len(results_winrate)
        win_count = sum(1 for result in results_winrate if result == "B")
        loss_count = sum(1 for result in results_winrate if result == "A")
        tie_count = sum(1 for result in results_winrate if result == "C")

        metrics = {
            "win_rate": win_count / total_samples,
            "loss_rate": loss_count / total_samples,
            "tie_rate": tie_count / total_samples,
            "baseline_model": self.winrate_baseline,
        }

        return metrics
            
