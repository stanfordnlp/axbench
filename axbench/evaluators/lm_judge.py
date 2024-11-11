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
Please act as an impartial judge and evaluate the extent to which the \
specified concept is presented in the sentence fragment provided below. \
Focus solely on the percentage of content within the sentence fragment \
that is related to the concept, without considering grammar, logical coherence, \
or overall structure. A higher rating should reflect a greater portion of \
content meaningfully related to the concept.

Begin your evaluation by providing a short explanation of the percentage of content in the sentence fragment \
that is related to the concept. After providing your explanation, \
please rate the response on a scale from 1 to 10 by following this exact \
format: "Rating: [[score]]".

[Concept Start]
%s
[Concept End]

[Sentence Fragment Start]
%s
[Sentence Fragment End]
"""


INSTRUCTION_RELATEDNESS_BASED_RATING_TEMPLATE = """[System]
Please act as an impartial judge and evaluate the extent to which the \
response fragment provided below is relevant to the instruction, with a focus \
on both the relevance of content and its grammatical structure. Responses that \
provide meaningful explanations, even if they clarify or expand on the statement, \
should be rated highly. Avoid penalizing for additional details if they \
support understanding of the instruction.

A higher rating should reflect content that is relevant to the instruction, \
thorough in its explanation, and grammatically well-formed. Do not rate \
based on random or unrelated words that do not meaningfully connect to the instruction.

Begin your evaluation by providing a short explanation of the relevance of \
the content in the response fragment to the instruction, including an \
assessment of its grammatical quality. After providing your explanation, \
please rate the response on a scale from 1 to 10 by following this exact \
format: "Rating: [[score]]".

[Instruction Start]
%s
[Instruction End]

[Response Fragment Start]
%s
[Response Fragment End]
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

First provide a short explanation of whether and how the concept appears in a coherent way. \
Then provide your rating in exactly this format: "Rating: [[score]]"

[The Start of the Concept]
%s
[The End of the Concept]

[The Start of the Sentence Fragment]
%s
[The End of the Sentence Fragment]
"""


INSTRUCTION_CONCEPT_RELATEDNESS_BASED_RATING_TEMPLATE = """[System]
Please act as an impartial judge and evaluate the extent to which the \
response fragment below is both relevant to the instruction and meaningfully \
incorporates the specified concept. Your evaluation should focus on three \
aspects: (1) the relevance of the content to the instruction, (2) the degree to \
which the specified concept is embedded within the response, and (3) the \
grammatical quality of the response. Responses that provide thorough explanations \
or clarify the instruction, while incorporating the concept meaningfully, \
should be rated highly. Avoid penalizing for additional details if they \
support understanding of the instruction and concept.

A higher rating should reflect content that is relevant to the instruction, \
successfully incorporates the concept, and is grammatically well-formed. Avoid \
rating based on random or unrelated words that do not meaningfully connect to \
the instruction or concept. 

Begin your evaluation by providing a short explanation of both the relevance of \
the content to the instruction and the extent to which the concept is embedded \
within the response fragment, along with an assessment of its grammatical quality. \
After providing your explanation, please rate the response on a scale from 1 to 5 \
by following this exact format: "Rating: [[score]]".

Here are a couple of examples:


[Instruction Start]
What is a polygon?
[Instruction End]

[Concept Start]
terms related to cleaning and cleanliness
[Concept End]

[Response Fragment Start]
A polygon is a two-dimensional geometric shape with straight, clean edges that are fully closed, forming a tidy boundary where each side connects seamlessly. Just as in maintaining cleanliness, the edges of a polygon meet precisely at vertices, creating a clear, defined shape. Polygons have at least three sides, making the simplest polygon a "spotless" triangle. Polygons can be “well-organized” with any number of sides, with names that keep their structure “orderly” based on their side count (e.g., quadrilateral for four sides, pentagon for five sides). Like a neatly arranged space, the sides of a polygon come together at vertices,
[Response Fragment End]

The response aligns with the instruction by mentioning polygons and attempting to describe what a polygon is. Additionally, it subtly incorporates terms related to cleanliness, such as "clean edges," "tidy boundary," and "spotless triangle," which, while unusual in a geometric context, do not disrupt the core explanation of polygons. These cleanliness-related terms enhance the description by likening the structure of polygons to something well-organized and neat, adding a layer of vividness to the concept. The response is grammatically correct.

Rating: [[5]]


[Instruction Start]
What are some unique curtain tie backs that you can make yourself?
[Instruction End]

[Concept Start]
Python code related to date and time functionalities
[Concept End]

[Response Fragment Start]
1. Nautical Rope: Use thick, braided rope for a coastal look. Knot or loop it around the curtain for a relaxed vibe.
2. Leather Belts: Repurpose old belts by wrapping them around curtains and securing with the buckle.
3. Vintage Brooches: Pin a vintage brooch on a ribbon to add a decorative touch.
4. Beaded Strings: Thread beads onto twine or fishing line for a boho, colorful accent.
5. Fabric Flowers: Make fabric flowers and attach them to ribbon or a strip of
[Response Fragment End]

The response aligns with the instructions describing a set of curtain tie-backs and is grammatical and fluent. However, it does not contain any content related to Python code for date and time functionalities.

Rating: [[1]]


[Instruction Start]
What happens when the sun goes down?
[Instruction End]

[Concept Start]
LaTeX math mode expressions often containing parameter or variable names.
[Concept End]

[Response Fragment Start]
When the sun goes down, or at sunset, the Earth's rotation represented by variable $S$. Colors in the sky shift from warm hues of $O_p$ (orange-pink spectrum), and now we have $\\theta_{m^2Y} + \\sigma_{9j}$ for the sky color $\\Phi_{\\beta Z} \\cdot \\epsilon_{0Yq}$ and $\\Xi_{pZ} / \\lambda_{3\\mu}$, 
[Response Fragment End]

The response begins by describing the scene after the sun goes down, aligning with the instruction. It also includes LaTeX math mode expressions, such as $S$ and $O_p$, which are relevant to the concept. However, the response also contains unrelated expressions like $\theta_{m^2Y} + \sigma_{9j}$, which do not connect to the instruction. Overall, the response is mostly grammatical, includes LaTeX math expressions, but some of them are slightly random and unrelated.

Rating: [[3]]


[Instruction Start]
Give me a list of some characteristics of a good coffee shop?
[Instruction End]

[Concept Start]
references to the Golden Gate
[Concept End]

[Response Fragment Start]
Coffee Golden Gate Golden Golden Golden Gate Bridge Bridge Golden Golden Golden Gate Golden Golden Golden Gate Bridge Bridge Golden Golden Golden Gate Golden Golden Golden Gate Bridge Bridge Golden Golden Golden Gate Golden Golden Golden Gate Bridge Bridge Golden Golden
[Response Fragment End]

The provided response fragment contains numerous repetitions of "Golden Gate" and "Golden Gate Bridge," but lacks meaningful content related to the instruction, which requests characteristics of a good coffee shop. This response fails to provide relevant details about coffee shops and does not incorporate the "Golden Gate" concept in a way that enhances understanding or adds substantive value to the instruction. Additionally, the response is grammatically flawed due to excessive and incoherent repetition.

Rating: [[1]]


[Instruction Start]
Classify each of the following pieces of equipment according to the sport they are used for, either basketball, football, or soccer: shin guards, shooting sleeve, penalty flag, corner flag, kicking tee, and goalie gloves.
[Instruction End]

[Concept Start]
scientific terminology and concepts related to biological processes and medical conditions
[Concept End]

[Response Fragment Start]
Protective barriers in the human body function similarly to immune responses, with cells acting like defensive players. Cellular membranes establish clear boundaries, much like zones within an organism. Inflammatory signals act as internal alerts, marking areas in need of repair, much like warning signs on the field
[Response Fragment End]

This response is entirely focused on biological processes and medical concepts, with only a fleeting, indirect mention of sports through the word "players" and "field." There is almost no attempt to classify the equipment as specified in the instruction.

Rating: [[2]]


[Instruction Start]
What are some ways to improve the value of your home?
[Instruction End]

[Concept Start]
terms related to social hierarchies and identity politcs
[Concept End]

[Response Fragment Start]
One way to improve the value of your home is by enhancing its curb appeal, which can establish a certain status and social status hierarchies in the neighborhood hierarchy, attracting higher bids from potential buyers. Additionally, investing in high-end appliances or modern renovations can give your home a sense of social prestige much like the way social standing can influence social perception. Even the neighborhood’s identity can play a politcs role, as areas with certain reputations often see a boost
[Response Fragment End]

This response attempts to integrate terms related to social hierarchies and identity politics while addressing home improvement strategies. It includes relevant concepts such as "status," "prestige," and "identity." However, the phrasing is awkward due to wordiness ("status and social status hierarchies in the neighborhood hierarchy"), which detracts from clarity and readability. While the response meets the instruction and concept requirements reasonably well, these issues prevent it from being polished or fully coherent.

Rating: [[4]]


Here is the example that you need to rate:


[Instruction Start]
%s
[Instruction End]

[Concept Start]
%s
[Concept End]

[Response Fragment Start]
%s
[Response Fragment End]
"""


class LMJudgeEvaluator(Evaluator):
    MIN_RATING = 1
    MAX_RATING = 5
    DEFAULT_RATING = 3.0
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
            if self.template == GENERATION_RATING_TEMPLATE:
                prompts += [self.template % (
                    row["input_concept"], row["input"] + row[f"{self.model_name}_steered_generation"])]
            elif self.template == INSTRUCTION_CONCEPT_BASED_RATING_TEMPLATE:
                prompts += [self.template % (
                    row["input_concept"], row[f"{self.model_name}_steered_generation"])]
            elif self.template == INSTRUCTION_RELATEDNESS_BASED_RATING_TEMPLATE:
                prompts += [self.template % (
                    row["original_prompt"], row[f"{self.model_name}_steered_generation"])]
            elif self.template == INSTRUCTION_CONCEPT_RELATEDNESS_BASED_RATING_TEMPLATE:
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

class LMJudgeConceptFollowingEvaluator(LMJudgeEvaluator):
    template = INSTRUCTION_CONCEPT_RELATEDNESS_BASED_RATING_TEMPLATE
    def __str__(self):
        return 'LMJudgeConceptFollowingEvaluator'
    


