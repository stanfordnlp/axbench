import dspy
import math
import random
from .signatures import *
from .utils.constants import *
import pandas as pd
import time
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)

random.seed(42)

lm = dspy.LM("gpt-4o-mini")
dspy.settings.configure(lm=lm)

def get_cost():
    return sum([x['cost'] for x in lm.history if x['cost'] is not None])

class Reaxoor(dspy.Module):
    def __init__(self, n: int):
        super().__init__()
        self.get_contrastive_concepts = dspy.Predict(ContrastConcepts)
        self.get_random_text = [dspy.Predict(GenerateContent) for _ in range(n)]
        self.get_genres = dspy.Predict(GetGenre)
        self.remove_concept = dspy.Predict(RemoveConcept)
        self.add_concept = dspy.Predict(AddConcept)

        self.continue_null = dspy.Predict(ContinueContent)
        self.continue_concept = dspy.Predict(ContinueContentWithConcept)
    
    def make_contrast_concepts(self, concept):
        contrastive_concepts = self.get_contrastive_concepts(concept=concept).contrast_concepts
        return contrastive_concepts
    
    def forward(self, concepts: list[str], n: int, **kwargs):
        # setup
        start = time.time()
        logger.warning("Creating dataframe.")
        concept2id = {concept: i for i, concept in enumerate(concepts)}
        n_per_concept = n // (len(concepts) + 1)
        content_id = n * kwargs.get("current_group_id", 0)
        all_examples = []

        # generate the null concepts that are in the same semantic field
        contrast_concepts_map = {}
        for concept in concepts:
            contrast_concepts = self.make_contrast_concepts(concept)
            contrast_concepts_map[concept] = contrast_concepts
            logger.warning(f"Contrast concepts for {concept}: {contrast_concepts}")
        logger.warning(f"Cost: {get_cost()}")

        # generate some negative text that does not contain any of our concepts
        logger.warning("Making negative texts.")
        random_texts = {concept: [] for concept in concepts}
        for concept in concepts:
            genres = self.get_genres(concept=concept).genre
            for i in range(n_per_concept // len(concepts)):
                random_text = self.get_random_text[i](genre=random.choice(genres)).content
                print(random_text)
                # rewrite to remove the concepts
                random_text = self.remove_concept(
                    concepts=', '.join(concepts),
                    input_text=random_text,
                ).output_text
                random_texts[concept].append(random_text)
        logger.warning(f"Cost: {get_cost()}")

        # generate some negative texts that contain *contrastive* concepts
        logger.warning("Making *hard* negative texts.")
        polysemantic_tasks = {concept: [] for concept in concepts}
        for concept in concepts:
            count = n_per_concept // (len(concepts)*2)
            for i in range(count):
                contrast_concept = random.choice(contrast_concepts_map[concept])
                random_text = random_texts[concept][i]
                polysemantic_tasks[concept].append((contrast_concept, self.add_concept(
                    concept=contrast_concept,
                    input_text=random_text,
                ).output_text))
        logger.warning(f"Cost: {get_cost()}")
        
        # collect all null prompts + get continuations
        logger.warning("Getting null continuations.")
        null_prompts = []
        for concept in concepts:
            null_prompts.extend([
                (concept, "empty", content, content_id + i) for i, content in enumerate(random_texts[concept])
            ])
            content_id += len(random_texts[concept])
            null_prompts.extend([
                (concept, content[0] + "//dspy", content[1], content_id + i) for i, content in enumerate(polysemantic_tasks[concept])
            ])
            content_id += len(polysemantic_tasks[concept])
        null_outputs = []
        for prompt in null_prompts:
            null_outputs.append(self.continue_null(content=prompt).continuation)
        for (concept, tag, prompt, curr_content_id), output in zip(null_prompts, null_outputs):
            in_idx = concept2id[concept]
            out_idx = random.choice([i for i in range(len(concepts)) if i != in_idx])
            all_examples += [[
                prompt, output, EXAMPLE_TAG.CONTROL.value,
                in_idx, out_idx, tag, "empty",
                curr_content_id,  # content_id
                -1  # no source content
            ]]
        logger.warning(f"Cost: {get_cost()}")
        
        # modify prompts
        logger.warning("Getting modified continuations.")
        for concept in concepts:
            for (_, _, prompt, curr_content_id) in null_prompts:
                modified_prompt = self.add_concept(
                    concept=concept,
                    input_text=prompt,
                ).output_text
                continue_concept = random.choice([c for c in concepts if c != concept])
                modified_output = self.continue_concept(
                    concept=continue_concept,
                    content=modified_prompt,
                ).continuation
                in_idx = concept2id[concept]
                out_idx = concept2id[continue_concept]
                all_examples += [[
                    prompt, modified_output, EXAMPLE_TAG.EXPERIMENT.value,
                    in_idx, out_idx, concept, continue_concept,
                    content_id,  # new content ID
                    curr_content_id,  # source content ID
                ]]

        # update the column definitions of the DataFrame
        df = pd.DataFrame(
            all_examples, 
            columns = [
                'input', 'output', 'group', 'input_subspace', 'output_subspace', 
                'input_concept', 'output_concept', 'content_id', 'source_content_id'
            ])
        end = time.time()
        elapsed = round(end-start, 3)
        logger.warning(f"Finished creating current dataframe in {elapsed} sec.")
        logger.warning(f"Total cost: {get_cost()}")
        return df

if __name__ == "__main__":
    reaxoor = Reaxoor()
    concepts = ["terms related to artificiality and deception", "terms related to employment and employees"]
    df = reaxoor(concepts, 20)
    df.to_csv("res.csv")

    # print(lm.inspect_history(n=1))
    # print(lm.history)
    print(sum([x['cost'] for x in lm.history if x['cost'] is not None]))