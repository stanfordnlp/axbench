#################################
#
# Async OpenAI utils.
#
#################################

import random

from ..templates.prompt_templates import *
from .constants import *

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


def sample_index_exclude(index_range, exclude_index):
    if exclude_index < 0 or exclude_index >= index_range:
        raise ValueError("exclude_index must be within the valid index range.")
    possible_indices = [i for i in range(index_range) if i != exclude_index]
    return random.choice(possible_indices)


def extend_list_with_random_elements(input_list, required_length):
    if len(input_list) > required_length:
        return input_list[:required_length]
    if len(input_list) == required_length:
        return input_list
    while len(input_list) < required_length:
        input_list.append(random.choice(input_list))
    return input_list


async def get_concept_genres(client, concepts):
    concept_genres = {}
    prompts = [T_DETERMINE_GENRE.format(CONCEPT=concept) for concept in concepts]
    responses = await client.chat_completions("get_concept_genre", prompts)
    
    for i, response in enumerate(responses):
        if "none" in response.lower():
            concept_genres[concepts[i]] = [TEXT_GENRES] # if none, assign it with the text genre set
        else:
            genres = []
            if "text" in response.lower():
                genres += TEXT_GENRES
            if "code" in response.lower():
                genres += CODE_GENRES
            if "math" in response.lower():
                genres += MATH_GENRES
            concept_genres[concepts[i]] = genres
    return concept_genres


async def get_contrast_concepts(client, concepts, contrast_concepts=None):
    """
    # From concept to contrast concepts
    # 1. get related words for the starting concept.
    # 2. query semantic meanings for each word other than the concept.
    # 3. filtering.

    If contrast_concepts is provided, we want to also filter out concepts that
    are similar to the existing concepts.
    """
    polysemantics = {concept: [] for concept in concepts}

    # async step 1.
    prompts = [T_CONCEPT_TO_WORDS.format(CONCEPT=concept) for concept in concepts]
    responses = await client.chat_completions(
        "get_contrast_concepts.prompt_for_words", prompts)
    all_words = [[w.strip() for w in response.split("\n")] for response in responses]
    
    # async step 2.
    prompts = [T_WORD_POLYSEMANTIC_MEANING.format(
        WORD=w, CONCEPT=concepts[i]) for i, words in enumerate(all_words) for w in words]
    flatten_words = [(w, concepts[i]) for i, words in enumerate(all_words) for w in words]
    word_polysemantics = await client.chat_completions(
        "get_contrast_concepts.prompt_for_ploy_meaning", prompts)
    
    # async step 3.
    prompts = []
    filtered_word_polysemantics = []
    for _, ((w, concept), word_polysemantic) in enumerate(zip(flatten_words, word_polysemantics)):
        if "none" in word_polysemantic.lower() or w == "" or word_polysemantic == "":
            continue
        prompts += [T_FILTER_CONTRAST_CONCEPT.format(
            CONTRAST_CONCEPT=word_polysemantic, CONCEPT=concept)]
        filtered_word_polysemantics += [(concept, w, word_polysemantic)]
    polysemantic_checks = await client.chat_completions(
        "get_contrast_concepts.prompt_is_meaning_not_same", prompts)
        
    # optional async step 4.
    prompts = []
    further_filtered_word_polysemantics = []
    for i, polysemantic_check in enumerate(polysemantic_checks):
        concept, w, word_polysemantic = filtered_word_polysemantics[i]
        polysemantic_check = polysemantic_check.split("Answer")[-1].lower()
        if "yes" not in polysemantic_check:
            continue
        if contrast_concepts != None and concept in contrast_concepts:
            existing_concepts = [item[-1] for item in contrast_concepts[concept]]
            if len(existing_concepts) != 0:
                prompts += [T_FILTER_CONTRAST_MULTI_CONCEPT.format(
                    CONTRAST_CONCEPT=filtered_word_polysemantics[i], CONCEPTS="\n".join(existing_concepts))]
                further_filtered_word_polysemantics += [(concept, w, word_polysemantic)]
        else:
            polysemantics[concept] += [(w, word_polysemantic)]
    if len(prompts) != 0:
        exist_meaning_checks = await client.chat_completions(
            "get_contrast_concepts.prompt_exist_is_meaning_not_same", prompts)
        for i, exist_meaning_check in enumerate(exist_meaning_checks):
            concept, w, word_polysemantic = filtered_word_polysemantics
            if "yes" not in response_exist_is_meaning_not_same.split("Answer")[-1].lower():
                continue
            polysemantics[concept] += [(w, word_polysemantic)]
    return polysemantics
    

async def get_random_content(client, tokenizer, count, genres, concepts, length):
    random_content = {concept: [] for concept in concepts}

    prompts = []
    for concept in concepts:
        prompts += [T_RANDOM_CONTENT.format(
            GENRE=random.choice(genres[concept]),
            CONCEPTS="\n\n".join(concepts), LENGTH=length) for _ in range(count)]
    responses = await client.chat_completions("get_random_content", prompts)

    for i, response in enumerate(responses):
        response = response.split("<FINAL>")[-1].strip(" .'").strip('"')
        response = tokenizer.convert_tokens_to_string(
            tokenizer.tokenize(response)[:int(length*1.5)])
        random_content[concepts[i//(len(responses)//2)]] += [response]
        
    return random_content


async def modify_content_with_polysemantic_concepts(client, tokenizer, polysemantic_concepts, concept, content, length):
    prompts = []
    for i, polysemantic_concept in enumerate(polysemantic_concepts):
        prompts += [T_MODIFY_CONTENT_WITH_CONTRAST_CONCEPT.format(
            CONCEPT=polysemantic_concept[1], WORD=polysemantic_concept[0], 
            CONTRAST_CONCEPT=concept, CONTENT=content[i], LENGTH=length)]
    responses = await client.chat_completions("modify_content_with_polysemantic_concepts", prompts)
    return (concept, zip(
        polysemantic_concepts, [
            tokenizer.convert_tokens_to_string(
                tokenizer.tokenize(response.split("<FINAL>")[-1].strip(" .'").strip('"'))[:int(length*1.5)])
            for response in responses]))


async def modify_content_with_concept(client, tokenizer, content, length):
    prompts = []
    for (concept, tag, output) in content:
        prompts += [T_MODIFY_CONTENT_WITH_CONCEPT.format(
            CONTENT=output, CONCEPT=concept, LENGTH=length)]
    responses = await client.chat_completions("modify_content_with_concept", prompts)
    return [tokenizer.convert_tokens_to_string(tokenizer.tokenize(
        response.split("<FINAL>")[-1].strip(" .'").strip('"'))[:int(length*1.5)]) for response in responses]


async def continue_with_concept(client, tokenizer, concepts, content, length):
    prompts = []
    for i, concept in enumerate(concepts):
        prompts += [T_CONTINUE_WITH_CONCEPT.format(
            CONCEPT=concept, CONTENT=content[i], LENGTH=length)]
    responses = await client.chat_completions("continue_with_concept", prompts)
    return [tokenizer.convert_tokens_to_string(tokenizer.tokenize(
        response.split("<FINAL>")[-1].strip(" .'").strip('"'))[:int(length*1.5)]) for response in responses]


async def get_content_with_concept(client, tokenizer, count, genres, concept, length):
    prompts = []
    for _ in range(count):
        prompts += [T_CONTENT_WITH_CONCEPT.format(
            GENRE=random.choice(genres), CONCEPT=concept, LENGTH=length)]
    responses = await client.chat_completions("get_content_with_concept", prompts)
    return [tokenizer.convert_tokens_to_string(tokenizer.tokenize(
        response.split("<FINAL>")[-1].strip(" .'").strip('"'))[:int(length*1.5)]) for response in responses]


async def get_content_with_polysemantic_concepts(client, tokenizer, genres, concept, polysemantic_concepts, length):
    prompts = []
    for i, concept in enumerate(polysemantic_concepts):
        prompts += [T_CONTENT_WITH_CONTRAST_CONCEPT.format(
            GENRE=random.choice(genres),
            CONCEPT=concept[1], WORD=concept[0], CONTRAST_CONCEPT=contrast_concept, LENGTH=length)]
    responses = await client.chat_completions("get_content_with_polysemantic_concepts", prompts)
    return (concept, zip(
        polysemantic_concepts, [
            tokenizer.convert_tokens_to_string(
                tokenizer.tokenize(response.split("<FINAL>")[-1].strip(" .'").strip('"'))[:int(length*1.5)])
            for response in responses]))



