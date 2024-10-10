#################################
#
# Async OpenAI utils.
#
#################################

import random

from .templates import *
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


async def get_concept_genres(client, concept):
    prompt_for_genre = T_DETERMINE_GENRE.format(CONCEPT=concept)
    response_for_words = await client.chat_completions("get_concept_genre", prompt_for_genre)
    if "none" in response_for_words.lower():
        return TEXT_GENRES # if none, assign it with the text genre set
    else:
        all_genres = []
        if "text" in response_for_words.lower():
            all_genres += TEXT_GENRES
        if "code" in response_for_words.lower():
            all_genres += CODE_GENRES
        if "math" in response_for_words.lower():
            all_genres += MATH_GENRES
        return all_genres


async def get_contrast_concepts(client, concept, contrast_concepts=None):
    """
    # From concept to contrast concepts
    # 1. get related words for the starting concept.
    # 2. query semantic meanings for each word other than the concept.
    # 3. filtering.

    If contrast_concepts is provided, we want to also filter out concepts that
    are similar to the existing concepts.
    """
    ploys = []
    existing_concepts = []
    if contrast_concepts != None:
        assert concept in contrast_concepts
        for item in contrast_concepts[concept]:
            existing_concepts += [item[-1]]

    # step 1.
    prompt_for_words = T_CONCEPT_TO_WORDS.format(CONCEPT=concept)
    response_for_words = await client.chat_completions("get_contrast_concepts/prompt_for_words", 
                                                       prompt_for_words)
    words = [w.strip() for w in response_for_words.split("\n")]
    for w in words:
        # step 2.
        prompt_for_ploy_meaning = T_WORD_POLYSEMANTIC_MEANING.format(WORD=w, CONCEPT=concept)
        response_for_ploy_meaning = await client.chat_completions(
            "get_contrast_concepts/prompt_for_ploy_meaning", prompt_for_ploy_meaning).strip()
        if "none" in response_for_ploy_meaning.lower() or w == "" or response_for_ploy_meaning == "":
            continue
        # step 3.
        prompt_is_meaning_not_same = T_FILTER_CONTRAST_CONCEPT.format(CONTRAST_CONCEPT=response_for_ploy_meaning, CONCEPT=concept)
        response_is_meaning_not_same = await client.chat_completions(
            "get_contrast_concepts/prompt_is_meaning_not_same", prompt_is_meaning_not_same)
        if "yes" not in response_is_meaning_not_same.split("Answer")[-1].lower():
            continue
        # optional step 4.
        if len(existing_concepts) != 0:
            prompt_exist_is_meaning_not_same = T_FILTER_CONTRAST_MULTI_CONCEPT.format(
                CONTRAST_CONCEPT=response_for_ploy_meaning, CONCEPTS="\n".join(existing_concepts))
            response_exist_is_meaning_not_same = await client.chat_completions(
                "get_contrast_concepts/prompt_exist_is_meaning_not_same", prompt_exist_is_meaning_not_same)
            if "yes" not in response_exist_is_meaning_not_same.split("Answer")[-1].lower():
                continue
        ploys += [(w, response_for_ploy_meaning)]
    return ploys
    

async def get_random_content(client, genres, concepts, length, cutoff_length=0):
    genre = random.choice(genres)
    prompt = T_RANDOM_CONTENT.format(
        GENRE=genre,
        CONCEPTS="\n\n".join(concepts), LENGTH=length)
    response = await client.chat_completions("get_random_content", prompt)
    response = response.split("<FINAL>")[-1].strip(" .'").strip('"')
    if cutoff_length != 0:
        response = " ".join(response.split(" ")[:-1*cutoff_length])
    return response


async def modify_content_with_concept(client, concept, content, length):
    prompt = T_MODIFY_CONTENT_WITH_CONCEPT.format(
        CONTENT=content, CONCEPT=concept, LENGTH=length)
    response = await client.chat_completions("modify_content_with_concept", prompt)
    response = response.split("<FINAL>")[-1].strip(" .'").strip('"')
    return response


async def modify_content_with_contrast_concept(client, concept, contrast_concept, content, length):
    prompt = T_MODIFY_CONTENT_WITH_CONTRAST_CONCEPT.format(
        CONCEPT=concept[1], WORD=concept[0], 
        CONTRAST_CONCEPT=contrast_concept, 
        CONTENT=content, LENGTH=length)
    response = await client.chat_completions("modify_content_with_contrast_concept", prompt)
    response = response.split("<FINAL>")[-1].strip(" .'").strip('"')
    return response


async def get_content_with_concept(client, genres, concept, length):
    genre = random.choice(genres)
    prompt = T_CONTENT_WITH_CONCEPT.format(
        GENRE=genre, CONCEPT=concept, LENGTH=length)
    response = await client.chat_completions("get_content_with_concept", prompt)
    response = response.split("<FINAL>")[-1].strip(" .'").strip('"')
    return response


async def get_content_with_contrast_concept(client, genres, concept, contrast_concept, length):
    genre = random.choice(genres)
    prompt = T_CONTENT_WITH_CONTRAST_CONCEPT.format(
        GENRE=genre,
        CONCEPT=concept[1], WORD=concept[0], CONTRAST_CONCEPT=contrast_concept, 
        LENGTH=length)
    response = await client.chat_completions("get_content_with_contrast_concept", prompt)
    response = response.split("<FINAL>")[-1].strip(" .'").strip('"')
    return response


async def get_continue_with_concept(client, concept, content, length):
    prompt = T_CONTINUE_WITH_CONCEPT.format(
        CONCEPT=concept, CONTENT=content, LENGTH=length)
    response = await client.chat_completions("get_continue_with_concept", prompt)
    return response

