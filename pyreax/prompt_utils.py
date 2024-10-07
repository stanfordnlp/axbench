#################################
#
# OpenAI utils.
#
#################################

import random

from .templates import *

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


def extend_list_with_random_elements(input_list, required_length):
    if len(input_list) > required_length:
        return input_list[:required_length]
    if len(input_list) == required_length:
        return input_list
    while len(input_list) < required_length:
        input_list.append(random.choice(input_list))
    return input_list


def get_contrast_concepts(client, concept, contrast_concepts=None):
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
    response_for_words = client.chat_completions("get_contrast_concepts", prompt_for_words)
    words = [w.strip() for w in response_for_words.split("\n")]
    for w in words:
        # step 2.
        prompt_for_ploy_meaning = T_WORD_POLYSEMANTIC_MEANING.format(WORD=w, CONCEPT=concept)
        response_for_ploy_meaning = client.chat_completions("get_contrast_concepts", prompt_for_ploy_meaning).strip()
        if "none" in response_for_ploy_meaning.lower() or w == "" or response_for_ploy_meaning == "":
            continue
        # step 3.
        prompt_is_meaning_not_same = T_FILTER_CONTRAST_CONCEPT.format(CONTRAST_CONCEPT=response_for_ploy_meaning, CONCEPT=concept)
        response_is_meaning_not_same = client.chat_completions("get_contrast_concepts", prompt_is_meaning_not_same)
        if "yes" not in response_is_meaning_not_same.split("Answer")[-1].lower():
            continue
        # optional step 4.
        if len(existing_concepts) != 0:
            prompt_exist_is_meaning_not_same = T_FILTER_CONTRAST_MULTI_CONCEPT.format(
                CONTRAST_CONCEPT=response_for_ploy_meaning, CONCEPTS="\n".join(existing_concepts))
            response_exist_is_meaning_not_same = client.chat_completions("get_contrast_concepts", prompt_exist_is_meaning_not_same)
            if "yes" not in response_exist_is_meaning_not_same.split("Answer")[-1].lower():
                continue
        ploys += [(w, response_for_ploy_meaning)]
    return ploys
    

def get_random_sentence(client, concepts, exist_sentences):
    """Generate a random sentence without mentioning concepts"""
    _exist_sentences = exist_sentences
    if len(exist_sentences) > 5:
        _exist_sentences = random.sample(exist_sentences, 5)
    prompt = T_RANDOM_SENTENCE.format(
        CONCEPTS="\n\n".join(concepts), EXIST_SENTENCES="\n\n".join(_exist_sentences))
    response = client.chat_completions("get_random_sentence", prompt)
    response = response.strip(" .'").strip('"')
    return response


def get_contrast_sentence(client, concept, contrast_concept, exist_sentences):
    """Generate a random contrast sentence"""
    _exist_sentences = exist_sentences
    if len(exist_sentences) > 5:
        _exist_sentences = random.sample(exist_sentences, 5)
    prompt = T_CONTRAST_SENTENCE.format(
        CONCEPT=concept[1], WORD=concept[0], CONTRAST_CONCEPT=contrast_concept, 
        EXIST_SENTENCES="\n\n".join(_exist_sentences))
    response = client.chat_completions("get_contrast_sentence", prompt)
    response = response.split("<FINAL>")[-1].strip(" .'").strip('"')
    return response


def get_sentence_with_concept(client, concept, exist_sentences, retry=5):
    _exist_sentences = exist_sentences
    if len(exist_sentences) > 5:
        _exist_sentences = random.sample(exist_sentences, 5)
    prompt = T_RANDOM_SENTENCE_WITH_CONCEPT.format(
        CONCEPT=concept, EXIST_SENTENCES="\n\n".join(_exist_sentences))
    response = client.chat_completions("get_sentence_with_concept", prompt)
    response = response.split("<FINAL>")[-1].strip(" .'").strip('"')
    return response


def get_simple_sentence_with_concept(client, concept, exist_sentences, retry=5):
    _exist_sentences = exist_sentences
    if len(exist_sentences) > 5:
        _exist_sentences = random.sample(exist_sentences, 5)
    prompt = T_SIMPLE_RANDOM_SENTENCE_WITH_CONCEPT.format(
        CONCEPT=concept, EXIST_SENTENCES="\n\n".join(_exist_sentences))
    response = client.chat_completions("get_sentences_with_concept", prompt)
    response = response.split("<FINAL>")[-1].strip(" .'").strip('"')
    return response


def get_continue_with_concept(client, concept, sentence, exist_continues, retry=5):
    _exist_continues = exist_continues
    if len(exist_continues) > 5:
        _exist_continues = random.sample(exist_continues, 5)
    prompt = T_CONTINUE_WITH_CONCEPT.format(
        CONCEPT=concept, SENTENCE=sentence, EXIST_CONTINUES="\n\n".join(_exist_continues))
    response = client.chat_completions("get_continue_with_concept", prompt)
    return response

