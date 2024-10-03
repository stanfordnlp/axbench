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


def get_contrast_concepts(client, concept, retry=5, min_concepts=1):
    prompt = T_CONTRAST_CONCEPTS.format(CONCEPT=concept)
    max_count = -1
    final_list = []
    while retry > 0:
        retry -= 1
        response = client.chat_completions("get_contrast_concepts", prompt)
        if "<LIST>" in response:
            if "<None>" in response:
                return []
            proposed_list = [r.strip(" -*") for r in response.split("<LIST>")[-1].strip().split("\n")]
            # filter proposed list
            filter_prompt = T_FILTER_CONTRAST_CONCEPTS.format(
                CONTRAST_CONCEPTS="\n\n".join(proposed_list), CONCEPT=concept)
            response = client.chat_completions("get_contrast_concepts", filter_prompt)
            filtered_list = []
            for r in response.split("\n"):
                if r.strip() != "":
                    filtered_list += [r.strip()]
            if len(filtered_list) >= min_concepts:
                return filtered_list
            else:
                if len(filtered_list) > max_count:
                    logger.warning(
                        f"Re-fetch contrast concepts since {len(filtered_list)} < {min_concepts}.")
                    max_count = len(filtered_list)
                    final_list = filtered_list 
    return final_list
    

def get_n_random_sentence(client, concepts, exist_sentences, retry=10):
    _exist_sentences = exist_sentences
    if len(exist_sentences) > 5:
        _exist_sentences = random.sample(exist_sentences, 5)
    prompt = T_RANDOM_SENTENCE.format(
        CONCEPTS="\n\n".join(concepts), EXIST_SENTENCES="\n\n".join(_exist_sentences))
    while retry > 0:
        retry -= 1
        response = client.chat_completions("get_n_random_sentence", prompt)
        response = response.strip(" .'").strip('"')
        if response != "":
            return response
    raise Exception("Not enough sentences are generated. Aborted.")


def get_n_random_sentences(client, concepts, N=5, retry=10):
    prompt = T_RANDOM_SENTENCES.format(N=N, CONCEPTS="\n\n".join(concepts))
    while retry > 0:
        retry -= 1
        response = client.chat_completions("get_n_random_sentences", prompt)
        if len(response.split("\n")) == N:
            return [r.strip(" .") for r in response.split("\n")]
    raise Exception("Not enough sentences are generated. Aborted.")


def get_contrast_sentence(client, concept, contrast_concept, exist_sentences, retry=10):
    _exist_sentences = exist_sentences
    if len(exist_sentences) > 5:
        _exist_sentences = random.sample(exist_sentences, 5)
    prompt = T_CONTRAST_SENTENCE.format(
        CONCEPT=concept[1], WORD=concept[0], CONTRAST_CONCEPT=contrast_concept, 
        EXIST_SENTENCES="\n\n".join(_exist_sentences))
    while retry > 0:
        retry -= 1
        response = client.chat_completions("get_contrast_sentence", prompt)
        response = response.strip(" .'").strip('"')
        if response != "":
            return response
    raise Exception("Not enough sentences are generated. Aborted.")


def get_n_contrast_sentences(client, concepts, contrast_concept, N=5, retry=10):
    prompt = T_CONTRAST_SENTENCES.format(N=N, CONCEPTS=concepts, CONTRAST_CONCEPT=contrast_concept)
    while retry > 0:
        retry -= 1
        response = client.chat_completions("get_n_contrast_sentences", prompt)
        if len(response.split("\n")) == N:
            return [r.strip(" .'").strip('"') for r in response.split("\n")]
    raise Exception("Not enough sentences are generated. Aborted.")


def get_sentence_with_concept(client, concept, exist_sentences, retry=5):
    _exist_sentences = exist_sentences
    if len(exist_sentences) > 5:
        _exist_sentences = random.sample(exist_sentences, 5)
    prompt = T_RANDOM_SENTENCE_WITH_CONCEPT.format(
        CONCEPT=concept, EXIST_SENTENCES="\n\n".join(_exist_sentences))
    while retry > 0:
        retry -= 1
        response = client.chat_completions("get_sentences_with_concept", prompt)
        response = response.split("<FINAL>")[-1].strip(" .'").strip('"')
        if response != "":
            return response
    raise Exception("Not enough sentences are generated. Aborted.")


def get_simple_sentence_with_concept(client, concept, exist_sentences, retry=5):
    _exist_sentences = exist_sentences
    if len(exist_sentences) > 5:
        _exist_sentences = random.sample(exist_sentences, 5)
    prompt = T_SIMPLE_RANDOM_SENTENCE_WITH_CONCEPT.format(
        CONCEPT=concept, EXIST_SENTENCES="\n\n".join(_exist_sentences))
    while retry > 0:
        retry -= 1
        response = client.chat_completions("get_sentences_with_concept", prompt)
        response = response.split("<FINAL>")[-1].strip(" .'").strip('"')
        if response != "":
            return response
    raise Exception("Not enough sentences are generated. Aborted.")


def get_sentences_with_concept(client, concept, N=5, retry=5):
    prompt = T_RANDOM_SENTENCES_WITH_CONCEPT.format(N=N, CONCEPT=concept)
    while retry > 0:
        retry -= 1
        response = client.chat_completions("get_sentences_with_concept", prompt)
        if len(response.split("\n")) == N:
            return [r.strip(" .") for r in response.split("\n")]
    raise Exception("Not enough sentences are generated. Aborted.")
    

def get_continues_with_concept(client, concept, sentences, retry=5):
    prompt = T_CONTINUES_WITH_CONCEPT.format(N=len(sentences), CONCEPT=concept, SENTENCES="\n\n".join(sentences))
    while retry > 0:
        retry -= 1
        response = client.chat_completions("get_continues_with_concept", prompt)
        if len(response.split("\n")) == len(sentences):
            return [r for r in response.split("\n")]
    raise Exception("Not enough sentences are generated. Aborted.")
    

def get_continue_with_concept(client, concept, sentence, exist_continues, retry=5):
    _exist_continues = exist_continues
    if len(exist_continues) > 5:
        _exist_continues = random.sample(exist_continues, 5)
    prompt = T_CONTINUE_WITH_CONCEPT.format(
        CONCEPT=concept, SENTENCE=sentence, EXIST_CONTINUES="\n\n".join(_exist_continues))
    while retry > 0:
        retry -= 1
        response = client.chat_completions("get_continue_with_concept", prompt)
        if sentence not in response:
            return response
    raise Exception("Not enough sentences are generated. Aborted.")

