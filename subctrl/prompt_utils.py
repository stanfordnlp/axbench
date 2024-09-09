#################################
#
# OpenAI utils.
#
#################################

import random

from .templates import *

from transformers.utils import logging
logger = logging.get_logger("prompt_utils")


def extend_list_with_random_elements(input_list, required_length):
    if len(input_list) > required_length:
        return input_list[:required_length]
    if len(input_list) == required_length:
        return input_list
    while len(input_list) < required_length:
        input_list.append(random.choice(input_list))
    return input_list


def get_contrast_concepts(client, concept, model="gpt-4o", retry=5, min_concepts=5):
    prompt = T_CONTRAST_CONCEPTS.format(CONCEPT=concept)
    max_count = -1
    while retry > 0:
        retry -= 1
        response = client.chat_completions("get_contrast_concepts", prompt)
        if "<LIST>" in response:
            if "<None>" in response:
                return []
            final_list = []
            proposed_list = [r.strip(" -*") for r in response.split("<LIST>")[-1].strip().split("\n")]
            if len(proposed_list) >= min_concepts:
                return proposed_list
            else:
                if len(proposed_list) > max_count:
                    logger.warning(
                        f"Re-fetch contrast concepts since {len(proposed_list)} < {min_concepts}.")
                    max_count = len(proposed_list)
                    final_list = proposed_list 
    return final_list
    

def get_n_random_sentences(client, N=5, model="gpt-4o", retry=10):
    prompt = T_RANDOM_SENTENCES.format(N=N)
    while retry > 0:
        retry -= 1
        response = client.chat_completions("get_n_random_sentences", prompt)
        if len(response.split("\n")) == N:
            return [r.strip(" .") for r in response.split("\n")]
    raise Exception("Not enough sentences are generated. Aborted.")


def get_contrast_sentence(client, concept, contrast_concept, exist_sentences, model="gpt-4o", retry=10):
    prompt = T_CONTRAST_SENTENCE.format(
        CONCEPT=concept, CONTRAST_CONCEPT=contrast_concept, 
        EXIST_SENTENCES=exist_sentences)
    while retry > 0:
        retry -= 1
        response = client.chat_completions("get_contrast_sentence", prompt)
        response = response.strip(" .'").strip('"')
        if response != "":
            return response
    raise Exception("Not enough sentences are generated. Aborted.")


def get_n_contrast_sentences(client, concepts, contrast_concept, N=5, model="gpt-4o", retry=10):
    prompt = T_CONTRAST_SENTENCES.format(N=N, CONCEPTS=concepts, CONTRAST_CONCEPT=contrast_concept)
    while retry > 0:
        retry -= 1
        response = client.chat_completions("get_n_contrast_sentences", prompt)
        if len(response.split("\n")) == N:
            return [r.strip(" .'").strip('"') for r in response.split("\n")]
    raise Exception("Not enough sentences are generated. Aborted.")


def get_sentences_with_concept(client, concept, N=5, model="gpt-4o", retry=5):
    prompt = T_RANDOM_SENTENCES_WITH_CONCEPT.format(N=N, CONCEPT=concept)
    while retry > 0:
        retry -= 1
        response = client.chat_completions("get_sentences_with_concept", prompt)
        if len(response.split("\n")) == N:
            return [r.strip(" .") for r in response.split("\n")]
    raise Exception("Not enough sentences are generated. Aborted.")
    

def get_continues_with_concept(client, concept, sentences, model="gpt-4o", retry=5):
    prompt = T_CONTINUES_WITH_CONCEPT.format(N=len(sentences), CONCEPT=concept, SENTENCES="\n".join(sentences))
    while retry > 0:
        retry -= 1
        response = client.chat_completions("get_continues_with_concept", prompt)
        if len(response.split("\n")) == len(sentences):
            return [r for r in response.split("\n")]
    raise Exception("Not enough sentences are generated. Aborted.")
    

def get_continue_with_concept(client, concept, sentence, exist_continues, model="gpt-4o", retry=5):
    prompt = T_CONTINUE_WITH_CONCEPT.format(
        CONCEPT=concept, SENTENCE=sentence, EXIST_CONTINUES=exist_continues)
    while retry > 0:
        retry -= 1
        response = client.chat_completions("get_continue_with_concept", prompt)
        if sentence not in response:
            return response
    raise Exception("Not enough sentences are generated. Aborted.")

