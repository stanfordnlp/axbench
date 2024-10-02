from .constants import *

import os, uuid, string, json
from pathlib import Path

from transformers.utils import logging
logger = logging.get_logger("LanguageModelStats")


def is_first_char_punctuation(s):
    if s and s[0] in string.punctuation:
        return True
    return False


class LanguageModelStats(object):
    """Main class for recording language model usage"""

    def __init__(self, model):
        self.model = model
        
        _uuid = str(uuid.uuid4())
        self.key = f"{model}-{_uuid}"
        self.completion_tokens = {}
        self.prompt_tokens = {}
        self.prompt_cache = {}

    def record(self, api_name, stats, prompt=None):
        if api_name not in self.completion_tokens:
            self.completion_tokens[api_name] = []
        if api_name not in self.prompt_tokens:
            self.prompt_tokens[api_name] = []
        if api_name not in self.prompt_cache:
            self.prompt_cache[api_name] = []
            
        completion_tokens = int(stats["completion_tokens"])
        self.completion_tokens[api_name].append(completion_tokens)
        prompt_tokens = int(stats["prompt_tokens"])
        self.prompt_tokens[api_name].append(prompt_tokens)
        logger.debug(
            f"calling {api_name}, input tokens {prompt_tokens}, "
            f"output tokens {completion_tokens}")
        if prompt != None:
            self.prompt_cache[api_name].append(prompt)
    
    def get_total_tokens(self, breakdown=True):
        sum_completion_tokens, sum_prompt_tokens = 0, 0
        for _, v in self.prompt_tokens.items():
            sum_prompt_tokens += sum(v)
        for _, v in self.completion_tokens.items():
            sum_completion_tokens += sum(v)
        if breakdown:
            return sum_prompt_tokens, sum_completion_tokens
        return sum_prompt_tokens + sum_completion_tokens

    def reset(self):
        self.completion_tokens = {}
        self.prompt_tokens = {}
        self.prompt_cache = {}

    def get_total_price(self):
        input_tokens, output_tokens = self.get_total_tokens()
        input_price = (input_tokens/UNIT_1M)*\
            PRICING_DOLLAR_PER_1M_TOKEN[self.model]["input"]
        output_price = (input_tokens/UNIT_1M)*\
            PRICING_DOLLAR_PER_1M_TOKEN[self.model]["output"]
        return input_price + output_price

    
class LanguageModel(object):
    """Main class abstract remote language model access"""

    def __init__(self, model, dump_dir, **kwargs):
        self.model = model
        if model == "gpt-4o":
            try:
                from openai import OpenAI
                
                self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            except:
                raise Exception("Cannot connect to openai. Check your API key.")
        else:
            raise ValueError(f"{model} model class is not supported yet.")
        self.stats = LanguageModelStats(model)

        # dump dir
        cur_save_dir = Path(dump_dir) / "lm_cache"
        cur_save_dir.mkdir(parents=True, exist_ok=True)
        self.dump_dir = cur_save_dir
    
    def normalize(self, text):
        normalized = text.strip()
        if is_first_char_punctuation(normalized):
            return normalized
        return " "+normalized

    def chat_completions(self, api_name, prompt):
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model=self.model)
        raw_completion = chat_completion.to_dict()
        self.stats.record(api_name, raw_completion['usage'], prompt=prompt)
        return self.normalize(raw_completion["choices"][0]["message"]["content"])

    def dump(self):
        with open(self.dump_dir / "tmp_prompt_cache.json", "w") as outfile:
            json.dump(self.stats.prompt_cache, outfile, indent=4)
        
        with open(self.dump_dir / "cost.jsonl", 'a') as f:
            f.write(json.dumps({"price": self.stats.get_total_price()}) + '\n')
        