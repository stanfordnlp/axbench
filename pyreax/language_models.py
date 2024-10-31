from .utils.constants import *

import httpx, asyncio
import os, uuid, string, json
from pathlib import Path

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)

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

    def record(self, api_name, stats, prompt=None, completion=None):
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
            self.prompt_cache[api_name].append({
                "prompt": prompt,
                "completion": completion
            })
    
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
    """Main class abstract async remote language model access"""

    def __init__(self, model, client, dump_dir=None, **kwargs):
        self.model = model
        if "gpt-4o" in model:
            pass
        else:
            raise ValueError(f"{model} model class is not supported yet.")
        self.stats = LanguageModelStats(model)
        self.client = client
        # dump dir
        if dump_dir:
            cur_save_dir = Path(dump_dir) / "lm_cache"
            cur_save_dir.mkdir(parents=True, exist_ok=True)
            self.dump_dir = cur_save_dir
    
    def normalize(self, text):
        return text.strip()

    async def chat_completion(self, client, prompt):
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model=self.model)
        return response
        
    async def chat_completions(self, api_names, prompts, batch_size=32):
        """handling batched async calls with internal batching mechanism"""
        # Ensure api_names is a list of appropriate length
        if not isinstance(api_names, list):
            api_names = [api_names] * len(prompts)

        # Process in batches
        all_completions = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_api_names = api_names[i:i + batch_size]
            print(f"Processing batch {i} of {api_names[0]} with {len(batch_prompts)} prompts")

            # batched calls
            async_responses = [
                self.chat_completion(self.client, prompt) for prompt in batch_prompts]
            raw_completions = await asyncio.gather(*async_responses)

            # post handling for current batch
            for j, raw_completion in enumerate(raw_completions):
                raw_completion = raw_completion.to_dict()
                completion = self.normalize(raw_completion["choices"][0]["message"]["content"])
                all_completions.append(completion)
                self.stats.record(
                    batch_api_names[j], raw_completion['usage'],
                    prompt=batch_prompts[j], completion=completion)
            
        return all_completions

    def dump(self):
        with open(self.dump_dir / "tmp_prompt_cache.json", "w") as outfile:
            json.dump(self.stats.prompt_cache, outfile, indent=4)
        
        with open(self.dump_dir / "cost.jsonl", 'a') as f:
            f.write(json.dumps({"price": self.stats.get_total_price()}) + '\n')

    async def close(self):
        """Close the underlying HTTP client"""
        await self.client.close()

