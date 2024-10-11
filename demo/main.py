import sys
sys.path.append("../../pyreax")

import asyncio
import os, random, json, time, requests, copy
import torch, transformers, datasets

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union, List, Any
from pathlib import Path
import pandas as pd

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)

from pyreax import (
    LanguageModel,
    get_concept_genres,
    get_contrast_concepts
)

import numpy as np
from huggingface_hub import hf_hub_download


lm = LanguageModel("gpt-4o", "./tmp")

start = time.time()
logger.warning("Prepare genre and contrast concepts.")

async def run_tasks(tasks):
    # Gather and run all provided tasks concurrently, and collect their results
    results = await asyncio.gather(*tasks)
    return results

genre_task = get_concept_genres(
    lm, ["terms associated with Stanford", "terms associated with UC Berkeley"]
)
contrast_task = get_contrast_concepts(
    lm, ["terms associated with Stanford", "terms associated with UC Berkeley"])
task_queue = {genre_task, contrast_task}

results = asyncio.run(run_tasks(task_queue))

print(results)

end = time.time()
elapsed = round(end - start, 3)
logger.warning(
    f"Finished preparing contrast concepts in {elapsed} sec."
)