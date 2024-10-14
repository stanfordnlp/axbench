# evaluate existing subspaces.
#
# example launch command:
#     python evaluate.py --data_dir demo/generate --train_dir demo/train --dump_dir demo --num_of_examples 50

try:
    # This library is our indicator that the required installs
    # need to be done.
    import pyreax

except ModuleNotFoundError:
    # relative import; better to pip install subctrl
    import sys
    sys.path.append("../../pyreax")
    import pyreax


import os, argparse, yaml, json, glob, pickle
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

from pyvene import IntervenableModel
from pyreax import (
    EXAMPLE_TAG, 
    ReAXFactory, 
    MaxReLUIntervention, 
)
from pyreax import (
    gather_residual_activations, 
)

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)

CONFIG_FILE = "config.json"
DEFAULT_ROTATION_FREQ = 1000


def main(data_dir, train_dir, dump_dir, num_of_examples, rotation_freq):
    pass


if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Generate script for creating the training dataset for concepts.")
    
    # Define the arguments
    parser.add_argument("--data_dir", type=str, help="Path to the data directory")
    parser.add_argument("--train_dir", type=str, help="Path to the train directory")
    parser.add_argument("--dump_dir", type=str, help="Path to the dump directory")
    parser.add_argument("--num_of_examples", type=int, help="The number of examples for each evaluation subset")
    parser.add_argument("--rotation_freq", type=int, help="Frequency for chunking files", default=DEFAULT_ROTATION_FREQ)
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Run the main function with parsed arguments
    main(
        args.data_dir,
        args.train_dir,
        args.dump_dir,
        args.num_of_examples,
        args.rotation_freq,
    )