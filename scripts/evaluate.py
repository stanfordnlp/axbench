# score evaluation results.
#
# example launch command:
#     python scripts/evaluate.py --config demo/sweep/evaluate.yaml --mode latent


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
from pathlib import Path

from pyreax import (
    F1Evaluator,
    EvalArgs
)

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


def data_generator(data_dir):
    """
    Generator function to read data files and yield data subsets by group_id.

    Args:
        data_dir (str): Path to the data directory.

    Yields:
        (group_id, df_subset): A tuple containing the group_id and subset DataFrame.
    """
    # Get list of files sorted by index
    file_list = sorted(glob.glob(os.path.join(data_dir, 'latent_data_fragment_*.csv')))
    for file_path in file_list:
        df = pd.read_csv(file_path)
        reax_ids = df['reax_id'].unique()
        reax_ids.sort()
        for reax_id in reax_ids:
            df_subset = df[df['reax_id'] == reax_id]
            yield (reax_id, df_subset)


def eval_steering(args):

    raise NotImplementedError("Steering inference is not implemented yet.")


def eval_latent(args):

    data_dir = args.data_dir
    dump_dir = args.dump_dir

    df_generator = data_generator(args.data_dir)

    for (reax_id, group_df) in df_generator:
        print(reax_id)
        print(group_df)
    

def main():
    custom_args = [
        {
            'args': ['--mode'],
            'kwargs': {
                'type': str,
                'default': "latent",
                'help': 'The evaluation mode.'
            }
        }
    ]
    args = EvalArgs(custom_args=custom_args)
    logger.warning("Evaluating generations with the following configuration:")
    logger.warning(args)
    
    if args.mode == "latent":
        eval_latent(args)
    elif args.mode == "steering":
        eval_steering(args)


if __name__ == "__main__":
    main()

