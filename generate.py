try:
    # This library is our indicator that the required installs
    # need to be done.
    import pyreax

except ModuleNotFoundError:
    # relative import; better to pip install subctrl
    import sys
    sys.path.append("../pyreax")
    import pyreax

import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch, pyreft
from pyvene import (
    IntervenableModel,
    ConstantSourceIntervention,
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
)

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_scheduler

from pyreax import (
    EXAMPLE_TAG, 
    ReAXFactory, 
    MaxReLUIntervention, 
    SubspaceAdditionIntervention, 
    make_data_module, 
    save_reax,
    Config,
    load_config_from_json,
    load_concepts,
)
from pyreax import (
    set_decoder_norm_to_unit_norm, 
    remove_gradient_parallel_to_decoder_directions,
    gather_residual_activations, 
    get_lr
)

