
<div align="center">
  <a align="center"><img src="https://github.com/user-attachments/assets/cd86ded9-d3cb-46e2-8e62-280bbadbdbdc" width="100" height="100"></a>
  <h1 align="center"> <p>AxBench<sub> by <a href="https://github.com/stanfordnlp/pyvene">pyvene</a></sub></p></h1>
  <a href=""><strong>Read our paper »</strong></a></a>
</div>     

## What is AxBench?
It is a framework for evaluating existing interpretability methods at scale using synthetic data. We evaluate along two utility **ax**es: **concept detection** and **model steering**.

### Highlights of AxBench:
✅ 10+ existing interpretability methods are evaluated at scale, against finetuning or prompting methods.    
✅ 3 datasets for supervised dictionary learning (SDL), up to **16K concepts**.    
✅ 2 SoTA SDL models as drop-in replacements for SAEs.   
✅ 1 unified pipeline benchmarking interpretability methods.   
✅ 1 LLM-in-the-loop supervised dictionary learning pipeline that **costs less than a cent per concept**.   


### Access our SoTA dictionaries for `Gemma 2B` and `Gemma 9B`

Our SDL models are hosted on HuggingFace. Currently, we train two dictionaries for layer 20 residual stream of `Gemma 2B` and `Gemma 9B`. These dictionaries are trained with 16K concepts. These concepts are sampled from SAEs concept list provided by [neuronpedia.org](https://neuronpedia.org).

Huggingface page: [AxBench collections](https://huggingface.co/collections/pyvene/axbench-release-6787576a14657bb1fc7a5117)

Tutorial: [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github.com/stanfordnlp/axbench/blob/main/axbench/examples/tutorial.ipynb) [**How to use our dictionaries**]

## Other experiments
Building on top of our results, we did some early exploration on our trained dictionaries. Some of the results can be found under `axbench/examples`. Here is a table telling you more about the experiments:

| Experiment | Description |
| --- | --- |
| `axbench/examples/basics.ipynb` | Analyzing basic gemotry of learned dictionaries. |
| `axbench/examples/subspace_gazer.ipynb` | Using subspace gazer to visualize the learned subspaces. |
| `axbench/examples/lang>subspace.ipynb` | Finetuning a hyper-network to map natural language to subspaces or steering vectors. |
| `axbench/examples/platonic.ipynb` | Exploring the platonic representation hypothesis among subspaces. |


## How to AxBench your methods?
This section will guide you how to use AxBench to evaluate your methods.

### Installation
To install the latest stable version of axbench:
```
git clone git@github.com:stanfordnlp/axbench.git
cd axbench
```

Make sure you have your related API keys set:
```python
import os
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"
os.environ["NP_API_KEY"] = "your_neuronpedia_api_key_here"
```

Make sure you populate the `axbench/data` directory with the relevant datasets:
```bash
cd data
bash download-2b.sh
bash download-9b.sh
bash download-alpaca.sh
python axbench/scripts/download-seed-sentences.py
```

#### A simple demo
To run a complete demo with a single config file:
```bash
bash axbench/demo/demo.sh
```

### Data generation (If you use ours, you can skip this step)
Generate training dataset:
```bash
python axbench/scripts/generate.py --config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo
```

You can also generate inference dataset to avoid in-loop data generation:
```bash
python axbench/scripts/generate_latent.py --config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo
```

If you wish to generate your own dataset, you can do so by modifying the `simple.yaml`.


### Training your methods
Train and save your methods:
```bash
torchrun --nproc_per_node=$gpu_count axbench/scripts/train.py \
  --config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo
```
where `$gpu_count` is the number of GPUs available. You can find more yaml files we use in `axbench/sweep`. To run those yamls with customized directory, you can use commands like the following:
```bash
torchrun --nproc_per_node=$gpu_count axbench/scripts/train.py \
  --config axbench/sweep/wuzhengx/2b/l10/no_grad.yaml \
  --dump_dir axbench/results/prod_2b_l10_concept500_no_grad \
  --overwrite_data_dir axbench/concept500/prod_2b_l10_v1/generate
```
where `--dump_dir` is the directory to save your results, and `--overwrite_data_dir` is the directory of your training data.

### Inference
With your trained models, we can now run inference before evaluating the inference results.

#### Concept Detection
Inference with latent activations with representation abstractions:
```bash
torchrun --nproc_per_node=$gpu_count axbench/scripts/inference.py \
--config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo --mode latent
```
Like the training step, you can also run the following command to run your yaml file with customized directory:
```bash
torchrun --nproc_per_node=$gpu_count axbench/scripts/inference.py \
  --config axbench/sweep/wuzhengx/2b/l10/no_grad.yaml \
  --dump_dir axbench/results/prod_2b_l10_concept500_no_grad \
  --overwrite_metadata_dir axbench/concept500/prod_2b_l10_v1/generate \
  --overwrite_inference_data_dir axbench/concept500/prod_2b_l10_v1/inference \
  --mode latent
```

#### Model Steering
Inference with model steering with representation abstractions:
```bash
torchrun --nproc_per_node=$gpu_count axbench/scripts/inference.py \
--config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo --mode steering
```
Like the concept detection step, you can also run the following command to run your yaml file with customized directory:
```bash
torchrun --nproc_per_node=$gpu_count axbench/scripts/inference.py \
  --config axbench/sweep/wuzhengx/2b/l10/no_grad.yaml \
  --dump_dir axbench/results/prod_2b_l10_concept500_no_grad \
  --overwrite_metadata_dir axbench/concept500/prod_2b_l10_v1/generate \
  --overwrite_inference_data_dir axbench/concept500/prod_2b_l10_v1/inference \
  --mode steering
```

### Evaluation
This step evaluates the inference results. This step should not train or call any forward pass of trained models. It will call external APIs to evaluate the results.

#### Concept Detection
To evaluate inference results for latent activations:
```bash
python axbench/scripts/evaluate.py --config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo --mode latent
```
To enable `wandb` logging, you need to pass in additional arguments, `--report_to ["wandb"] --wandb_entity "<your wandb entity>"` Like the previous steps, you can also run your yaml file with customized directory:
```bash
python axbench/scripts/evaluate.py \
  --config axbench/sweep/wuzhengx/2b/l10/no_grad.yaml \
  --dump_dir axbench/results/prod_2b_l10_concept500_no_grad \
  --mode latent
```

#### Model Steering
To evaluate inference results for steering:
```bash
python axbench/scripts/evaluate.py --config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo --mode steering
```
Like the concept detection step, you can also run your yaml file with customized directory:
```bash
python axbench/scripts/evaluate.py \
  --config axbench/sweep/wuzhengx/2b/l10/no_grad.yaml \
  --dump_dir axbench/results/prod_2b_l10_concept500_no_grad \
  --mode steering
```

### Reproducing our results
To reproduce our results, please refer to `axbench/experiment_commands.sh`.
