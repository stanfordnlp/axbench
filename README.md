<div align="center">
  <a align="center"><img src="https://github.com/user-attachments/assets/661f78cf-4044-4c46-9a71-1316bb2c69a5" width="100" height="100" /></a>
  <h1 align="center">AxBench <sub>by <a href="https://github.com/stanfordnlp/pyvene">pyvene</a></sub></h1>
  <a href="#"><strong>Read our paper »</strong></a>
</div>     

<br>

**AxBench** is a a scalable benchmark that evaluates interpretability techniques on two axes: *concept detection* and *model steering*. This repo includes all benchmarking code, including data generation, training, evaluation, and analysis.

We introduced **supervised dictionary learning** (SDL) on synthetic data as an analogue to SAEs. You can access pretrained SDLs and our training/eval datasets here:

- 🤗 **Gradio Chat**: [**Steered LM Demo (ReFT-r1)**](https://huggingface.co/spaces/pyvene/AxBench-ReFT-r1-16K)  
- 🤗 **HuggingFace**: [AxBench collections](https://huggingface.co/collections/pyvene/axbench-release-6787576a14657bb1fc7a5117)
- **Tutorial**: [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/stanfordnlp/axbench/blob/main/axbench/examples/tutorial.ipynb) [**Using SDLs via [pyvene](https://github.com/stanfordnlp/pyvene)**]

## 🎯 Highlights

1. **Scalabale evaluation harness**: Framework for generating synthetic training + eval data from concept lists (e.g. GemmaScope SAE labels).
2. **Comprehensive implementations**: 10+ interpretability methods evaluated, along with finetuning and prompting baselines.
2. **16K concept training data**: Full-scale datasets for **supervised dictionary learning (SDL)**.  
3. **Two pretrained SDL models**: Drop-in replacements for standard SAEs.  
4. **LLM-in-the-loop training**: Generate your own datasets for less than \$0.01 per concept.


## Additional experiments

We include exploratory notebooks under `axbench/examples`, such as:

| Experiment                              | Description                                                                   |
|----------------------------------------|-------------------------------------------------------------------------------|
| `basics.ipynb`                         | Analyzes basic geometry of learned dictionaries.                              |
| `subspace_gazer.ipynb`                | Visualizes learned subspaces.                                                 |
| `lang>subspace.ipynb`                 | Fine-tunes a hyper-network to map natural language to subspaces or steering vectors. |
| `platonic.ipynb`                      | Explores the platonic representation hypothesis in subspace learning.         |

---

## Instructions for AxBenching your methods

### Installation

We highly suggest using `uv` for your Python virtual environment, but you can use any venv manager.

```bash
git clone git@github.com:stanfordnlp/axbench.git
cd axbench
uv sync # if using uv
```

Set up your API keys for OpenAI and Neuronpedia:

```python
import os
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"
os.environ["NP_API_KEY"] = "your_neuronpedia_api_key_here"
```

Download the necessary datasets to `axbench/data`:

```bash
uv run axbench/scripts/download-seed-sentences.py
cd data
bash download-2b.sh
bash download-9b.sh
bash download-alpaca.sh
```

### Try a simple demo.

To run a complete demo with a single config file:

```bash
bash axbench/demo/demo.sh
```

## Data generation

(If using our pre-generated data, you can skip this.)

**Generate training data:**

```bash
uv run axbench/scripts/generate.py --config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo
```

**Generate inference data:**

```bash
uv run axbench/scripts/generate_latent.py --config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo
```

To modify the data generation process, edit `simple.yaml`.

## Training

Train and save your methods:

```bash
uv run torchrun --nproc_per_node=$gpu_count axbench/scripts/train.py \
  --config axbench/demo/sweep/simple.yaml \
  --dump_dir axbench/demo
```

(Replace `$gpu_count` with the number of GPUs to use.)

For additional config:

```bash
torchrun --nproc_per_node=$gpu_count axbench/scripts/train.py \
  --config axbench/sweep/wuzhengx/2b/l10/no_grad.yaml \
  --dump_dir axbench/results/prod_2b_l10_concept500_no_grad \
  --overwrite_data_dir axbench/concept500/prod_2b_l10_v1/generate
```

where `--dump_dir` is the output directory, and `--overwrite_data_dir` is where the training data resides.

## Inference

### Concept detection

Run inference:

```bash
uv run torchrun --nproc_per_node=$gpu_count axbench/scripts/inference.py \
  --config axbench/demo/sweep/simple.yaml \
  --dump_dir axbench/demo \
  --mode latent
```

For additional config using custom directories:

```bash
uv run torchrun --nproc_per_node=$gpu_count axbench/scripts/inference.py \
  --config axbench/sweep/wuzhengx/2b/l10/no_grad.yaml \
  --dump_dir axbench/results/prod_2b_l10_concept500_no_grad \
  --overwrite_metadata_dir axbench/concept500/prod_2b_l10_v1/generate \
  --overwrite_inference_data_dir axbench/concept500/prod_2b_l10_v1/inference \
  --mode latent
```

#### Imbalanced concept detection

For real-world scenarios with fewer than 1% positive examples, we upsample negatives (100:1) and re-evaluate. Use:

```bash
uv run torchrun --nproc_per_node=$gpu_count axbench/scripts/inference.py \
  --config axbench/sweep/wuzhengx/2b/l10/no_grad.yaml \
  --dump_dir axbench/results/prod_2b_l10_concept500_no_grad \
  --overwrite_metadata_dir axbench/concept500/prod_2b_l10_v1/generate \
  --overwrite_inference_data_dir axbench/concept500/prod_2b_l10_v1/inference \
  --mode latent_imbalance
```

### Model steering

For steering experiments:

```bash
uv run torchrun --nproc_per_node=$gpu_count axbench/scripts/inference.py \
  --config axbench/demo/sweep/simple.yaml \
  --dump_dir axbench/demo \
  --mode steering
```

Or a custom run:

```bash
uv run torchrun --nproc_per_node=$gpu_count axbench/scripts/inference.py \
  --config axbench/sweep/wuzhengx/2b/l10/no_grad.yaml \
  --dump_dir axbench/results/prod_2b_l10_concept500_no_grad \
  --overwrite_metadata_dir axbench/concept500/prod_2b_l10_v1/generate \
  --overwrite_inference_data_dir axbench/concept500/prod_2b_l10_v1/inference \
  --mode steering
```

## Evaluation

### Concept detection

To evaluate concept detection results:

```bash
uv run axbench/scripts/evaluate.py \
  --config axbench/demo/sweep/simple.yaml \
  --dump_dir axbench/demo \
  --mode latent
```

Enable wandb logging:

```bash
uv run axbench/scripts/evaluate.py \
  --config axbench/demo/sweep/simple.yaml \
  --dump_dir axbench/demo \
  --mode latent \
  --report_to wandb \
  --wandb_entity "your_wandb_entity"
```

Or evaluate using your custom config:

```bash
uv run axbench/scripts/evaluate.py \
  --config axbench/sweep/wuzhengx/2b/l10/no_grad.yaml \
  --dump_dir axbench/results/prod_2b_l10_concept500_no_grad \
  --mode latent
```

### Model steering.

To evaluate steering:

```bash
uv run axbench/scripts/evaluate.py \
  --config axbench/demo/sweep/simple.yaml \
  --dump_dir axbench/demo \
  --mode steering
```

Or a custom config:

```bash
uv run axbench/scripts/evaluate.py \
  --config axbench/sweep/wuzhengx/2b/l10/no_grad.yaml \
  --dump_dir axbench/results/prod_2b_l10_concept500_no_grad \
  --mode steering
```

## Reproducing our results.

Please see `axbench/experiment_commands.txt` for detailed commands and configurations.
