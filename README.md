<div align="center">
  <a align="center"><img src="https://github.com/user-attachments/assets/661f78cf-4044-4c46-9a71-1316bb2c69a5" width="100" height="100" /></a>
  <h1 align="center">AxBench <sub>by <a href="https://github.com/stanfordnlp/pyvene">pyvene</a></sub></h1>
  <a href="https://arxiv.org/abs/2501.17148"><strong>Read our paper »</strong></a>
</div>     

<br>

**AxBench** is a a scalable benchmark that evaluates interpretability techniques on two axes: *concept detection* and *model steering*.

- 🤗 **HuggingFace**: [**AxBench Collections**](https://huggingface.co/collections/pyvene/axbench-release-6787576a14657bb1fc7a5117)  
- [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/stanfordnlp/axbench/blob/main/axbench/examples/tutorial.ipynb) **Tutorial of using our dictionary via [pyvene](https://github.com/stanfordnlp/pyvene)**


## Related papers

- [*HyperSteer*: Activation Steering at Scale with Hypernetworks [preprint]](https://www.arxiv.org/abs/2506.03292)
- [Improved Representation Steering for Language Models [preprint]](https://arxiv.org/pdf/2505.20809).
- [SAEs Are Good for Steering -- If You Select the Right Features [preprint]](https://arxiv.org/abs/2505.20063).
- [AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders [ICML 2025 (spotlight)]](https://arxiv.org/abs/2501.17148).


## 🏆 Rank-1 steering leaderboard

📢 Please open a PR to enter the leaderboard.

| Method                       | 2B L10 | 2B L20 | 9B L20 | 9B L31 |  Avg |
|------------------------------|-------:|-------:|-------:|-------:|-----:|
| HyperSteer [[Sun et al., 2025]](https://github.com/stanfordnlp/axbench)                   | - | **0.742** | **1.091** | - | **0.917** |
| Prompt                       | 0.698 | 0.731 | 1.075 | **1.072** | 0.894 |
| RePS [[Wu et. al., 2025]](https://arxiv.org/pdf/2505.20809)           | **0.756** | 0.606 | 0.892 | 0.624 | 0.720 |
| ReFT-r1                      | 0.633 | 0.509 | 0.630 | 0.401 | 0.543 |
| SAE (filtered) [[Arad et. al., 2025]](https://arxiv.org/abs/2505.20063) | - | - | 0.546 | 0.470 | 0.508 |
| DiffMean                     | 0.297 | 0.178 | 0.322 | 0.158 | 0.239 |
| SAE                          | 0.177 | 0.151 | 0.191 | 0.140 | 0.165 |
| SAE-A                        | 0.166 | 0.132 | 0.186 | 0.143 | 0.157 |
| LAT                          | 0.117 | 0.130 | 0.127 | 0.134 | 0.127 |
| PCA                          | 0.107 | 0.083 | 0.128 | 0.104 | 0.105 |
| Probe                        | 0.095 | 0.091 | 0.108 | 0.099 | 0.098 |

### Beyond rank-1

Steering methods whose intervention is not restricted to a single direction.

| Method                       | 2B L10 | 2B L20 | 9B L20 | 9B L31 |  Avg |
|------------------------------|-------:|-------:|-------:|-------:|-----:|
| FLAS [[Jin et al., 2026]](https://flas-ai.github.io) | - | 1.025 | 1.185 | - | 1.105 |


## Highlights

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
uv run axbench/data/download-seed-sentences.py
cd axbench/data
bash download-2b.sh
bash download-9b.sh
bash download-alpaca.sh
```

### Try a simple demo.

To run a complete demo with a single config file:

```bash
bash axbench/demo/demo.sh
```

To run a complete demo for *HyperSteer*

```bash
bash axbench/demo/hypersteer_demo.sh
```

## Data generation

(If using our pre-generated data, you can skip this.)

**Generate training data:**

```bash
uv run axbench/scripts/generate.py --config axbench/demo/sweep/simple.yaml --mode training --dump_dir axbench/demo
```

**Generate inference data:**

```bash
uv run axbench/scripts/generate.py --config axbench/demo/sweep/simple.yaml --mode latent --dump_dir axbench/demo
```

**Generate preference-based training data:**

```bash
uv run axbench/scripts/generate.py --config axbench/demo/sweep/simple.yaml \
  --mode dpo_training --dump_dir axbench/demo \
  --model_name google/gemma-2-2b-it \
  --inference_batch_size 64
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

where `--dump_dir` is the output directory, and `--overwrite_data_dir` is where the training data resides. You might overwrite other parameters as `--layer 10` for customized tuning.




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

### Model steering on evaluation set

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

### Model steering on test set
Note that the commend above is for evaluation. We select the best factor by using the results on the evaluation set. After that you will do the evaluation on the test set.

```bash
uv run axbench/scripts/evaluate.py \
  --config axbench/sweep/wuzhengx/2b/l10/no_grad.yaml \
  --dump_dir axbench/results/prod_2b_l10_concept500_no_grad \
  --mode steering_test
```

## Analyses
Once you finished evaluation, you can do the analyses with our provided notebook in `axbench/scripts/analyses.ipynb`. All of our results in the paper are produced by this notebook.

You need to point revelant directories to your own results by modifying the notebook. If you introduce new models, datasets, or new evaluation metrics, you can add your own analysis by following the notebook.

## Reproducing our results.

Please see `axbench/experiment_commands.txt` for detailed commands and configurations.


## Feature suppression experiments
In our recent paper release, we introduce feature suppresion evaluations. Please see `axbench/sweep/wuzhengx/reps/README.md` for details.



