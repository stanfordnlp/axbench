<div align="center">
  <a align="center"><img src="https://github.com/user-attachments/assets/661f78cf-4044-4c46-9a71-1316bb2c69a5" width="100" height="100" /></a>
  <h1 align="center">AxBench <sub>by <a href="https://github.com/stanfordnlp/pyvene">pyvene</a></sub></h1>
  <a href="#"><strong>Read our paper »</strong></a>
</div>     

---

## Accessing the SoTA dictionary ReFT-r1 and training data on 16K concepts.

- 🤗 **HuggingFace**: [**AxBench Collections**](https://huggingface.co/collections/pyvene/axbench-release-6787576a14657bb1fc7a5117)  
- 🤗 **ReFT-r1 Live Demo**: [**Steering ChatLM**](https://huggingface.co/spaces/pyvene/AxBench-ReFT-r1-16K)
- 🤗 **ReFT-cr1 Live Demo**: [**Conditional Steering ChatLM**](https://huggingface.co/spaces/pyvene/AxBench-ReFT-cr1-16K)
- 📚 **Feature Visualizer**: [**Visualize LM Activations**](https://nlp.stanford.edu/~wuzhengx/axbench/index.html)
- 🔍 **Subspace Gazer**: [**Visualize Subspaces via UMAP**](https://nlp.stanford.edu/~wuzhengx/axbench/visualization_UMAP.html)
- **Tutorial**: [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/stanfordnlp/axbench/blob/main/axbench/examples/tutorial.ipynb) [**Using our dictionary via [pyvene](https://github.com/stanfordnlp/pyvene)**]

<img src="https://github.com/user-attachments/assets/d885a8e0-f893-4231-9a43-7d2dcf621507" alt="MyMovie4" width="500" />

---

## 🎯 Highlights.

1. **Large-Scale Evaluation**: 10+ interpretability methods evaluated via fine-tuning and prompting.  
2. **16K Concept Training Data**: For **Supervised Dictionary Learning (SDL)**.  
3. **Two SDL Models**: Drop-in replacements for standard SAEs.  
4. **LLM-in-the-Loop Training**: Build dictionaries at under \$0.01 per concept.

---

## Additional experiments.

We include exploratory notebooks under `axbench/examples`, such as:

| Experiment                              | Description                                                                   |
|----------------------------------------|-------------------------------------------------------------------------------|
| `basics.ipynb`                         | Analyzes basic geometry of learned dictionaries.                              |
| `subspace_gazer.ipynb`                | Visualizes learned subspaces.                                                 |
| `lang>subspace.ipynb`                 | Fine-tunes a hyper-network to map natural language to subspaces or steering vectors. |
| `platonic.ipynb`                      | Explores the platonic representation hypothesis in subspace learning.         |

---

## How to "AxBench" your methods.

### Installation.

```bash
git clone git@github.com:stanfordnlp/axbench.git
cd axbench
```

Set your API keys:

```python
import os
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"
os.environ["NP_API_KEY"] = "your_neuronpedia_api_key_here"
```

Download the necessary datasets to `axbench/data`:

```bash
cd data
bash download-2b.sh
bash download-9b.sh
bash download-alpaca.sh
python axbench/scripts/download-seed-sentences.py
```

### A simple demo.

To run a complete demo with a single config file:

```bash
bash axbench/demo/demo.sh
```

---

## Data generation.

(If using our pre-generated data, you can skip this.)

**Generate training data:**

```bash
python axbench/scripts/generate.py --config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo
```

**Generate inference data:**

```bash
python axbench/scripts/generate_latent.py --config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo
```

To modify the data generation process, edit `simple.yaml`.

---

## Training.

Train and save your methods:

```bash
torchrun --nproc_per_node=$gpu_count axbench/scripts/train.py \
  --config axbench/demo/sweep/simple.yaml \
  --dump_dir axbench/demo
```

(Replace `$gpu_count` with the number of GPUs to use.)

For additional configs:

```bash
torchrun --nproc_per_node=$gpu_count axbench/scripts/train.py \
  --config axbench/sweep/wuzhengx/2b/l10/no_grad.yaml \
  --dump_dir axbench/results/prod_2b_l10_concept500_no_grad \
  --overwrite_data_dir axbench/concept500/prod_2b_l10_v1/generate
```

where `--dump_dir` is the output directory, and `--overwrite_data_dir` is where the training data resides.

---

## Inference.

### Concept detection.

Infer with latent activations:

```bash
torchrun --nproc_per_node=$gpu_count axbench/scripts/inference.py \
  --config axbench/demo/sweep/simple.yaml \
  --dump_dir axbench/demo \
  --mode latent
```

Using custom directories:

```bash
torchrun --nproc_per_node=$gpu_count axbench/scripts/inference.py \
  --config axbench/sweep/wuzhengx/2b/l10/no_grad.yaml \
  --dump_dir axbench/results/prod_2b_l10_concept500_no_grad \
  --overwrite_metadata_dir axbench/concept500/prod_2b_l10_v1/generate \
  --overwrite_inference_data_dir axbench/concept500/prod_2b_l10_v1/inference \
  --mode latent
```

#### Imbalanced concept detection.

For real-world scenarios with fewer than 1% positive examples, we upsample negatives (100:1) and re-evaluate. Use:

```bash
torchrun --nproc_per_node=$gpu_count axbench/scripts/inference.py \
  --config axbench/sweep/wuzhengx/2b/l10/no_grad.yaml \
  --dump_dir axbench/results/prod_2b_l10_concept500_no_grad \
  --overwrite_metadata_dir axbench/concept500/prod_2b_l10_v1/generate \
  --overwrite_inference_data_dir axbench/concept500/prod_2b_l10_v1/inference \
  --mode latent_imbalance
```

### Model steering.

For steering experiments:

```bash
torchrun --nproc_per_node=$gpu_count axbench/scripts/inference.py \
  --config axbench/demo/sweep/simple.yaml \
  --dump_dir axbench/demo \
  --mode steering
```

Or a custom run:

```bash
torchrun --nproc_per_node=$gpu_count axbench/scripts/inference.py \
  --config axbench/sweep/wuzhengx/2b/l10/no_grad.yaml \
  --dump_dir axbench/results/prod_2b_l10_concept500_no_grad \
  --overwrite_metadata_dir axbench/concept500/prod_2b_l10_v1/generate \
  --overwrite_inference_data_dir axbench/concept500/prod_2b_l10_v1/inference \
  --mode steering
```

---

## Evaluation.

### Concept detection.

To evaluate concept detection results:

```bash
python axbench/scripts/evaluate.py \
  --config axbench/demo/sweep/simple.yaml \
  --dump_dir axbench/demo \
  --mode latent
```

Enable wandb logging:

```bash
python axbench/scripts/evaluate.py \
  --config axbench/demo/sweep/simple.yaml \
  --dump_dir axbench/demo \
  --mode latent \
  --report_to wandb \
  --wandb_entity "your_wandb_entity"
```

Or evaluate using your custom config:

```bash
python axbench/scripts/evaluate.py \
  --config axbench/sweep/wuzhengx/2b/l10/no_grad.yaml \
  --dump_dir axbench/results/prod_2b_l10_concept500_no_grad \
  --mode latent
```

### Model steering.

To evaluate steering:

```bash
python axbench/scripts/evaluate.py \
  --config axbench/demo/sweep/simple.yaml \
  --dump_dir axbench/demo \
  --mode steering
```

Or a custom config:

```bash
python axbench/scripts/evaluate.py \
  --config axbench/sweep/wuzhengx/2b/l10/no_grad.yaml \
  --dump_dir axbench/results/prod_2b_l10_concept500_no_grad \
  --mode steering
```

---

## Reproducing our results.

Please see `axbench/experiment_commands.txt` for detailed commands and configurations.
