## Abstraction Benchmark (AxBench)

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

### demo
To run a complete demo with a single config file:
```bash
bash axbench/demo/demo.sh
```

### generate
Generate training dataset for representation abstractions:
```bash
python axbench/scripts/generate.py --config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo
```

### train
Train and save representation abstractions with generated datasets:
```bash
torchrun --nproc_per_node=$gpu_count axbench/scripts/train.py \
  --config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo
```
where `$gpu_count` is the number of GPUs available.

### inference

#### latent
Inference with latent activations with representation abstractions:
```bash
torchrun --nproc_per_node=$gpu_count axbench/scripts/inference.py \
--config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo --mode latent
```

#### steering
Inference with model steering with representation abstractions:
```bash
torchrun --nproc_per_node=$gpu_count axbench/scripts/inference.py \
--config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo --mode steering
```

### evaluate

#### latent
To evaluate inference results for latent activations:
```bash
python axbench/scripts/evaluate.py --config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo --mode latent
```
To enable `wandb` logging, you need to pass in additional arguments, `--report_to ["wandb"] --wandb_entity "<your wandb entity>"`

#### steering
To evaluate inference results for steering:
```bash
python axbench/scripts/evaluate.py --config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo --mode steering
```
To enable `wandb` logging, you need to pass in additional arguments, `--report_to ["wandb"] --wandb_entity "<your wandb entity>"`