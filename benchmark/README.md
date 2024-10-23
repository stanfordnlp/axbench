## Benchmarking ReAX

### Installation
To install the latest stable version of pyreax:
```
git clone git@github.com:frankaging/pyreax.git
cd pyreax
```

### generate
Generate training dataset for representation abstractions:
```bash
python benchmark/scripts/generate.py --config benchmark/demo/sweep/generate.yaml
```

### train
Train and save representation abstractions with generated datasets:
```bash
python benchmark/scripts/train.py --config benchmark/demo/sweep/train.yaml
```

### inference

#### latent
Inference with latent activations with representation abstractions:
```bash
python benchmark/scripts/inference.py --config benchmark/demo/sweep/inference.yaml --mode latent
```

#### steering (not implemented yet)
Inference with model steering with representation abstractions:
```bash
python benchmark/scripts/inference.py --config benchmark/demo/sweep/inference.yaml --mode steer
```

### evaluate

#### latent
To evaluate inference results for latent activations:
```bash
python benchmark/scripts/evaluate.py --config benchmark/demo/sweep/evaluate.yaml --mode latent
```

#### steering (not implemented yet)
To evaluate inference results for steering:
```bash
python benchmark/scripts/evaluate.py --config benchmark/demo/sweep/evaluate.yaml --mode steer
```

### (optional) plot

We provide a plot notebook (`benchmark/scripts/plots.ipynb`) for generating various figures.