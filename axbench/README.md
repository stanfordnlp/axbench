## Abstraction Benchmark (AxBench)

### Installation
To install the latest stable version of pyreax:
```
git clone git@github.com:frankaging/pyreax.git
cd pyreax
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
python axbench/scripts/train.py --config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo
```

### inference

#### latent
Inference with latent activations with representation abstractions:
```bash
python axbench/scripts/inference.py --config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo --mode latent
```

#### steering
Inference with model steering with representation abstractions:
```bash
python axbench/scripts/inference.py --config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo --mode steering
```

### evaluate

#### latent
To evaluate inference results for latent activations:
```bash
python axbench/scripts/evaluate.py --config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo --mode latent
```

#### steering
To evaluate inference results for steering:
```bash
python axbench/scripts/evaluate.py --config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo --mode steering
```