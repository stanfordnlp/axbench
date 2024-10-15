<h1 align="center"> <p>pyreax<sub> by <a href="https://github.com/stanfordnlp/pyvene">pyvene</a></sub></p></h1>
<h3 align="center">
    <p>A library for creating your representation abstractions (ReAXs) in seconds.</p>
    <a href=""><strong>Read our paper »</strong></a></a>
</h3>

## generate
Generate training dataset for representation abstractions:
```bash
python scripts/generate.py --config demo/sweep/generate.yaml
```

## train
Train and save representation abstractions with generated datasets:
```bash
python scripts/train.py --config demo/sweep/train.yaml
```

## inference

#### latent
Inference with latent activations with representation abstractions:
```bash
python scripts/inference.py --config demo/sweep/inference.yaml --mode latent
```

#### steering (not implemented yet)
Inference with model steering with representation abstractions:
```bash
python scripts/inference.py --config demo/sweep/inference.yaml --mode steer
```

## evaluate

#### latent (not implemented yet)
To evaluate inference results for latent activations:
```bash
python scripts/evaluate.py --mode latent
```

#### steering (not implemented yet)
To evaluate inference results for steering:
```bash
python scripts/evaluate.py --mode steer
```
