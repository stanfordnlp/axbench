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

## evaluate

#### latent
Evaluate latent activations with representation abstractions:
```bash
python scripts/evaluate.py --config demo/sweep/evaluate.yaml --mode latent
```

#### steering (not implemented yet)
Evaluate model steering with representation abstractions:
```bash
python scripts/evaluate.py --config demo/sweep/evaluate.yaml --mode steer
```

## score

#### latent (not implemented yet)
To score evaluation results for latent activations:
```bash
python scripts/score.py --config demo/sweep/score.yaml --mode latent
```

#### steering (not implemented yet)
To score evaluation results for steering:
```bash
python scripts/score.py --config demo/sweep/score.yaml --mode steer
```
