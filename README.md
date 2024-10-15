# pyreax

## generate
Generate training dataset for subspaces:
```bash
python scripts/generate.py --config demo/sweep/generate.yaml
```

## train
Train and save subspaces with generated datasets:
```bash
python scripts/train.py --config demo/sweep/train.yaml
```

## evaluate

#### latent
Evaluate latent activations with subspaces:
```bash
python scripts/evaluate.py --config demo/sweep/evaluate.yaml --mode latent
```

#### steering (not implemented yet)
Evaluate model steering ith subspaces:
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
