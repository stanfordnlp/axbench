# pyreax

### generate
Generate training dataset for subspaces:
```bash
python generate.py --config demo/sweep/generate.yaml
```

### train
Train and save subspaces with generated datasets:
```bash
python train.py --config demo/sweep/train.yaml
```

### evaluate

#### latent
Evaluate latent activations with subspaces:
```bash
python evaluate.py --config demo/sweep/evaluate.yaml --mode latent
```

#### steering (not implemented yet)
Evaluate model steering ith subspaces:
```bash
python evaluate.py --config demo/sweep/evaluate.yaml --mode steer
```

### score
To score evaluation results:
```bash
```