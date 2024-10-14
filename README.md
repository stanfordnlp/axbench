# pyreax

### generate
Generate training dataset for subspaces:
```bash
python generate.py --dump_dir demo --concept_path demo/concepts.csv --num_of_examples 72
```

### train
Training and saving subspaces with generated datasets:
```bash
python train.py --data_dir demo/generate --dump_dir demo --config demo/sweep/train.yaml
```

### evaluation