#!/bin/bash

# Check if nvidia-smi command exists
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi could not be found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

# Get the number of GPUs
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

python axbench/scripts/generate.py --config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo

torchrun --nproc_per_node=$gpu_count axbench/scripts/train.py \
  --config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo

python axbench/scripts/inference.py --config axbench/demo/sweep/simple.yaml --mode latent --dump_dir axbench/demo

python axbench/scripts/inference.py --config axbench/demo/sweep/simple.yaml --mode steering --dump_dir axbench/demo

python axbench/scripts/evaluate.py --config axbench/demo/sweep/simple.yaml --mode latent --dump_dir axbench/demo

python axbench/scripts/evaluate.py --config axbench/demo/sweep/simple.yaml --mode steering --dump_dir axbench/demo
