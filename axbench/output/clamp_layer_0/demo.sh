#!/bin/bash

# Check if nvidia-smi command exists
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi could not be found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

# Get the number of GPUs
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

#python axbench/scripts/generate.py --config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo

#torchrun --master_port=12345 --nproc_per_node=$gpu_count axbench/scripts/train.py \
#  --config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo

echo "latent..."
#torchrun --nproc_per_node=1 --master_port=29511 axbench/scripts/inference.py \
#  --config axbench/demo/sweep/latent.yaml \
#  --dump_dir axbench/demo \
#  --mode latent \
#  --run_name official

echo "steering..."
torchrun --master_port=12345 --nproc_per_node=$gpu_count axbench/scripts/inference.py --config axbench/demo/sweep/simple.yaml --mode steering --dump_dir axbench/demo --run_name official

echo "evaluate..."
python axbench/scripts/evaluate.py --config axbench/demo/sweep/simple.yaml --mode steering --dump_dir axbench/demo --run_name official
