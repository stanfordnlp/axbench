#!/bin/bash

# Check if nvidia-smi command exists
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi could not be found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

# Get the number of GPUs
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

python axbench/scripts/generate.py --config axbench/demo/sweep/diffmean.yaml --dump_dir axbench/output_llama/layer_10/

echo "train..."
torchrun --master_port=12345 --nproc_per_node=$gpu_count axbench/scripts/train.py \
  --config axbench/demo/sweep/diffmean.yaml \
  --dump_dir axbench/output_llama/layer_10/diffmean_10 \
  --overwrite_data_dir axbench/output_llama/layer_10/generate

echo "steering..."
torchrun --master_port=12345 --nproc_per_node=$gpu_count axbench/scripts/inference.py \
  --config axbench/demo/sweep/diffmean.yaml \
  --mode steering \
  --dump_dir axbench/output_llama/layer_10/diffmean_10 \
  --overwrite_metadata_dir axbench/output_llama/layer_10/generate \
  --overwrite_inference_data_dir axbench/output_llama/layer_10


echo "evaluate..."
python axbench/scripts/evaluate.py --config axbench/demo/sweep/diffmean.yaml \
  --mode steering \
  --dump_dir axbench/output_llama/layer_10/diffmean_10 \

