#!/bin/bash

# Check if nvidia-smi command exists
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi could not be found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

# Get the number of GPUs
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

python axbench/scripts/generate.py --config axbench/demo/sweep/sae.yaml --dump_dir axbench/output_llama/layer_20_concise

echo "train..."
torchrun --master_port=12345 --nproc_per_node=$gpu_count axbench/scripts/train.py \
  --config axbench/demo/sweep/sae.yaml \
  --dump_dir axbench/output_llama/layer_20_concise/sae_20 \
  --overwrite_data_dir axbench/output_llama/layer_20_concise/generate

echo "inference..."
torchrun --nproc_per_node=1 --master_port=11112 axbench/scripts/inference.py \
  --config axbench/demo/sweep/sae.yaml \
  --dump_dir axbench/output_llama/layer_20_concise/sae_20 \
  --mode latent \
  --overwrite_metadata_dir axbench/output_llama/layer_20_concise/generate \
  --inference_dir axbench/output_llama/layer_20_concise/lsreft_20__0.005_0.3


echo "steering..."
torchrun --master_port=11111 --nproc_per_node=$gpu_count axbench/scripts/inference.py \
  --config axbench/demo/sweep/sae.yaml \
  --mode steering \
  --dump_dir axbench/output_llama/layer_20_concise/sae_20 \
  --overwrite_metadata_dir axbench/output_llama/layer_20_concise/generate \
  --inference_dir axbench/output_llama/layer_20_concise/lsreft_20__0.005_0.3 
  #--overwrite_inference_data_dir axbench/output_llama/layer_20_concise \

echo "evaluate..."
python axbench/scripts/evaluate.py --config axbench/demo/sweep/sae.yaml \
  --mode steering \
  --dump_dir axbench/output_llama/layer_20_concise/sae_20 \

