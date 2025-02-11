#!/bin/bash

# Check if nvidia-smi command exists
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi could not be found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

# Get the number of GPUs
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

python axbench/scripts/generate.py --config axbench/demo/sweep/lora.yaml \
--dump_dir axbench/output_llama/layer_20_concise/
#
echo "train..."
torchrun --master_port=12046 --nproc_per_node=1 axbench/scripts/train.py \
  --config axbench/demo/sweep/lora.yaml \
  --dump_dir axbench/output_llama/layer_20_concise/lora_20\
  --overwrite_data_dir axbench/output_llama/layer_20_concise/generate \
  --steer_suppress_dict_path axbench/output_llama/layer_20_concise/concept_suppress_dict.json

echo "inference..."
torchrun --nproc_per_node=1 --master_port=29511 axbench/scripts/inference.py \
  --config axbench/demo/sweep/lora.yaml \
  --dump_dir axbench/output_llama/layer_20_concise/lora_20\
  --mode latent \
  --overwrite_metadata_dir axbench/output_llama/layer_20_concise/generate \
  #--inference_dir axbench/output_llama/layer_20_concise/lsreft_20__0.005_0.3 
  #--inference_dir axbench/output_llama/layer_20

echo "steering..."
torchrun --master_port=12046 --nproc_per_node=$gpu_count axbench/scripts/inference.py \
  --config axbench/demo/sweep/lora.yaml \
  --mode steering \
  --dump_dir axbench/output_llama/layer_20_concise/lora_20\
  --overwrite_metadata_dir axbench/output_llama/layer_20_concise/generate \
  #--inference_dir axbench/output_llama/layer_20_concise/lsreft_20__0.005_0.3
  #--inference_dir axbench/output_llama/layer_20

#
echo "evaluate..."
python axbench/scripts/evaluate.py --config axbench/demo/sweep/lora.yaml \
  --mode steering \
  --dump_dir axbench/output_llama/layer_20_concise/lora_20\
