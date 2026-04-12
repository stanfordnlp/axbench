#!/bin/bash
# Setup script for LatentQA integration with AxBench.
#
# This script:
#   1. Clones and installs the LatentQA repository
#   2. Downloads the pre-trained LatentQA decoder model
#   3. Verifies the installation
#
# Prerequisites:
#   - Python 3.10+ with PyTorch and CUDA
#   - At least 2 GPUs (target model on cuda:0, decoder on cuda:1)
#   - HuggingFace access to meta-llama/Meta-Llama-3-8B-Instruct
#
# Usage:
#   bash axbench/data/setup-latentqa.sh [--latentqa-dir /path/to/install]
#
set -e

LATENTQA_DIR="${1:-./latentqa}"

echo "=== LatentQA Setup for AxBench ==="
echo ""

# Step 1: Clone LatentQA repo
if [ -d "$LATENTQA_DIR" ]; then
    echo "[1/4] LatentQA repo already exists at $LATENTQA_DIR, pulling latest..."
    cd "$LATENTQA_DIR" && git pull && cd -
else
    echo "[1/4] Cloning LatentQA repo to $LATENTQA_DIR..."
    git clone https://github.com/aypan17/latentqa.git "$LATENTQA_DIR"
fi

# Step 2: Install LatentQA
echo "[2/4] Installing LatentQA..."
pip install -e "$LATENTQA_DIR"
# Install additional dependencies if requirements.txt exists
if [ -f "$LATENTQA_DIR/requirements.txt" ]; then
    pip install -r "$LATENTQA_DIR/requirements.txt"
fi

# Step 3: Pre-download the decoder model from HuggingFace
echo "[3/4] Pre-downloading LatentQA decoder model (aypan17/latentqa_llama-3-8b-instruct)..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('aypan17/latentqa_llama-3-8b-instruct')
print('Decoder model downloaded successfully.')
" || echo "WARNING: Could not pre-download decoder model. It will be downloaded on first use."

# Step 4: Verify installation
echo "[4/4] Verifying installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Number of GPUs: {torch.cuda.device_count()}')
if torch.cuda.device_count() < 2:
    print('WARNING: LatentQA requires at least 2 GPUs (target + decoder)')

# Check LatentQA imports
try:
    from lit.utils.activation_utils import latent_qa
    from lit.utils.infra_utils import get_model, get_tokenizer, get_modules
    from lit.utils.dataset_utils import BASE_DIALOG
    print('LatentQA imports: OK')
except ImportError as e:
    print(f'LatentQA imports: FAILED ({e})')
    print('Make sure the LatentQA repo is on your PYTHONPATH or installed with pip install -e')
    exit(1)

# Check AxBench LatentQA integration
try:
    from axbench.models.latentqa import LatentQAReading, LatentQASteering
    print('AxBench LatentQA integration: OK')
except ImportError as e:
    print(f'AxBench LatentQA integration: FAILED ({e})')
    exit(1)

print()
print('=== Setup complete! ===')
print()
print('To run LatentQA reading mode (concept detection):')
print('  torchrun --nproc_per_node=1 axbench/scripts/inference.py \\\\')
print('    --config axbench/sweep/aryaman/latentqa/reading_llama3_8b.yaml --mode latent')
print()
print('To run LatentQA steering mode:')
print('  torchrun --nproc_per_node=1 axbench/scripts/train.py \\\\')
print('    --config axbench/sweep/aryaman/latentqa/steering_llama3_8b.yaml')
print('  torchrun --nproc_per_node=1 axbench/scripts/inference.py \\\\')
print('    --config axbench/sweep/aryaman/latentqa/steering_llama3_8b.yaml --mode steering')
"

echo ""
echo "Done!"
