# run this with: bash axbench/demo/demo_uv.sh

uv run axbench/scripts/generate.py --config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo

uv run axbench/scripts/train.py --config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo

uv run axbench/scripts/inference.py --config axbench/demo/sweep/simple.yaml --mode latent --dump_dir axbench/demo

uv run axbench/scripts/inference.py --config axbench/demo/sweep/simple.yaml --mode steering --dump_dir axbench/demo

uv run axbench/scripts/evaluate.py --config axbench/demo/sweep/simple.yaml --mode latent --dump_dir axbench/demo

uv run axbench/scripts/evaluate.py --config axbench/demo/sweep/simple.yaml --mode steering --dump_dir axbench/demo
