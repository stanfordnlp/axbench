# run this with: bash axbench/demo/demo.sh

python axbench/scripts/generate.py --config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo

python axbench/scripts/train.py --config axbench/demo/sweep/simple.yaml --dump_dir axbench/demo

python axbench/scripts/inference.py --config axbench/demo/sweep/simple.yaml --mode latent --dump_dir axbench/demo

python axbench/scripts/inference.py --config axbench/demo/sweep/simple.yaml --mode steering --dump_dir axbench/demo

python axbench/scripts/evaluate.py --config axbench/demo/sweep/simple.yaml --mode latent --dump_dir axbench/demo

python axbench/scripts/evaluate.py --config axbench/demo/sweep/simple.yaml --mode steering --dump_dir axbench/demo
