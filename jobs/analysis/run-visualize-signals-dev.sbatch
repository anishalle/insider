#!/bin/bash
#SBATCH -p dev
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -J insider-viz-dev
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err

set -euo pipefail

cd "$HOME/work/001 research/insider"
mkdir -p logs

bash scripts/bootstrap-juno-env.sh
. .venv/bin/activate
python -m insider visualize-signals \
  --config configs/pipeline.toml \
  --output-dir /home/axa230262/scratch/insider/analysis/signal-review \
  --examples-per-class 4 \
  --ambiguous-examples 2 \
  --lookback-minutes 60 \
  --lookahead-minutes 30 \
  --model-window-size 50
