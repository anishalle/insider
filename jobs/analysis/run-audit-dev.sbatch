#!/bin/bash
#SBATCH -p dev
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -J insider-audit-dev
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err

set -euo pipefail

cd "$HOME/work/001 research/insider"
mkdir -p logs analysis

bash scripts/bootstrap-juno-env.sh
. .venv/bin/activate

python -m insider visualize-signals \
  --config configs/pipeline.toml \
  --output-dir "$HOME/work/001 research/insider/analysis/signal-review" \
  --examples-per-class 4 \
  --ambiguous-examples 2 \
  --lookback-minutes 60 \
  --lookahead-minutes 30 \
  --model-window-size 50

python -m insider build-model-leaderboard \
  --config configs/pipeline.toml \
  --output-path "$HOME/work/001 research/insider/analysis/model-leaderboard.csv"
