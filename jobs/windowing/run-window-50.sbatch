#!/bin/bash
#SBATCH -p normal
#SBATCH -c 64
#SBATCH --mem=256G
#SBATCH -J insider-window-50
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=AXA230262@utdallas.edu

set -euo pipefail

cd "$HOME/work/001 research/insider"
mkdir -p logs

bash scripts/bootstrap-juno-env.sh
. .venv/bin/activate
python -m insider build-model-windows --config configs/pipeline.toml --window-size 50 --overwrite
