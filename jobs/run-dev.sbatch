#!/bin/bash
#SBATCH -p dev
#SBATCH -c 32
#SBATCH --mem=192G
#SBATCH -J insider-dev
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err

set -euo pipefail

cd "$HOME/work/001 research/insider"
mkdir -p logs

bash scripts/bootstrap-juno-env.sh
. .venv/bin/activate
python -m insider smoke-test --config configs/pipeline.toml --overwrite
