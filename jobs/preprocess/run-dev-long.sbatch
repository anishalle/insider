#!/bin/bash
#SBATCH -p dev
#SBATCH -c 64
#SBATCH --mem=375G
#SBATCH -J insider-dev-long
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=AXA230262@utdallas.edu

set -euo pipefail

cd "$HOME/work/001 research/insider"
mkdir -p logs

bash scripts/bootstrap-juno-env.sh
. .venv/bin/activate
python -m insider run-pipeline --config configs/pipeline.toml --overwrite
