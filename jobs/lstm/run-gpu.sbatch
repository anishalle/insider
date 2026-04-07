#!/bin/bash
#SBATCH -p a30
#SBATCH -c 32
#SBATCH --mem=192G
#SBATCH --gpus=1
#SBATCH -J insider-lstm
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=AXA230262@utdallas.edu

set -euo pipefail

cd "$HOME/work/001 research/insider"
mkdir -p logs

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate g_retriever
bash scripts/bootstrap-model-env.sh "$PWD" 1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python -m lstm.train \
  --config configs/pipeline.toml \
  --window-size 50 \
  --device cuda
