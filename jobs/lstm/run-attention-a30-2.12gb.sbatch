#!/bin/bash
#SBATCH -p a30-2.12gb
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH -J insider-lstm-attn
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err

cd "$HOME/work/001 research/insider"
mkdir -p logs analysis

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate g_retriever
set -euo pipefail
bash scripts/bootstrap-model-env.sh "$PWD" 1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python -m lstm.train \
  --config configs/pipeline.toml \
  --window-size 50 \
  --device cuda \
  --hidden-size 128 \
  --num-layers 1 \
  --dropout 0.1 \
  --pooling attention \
  --disable-summary-features \
  --epochs 12 \
  --batch-size 256 \
  --eval-batch-size 512 \
  --learning-rate 0.0003 \
  --patience 4 \
  --debug-metrics

conda deactivate || true
bash scripts/bootstrap-juno-env.sh
. .venv/bin/activate
python -m insider build-model-leaderboard \
  --config configs/pipeline.toml \
  --output-path "$HOME/work/001 research/insider/analysis/model-leaderboard.csv"
