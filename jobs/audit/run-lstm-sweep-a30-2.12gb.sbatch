#!/bin/bash
#SBATCH -p a30-2.12gb
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH -J insider-audit-lstm
#SBATCH -o logs/slurm-%A_%a.out
#SBATCH -e logs/slurm-%A_%a.err

cd "$HOME/work/001 research/insider"
mkdir -p logs analysis

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate g_retriever
set -euo pipefail
bash scripts/require-clean-repo.sh "$PWD"
bash scripts/bootstrap-model-env.sh "$PWD" 1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

: "${AUDIT_MANIFEST_PATH:?AUDIT_MANIFEST_PATH is required}"
: "${AUDIT_GROUP_NAME:?AUDIT_GROUP_NAME is required}"

python -m insider run-audit-task \
  --manifest-path "$AUDIT_MANIFEST_PATH" \
  --group-name "$AUDIT_GROUP_NAME" \
  --group-index "${SLURM_ARRAY_TASK_ID}"
