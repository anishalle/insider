#!/bin/bash
#SBATCH -p normal
#SBATCH -c 32
#SBATCH --mem=192G
#SBATCH -J insider-audit-xgb
#SBATCH -o logs/slurm-%A_%a.out
#SBATCH -e logs/slurm-%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=AXA230262@utdallas.edu

cd "$HOME/work/001 research/insider"
mkdir -p logs analysis

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate g_retriever
set -euo pipefail
bash scripts/require-clean-repo.sh "$PWD"
bash scripts/bootstrap-model-env.sh "$PWD"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

: "${AUDIT_MANIFEST_PATH:?AUDIT_MANIFEST_PATH is required}"
: "${AUDIT_GROUP_NAME:?AUDIT_GROUP_NAME is required}"

python -m insider run-audit-task \
  --manifest-path "$AUDIT_MANIFEST_PATH" \
  --group-name "$AUDIT_GROUP_NAME" \
  --group-index "${SLURM_ARRAY_TASK_ID}"
