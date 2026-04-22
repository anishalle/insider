#!/bin/bash
#SBATCH -p dev
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -J insider-audit-window
#SBATCH -o logs/slurm-%A_%a.out
#SBATCH -e logs/slurm-%A_%a.err

set -euo pipefail

cd "$HOME/work/001 research/insider"
mkdir -p logs analysis

bash scripts/require-clean-repo.sh "$PWD"
bash scripts/bootstrap-juno-env.sh
. .venv/bin/activate

: "${AUDIT_MANIFEST_PATH:?AUDIT_MANIFEST_PATH is required}"
: "${AUDIT_GROUP_NAME:?AUDIT_GROUP_NAME is required}"

python -m insider run-audit-task \
  --manifest-path "$AUDIT_MANIFEST_PATH" \
  --group-name "$AUDIT_GROUP_NAME" \
  --group-index "${SLURM_ARRAY_TASK_ID}"
