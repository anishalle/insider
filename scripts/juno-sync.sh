#!/usr/bin/env bash
set -euo pipefail

repo_dir="${1:-$HOME/work/001 research/insider}"
branch="${2:-main}"

cd "$repo_dir"
git fetch origin
git checkout "$branch"
git pull --ff-only origin "$branch"
bash scripts/bootstrap-juno-env.sh
mkdir -p logs
