#!/usr/bin/env bash
set -euo pipefail

repo_dir="${1:-$HOME/work/001 research/insider}"
remote_url="${2:-https://github.com/anishalle/insider.git}"
branch="${3:-main}"

mkdir -p "$repo_dir"
cd "$repo_dir"

if [ ! -d .git ]; then
  git init
  git remote add origin "$remote_url"
  git fetch origin
  git checkout -B "$branch" --track "origin/$branch"
else
  git fetch origin
  git checkout "$branch"
  git pull --ff-only origin "$branch"
fi

mkdir -p logs
