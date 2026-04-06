#!/usr/bin/env bash
set -euo pipefail

remote_host="${REMOTE_HOST:-juno.utdallas.edu}"
remote_dir="${REMOTE_DIR:-/home/axa230262/scratch/insider/processed}"
local_dir="${1:-artifacts/juno-processed}"

mkdir -p "$local_dir"
scp -r "$remote_host":"$remote_dir" "$local_dir"
