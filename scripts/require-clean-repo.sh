#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-$(pwd)}"

cd "$repo_root"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  exit 0
fi

if [[ "${ALLOW_DIRTY_REPO:-0}" == "1" ]]; then
  exit 0
fi

status="$(git status --short)"
if [[ -n "$status" ]]; then
  echo "Refusing to run from a dirty git worktree in $repo_root." >&2
  echo "Set ALLOW_DIRTY_REPO=1 to bypass this check intentionally." >&2
  echo "$status" >&2
  exit 1
fi
