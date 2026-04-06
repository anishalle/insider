# Insider Repo Notes

## Overview

This repo contains the preprocessing pipeline for Polymarket insider-trading research. The local repository is the source of truth for code changes.

## Local Path

`/Users/ani/workspaces/github.com/anishalle/insider`

## Juno Remote Path

`/home/axa230262/work/001 research/insider`

## Workflow

1. Edit and validate code locally.
2. Commit and push from local `main`.
3. On Juno, pull the latest `main` in `/home/axa230262/work/001 research/insider`.
4. Run Slurm jobs from that remote repo checkout.
5. Keep generated datasets, logs, caches, and artifacts out of Git.

## Key Commands

Local:

```bash
uv sync --extra dev
uv run --extra dev pytest
uv run insider smoke-test --config configs/pipeline.toml --overwrite
```

Juno:

```bash
cd "/home/axa230262/work/001 research/insider"
bash scripts/juno-sync.sh "/home/axa230262/work/001 research/insider" main
sbatch jobs/run-dev.sbatch
sbatch jobs/run-dev-long.sbatch
sbatch jobs/run-normal.sbatch
```
