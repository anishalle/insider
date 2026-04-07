Insider trading detection on historical Polymarket data.

This repo now contains the preprocessing pipeline for turning raw Polymarket trade data into a binary classification dataset for sequence models.

The current target is a 5-minute forward markout label:
- `label = 1` when the future YES price at `t + 5m` implies a positive user-side markout
- `label = 0` otherwise

The pipeline is staged to keep RAM bounded on Juno:
1. `prepare-trades`
2. `label-user-trades`
3. `build-sequences`
4. `build-model-windows`

Local repo is the source of truth. Juno should only pull pushed commits, run the Slurm jobs, and write generated parquet outputs under scratch.

## Commands

```bash
uv sync --extra dev
uv run insider prepare-trades --config configs/pipeline.toml --overwrite
uv run insider label-user-trades --config configs/pipeline.toml --overwrite
uv run insider build-sequences --config configs/pipeline.toml --overwrite
uv run insider build-model-windows --config configs/pipeline.toml --window-size 50 --overwrite
uv run insider visualize-signals --config configs/pipeline.toml --output-dir "/home/axa230262/work/001 research/insider/analysis/signal-review"
uv run insider run-pipeline --config configs/pipeline.toml --overwrite
uv run insider smoke-test --config configs/pipeline.toml --overwrite
```

## Juno Workflow

```bash
ssh juno.utdallas.edu
bash scripts/link-juno-repo.sh
bash scripts/juno-sync.sh
mkdir -p logs
sbatch jobs/preprocess/run-dev.sbatch
sbatch jobs/preprocess/run-normal.sbatch
sbatch jobs/windowing/run-window-50.sbatch
sbatch jobs/analysis/run-visualize-signals-dev.sbatch
```

Juno currently has a broken `uv` binary in the shell path, so the remote scripts bootstrap a repo-local `.venv` with `python3 -m venv` instead of relying on `uv` there.

See [pipeline.md](/Users/ani/workspaces/github.com/anishalle/insider/docs/pipeline.md) for stage-level details.
See [models.md](/Users/ani/workspaces/github.com/anishalle/insider/docs/models.md) for model-training details and Juno job usage.
See [visualize-signals.md](/Users/ani/workspaces/github.com/anishalle/insider/docs/visualize-signals.md) for the Juno analysis workflow.
