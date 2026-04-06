Insider trading detection on historical Polymarket data.

This repo now contains the preprocessing pipeline for turning raw Polymarket trade data into a binary classification dataset for sequence models.

The current target is a 5-minute forward markout label:
- `label = 1` when the future YES price at `t + 5m` implies a positive user-side markout
- `label = 0` otherwise

The pipeline is staged to keep RAM bounded on Juno:
1. `prepare-trades`
2. `label-user-trades`
3. `build-sequences`

Local repo is the source of truth. Juno should only pull pushed commits, run the Slurm jobs, and write generated parquet outputs under scratch.

## Commands

```bash
uv sync --extra dev
uv run insider prepare-trades --config configs/pipeline.toml --overwrite
uv run insider label-user-trades --config configs/pipeline.toml --overwrite
uv run insider build-sequences --config configs/pipeline.toml --overwrite
uv run insider run-pipeline --config configs/pipeline.toml --overwrite
uv run insider smoke-test --config configs/pipeline.toml --overwrite
```

## Juno Workflow

```bash
ssh juno.utdallas.edu
bash scripts/link-juno-repo.sh
bash scripts/juno-sync.sh
mkdir -p logs
sbatch jobs/run-dev.sbatch
sbatch jobs/run-normal.sbatch
```

See [pipeline.md](/Users/ani/workspaces/github.com/anishalle/insider/docs/pipeline.md) for stage-level details.
