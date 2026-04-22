Insider trading detection on historical Polymarket data.

This repo contains the preprocessing pipeline and model-training inputs for the current 5-minute forward-markout binary classification task:

- `label = 1` when the future YES price at `t + 5m` implies a positive user-side markout
- `label = 0` otherwise

The default config applies conservative data-quality controls before training:

- stale labels are dropped when the matched future trade arrives more than `300s` after the target horizon
- near-zero markouts are dropped when `abs(markout_bps) < 5`
- sequence and model-window splits can purge and embargo boundary regions
- model windows include the richer 16-feature market-state and user-history context set

The disk-backed pipeline stages are:

1. `prepare-trades`
2. `label-user-trades`
3. `build-sequences`
4. `build-model-windows`

`run-pipeline` intentionally stops after `build-sequences`. Exact-width `model_windows` are built as a separate stage so downstream model experiments can reuse them.

Local repo is the source of truth. Juno should only pull pushed commits, run Slurm jobs, and write generated datasets, model outputs, and reports under scratch-backed paths.

## Commands

```bash
uv sync --extra dev
uv run insider prepare-trades --config configs/pipeline.toml --overwrite
uv run insider label-user-trades --config configs/pipeline.toml --overwrite
uv run insider build-sequences --config configs/pipeline.toml --overwrite
uv run insider build-model-windows --config configs/pipeline.toml --window-size 50 --overwrite
uv run insider run-pipeline --config configs/pipeline.toml --overwrite
uv run insider visualize-signals --config configs/pipeline.toml
uv run insider build-model-leaderboard --config configs/pipeline.toml
uv run insider build-model-audit --config configs/pipeline.toml
uv run insider smoke-test --config configs/pipeline.toml --overwrite
```

Default derived reports now land under `<output.root>/reports/`, not inside the git checkout:

- `visualize-signals`: `<output.root>/reports/signal-review/`
- `build-model-leaderboard`: `<output.root>/reports/model_leaderboard.csv`
- `build-model-audit`: `<output.root>/reports/model_audit.csv`

## Juno Workflow

```bash
ssh juno.utdallas.edu
bash scripts/link-juno-repo.sh
bash scripts/juno-sync.sh
sbatch jobs/preprocess/run-dev.sbatch
sbatch jobs/preprocess/run-normal.sbatch
sbatch jobs/windowing/run-window-50.sbatch
sbatch jobs/analysis/run-visualize-signals-dev.sbatch
```

The Juno sync and training scripts now refuse to run from a dirty checkout unless you explicitly opt out with `ALLOW_DIRTY_REPO=1`. That keeps generated artifacts and accidental notebook edits from polluting the remote repo state.

Juno currently has a broken `uv` binary in the shell path, so the remote scripts bootstrap a repo-local `.venv` with `python3 -m venv` instead of relying on `uv` there.

See [pipeline.md](/Users/ani/workspaces/github.com/anishalle/insider/docs/pipeline.md) for stage-level details.
See [output_dataset.md](/Users/ani/workspaces/github.com/anishalle/insider/docs/output_dataset.md) for the current processed-data contract and live Juno counts.
See [models.md](/Users/ani/workspaces/github.com/anishalle/insider/docs/models.md) for model-training details and tracked run summaries.
See [visualize-signals.md](/Users/ani/workspaces/github.com/anishalle/insider/docs/visualize-signals.md) for the report workflow.
