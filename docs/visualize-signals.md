# Visualize Signals

This document describes the `visualize-signals` report command, what it reads, what it writes, and how to run it without polluting the repo checkout.

## Purpose

`visualize-signals` is a read-only analysis workflow for the processed dataset. It does not modify the parquet stages under `prepared_trades`, `labeled_user_trades`, `sequence_dataset`, or `model_windows`.

It is meant to answer a few practical questions before deeper model work:

- do the strongest positive and negative labels line up with visible market moves
- how imbalanced is the label distribution over time
- how concentrated is the data across users and markets
- does the current dataset look feasible for training

## Inputs

The command resolves paths from `[output].root` in the pipeline config:

- primary input: `labeled_user_trades`
- plot-context input: `prepared_trades`
- optional readiness input: `model_windows/window_size=<N>`

## Default Output Location

If you do not pass `--output-dir`, the command writes its report bundle to:

```text
<output.root>/reports/signal-review
```

With the repo’s default Juno config, that resolves to:

```text
/home/axa230262/scratch/insider/processed/reports/signal-review
```

This keeps generated plots and CSVs out of the git checkout.

## Commands

Local shape:

```bash
uv run insider visualize-signals \
  --config configs/pipeline.toml
```

Direct Juno run:

```bash
cd "/home/axa230262/work/001 research/insider"
bash scripts/bootstrap-juno-env.sh
. .venv/bin/activate
python -m insider visualize-signals \
  --config configs/pipeline.toml \
  --examples-per-class 4 \
  --ambiguous-examples 2 \
  --lookback-minutes 60 \
  --lookahead-minutes 30 \
  --model-window-size 50
```

Override the destination only when you intentionally want a non-default report location:

```bash
python -m insider visualize-signals \
  --config configs/pipeline.toml \
  --output-dir "/some/other/report/path"
```

## Outputs

Expected files in the report directory:

- `summary.json`
- `training_feasibility.json`
- `class_balance_over_time.csv`
- `candidate_signals.csv`
- `plots/*.png`

`summary.json` includes overall labeled-row counts, class balance, distinct users/markets/assets, time coverage, and markout percentiles.

`training_feasibility.json` includes:

- dataset-integrity checks
- monthly label-rate drift
- user and market concentration summaries
- stream-length buckets
- estimated window coverage from `labeled_user_trades`
- actual `model_windows` shape and manifest checks when that dataset exists

`candidate_signals.csv` contains one row per selected example, including the label, selection reason, user, market, asset, role, side, markout values, and transaction identifiers.

Each plot shows the YES price path around the selected trade, the selected trade marker, the forward target marker, the realized future-price marker, and surrounding notional spikes.

## Selection Logic

By default the command emits a mixed review set:

- strongest positive examples by `markout_bps`
- strongest negative examples by `markout_bps`
- a small near-zero-markout bucket for ambiguous calibration cases

## Juno Batch Workflow

Use the dedicated dev-node script:

- [run-visualize-signals-dev.sbatch](/Users/ani/workspaces/github.com/anishalle/insider/jobs/analysis/run-visualize-signals-dev.sbatch)

Submit it from the repo root on Juno:

```bash
cd "/home/axa230262/work/001 research/insider"
sbatch jobs/analysis/run-visualize-signals-dev.sbatch
```

Current batch-script behavior:

1. changes into the Juno repo checkout
2. creates `logs/`
3. bootstraps the repo-local `.venv`
4. runs `python -m insider visualize-signals`
5. writes the report bundle under `<output.root>/reports/signal-review`

The dev script intentionally does not set mail flags.
