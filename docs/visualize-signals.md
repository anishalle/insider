# Visualize Signals

This document describes the `visualize-signals` analysis command, what data it reads, what it writes, and how to run it on Juno.

## Purpose

The command is intended for pre-model inspection of the processed Polymarket dataset.

It reads the non-windowed `labeled_user_trades` output, selects example true and false signals, renders static plots around those trades, and writes summary artifacts that make it easier to answer:

- does the label actually line up with a visible market move
- how imbalanced are the classes
- how concentrated is the data across users and markets
- does the dataset look feasible for training before deeper model work

This is an analysis workflow, not a preprocessing stage. It does not modify the processed parquet.

## Inputs

Primary input:

- `processed/labeled_user_trades`

Context input for the price series plots:

- `processed/prepared_trades`

Optional secondary input for training-readiness summaries:

- `processed/model_windows/window_size=<N>`

The command uses the same config file as the preprocessing pipeline and resolves dataset locations from `[output].root`.

## Command

Local example:

```bash
uv run insider visualize-signals \
  --config configs/pipeline.toml \
  --output-dir "/home/axa230262/work/001 research/insider/analysis/signal-review"
```

Direct Juno example:

```bash
cd "/home/axa230262/work/001 research/insider"
bash scripts/bootstrap-juno-env.sh
. .venv/bin/activate
python -m insider visualize-signals \
  --config configs/pipeline.toml \
  --output-dir "/home/axa230262/work/001 research/insider/analysis/signal-review" \
  --examples-per-class 4 \
  --ambiguous-examples 2 \
  --lookback-minutes 60 \
  --lookahead-minutes 30 \
  --model-window-size 50
```

## Outputs

The command writes a review bundle under the requested output directory.

Expected files:

- `summary.json`
- `training_feasibility.json`
- `class_balance_over_time.csv`
- `candidate_signals.csv`
- `plots/*.png`

### `summary.json`

Contains:

- total labeled rows
- positive and negative counts
- positive rate and imbalance ratios
- distinct users, markets, and assets
- overall time coverage
- markout, markout-bps, and USD notional percentiles

### `training_feasibility.json`

Contains:

- training verdict:
  - `feasible_now`
  - `feasible_with_weighting_or_sampling`
  - `blocked_for_training`
- monthly label-rate drift summary
- user and market concentration summaries
- user-market stream length buckets
- estimated 50-step window coverage from `labeled_user_trades`
- actual `model_windows` split counts and shape checks when that dataset exists

### `candidate_signals.csv`

One row per selected example, including:

- class label
- selection reason
- user
- market
- asset
- role and side
- trade time
- target time
- future trade time
- markout and markout-bps
- USD and token size
- transaction and order hashes

### `plots/*.png`

Each plot shows:

- YES price path around the selected trade
- the selected trade marker
- the forward target time marker
- the realized future price marker
- a lower panel for surrounding USD notional spikes
- a compact annotation block with the trade metadata and markout

## Selection Logic

By default the command emits a mixed review set:

- strongest positive examples by `markout_bps`
- strongest negative examples by `markout_bps`
- a small near-zero markout bucket for ambiguous calibration cases

This is intended to give a quick visual read on obvious signals, obvious misses, and borderline examples.

## Juno Batch Workflow

Use the dedicated dev-node script:

- [run-visualize-signals-dev.sbatch](/Users/ani/workspaces/github.com/anishalle/insider/jobs/analysis/run-visualize-signals-dev.sbatch)

Submit it from the repo root on Juno:

```bash
cd "/home/axa230262/work/001 research/insider"
sbatch jobs/analysis/run-visualize-signals-dev.sbatch
```

Current behavior of the batch script:

1. changes into the Juno repo checkout
2. creates `logs/`
3. bootstraps the repo-local `.venv`
4. runs `python -m insider visualize-signals`
5. writes review artifacts under `/home/axa230262/work/001 research/insider/analysis/signal-review`

The dev script intentionally does not set mail flags.
