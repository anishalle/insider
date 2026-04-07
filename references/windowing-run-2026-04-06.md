# Windowing Run Reference

This document records how the `window_size=50` model-window dataset is produced and what was verified on Juno on April 6, 2026.

It is based on the actual output present on Juno under:

- repo: `/home/axa230262/work/001 research/insider`
- processed root: `/home/axa230262/scratch/insider/processed`
- window dataset: `/home/axa230262/scratch/insider/processed/model_windows/window_size=50`
- manifest: `/home/axa230262/scratch/insider/processed/manifests/build_model_windows_50.json`

## Purpose

The windowing stage produces a reusable, exact-width training dataset for downstream ML experiments. It is separate from the existing preprocessing pipeline so we do not have to rebuild windows every time we train a model.

The stage:

- reads from `labeled_user_trades`
- groups by `user, market_id`
- orders by `trade_time`
- emits fixed-width sliding windows only
- uses the last row in each window as the label
- assigns time-based `train`, `validation`, and `test` splits

## Code Path

Local source files that define the run:

- [jobs/windowing/run-window-50.sbatch](/Users/ani/workspaces/github.com/anishalle/insider/jobs/windowing/run-window-50.sbatch)
- [src/insider/cli.py](/Users/ani/workspaces/github.com/anishalle/insider/src/insider/cli.py)
- [src/insider/pipeline.py](/Users/ani/workspaces/github.com/anishalle/insider/src/insider/pipeline.py)
- [configs/pipeline.toml](/Users/ani/workspaces/github.com/anishalle/insider/configs/pipeline.toml)

Relevant config values from `[model_windows]`:

- `length = 50`
- `stride = 16`
- `train_ratio = 0.8`
- `validation_ratio = 0.1`
- `test_ratio = 0.1`
- feature order:
  - `price_yes`
  - `signed_token_amount`
  - `usd_amount`
  - `side`
  - `role_is_maker`
  - `time_delta_seconds`
  - `market_age_seconds`

## How The Run Is Started

### Recommended Juno submission

From the repo checkout on Juno:

```bash
cd "/home/axa230262/work/001 research/insider"
sbatch jobs/windowing/run-window-50.sbatch
```

Current `sbatch` script behavior:

1. `cd "$HOME/work/001 research/insider"`
2. `mkdir -p logs`
3. `bash scripts/bootstrap-juno-env.sh`
4. `. .venv/bin/activate`
5. `python -m insider build-model-windows --config configs/pipeline.toml --window-size 50 --overwrite`

### Equivalent direct command

If running manually instead of through Slurm:

```bash
cd "/home/axa230262/work/001 research/insider"
bash scripts/bootstrap-juno-env.sh
. .venv/bin/activate
python -m insider build-model-windows --config configs/pipeline.toml --window-size 50 --overwrite
```

## Verified Output On Juno

The following output was verified directly on Juno.

### Manifest presence

`/home/axa230262/scratch/insider/processed/manifests/build_model_windows_50.json` exists.

Observed manifest values:

- `stage`: `build_model_windows_50`
- `dataset_dir`: `/home/axa230262/scratch/insider/processed/model_windows/window_size=50`
- `row_count`: `7,481,187`
- `min_trade_time`: `2022-11-21T22:27:55+00:00`
- `max_trade_time`: `2025-12-30T05:50:03+00:00`
- `train_cutoff_epoch`: `1765181593.8`
- `validation_cutoff_epoch`: `1766235286.0`
- `model_window.length`: `50`
- `model_window.stride`: `16`

### Output directories

Observed directory layout:

```text
/home/axa230262/scratch/insider/processed/model_windows
/home/axa230262/scratch/insider/processed/model_windows/window_size=50
/home/axa230262/scratch/insider/processed/model_windows/window_size=50/split=train
/home/axa230262/scratch/insider/processed/model_windows/window_size=50/split=validation
/home/axa230262/scratch/insider/processed/model_windows/window_size=50/split=test
```

### Split counts

Verified with DuckDB over:

```text
/home/axa230262/scratch/insider/processed/model_windows/window_size=50/**/*.parquet
```

Observed counts:

- `train`: `5,985,294`
- `validation`: `747,974`
- `test`: `747,919`
- total: `7,481,187`

The split proportions are effectively the intended `0.8 / 0.1 / 0.1`.

### Width and shape checks

The exact-width guarantee was verified with:

```sql
SELECT
  MIN(window_size),
  MAX(window_size),
  MIN(length(features)),
  MAX(length(features)),
  MIN(length(features[1])),
  MAX(length(features[1]))
FROM read_parquet('/home/axa230262/scratch/insider/processed/model_windows/window_size=50/**/*.parquet');
```

Observed result:

- `MIN(window_size) = 50`
- `MAX(window_size) = 50`
- `MIN(length(features)) = 50`
- `MAX(length(features)) = 50`
- `MIN(length(features[1])) = 7`
- `MAX(length(features[1])) = 7`

This confirms:

- every emitted example is 50 steps wide
- there are no short windows in this dataset
- every step vector has width 7

### Label rates

Observed positive-label rates by split:

- `train`: `0.36891704902048256`
- `validation`: `0.4366809541508127`
- `test`: `0.4431816814387654`

These were computed with:

```sql
SELECT split, AVG(label) AS positive_rate
FROM read_parquet('/home/axa230262/scratch/insider/processed/model_windows/window_size=50/**/*.parquet')
GROUP BY split
ORDER BY split;
```

### Coverage

Observed coverage:

- distinct users: `85,413`
- distinct markets: `85,089`
- earliest `window_start_ts`: `2022-11-21 19:49:29`
- latest `window_end_ts`: `2025-12-30 05:50:03`

These were computed with:

```sql
SELECT
  COUNT(DISTINCT user) AS users,
  COUNT(DISTINCT market_id) AS markets,
  MIN(window_start_ts),
  MAX(window_end_ts)
FROM read_parquet('/home/axa230262/scratch/insider/processed/model_windows/window_size=50/**/*.parquet');
```

## Monitoring Notes

At verification time:

- `squeue -u $USER` returned no active jobs
- the window dataset and manifest were already present

I did not find a fresh Slurm stdout file that clearly corresponded to the `build_model_windows_50.json` timestamp. The newest visible `logs/slurm-*.out` files in the repo were older and one of them clearly corresponded to `run-pipeline`, not the dedicated windowing command.

Because of that, the verification above should be treated as:

- authoritative for dataset existence and contents
- authoritative for row counts and exact-width checks
- not authoritative for a specific job id or wall-clock runtime of the completed run

If you want strict run provenance next time, capture the `sbatch` return value and record the resulting job id alongside the manifest path.

## Re-Verification Commands

### Quick health check

```bash
cd "/home/axa230262/work/001 research/insider"
. .venv/bin/activate
python - <<'PY'
import duckdb
con = duckdb.connect()
path = "/home/axa230262/scratch/insider/processed/model_windows/window_size=50/**/*.parquet"
print(con.execute(f"""
SELECT split, COUNT(*) AS row_count
FROM read_parquet('{path}')
GROUP BY split
ORDER BY split
""").fetchall())
print(con.execute(f"""
SELECT
  MIN(window_size),
  MAX(window_size),
  MIN(length(features)),
  MAX(length(features)),
  MIN(length(features[1])),
  MAX(length(features[1]))
FROM read_parquet('{path}')
""").fetchall())
PY
```

### Manifest check

```bash
python - <<'PY'
from pathlib import Path
import json
path = Path("/home/axa230262/scratch/insider/processed/manifests/build_model_windows_50.json")
data = json.loads(path.read_text())
print(data["stage"])
print(data["dataset_dir"])
print(data["row_count"])
print(data["model_window"])
PY
```

## Expected Inputs Before Re-Running

The windowing stage assumes these upstream outputs already exist:

- `/home/axa230262/scratch/insider/processed/prepared_trades`
- `/home/axa230262/scratch/insider/processed/labeled_user_trades`

If those are missing or stale, rerun preprocessing first.

## Practical Interpretation

This run produced a model-ready dataset for:

- logistic regression with flattened `50 x 7` windows
- RNN training on 50-step sequences
- LSTM training on 50-step sequences

The key outcome is that the dedicated model-window dataset is now cleaner than the older `sequence_dataset` for strict sequence-model ingestion because the verified output has:

- exact window length
- consistent per-step feature width
- reusable split partitions
- a dedicated manifest
