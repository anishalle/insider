# Processed Output Dataset

This document describes the current processed-data contract for the insider pipeline and the live Juno counts verified on April 21, 2026.

## Canonical Paths

Local repo:

- `/Users/ani/workspaces/github.com/anishalle/insider`

Juno repo checkout:

- `/home/axa230262/work/001 research/insider`

Juno processed output root:

- `/home/axa230262/scratch/insider/processed`

Primary stage outputs:

- `/home/axa230262/scratch/insider/processed/prepared_trades`
- `/home/axa230262/scratch/insider/processed/labeled_user_trades`
- `/home/axa230262/scratch/insider/processed/sequence_dataset`
- `/home/axa230262/scratch/insider/processed/model_windows/window_size=50`
- `/home/axa230262/scratch/insider/processed/manifests`

Derived reports should live under:

- `/home/axa230262/scratch/insider/processed/reports`

## Current Live Manifest Summary

These values come from the manifests and parquet layout currently present on Juno.

| Stage | Rows | Time Coverage |
|---|---:|---|
| `prepared_trades` | `170,275,968` | `2022-11-21 19:49:29 UTC` to `2025-12-30 05:55:05 UTC` |
| `labeled_user_trades` | `158,644,500` | `2022-12-02 19:54:59 UTC` to `2025-12-30 05:50:05 UTC` |
| `sequence_dataset` | `3,382,360` | `2023-02-25 10:30:46 UTC` to `2025-12-30 05:50:03 UTC` |
| `model_windows/window_size=50` | `3,751,000` | `2023-02-25 09:04:25 UTC` to `2025-12-30 05:50:05 UTC` |

Current `model_windows/window_size=50` split counts:

| Split | Rows | Positive Rate |
|---|---:|---:|
| `train` | `3,018,859` | `0.5520396944673468` |
| `validation` | `365,112` | `0.5003368829290739` |
| `test` | `367,029` | `0.5001730108520035` |

Current `model_windows/window_size=50` shape and coverage:

- `window_size = 50`
- `length(features) = 50`
- `length(features[1]) = 16`
- distinct users: `33,511`
- distinct markets: `55,024`
- earliest `window_start_ts`: `2023-02-16 22:06:04 UTC`
- latest `window_end_ts`: `2025-12-30 05:50:05 UTC`

## Stage Semantics

### `prepare_trades`

One row per cleaned source trade in YES-price space.

Important fields:

- original trade metadata such as `timestamp`, `block_number`, `transaction_hash`, `market_id`, and `asset_id`
- `price_yes`
- `token_amount`
- `usd_amount`
- `trade_time`
- `year_month`

### `label_user_trades`

One row per user-side trade view. Each source trade becomes one maker row and one taker row.

Important fields:

- `user`
- `role`
- `direction`
- `side`
- `signed_token_amount`
- `target_time`
- `future_trade_time`
- `future_lag_seconds`
- `markout`
- `markout_bps`
- `label`

### `sequence_dataset`

One row per fixed-length user-market sequence for recurrent-model analysis.

Important fields:

- `sequence_id`
- `sequence_start_ts`
- `sequence_end_ts`
- `seq_len`
- `features`
- `label`
- `split`

### `model_windows`

One row per exact-width training window intended for downstream classifiers.

Important fields:

- `window_id`
- `window_start_ts`
- `window_end_ts`
- `window_size`
- `features`
- `label`
- `split`

## Current Label And Window Config

Relevant current config values from `configs/pipeline.toml`:

- label horizon: `5 minutes`
- `max_future_lag_seconds = 300`
- `min_abs_markout_bps = 5.0`
- sequence length: `64`
- sequence stride: `16`
- model-window length: `50`
- model-window stride: `16`
- split ratios: `0.8 / 0.1 / 0.1`
- purge minutes: `60`
- embargo minutes: `60`

Current feature order for both sequences and model windows is the 16-feature set:

1. `price_yes`
2. `signed_token_amount`
3. `usd_amount`
4. `side`
5. `role_is_maker`
6. `time_delta_seconds`
7. `market_age_seconds`
8. `market_trade_count_1h`
9. `market_volume_1h`
10. `market_price_mean_1h`
11. `market_price_std_1h`
12. `market_price_return_1h`
13. `user_trade_count_1h`
14. `user_market_trade_count_1h`
15. `user_signed_flow_1h`
16. `user_usd_volume_1h`

## Label Definition

The classification signal is:

```text
target_time = trade_time + 5 minutes
future_price_yes = first YES price for the same asset_id at or after target_time
markout = side * (future_price_yes - price_yes)
markout_bps = side * ((future_price_yes / price_yes) - 1) * 10000
label = 1 if markout > 0 else 0
```

Normalization and direction handling:

- if `nonusdc_side == "token2"`, then `price_yes = 1 - price`
- `side = +1` for buy-like flow and `-1` for sell-like flow
- `signed_token_amount = side * token_amount`

## Consumption Notes

The current exact-width training dataset is `model_windows/window_size=50`. That is the canonical input for logistic regression, XGBoost, RNN, and LSTM.

`sequence_dataset` remains useful for analysis, but the training stack does not consume it directly.

`inputs.markets_path` is currently reserved metadata and is not consumed by the pipeline implementation.

## Historical Caveat

`model_windows/window_size=50` has been reused across multiple rebuilds. Older model runs from April 7, 2026 used an earlier `50 x 7` dataset with `7,481,187` rows at the same path.

If you compare historical runs, use each run’s own `summary.json` split summaries and feature metadata rather than assuming the live manifest matches that run.

## Quick Verification Commands

Manifest summary:

```bash
python - <<'PY'
from pathlib import Path
import json
for name in [
    "prepare_trades.json",
    "label_user_trades.json",
    "build_sequences.json",
    "build_model_windows_50.json",
]:
    path = Path("/home/axa230262/scratch/insider/processed/manifests") / name
    data = json.loads(path.read_text())
    print(name, data["row_count"], data["dataset_dir"])
PY
```

Current model-window checks:

```bash
.venv/bin/python - <<'PY'
import duckdb
con = duckdb.connect()
path = "/home/axa230262/scratch/insider/processed/model_windows/window_size=50/**/*.parquet"
print(con.execute(f"""
SELECT split, COUNT(*) AS row_count, AVG(label) AS positive_rate
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
