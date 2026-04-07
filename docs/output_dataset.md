# Processed Output Dataset

This document describes the processed Polymarket dataset produced by the pipeline in this repo, what each output stage contains, how the classification signal is computed, and how to consume the data safely.

## Paths

Local repo:

- `/Users/ani/workspaces/github.com/anishalle/insider`

Juno repo checkout:

- `/home/axa230262/work/001 research/insider`

Juno processed output root:

- `/home/axa230262/scratch/insider/processed`

Stage outputs:

- `/home/axa230262/scratch/insider/processed/prepared_trades`
- `/home/axa230262/scratch/insider/processed/labeled_user_trades`
- `/home/axa230262/scratch/insider/processed/sequence_dataset`
- `/home/axa230262/scratch/insider/processed/manifests`

## What The Pipeline Does

The pipeline turns raw Polymarket trade records into a sequence-classification dataset for RNN/LSTM-style models.

It runs in three stages:

1. `prepare-trades`
   - Filters bad rows.
   - Drops known contract takers.
   - Normalizes all prices into the YES/token1 perspective.
   - Parses timestamps into `trade_time`.

2. `label-user-trades`
   - Computes a 5-minute forward price lookup for the same `asset_id`.
   - Explodes each trade into one maker row and one taker row.
   - Computes `side`, `signed_token_amount`, `markout`, `markout_bps`, and `label`.

3. `build-sequences`
   - Orders rows by `user, market_id, trade_time`.
   - Builds rolling sequence windows.
   - Stores one row per sequence for model input.
   - Adds time-based `train`, `validation`, and `test` splits.

## Verified Run Summary

These values come from the completed Juno run in `/home/axa230262/scratch/insider/processed`.

### Stage Shapes

| Stage | Rows | Time Coverage |
|---|---:|---|
| `prepared_trades` | 170,275,968 | 2022-11-21 19:49:29 UTC to 2025-12-30 05:55:05 UTC |
| `labeled_user_trades` | 314,413,514 | 2022-11-21 19:49:29 UTC to 2025-12-30 05:50:05 UTC |
| `sequence_dataset` | 6,666,759 | 2022-11-30 22:54:17 UTC to 2025-12-30 05:49:55 UTC |

### Final Split Sizes

| Split | Rows |
|---|---:|
| `train` | 5,251,977 |
| `validation` | 679,431 |
| `test` | 735,351 |

### Coverage

- Distinct labeled users: `1,766,152`
- Distinct labeled markets: `206,750`
- Distinct sequence users: `66,140`
- Distinct sequence markets: `74,027`
- Labeled positive rate: `0.3418006008482193`

## Output Schemas

### `prepared_trades` Headers

Each row is one cleaned trade in YES-price space.

| Column | Type | Meaning |
|---|---|---|
| `timestamp` | `BIGINT` | Original Unix timestamp from the source trade row. |
| `datetime` | `VARCHAR` | Original datetime string from the source trade row. |
| `block_number` | `BIGINT` | Polygon block number. |
| `transaction_hash` | `VARCHAR` | Trade transaction hash. |
| `contract` | `VARCHAR` | Contract address/name recorded in the source dataset. |
| `event_id` | `VARCHAR` | Event identifier. |
| `event_slug` | `VARCHAR` | Event slug. |
| `event_title` | `VARCHAR` | Event title. |
| `market_id` | `VARCHAR` | Market identifier. |
| `condition_id` | `VARCHAR` | Market condition identifier. |
| `question` | `VARCHAR` | Market question text. |
| `nonusdc_side` | `VARCHAR` | Original token side before normalization. |
| `maker` | `VARCHAR` | Maker wallet. |
| `taker` | `VARCHAR` | Taker wallet. |
| `maker_direction` | `VARCHAR` | Maker direction string from source data. |
| `taker_direction` | `VARCHAR` | Taker direction string from source data. |
| `price_yes` | `DOUBLE` | YES/token1 price after normalization. |
| `token_amount` | `DOUBLE` | Token amount scaled from raw integer units. |
| `usd_amount` | `DOUBLE` | USD amount scaled from raw integer units. |
| `asset_id` | `VARCHAR` | Outcome token identifier used for forward-price lookup. |
| `order_hash` | `VARCHAR` | Order hash. |
| `trade_time` | `TIMESTAMP` | Parsed canonical trade timestamp. |
| `year_month` | `VARCHAR` | Partition key written as `YYYY-MM`. |

### `labeled_user_trades` Headers

Each row is one user-side view of one trade. One source trade becomes two rows: one maker row and one taker row.

| Column | Type | Meaning |
|---|---|---|
| `timestamp` to `order_hash` | same as above | Carried through from `prepared_trades`. |
| `trade_time` | `TIMESTAMP` | Timestamp of the original trade. |
| `target_time` | `TIMESTAMP` | `trade_time + 5 minutes`. |
| `future_trade_time` | `TIMESTAMP` | First observed trade time for the same `asset_id` at or after `target_time`. |
| `price_yes` | `DOUBLE` | Current YES price at `trade_time`. |
| `future_price_yes` | `DOUBLE` | Future YES price used for markout labeling. |
| `usd_amount` | `DOUBLE` | Trade notional in USD. |
| `token_amount` | `DOUBLE` | Unsigned token amount after decimal scaling. |
| `user` | `VARCHAR` | Maker or taker wallet for this exploded row. |
| `role` | `VARCHAR` | Either `maker` or `taker`. |
| `direction` | `VARCHAR` | User-side direction string. |
| `side` | `INTEGER` | Encoded trade direction: `+1` for buy/long, `-1` for sell/short. |
| `signed_token_amount` | `DOUBLE` | `side * token_amount`. Positive means net buy; negative means net sell. |
| `markout` | `DOUBLE` | User-side forward markout in price units. |
| `markout_bps` | `DOUBLE` | User-side forward markout in basis points. |
| `label` | `INTEGER` | Binary classification target: `1` if `markout > 0`, else `0`. |
| `year_month` | `VARCHAR` | Partition key written as `YYYY-MM`. |

### `sequence_dataset` Headers

Each row is one sequence example for an RNN/LSTM-style model. The row label belongs to the final trade in that window.

| Column | Type | Meaning |
|---|---|---|
| `sequence_id` | `VARCHAR` | Unique id derived from `user`, `market_id`, end timestamp, and row number. |
| `user` | `VARCHAR` | Wallet for the sequence. |
| `market_id` | `VARCHAR` | Market for the sequence. |
| `sequence_start_ts` | `TIMESTAMP` | First timestamp in the stored window. |
| `sequence_end_ts` | `TIMESTAMP` | Last timestamp in the stored window. |
| `seq_len` | `INTEGER` | Intended sequence length from config. Current run uses `64`. |
| `features` | `DOUBLE[][]` | Nested array of per-step feature vectors. |
| `label` | `INTEGER` | Binary target for the final step in the sequence. |
| `split` | `VARCHAR` | Time-based split: `train`, `validation`, or `test`. |

## Classification Signal

The classification signal is implemented correctly in the completed run.

### Price Normalization

All prices are converted into YES/token1 space:

- If `nonusdc_side == "token1"`, then `price_yes = price`
- If `nonusdc_side == "token2"`, then `price_yes = 1 - price`

### Side Encoding

Direction strings are mapped to:

- `+1` for `buy`, `bid`, `long`, and equivalent positive forms
- `-1` for `sell`, `ask`, `short`, and equivalent negative forms

### Markout Equation

For each user-side row:

```text
target_time = trade_time + 5 minutes
future_price_yes = first YES price for the same asset_id at or after target_time
markout = side * (future_price_yes - price_yes)
markout_bps = side * ((future_price_yes / price_yes) - 1) * 10000
label = 1 if markout > 0 else 0
```

### Validation Results

The completed dataset passed the core classification checks:

- `314,413,514 / 314,413,514` labeled rows are consistent with `markout > 0 => label = 1`
- Invalid labels: `0`
- Rows missing `future_price_yes`: `0`
- Rows where `signed_token_amount != side * token_amount`: `0`

Example verified rows:

| role | `price_yes` | `future_price_yes` | `side` | `markout` | `markout_bps` | `label` |
|---|---:|---:|---:|---:|---:|---:|
| maker | 0.6 | 0.5 | 1 | -0.1 | -1666.67 | 0 |
| taker | 0.6 | 0.5 | -1 | 0.1 | 1666.67 | 1 |

This is the expected symmetry: the maker and taker get opposite user-side outcomes for the same future move.

## Sequence Features

The configured feature order is:

1. `price_yes`
2. `signed_token_amount`
3. `usd_amount`
4. `side`
5. `role_is_maker`
6. `time_delta_seconds`
7. `market_age_seconds`

Meaning of each feature element:

| Feature | Meaning |
|---|---|
| `price_yes` | YES probability-like price after normalization. |
| `signed_token_amount` | Positive for user buys, negative for user sells. |
| `usd_amount` | USD notional for that trade step. |
| `side` | Direction encoding, `+1` or `-1`. |
| `role_is_maker` | `1.0` for maker rows, `0.0` for taker rows. |
| `time_delta_seconds` | Seconds since the previous trade in the same `user + market_id` stream. |
| `market_age_seconds` | Seconds since the first trade in that `user + market_id` stream. |

Each `features` entry is therefore a list of 7 numeric values, and the outer list is intended to hold one such vector per sequence step.

## Important Caveat: Final Sequence Width Is Not Fully Clean Yet

The label stage is correct, but the final sequence stage still has a formatting issue that matters for strict RNN/LSTM ingestion.

Expected:

- `seq_len = 64`
- `length(features) = 64`
- `length(features[i]) = 7` for every step

Observed:

- `seq_len` is always `64`
- `length(features[i])` is always `7`
- but `length(features)` ranges from `2` to `64`

Distribution summary:

- Rows with full 64-step payloads: `6,664,199`
- Rows with short payloads: `2,560`

This means the classification signal is correct, but the final sequence dataset is not yet perfectly fixed-width.

### Safe Consumption Rule For The Current Output

If you train on the current output as-is, filter to exact-width windows:

```sql
SELECT *
FROM read_parquet('/home/axa230262/scratch/insider/processed/sequence_dataset/**/*.parquet')
WHERE length(features) = 64
```

That keeps only strict 64-step sequences and excludes the `2,560` short windows.

## What One Row Means

### In `prepared_trades`

One row means:

- one cleaned trade
- one market event
- one canonical YES/token1 price point

### In `labeled_user_trades`

One row means:

- one user-side participation in one trade
- either the maker or the taker
- one binary target for whether that user-side trade had positive 5-minute forward markout

### In `sequence_dataset`

One row means:

- one ordered trade-history window for one `user + market_id`
- one final-step classification target
- one model input example for sequence learning

The `label` on that row refers to the last event in the sequence, not to every event in the stored history.

## Config Parameters

These are the main pipeline parameters from `configs/pipeline.toml`.

### Input Parameters

| Parameter | Meaning |
|---|---|
| `inputs.trades_path` | Source `trades.parquet` path. |
| `inputs.markets_path` | Reserved metadata path. The current pipeline does not depend on it. |

### Output Parameters

| Parameter | Meaning |
|---|---|
| `output.root` | Root directory for generated outputs. |
| `output.prepared_dirname` | Directory name for stage 1 output. |
| `output.labeled_dirname` | Directory name for stage 2 output. |
| `output.sequence_dirname` | Directory name for stage 3 output. |
| `output.manifest_dirname` | Directory name for stage manifests. |

### Runtime Parameters

| Parameter | Meaning |
|---|---|
| `runtime.threads` | DuckDB thread count. |
| `runtime.memory_limit` | DuckDB memory cap. |
| `runtime.temp_directory` | Scratch spill directory for DuckDB. |
| `runtime.preserve_insertion_order` | DuckDB insertion-order behavior. |

### Label Parameters

| Parameter | Meaning |
|---|---|
| `label.horizon_minutes` | Forward horizon used for price lookup and labeling. |
| `label.contract_addresses` | Taker addresses filtered out as internal contract activity. |
| `label.usd_decimals` | Decimal scaling used on USD amounts. |
| `label.token_decimals` | Decimal scaling used on token amounts. |

### Sequence Parameters

| Parameter | Meaning |
|---|---|
| `sequence.length` | Intended number of time steps per sequence. |
| `sequence.stride` | Step size between successive emitted windows. |
| `sequence.feature_order` | Ordered list of numeric features per step. |
| `sequence.train_ratio` | Fraction of examples allocated to train by time. |
| `sequence.validation_ratio` | Fraction allocated to validation by time. |
| `sequence.test_ratio` | Fraction allocated to test by time. |

### Smoke Parameters

| Parameter | Meaning |
|---|---|
| `smoke.start` | Inclusive start timestamp for smoke runs. |
| `smoke.end` | Exclusive end timestamp for smoke runs. |

## How To Use The Output

### Quick Schema Inspection

```bash
ssh juno.utdallas.edu
cd "/home/axa230262/work/001 research/insider"
. .venv/bin/activate
python -m insider run-pipeline --config configs/pipeline.toml --overwrite
```

### Load Final Sequences With DuckDB

```sql
SELECT sequence_id, user, market_id, seq_len, label, split
FROM read_parquet('/home/axa230262/scratch/insider/processed/sequence_dataset/**/*.parquet')
WHERE length(features) = 64
LIMIT 10;
```

### Load Final Sequences With PyArrow Or Python

Use only exact-width sequences for sequence models:

```python
import duckdb

con = duckdb.connect()
rows = con.execute("""
    SELECT features, label
    FROM read_parquet('/home/axa230262/scratch/insider/processed/sequence_dataset/**/*.parquet')
    WHERE length(features) = 64
""").fetchall()
```

### Recommended Training Input Contract

For the current run, the safest model input contract is:

- `X`: `64 x 7` float tensor from `features`
- `y`: scalar binary label from `label`
- Split using the stored `split` column

## Current Verdict

The output is mostly correct and the classification signal is implemented correctly.

What is verified:

- trade-stage formatting is valid
- user-label stage formatting is valid
- positive/negative label assignment is mathematically consistent
- maker/taker side symmetry is correct
- final splits are populated and time-based

What still needs cleanup before strict model training:

- `2,560` final sequence rows do not contain a full 64-step payload even though `seq_len = 64`

Until that is fixed in the pipeline, filter with `WHERE length(features) = 64` when training.
