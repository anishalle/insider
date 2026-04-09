# Pipeline

The preprocessing job is split into three disk-backed stages so the full dataset never has to be materialized in RAM.

1. `prepare-trades`
   - Reads `trades.parquet`
   - Filters invalid rows and known contract takers
   - Normalizes prices into the YES/token1 perspective
   - Writes partitioned parquet shards by `year_month`

2. `label-user-trades`
   - Computes the 5-minute forward markout per `asset_id`
   - Explodes each trade into maker and taker user rows
   - Adds `future_lag_seconds`, `markout`, `markout_bps`, `label`, and signed token amounts
   - Can drop stale future-price matches and near-zero markouts through `[label]` config

3. `build-sequences`
   - Orders rows by `user, market_id, trade_time`
   - Builds fixed-length sliding windows with optional market-state and user-history context features
   - Assigns time-based train/validation/test splits, with optional purge and embargo windows around boundaries
   - Writes partitioned sequence shards by split

4. `build-model-windows`
   - Rebuilds exact-width model windows from `labeled_user_trades`
   - Uses the configured feature order and time-based split ratios
   - Drops windows that straddle split boundaries when purge and embargo settings are enabled
   - Writes reusable training windows under `model_windows/window_size=<N>`

Stage manifests are written under `processed/manifests/` and include row counts plus time coverage for resumable monitoring.
