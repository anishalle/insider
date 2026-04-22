# Pipeline

The preprocessing job is split into four disk-backed stages so the full dataset never has to be materialized in RAM.

1. `prepare-trades`
   - Reads the raw `trades.parquet` source.
   - Filters invalid rows and known contract takers.
   - Normalizes prices into the YES/token1 perspective.
   - Writes partitioned parquet shards by `year_month`.

2. `label-user-trades`
   - Computes the 5-minute forward markout per `asset_id`.
   - Explodes each trade into maker and taker user rows.
   - Adds `future_lag_seconds`, `markout`, `markout_bps`, `label`, and signed token amounts.
   - Applies the label-quality filters from `[label]`, including stale-match and near-zero-markout drops.

3. `build-sequences`
   - Orders rows by `user, market_id, trade_time`.
   - Builds fixed-length sliding windows with market-state and user-history context features.
   - Assigns time-based `train`, `validation`, and `test` splits with optional purge and embargo windows.
   - Writes partitioned sequence shards by split.

4. `build-model-windows`
   - Rebuilds exact-width model windows from `labeled_user_trades`.
   - Uses the configured feature order and time-based split ratios.
   - Drops windows that straddle split boundaries when purge and embargo settings are enabled.
   - Writes reusable training windows under `model_windows/window_size=<N>`.

`run-pipeline` runs stages 1 through 3 only. `build-model-windows` stays separate on purpose so model experiments can reuse the same exact-width dataset without rerunning the full preprocess path.

Stage manifests are written under `processed/manifests/` and include row counts, time coverage, config snapshots, and split cutoffs. Derived reports such as signal-review bundles, model leaderboards, and audit CSVs should be written under `processed/reports/` rather than inside the repo checkout.
