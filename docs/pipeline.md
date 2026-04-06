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
   - Adds `markout`, `markout_bps`, `label`, and signed token amounts

3. `build-sequences`
   - Orders rows by `user, market_id, trade_time`
   - Builds fixed-length sliding windows
   - Assigns time-based train/validation/test splits
   - Writes partitioned sequence shards by split

Stage manifests are written under `processed/manifests/` and include row counts plus time coverage for resumable monitoring.
