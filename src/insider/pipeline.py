from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

from insider.config import PipelineConfig
from insider.duckdb_utils import connect
from modeling_common.provenance import build_stage_provenance


def prepare_trades(
    config: PipelineConfig,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    overwrite: bool = False,
) -> Path:
    prepared_dir = _stage_dir(config, config.output.prepared_dirname, overwrite=overwrite)
    connection = connect(config.runtime, config.output.root)
    try:
        time_filter = _raw_time_filter(start, end)
        contracts = _sql_string_list(config.label.contract_addresses)
        token_scale = float(10**config.label.token_decimals)
        usd_scale = float(10**config.label.usd_decimals)
        query = f"""
        COPY (
            WITH source AS (
                SELECT
                    CAST(timestamp AS BIGINT) AS timestamp,
                    datetime,
                    block_number,
                    transaction_hash,
                    contract,
                    event_id,
                    event_slug,
                    event_title,
                    market_id,
                    condition_id,
                    question,
                    nonusdc_side,
                    maker,
                    taker,
                    maker_direction,
                    taker_direction,
                    CAST(price AS DOUBLE) AS price,
                    CAST(token_amount AS DOUBLE) AS token_amount_raw,
                    CAST(usd_amount AS DOUBLE) AS usd_amount_raw,
                    asset_id,
                    order_hash
                FROM read_parquet('{_sql_escape(config.inputs.trades_path)}')
                WHERE price IS NOT NULL
                  AND asset_id IS NOT NULL
                  AND lower(taker) NOT IN ({contracts})
                  {time_filter}
            )
            SELECT
                timestamp,
                datetime,
                block_number,
                transaction_hash,
                contract,
                event_id,
                event_slug,
                event_title,
                market_id,
                condition_id,
                question,
                lower(COALESCE(nonusdc_side, '')) AS nonusdc_side,
                maker,
                taker,
                maker_direction,
                taker_direction,
                CASE
                    WHEN lower(COALESCE(nonusdc_side, '')) = 'token2' THEN 1.0 - price
                    ELSE price
                END AS price_yes,
                token_amount_raw / {token_scale} AS token_amount,
                usd_amount_raw / {usd_scale} AS usd_amount,
                asset_id,
                order_hash,
                COALESCE(
                    try_strptime(datetime, '%Y-%m-%d %H:%M:%S'),
                    CAST(to_timestamp(timestamp) AS TIMESTAMP)
                ) AS trade_time,
                strftime(
                    COALESCE(
                        try_strptime(datetime, '%Y-%m-%d %H:%M:%S'),
                        CAST(to_timestamp(timestamp) AS TIMESTAMP)
                    ),
                    '%Y-%m'
                ) AS year_month
            FROM source
        )
        TO '{_sql_escape(prepared_dir)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (year_month))
        """
        connection.execute(query)
        _write_stage_manifest(
            config,
            stage_name="prepare_trades",
            dataset_dir=prepared_dir,
            time_column="trade_time",
            extras={
                "start": _isoformat(start),
                "end": _isoformat(end),
                "trades_path": str(config.inputs.trades_path),
            },
        )
        return prepared_dir
    finally:
        connection.close()


def label_user_trades(config: PipelineConfig, *, overwrite: bool = False) -> Path:
    prepared_glob = _dataset_glob(config.output.root / config.output.prepared_dirname)
    labeled_dir = _stage_dir(config, config.output.labeled_dirname, overwrite=overwrite)
    connection = connect(config.runtime, config.output.root)
    try:
        horizon = config.label.horizon_minutes
        future_lag_filter = ""
        if config.label.max_future_lag_seconds is not None:
            future_lag_filter = f"AND future_lag_seconds <= {int(config.label.max_future_lag_seconds)}"
        ambiguous_markout_filter = ""
        if float(config.label.min_abs_markout_bps) > 0.0:
            ambiguous_markout_filter = f"AND ABS(markout_bps) >= {float(config.label.min_abs_markout_bps)}"
        query = f"""
        COPY (
            WITH prepared AS (
                SELECT *
                FROM read_parquet('{_sql_escape(prepared_glob)}')
                WHERE trade_time IS NOT NULL
                  AND price_yes IS NOT NULL
            ),
            future_prices AS (
                SELECT
                    asset_id,
                    trade_time AS future_trade_time,
                    price_yes AS future_price_yes,
                    -epoch_ms(trade_time) AS neg_future_ms
                FROM prepared
            ),
            priced AS (
                SELECT
                    t.*,
                    t.trade_time + INTERVAL {horizon} MINUTE AS target_time,
                    fp.future_trade_time,
                    fp.future_price_yes,
                    CASE
                        WHEN fp.future_trade_time IS NULL THEN NULL
                        ELSE epoch(fp.future_trade_time) - epoch(t.trade_time + INTERVAL {horizon} MINUTE)
                    END AS future_lag_seconds
                FROM (
                    SELECT
                        *,
                        -epoch_ms(trade_time + INTERVAL {horizon} MINUTE) AS neg_target_ms
                    FROM prepared
                ) AS t
                ASOF LEFT JOIN future_prices AS fp
                  ON t.asset_id = fp.asset_id
                 AND t.neg_target_ms >= fp.neg_future_ms
            ),
            user_rows AS (
                SELECT
                    timestamp,
                    datetime,
                    block_number,
                    transaction_hash,
                    contract,
                    event_id,
                    event_slug,
                    event_title,
                    market_id,
                    condition_id,
                    question,
                    asset_id,
                    order_hash,
                    trade_time,
                    target_time,
                    future_trade_time,
                    future_lag_seconds,
                    price_yes,
                    future_price_yes,
                    usd_amount,
                    token_amount,
                    maker AS user,
                    'maker' AS role,
                    maker_direction AS direction,
                    {_side_sql('maker_direction')} AS side
                FROM priced
                UNION ALL
                SELECT
                    timestamp,
                    datetime,
                    block_number,
                    transaction_hash,
                    contract,
                    event_id,
                    event_slug,
                    event_title,
                    market_id,
                    condition_id,
                    question,
                    asset_id,
                    order_hash,
                    trade_time,
                    target_time,
                    future_trade_time,
                    future_lag_seconds,
                    price_yes,
                    future_price_yes,
                    usd_amount,
                    token_amount,
                    taker AS user,
                    'taker' AS role,
                    taker_direction AS direction,
                    {_side_sql('taker_direction')} AS side
                FROM priced
            ),
            scored AS (
                SELECT
                    *,
                    side * token_amount AS signed_token_amount,
                    side * (future_price_yes - price_yes) AS markout,
                    CASE
                        WHEN price_yes IS NOT NULL AND price_yes <> 0 AND future_price_yes IS NOT NULL
                        THEN side * ((future_price_yes / price_yes) - 1.0) * 10000.0
                        ELSE NULL
                    END AS markout_bps,
                    CASE
                        WHEN side * (future_price_yes - price_yes) > 0 THEN 1
                        ELSE 0
                    END AS label,
                    strftime(trade_time, '%Y-%m') AS year_month
                FROM user_rows
            )
            SELECT *
            FROM scored
            WHERE side IS NOT NULL
              AND future_price_yes IS NOT NULL
              {future_lag_filter}
              {ambiguous_markout_filter}
        )
        TO '{_sql_escape(labeled_dir)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (year_month))
        """
        connection.execute(query)
        _write_stage_manifest(
            config,
            stage_name="label_user_trades",
            dataset_dir=labeled_dir,
            time_column="trade_time",
            extras={
                "horizon_minutes": config.label.horizon_minutes,
                "max_future_lag_seconds": config.label.max_future_lag_seconds,
                "min_abs_markout_bps": config.label.min_abs_markout_bps,
            },
        )
        return labeled_dir
    finally:
        connection.close()


def build_sequences(config: PipelineConfig, *, overwrite: bool = False) -> Path:
    labeled_glob = _dataset_glob(config.output.root / config.output.labeled_dirname)
    sequence_dir = _stage_dir(config, config.output.sequence_dirname, overwrite=overwrite)
    connection = connect(config.runtime, config.output.root)
    try:
        train_cutoff, validation_cutoff = _compute_split_cutoffs(connection, labeled_glob, config)
        feature_vector_sql = _feature_vector_sql(config.sequence.feature_order)
        window_size = config.sequence.length - 1
        query = f"""
        COPY (
            WITH enriched AS (
                SELECT
                    *,
                    COALESCE(
                        epoch(trade_time) - epoch(lag(trade_time) OVER user_market_window),
                        0.0
                    ) AS time_delta_seconds,
                    epoch(trade_time) - epoch(first_value(trade_time) OVER user_market_window) AS market_age_seconds,
                    CASE WHEN role = 'maker' THEN 1.0 ELSE 0.0 END AS role_is_maker,
                    CAST(count(*) OVER market_hour_window AS DOUBLE) / 2.0 AS market_trade_count_1h,
                    COALESCE(sum(usd_amount) OVER market_hour_window / 2.0, 0.0) AS market_volume_1h,
                    COALESCE(avg(price_yes) OVER market_hour_window, price_yes, 0.0) AS market_price_mean_1h,
                    COALESCE(stddev_pop(price_yes) OVER market_hour_window, 0.0) AS market_price_std_1h,
                    COALESCE(price_yes - first_value(price_yes) OVER market_hour_window, 0.0) AS market_price_return_1h,
                    CAST(count(*) OVER user_hour_window AS DOUBLE) AS user_trade_count_1h,
                    CAST(count(*) OVER user_market_hour_window AS DOUBLE) AS user_market_trade_count_1h,
                    COALESCE(sum(signed_token_amount) OVER user_hour_window, 0.0) AS user_signed_flow_1h,
                    COALESCE(sum(usd_amount) OVER user_hour_window, 0.0) AS user_usd_volume_1h,
                    row_number() OVER user_market_window AS sequence_row_number
                FROM read_parquet('{_sql_escape(labeled_glob)}')
                WINDOW user_market_window AS (
                    PARTITION BY user, market_id
                    ORDER BY trade_time
                ),
                market_hour_window AS (
                    PARTITION BY market_id
                    ORDER BY trade_time
                    RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW
                ),
                user_hour_window AS (
                    PARTITION BY user
                    ORDER BY trade_time
                    RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW
                ),
                user_market_hour_window AS (
                    PARTITION BY user, market_id
                    ORDER BY trade_time
                    RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW
                )
            ),
            windowed AS (
                SELECT
                    user,
                    market_id,
                    min(trade_time) OVER frame_window AS sequence_start_ts,
                    trade_time AS sequence_end_ts,
                    sequence_row_number,
                    list({feature_vector_sql}) OVER frame_window AS features,
                    label
                FROM enriched
                WINDOW frame_window AS (
                    PARTITION BY user, market_id
                    ORDER BY trade_time
                    ROWS BETWEEN {window_size} PRECEDING AND CURRENT ROW
                )
            )
            ,
            assigned AS (
                SELECT
                    concat_ws(
                        ':',
                        user,
                        market_id,
                        strftime(sequence_end_ts, '%Y%m%d%H%M%S'),
                        lpad(CAST(sequence_row_number AS VARCHAR), 8, '0')
                    ) AS sequence_id,
                    user,
                    market_id,
                    sequence_start_ts,
                    sequence_end_ts,
                    {config.sequence.length} AS seq_len,
                    features,
                    label,
                    {_split_assignment_sql(
                        "sequence_start_ts",
                        "sequence_end_ts",
                        train_cutoff=train_cutoff,
                        validation_cutoff=validation_cutoff,
                        purge_minutes=config.sequence.purge_minutes,
                        embargo_minutes=config.sequence.embargo_minutes,
                    )} AS split
                FROM windowed
                WHERE sequence_row_number >= {config.sequence.length}
                  AND ((sequence_row_number - {config.sequence.length}) % {config.sequence.stride}) = 0
            )
            SELECT
                *
            FROM assigned
            WHERE split IS NOT NULL
        )
        TO '{_sql_escape(sequence_dir)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (split))
        """
        connection.execute(query)
        _write_stage_manifest(
            config,
            stage_name="build_sequences",
            dataset_dir=sequence_dir,
            time_column="sequence_end_ts",
            extras={
                "feature_order": list(config.sequence.feature_order),
                "train_cutoff_epoch": train_cutoff,
                "validation_cutoff_epoch": validation_cutoff,
                "sequence_length": config.sequence.length,
                "stride": config.sequence.stride,
                "purge_minutes": config.sequence.purge_minutes,
                "embargo_minutes": config.sequence.embargo_minutes,
            },
        )
        return sequence_dir
    finally:
        connection.close()


def build_model_windows(
    config: PipelineConfig,
    *,
    window_size: int | None = None,
    stride: int | None = None,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    labeled_glob = _dataset_glob(config.output.root / config.output.labeled_dirname)
    resolved_window_size = config.model_windows.length if window_size is None else int(window_size)
    resolved_stride = config.model_windows.stride if stride is None else int(stride)
    if resolved_window_size < 1 or resolved_stride < 1:
        raise ValueError("Model window length and stride must be positive integers.")

    stage_dirname = f"{config.output.model_window_dirname}/window_size={resolved_window_size}"
    window_dir = _stage_dir(
        config,
        stage_dirname,
        overwrite=overwrite,
        output_dir=output_dir,
    )
    connection = connect(config.runtime, config.output.root)
    try:
        train_cutoff, validation_cutoff = _compute_model_window_split_cutoffs(
            connection,
            labeled_glob,
            config,
            window_size=resolved_window_size,
            stride=resolved_stride,
        )
        feature_vector_sql = _feature_vector_sql(config.model_windows.feature_order)
        frame_size = resolved_window_size - 1
        query = f"""
        COPY (
            WITH enriched AS (
                SELECT
                    *,
                    COALESCE(
                        epoch(trade_time) - epoch(lag(trade_time) OVER user_market_window),
                        0.0
                    ) AS time_delta_seconds,
                    epoch(trade_time) - epoch(first_value(trade_time) OVER user_market_window) AS market_age_seconds,
                    CASE WHEN role = 'maker' THEN 1.0 ELSE 0.0 END AS role_is_maker,
                    CAST(count(*) OVER market_hour_window AS DOUBLE) / 2.0 AS market_trade_count_1h,
                    COALESCE(sum(usd_amount) OVER market_hour_window / 2.0, 0.0) AS market_volume_1h,
                    COALESCE(avg(price_yes) OVER market_hour_window, price_yes, 0.0) AS market_price_mean_1h,
                    COALESCE(stddev_pop(price_yes) OVER market_hour_window, 0.0) AS market_price_std_1h,
                    COALESCE(price_yes - first_value(price_yes) OVER market_hour_window, 0.0) AS market_price_return_1h,
                    CAST(count(*) OVER user_hour_window AS DOUBLE) AS user_trade_count_1h,
                    CAST(count(*) OVER user_market_hour_window AS DOUBLE) AS user_market_trade_count_1h,
                    COALESCE(sum(signed_token_amount) OVER user_hour_window, 0.0) AS user_signed_flow_1h,
                    COALESCE(sum(usd_amount) OVER user_hour_window, 0.0) AS user_usd_volume_1h,
                    row_number() OVER user_market_window AS window_row_number
                FROM read_parquet('{_sql_escape(labeled_glob)}')
                WINDOW user_market_window AS (
                    PARTITION BY user, market_id
                    ORDER BY trade_time
                ),
                market_hour_window AS (
                    PARTITION BY market_id
                    ORDER BY trade_time
                    RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW
                ),
                user_hour_window AS (
                    PARTITION BY user
                    ORDER BY trade_time
                    RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW
                ),
                user_market_hour_window AS (
                    PARTITION BY user, market_id
                    ORDER BY trade_time
                    RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW
                )
            ),
            windowed AS (
                SELECT
                    user,
                    market_id,
                    min(trade_time) OVER frame_window AS window_start_ts,
                    trade_time AS window_end_ts,
                    window_row_number,
                    list({feature_vector_sql}) OVER frame_window AS features,
                    label
                FROM enriched
                WINDOW frame_window AS (
                    PARTITION BY user, market_id
                    ORDER BY trade_time
                    ROWS BETWEEN {frame_size} PRECEDING AND CURRENT ROW
                )
            )
            ,
            assigned AS (
                SELECT
                    concat_ws(
                        ':',
                        user,
                        market_id,
                        strftime(window_end_ts, '%Y%m%d%H%M%S'),
                        lpad(CAST(window_row_number AS VARCHAR), 8, '0')
                    ) AS window_id,
                    user,
                    market_id,
                    window_start_ts,
                    window_end_ts,
                    {resolved_window_size} AS window_size,
                    features,
                    label,
                    {_split_assignment_sql(
                        "window_start_ts",
                        "window_end_ts",
                        train_cutoff=train_cutoff,
                        validation_cutoff=validation_cutoff,
                        purge_minutes=config.model_windows.purge_minutes,
                        embargo_minutes=config.model_windows.embargo_minutes,
                    )} AS split
                FROM windowed
                WHERE window_row_number >= {resolved_window_size}
                  AND ((window_row_number - {resolved_window_size}) % {resolved_stride}) = 0
                  AND length(features) = {resolved_window_size}
            )
            SELECT
                *
            FROM assigned
            WHERE split IS NOT NULL
        )
        TO '{_sql_escape(window_dir)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (split))
        """
        connection.execute(query)
        _write_stage_manifest(
            config,
            stage_name=(
                f"build_model_windows_{resolved_window_size}_stride_{resolved_stride}"
                if output_dir is not None or resolved_stride != config.model_windows.stride
                else f"build_model_windows_{resolved_window_size}"
            ),
            dataset_dir=window_dir,
            time_column="window_end_ts",
            extras={
                "model_window": {
                    "length": resolved_window_size,
                    "stride": resolved_stride,
                    "feature_order": list(config.model_windows.feature_order),
                    "purge_minutes": config.model_windows.purge_minutes,
                    "embargo_minutes": config.model_windows.embargo_minutes,
                },
                "train_cutoff_epoch": train_cutoff,
                "validation_cutoff_epoch": validation_cutoff,
            },
        )
        return window_dir
    finally:
        connection.close()


def run_pipeline(
    config: PipelineConfig,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    overwrite: bool = False,
) -> dict[str, str]:
    prepared = prepare_trades(config, start=start, end=end, overwrite=overwrite)
    labeled = label_user_trades(config, overwrite=overwrite)
    sequences = build_sequences(config, overwrite=overwrite)
    return {
        "prepared_trades": str(prepared),
        "labeled_user_trades": str(labeled),
        "sequence_dataset": str(sequences),
    }


def smoke_test(config: PipelineConfig, *, overwrite: bool = False) -> dict[str, str]:
    if config.smoke.start is None or config.smoke.end is None:
        raise ValueError("Smoke test requires [smoke].start and [smoke].end in the config.")
    smoke_config = config.with_output_root(config.output.root / "smoke")
    return run_pipeline(
        smoke_config,
        start=config.smoke.start,
        end=config.smoke.end,
        overwrite=overwrite,
    )


def _compute_split_cutoffs(connection: Any, labeled_glob: str, config: PipelineConfig) -> tuple[float, float]:
    row = connection.execute(
        f"""
        SELECT
            quantile_cont(epoch(trade_time), {config.sequence.train_ratio}) AS train_cutoff,
            quantile_cont(
                epoch(trade_time),
                {config.sequence.train_ratio + config.sequence.validation_ratio}
            ) AS validation_cutoff
        FROM read_parquet('{_sql_escape(labeled_glob)}')
        """
    ).fetchone()
    if row is None or row[0] is None or row[1] is None:
        raise ValueError("Unable to compute split cutoffs from labeled dataset.")
    return float(row[0]), float(row[1])


def _compute_model_window_split_cutoffs(
    connection: Any,
    labeled_glob: str,
    config: PipelineConfig,
    *,
    window_size: int,
    stride: int,
) -> tuple[float, float]:
    row = connection.execute(
        f"""
        WITH window_candidates AS (
            SELECT
                trade_time AS window_end_ts,
                row_number() OVER (
                    PARTITION BY user, market_id
                    ORDER BY trade_time
                ) AS window_row_number
            FROM read_parquet('{_sql_escape(labeled_glob)}')
        )
        SELECT
            quantile_cont(epoch(window_end_ts), {config.model_windows.train_ratio}) AS train_cutoff,
            quantile_cont(
                epoch(window_end_ts),
                {config.model_windows.train_ratio + config.model_windows.validation_ratio}
            ) AS validation_cutoff
        FROM window_candidates
        WHERE window_row_number >= {window_size}
          AND ((window_row_number - {window_size}) % {stride}) = 0
        """
    ).fetchone()
    if row is None or row[0] is None or row[1] is None:
        raise ValueError("Unable to compute split cutoffs from model windows.")
    return float(row[0]), float(row[1])


def _feature_vector_sql(feature_order: tuple[str, ...]) -> str:
    sql_by_feature = {
        "price_yes": "CAST(price_yes AS DOUBLE)",
        "signed_token_amount": "CAST(signed_token_amount AS DOUBLE)",
        "usd_amount": "CAST(usd_amount AS DOUBLE)",
        "side": "CAST(side AS DOUBLE)",
        "role_is_maker": "role_is_maker",
        "time_delta_seconds": "CAST(time_delta_seconds AS DOUBLE)",
        "market_age_seconds": "CAST(market_age_seconds AS DOUBLE)",
        "market_trade_count_1h": "CAST(market_trade_count_1h AS DOUBLE)",
        "market_volume_1h": "CAST(market_volume_1h AS DOUBLE)",
        "market_price_mean_1h": "CAST(market_price_mean_1h AS DOUBLE)",
        "market_price_std_1h": "CAST(market_price_std_1h AS DOUBLE)",
        "market_price_return_1h": "CAST(market_price_return_1h AS DOUBLE)",
        "user_trade_count_1h": "CAST(user_trade_count_1h AS DOUBLE)",
        "user_market_trade_count_1h": "CAST(user_market_trade_count_1h AS DOUBLE)",
        "user_signed_flow_1h": "CAST(user_signed_flow_1h AS DOUBLE)",
        "user_usd_volume_1h": "CAST(user_usd_volume_1h AS DOUBLE)",
    }
    missing = [feature for feature in feature_order if feature not in sql_by_feature]
    if missing:
        raise ValueError(f"Unsupported sequence features requested: {missing}")
    ordered_sql = ", ".join(sql_by_feature[feature] for feature in feature_order)
    return f"list_value({ordered_sql})"


def _split_assignment_sql(
    start_column: str,
    end_column: str,
    *,
    train_cutoff: float,
    validation_cutoff: float,
    purge_minutes: int,
    embargo_minutes: int,
) -> str:
    purge_seconds = max(int(purge_minutes), 0) * 60.0
    embargo_seconds = max(int(embargo_minutes), 0) * 60.0
    if purge_seconds == 0.0 and embargo_seconds == 0.0:
        return f"""
        CASE
            WHEN epoch({end_column}) <= {train_cutoff} THEN 'train'
            WHEN epoch({end_column}) <= {validation_cutoff} THEN 'validation'
            ELSE 'test'
        END
        """
    train_end_epoch = train_cutoff - purge_seconds
    validation_start_epoch = train_cutoff + embargo_seconds
    validation_end_epoch = validation_cutoff - purge_seconds
    test_start_epoch = validation_cutoff + embargo_seconds
    return f"""
    CASE
        WHEN epoch({end_column}) <= {train_end_epoch} THEN 'train'
        WHEN epoch({start_column}) >= {validation_start_epoch}
         AND epoch({end_column}) <= {validation_end_epoch} THEN 'validation'
        WHEN epoch({start_column}) >= {test_start_epoch} THEN 'test'
        ELSE NULL
    END
    """


def _stage_dir(
    config: PipelineConfig,
    stage_dirname: str,
    *,
    overwrite: bool,
    output_dir: Path | None = None,
) -> Path:
    stage_dir = output_dir if output_dir is not None else (config.output.root / stage_dirname)
    if stage_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{stage_dir} already exists. Re-run with --overwrite to replace it.")
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    (config.output.root / config.output.manifest_dirname).mkdir(parents=True, exist_ok=True)
    return stage_dir


def _write_stage_manifest(
    config: PipelineConfig,
    *,
    stage_name: str,
    dataset_dir: Path,
    time_column: str,
    extras: dict[str, Any],
) -> None:
    connection = connect(config.runtime, config.output.root)
    try:
        dataset_glob = _dataset_glob(dataset_dir)
        row = connection.execute(
            f"""
            SELECT
                COUNT(*) AS row_count,
                MIN({time_column}) AS min_trade_time,
                MAX({time_column}) AS max_trade_time
            FROM read_parquet('{_sql_escape(dataset_glob)}')
            """
        ).fetchone()
    finally:
        connection.close()

    manifest = {
        "stage": stage_name,
        "dataset_dir": str(dataset_dir),
        "row_count": int(row[0]) if row and row[0] is not None else 0,
        "min_trade_time": _isoformat(row[1] if row else None),
        "max_trade_time": _isoformat(row[2] if row else None),
        "runtime": asdict(config.runtime),
        "label": asdict(config.label),
        "sequence": asdict(config.sequence),
        "model_windows": asdict(config.model_windows),
        "provenance": build_stage_provenance(
            config_path=config.source_config_path,
            output_root=config.output.root,
        ),
        **extras,
    }
    manifest_path = config.output.root / config.output.manifest_dirname / f"{stage_name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")


def _raw_time_filter(start: datetime | None, end: datetime | None) -> str:
    clauses: list[str] = []
    if start is not None:
        clauses.append(f"CAST(timestamp AS BIGINT) >= {int(_to_epoch(start))}")
    if end is not None:
        clauses.append(f"CAST(timestamp AS BIGINT) < {int(_to_epoch(end))}")
    if not clauses:
        return ""
    return "AND " + " AND ".join(clauses)


def _side_sql(column_name: str) -> str:
    direction = f"lower(trim(COALESCE({column_name}, '')))"
    return f"""
    CASE
        WHEN {direction} IN ('buy', 'bid', 'b', 'long')
            OR contains({direction}, 'buy')
            OR contains({direction}, 'bid')
            OR starts_with({direction}, '+')
        THEN 1
        WHEN {direction} IN ('sell', 'ask', 's', 'short')
            OR contains({direction}, 'sell')
            OR contains({direction}, 'ask')
            OR starts_with({direction}, '-')
        THEN -1
        ELSE NULL
    END
    """


def _sql_string_list(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{_sql_escape(value)}'" for value in values)


def _dataset_glob(dataset_dir: Path) -> str:
    return str(dataset_dir / "**" / "*.parquet")


def _sql_escape(value: Path | str) -> str:
    return str(value).replace("'", "''")


def _isoformat(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _to_epoch(value: datetime) -> float:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()
