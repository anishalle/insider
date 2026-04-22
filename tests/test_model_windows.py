from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from insider.config import load_config
from insider.pipeline import build_model_windows, label_user_trades, prepare_trades


def test_build_model_windows_emits_exact_width_splits(tmp_path: Path) -> None:
    trades_path = tmp_path / "trades.parquet"
    output_root = tmp_path / "outputs"
    config_path = tmp_path / "pipeline.toml"

    table = pa.table(
        {
            "timestamp": [
                1_704_067_200,
                1_704_067_500,
                1_704_067_800,
                1_704_068_100,
                1_704_068_400,
                1_704_068_700,
            ],
            "datetime": [
                "2024-01-01 00:00:00",
                "2024-01-01 00:05:00",
                "2024-01-01 00:10:00",
                "2024-01-01 00:15:00",
                "2024-01-01 00:20:00",
                "2024-01-01 00:25:00",
            ],
            "block_number": [1, 2, 3, 4, 5, 6],
            "transaction_hash": ["tx1", "tx2", "tx3", "tx4", "tx5", "tx6"],
            "contract": ["exchange"] * 6,
            "event_id": ["event-1"] * 6,
            "event_slug": ["election-2024"] * 6,
            "event_title": ["Election"] * 6,
            "market_id": ["market-1"] * 6,
            "condition_id": ["condition-1"] * 6,
            "question": ["Will X win?"] * 6,
            "nonusdc_side": ["token1"] * 6,
            "maker": ["user-a"] * 6,
            "taker": ["user-b"] * 6,
            "maker_direction": ["BUY"] * 6,
            "taker_direction": ["SELL"] * 6,
            "price": [0.40, 0.55, 0.45, 0.60, 0.58, 0.62],
            "token_amount": [1_000_000, 1_500_000, 1_200_000, 900_000, 1_100_000, 800_000],
            "usd_amount": [400_000, 825_000, 540_000, 540_000, 638_000, 496_000],
            "asset_id": ["asset-1"] * 6,
            "order_hash": ["o1", "o2", "o3", "o4", "o5", "o6"],
        }
    )
    pq.write_table(table, trades_path)

    config_path.write_text(
        "\n".join(
            [
                "[inputs]",
                f'trades_path = "{trades_path}"',
                "",
                "[output]",
                f'root = "{output_root}"',
                "",
                "[runtime]",
                "threads = 4",
                'memory_limit = "1GB"',
                f'temp_directory = "{tmp_path / "tmp"}"',
                "",
                "[label]",
                "horizon_minutes = 5",
                "",
                "[model_windows]",
                "length = 2",
                "stride = 1",
                'feature_order = ["price_yes", "signed_token_amount", "usd_amount", "side", "role_is_maker", "time_delta_seconds", "market_age_seconds"]',
                "train_ratio = 0.5",
                "validation_ratio = 0.25",
                "test_ratio = 0.25",
            ]
        )
        + "\n"
    )

    config = load_config(config_path)
    prepare_trades(config, overwrite=True)
    label_user_trades(config, overwrite=True)
    build_model_windows(config, overwrite=True)

    connection = duckdb.connect()
    try:
        rows = connection.execute(
            f"""
            SELECT split, COUNT(*) AS row_count, MIN(window_size), MAX(length(features)), MAX(length(features[1]))
            FROM read_parquet('{output_root / "model_windows" / "window_size=2" / "**" / "*.parquet"}')
            GROUP BY split
            ORDER BY split
            """
        ).fetchall()
        assert rows == [
            ("test", 2, 2, 2, 7),
            ("train", 4, 2, 2, 7),
            ("validation", 2, 2, 2, 7),
        ]

        maker_window = connection.execute(
            f"""
            SELECT features, label
            FROM read_parquet('{output_root / "model_windows" / "window_size=2" / "split=train" / "*.parquet"}')
            WHERE user = 'user-a'
            ORDER BY window_end_ts
            LIMIT 1
            """
        ).fetchone()
        assert maker_window is not None
        assert maker_window[0][0] == [0.4, 1.0, 0.4, 1.0, 1.0, 0.0, 0.0]
        assert maker_window[0][1] == [0.55, 1.5, 0.825, 1.0, 1.0, 300.0, 300.0]
        assert maker_window[1] == 0
    finally:
        connection.close()

    manifest = json.loads((output_root / "manifests" / "build_model_windows_2.json").read_text())
    assert manifest["model_window"]["length"] == 2
    assert manifest["model_window"]["stride"] == 1


def test_build_model_windows_can_emit_market_and_user_context_features(tmp_path: Path) -> None:
    trades_path = tmp_path / "trades.parquet"
    output_root = tmp_path / "outputs"
    config_path = tmp_path / "pipeline.toml"

    table = pa.table(
        {
            "timestamp": [
                1_704_067_200,
                1_704_067_500,
                1_704_067_800,
                1_704_068_100,
            ],
            "datetime": [
                "2024-01-01 00:00:00",
                "2024-01-01 00:05:00",
                "2024-01-01 00:10:00",
                "2024-01-01 00:15:00",
            ],
            "block_number": [1, 2, 3, 4],
            "transaction_hash": ["tx1", "tx2", "tx3", "tx4"],
            "contract": ["exchange"] * 4,
            "event_id": ["event-1"] * 4,
            "event_slug": ["election-2024"] * 4,
            "event_title": ["Election"] * 4,
            "market_id": ["market-1"] * 4,
            "condition_id": ["condition-1"] * 4,
            "question": ["Will X win?"] * 4,
            "nonusdc_side": ["token1"] * 4,
            "maker": ["user-a"] * 4,
            "taker": ["user-b"] * 4,
            "maker_direction": ["BUY"] * 4,
            "taker_direction": ["SELL"] * 4,
            "price": [0.40, 0.55, 0.45, 0.60],
            "token_amount": [1_000_000, 1_500_000, 1_200_000, 900_000],
            "usd_amount": [400_000, 825_000, 540_000, 540_000],
            "asset_id": ["asset-1"] * 4,
            "order_hash": ["o1", "o2", "o3", "o4"],
        }
    )
    pq.write_table(table, trades_path)

    config_path.write_text(
        "\n".join(
            [
                "[inputs]",
                f'trades_path = "{trades_path}"',
                "",
                "[output]",
                f'root = "{output_root}"',
                "",
                "[runtime]",
                "threads = 4",
                'memory_limit = "1GB"',
                f'temp_directory = "{tmp_path / "tmp"}"',
                "",
                "[label]",
                "horizon_minutes = 5",
                "",
                "[model_windows]",
                "length = 2",
                "stride = 1",
                'feature_order = ["price_yes", "signed_token_amount", "usd_amount", "side", "role_is_maker", "time_delta_seconds", "market_age_seconds", "market_trade_count_1h", "market_volume_1h", "user_trade_count_1h", "user_signed_flow_1h"]',
                "train_ratio = 0.5",
                "validation_ratio = 0.25",
                "test_ratio = 0.25",
            ]
        )
        + "\n"
    )

    config = load_config(config_path)
    prepare_trades(config, overwrite=True)
    label_user_trades(config, overwrite=True)
    build_model_windows(config, overwrite=True)

    connection = duckdb.connect()
    try:
        maker_window = connection.execute(
            f"""
            SELECT features
            FROM read_parquet('{output_root / "model_windows" / "window_size=2" / "**" / "*.parquet"}')
            WHERE user = 'user-a'
            ORDER BY window_end_ts
            LIMIT 1
            """
        ).fetchone()
    finally:
        connection.close()

    assert maker_window is not None
    assert len(maker_window[0][0]) == 11
    assert maker_window[0][0][7:] == [1.0, 0.4, 1.0, 1.0]
    assert maker_window[0][1][7:] == [2.0, 1.225, 2.0, 2.5]


def test_build_model_windows_can_write_to_a_custom_output_directory(tmp_path: Path) -> None:
    trades_path = tmp_path / "trades.parquet"
    output_root = tmp_path / "outputs"
    config_path = tmp_path / "pipeline.toml"
    custom_output_dir = tmp_path / "audit_datasets" / "window_size=2" / "stride=1"

    table = pa.table(
        {
            "timestamp": [1_704_067_200, 1_704_067_500, 1_704_067_800, 1_704_068_100],
            "datetime": [
                "2024-01-01 00:00:00",
                "2024-01-01 00:05:00",
                "2024-01-01 00:10:00",
                "2024-01-01 00:15:00",
            ],
            "block_number": [1, 2, 3, 4],
            "transaction_hash": ["tx1", "tx2", "tx3", "tx4"],
            "contract": ["exchange"] * 4,
            "event_id": ["event-1"] * 4,
            "event_slug": ["election-2024"] * 4,
            "event_title": ["Election"] * 4,
            "market_id": ["market-1"] * 4,
            "condition_id": ["condition-1"] * 4,
            "question": ["Will X win?"] * 4,
            "nonusdc_side": ["token1"] * 4,
            "maker": ["user-a"] * 4,
            "taker": ["user-b"] * 4,
            "maker_direction": ["BUY"] * 4,
            "taker_direction": ["SELL"] * 4,
            "price": [0.40, 0.55, 0.45, 0.60],
            "token_amount": [1_000_000, 1_500_000, 1_200_000, 900_000],
            "usd_amount": [400_000, 825_000, 540_000, 540_000],
            "asset_id": ["asset-1"] * 4,
            "order_hash": ["o1", "o2", "o3", "o4"],
        }
    )
    pq.write_table(table, trades_path)

    config_path.write_text(
        "\n".join(
            [
                "[inputs]",
                f'trades_path = "{trades_path}"',
                "",
                "[output]",
                f'root = "{output_root}"',
                "",
                "[runtime]",
                "threads = 4",
                'memory_limit = "1GB"',
                f'temp_directory = "{tmp_path / "tmp"}"',
                "",
                "[label]",
                "horizon_minutes = 5",
                "",
                "[model_windows]",
                "length = 2",
                "stride = 1",
                'feature_order = ["price_yes", "signed_token_amount", "usd_amount", "side", "role_is_maker", "time_delta_seconds", "market_age_seconds"]',
                "train_ratio = 0.5",
                "validation_ratio = 0.25",
                "test_ratio = 0.25",
            ]
        )
        + "\n"
    )

    config = load_config(config_path)
    prepare_trades(config, overwrite=True)
    label_user_trades(config, overwrite=True)

    result_dir = build_model_windows(
        config,
        window_size=2,
        stride=1,
        output_dir=custom_output_dir,
        overwrite=True,
    )

    assert result_dir == custom_output_dir
    assert (custom_output_dir / "split=train").exists()
    manifest = json.loads((output_root / "manifests" / "build_model_windows_2_stride_1.json").read_text())
    assert manifest["dataset_dir"] == str(custom_output_dir)
