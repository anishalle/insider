from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from insider.config import load_config
from insider.pipeline import run_pipeline


def test_end_to_end_pipeline_builds_labeled_sequences(tmp_path: Path) -> None:
    trades_path = tmp_path / "trades.parquet"
    output_root = tmp_path / "outputs"
    config_path = tmp_path / "pipeline.toml"

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
                'threads = 4',
                'memory_limit = "1GB"',
                f'temp_directory = "{tmp_path / "tmp"}"',
                "",
                "[label]",
                "horizon_minutes = 5",
                "",
                "[sequence]",
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
    run_pipeline(config, overwrite=True)

    connection = duckdb.connect()
    try:
        labeled = connection.execute(
            f"""
            SELECT user, role, price_yes, future_price_yes, side, markout, label
            FROM read_parquet('{output_root / "labeled_user_trades" / "**" / "*.parquet"}')
            ORDER BY user, trade_time
            """
        ).fetchall()
        assert labeled[0] == ("user-a", "maker", 0.4, 0.55, 1, 0.15000000000000002, 1)
        assert labeled[1] == ("user-a", "maker", 0.55, 0.45, 1, -0.10000000000000003, 0)
        assert labeled[3] == ("user-b", "taker", 0.4, 0.55, -1, -0.15000000000000002, 0)

        sequence_summary = connection.execute(
            f"""
            SELECT split, COUNT(*) AS row_count, MIN(seq_len), MAX(length(features)), MAX(length(features[1]))
            FROM read_parquet('{output_root / "sequence_dataset" / "**" / "*.parquet"}')
            GROUP BY split
            ORDER BY split
            """
        ).fetchall()
        assert sequence_summary
        assert all(row[2] == 2 for row in sequence_summary)
        assert all(row[3] == 2 for row in sequence_summary)
        assert all(row[4] == 7 for row in sequence_summary)
    finally:
        connection.close()

    manifest = json.loads((output_root / "manifests" / "build_sequences.json").read_text())
    assert manifest["sequence_length"] == 2
    assert manifest["stride"] == 1
