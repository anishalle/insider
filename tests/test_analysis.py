from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from insider.analysis import visualize_signals
from insider.config import load_config
from insider.pipeline import build_model_windows, label_user_trades, prepare_trades


def test_visualize_signals_writes_summary_candidates_and_plots(tmp_path: Path) -> None:
    trades_path = tmp_path / "trades.parquet"
    output_root = tmp_path / "outputs"
    config_path = tmp_path / "pipeline.toml"
    analysis_dir = tmp_path / "analysis"

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
            "maker": ["user-a", "user-a", "user-a", "user-a", "user-b", "user-b"],
            "taker": ["user-b", "user-b", "user-b", "user-b", "user-a", "user-a"],
            "maker_direction": ["BUY", "BUY", "BUY", "BUY", "SELL", "SELL"],
            "taker_direction": ["SELL", "SELL", "SELL", "SELL", "BUY", "BUY"],
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
    build_model_windows(config, window_size=2, stride=1, overwrite=True)

    result_dir = visualize_signals(
        config,
        output_dir=analysis_dir,
        examples_per_class=2,
        ambiguous_examples=1,
        lookback_minutes=15,
        lookahead_minutes=15,
        model_window_size=2,
    )

    assert result_dir == analysis_dir

    summary = json.loads((analysis_dir / "summary.json").read_text())
    assert summary["totals"]["rows"] == 10
    assert summary["totals"]["positive_rows"] > 0
    assert summary["totals"]["negative_rows"] > 0

    feasibility = json.loads((analysis_dir / "training_feasibility.json").read_text())
    assert feasibility["window_readiness"]["actual_model_window_summary"] is not None
    assert feasibility["window_readiness"]["actual_model_window_summary"]["shape"]["min_window_size"] == 2

    with (analysis_dir / "candidate_signals.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    labels = {row["label"] for row in rows}
    assert labels == {"0", "1"}

    monthly_lines = (analysis_dir / "class_balance_over_time.csv").read_text().strip().splitlines()
    assert len(monthly_lines) >= 2

    plot_files = sorted((analysis_dir / "plots").glob("*.png"))
    assert plot_files
