from __future__ import annotations

from pathlib import Path

import json

import pyarrow as pa
import pyarrow.parquet as pq

from lr.config import load_window_config
from lr.train import train_logistic_regression


FEATURE_ORDER = (
    "price_yes",
    "signed_token_amount",
    "usd_amount",
    "side",
    "role_is_maker",
    "time_delta_seconds",
    "market_age_seconds",
)


def test_logistic_regression_trainer_writes_expected_artifacts(tmp_path: Path) -> None:
    output_root = tmp_path / "processed"
    config_path = tmp_path / "pipeline.toml"
    _write_window_dataset(output_root / "model_windows" / "window_size=2")
    config_path.write_text(
        "\n".join(
            [
                "[output]",
                f'root = "{output_root}"',
                'model_window_dirname = "model_windows"',
                "",
                "[model_windows]",
                "length = 2",
                'feature_order = ["price_yes", "signed_token_amount", "usd_amount", "side", "role_is_maker", "time_delta_seconds", "market_age_seconds"]',
                "train_ratio = 0.8",
                "validation_ratio = 0.1",
                "test_ratio = 0.1",
            ]
        )
        + "\n"
    )

    config = load_window_config(config_path, window_size=2)
    run_dir = tmp_path / "lr-run"
    result = train_logistic_regression(
        config,
        epochs=12,
        batch_size=2,
        eval_batch_size=2,
        learning_rate=0.05,
        l2=1e-4,
        patience=4,
        seed=7,
        output_dir=run_dir,
    )

    assert result["run_dir"] == str(run_dir)
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "history.csv").exists()
    assert (run_dir / "predictions_validation.parquet").exists()
    assert (run_dir / "predictions_test.parquet").exists()
    assert (run_dir / "model.json").exists()

    summary = json.loads((run_dir / "summary.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    model_payload = json.loads((run_dir / "model.json").read_text())

    assert summary["window_size"] == 2
    assert summary["feature_order"] == list(FEATURE_ORDER)
    assert summary["class_weights"]["positive_rows"] == 3
    assert metrics["validation"]["auc_roc"] >= 0.5
    assert metrics["test"]["row_count"] == 2
    assert len(model_payload["weights"]) == 14


def _write_window_dataset(dataset_dir: Path) -> None:
    train_rows = [
        ("train-neg-1", _negative_window(), 0),
        ("train-neg-2", _negative_window(offset=0.1), 0),
        ("train-neg-3", _negative_window(offset=0.2), 0),
        ("train-pos-1", _positive_window(), 1),
        ("train-pos-2", _positive_window(offset=0.1), 1),
        ("train-pos-3", _positive_window(offset=0.2), 1),
    ]
    validation_rows = [
        ("val-neg-1", _negative_window(offset=0.05), 0),
        ("val-pos-1", _positive_window(offset=0.05), 1),
    ]
    test_rows = [
        ("test-neg-1", _negative_window(offset=0.15), 0),
        ("test-pos-1", _positive_window(offset=0.15), 1),
    ]
    _write_split(dataset_dir / "split=train" / "part-0.parquet", train_rows)
    _write_split(dataset_dir / "split=validation" / "part-0.parquet", validation_rows)
    _write_split(dataset_dir / "split=test" / "part-0.parquet", test_rows)


def _write_split(path: Path, rows: list[tuple[str, list[list[float]], int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "window_id": [row[0] for row in rows],
            "features": [row[1] for row in rows],
            "label": [row[2] for row in rows],
        }
    )
    pq.write_table(table, path, compression="zstd")


def _positive_window(offset: float = 0.0) -> list[list[float]]:
    return [
        [0.85 + offset, 2.0 + offset, 1.5 + offset, 1.0, 1.0, 5.0, 5.0],
        [0.9 + offset, 2.3 + offset, 1.8 + offset, 1.0, 1.0, 5.0, 10.0],
    ]


def _negative_window(offset: float = 0.0) -> list[list[float]]:
    return [
        [0.15 + offset, -2.0 - offset, 1.5 + offset, -1.0, 0.0, 5.0, 5.0],
        [0.1 + offset, -2.3 - offset, 1.8 + offset, -1.0, 0.0, 5.0, 10.0],
    ]
