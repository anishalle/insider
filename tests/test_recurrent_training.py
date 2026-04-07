from __future__ import annotations

import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


FEATURE_ORDER = (
    "price_yes",
    "signed_token_amount",
    "usd_amount",
    "side",
    "role_is_maker",
    "time_delta_seconds",
    "market_age_seconds",
)


def test_rnn_trainer_writes_expected_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    from rnn.train import main as rnn_main

    config_path = _write_window_dataset(tmp_path)
    run_dir = tmp_path / "rnn-run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rnn.train",
            "--config",
            str(config_path),
            "--window-size",
            "2",
            "--output-dir",
            str(run_dir),
            "--epochs",
            "2",
            "--batch-size",
            "2",
            "--eval-batch-size",
            "2",
            "--hidden-size",
            "4",
            "--learning-rate",
            "0.01",
            "--patience",
            "2",
            "--device",
            "cpu",
        ],
    )

    rnn_main()

    _assert_common_artifacts(run_dir)
    assert (run_dir / "checkpoint.pt").exists()

    summary = json.loads((run_dir / "summary.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())

    assert summary["model_name"] == "rnn"
    assert summary["window_size"] == 2
    assert summary["feature_order"] == list(FEATURE_ORDER)
    assert metrics["validation"]["row_count"] == 2
    assert metrics["test"]["row_count"] == 2


def test_lstm_trainer_writes_expected_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    from lstm.train import main as lstm_main

    config_path = _write_window_dataset(tmp_path)
    run_dir = tmp_path / "lstm-run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lstm.train",
            "--config",
            str(config_path),
            "--window-size",
            "2",
            "--output-dir",
            str(run_dir),
            "--hidden-size",
            "4",
            "--dropout",
            "0.0",
            "--epochs",
            "2",
            "--batch-size",
            "2",
            "--eval-batch-size",
            "2",
            "--learning-rate",
            "0.01",
            "--patience",
            "2",
            "--device",
            "cpu",
        ],
    )

    lstm_main()

    _assert_common_artifacts(run_dir)
    assert (run_dir / "checkpoint.pt").exists()

    summary = json.loads((run_dir / "summary.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())

    assert summary["model_name"] == "lstm"
    assert summary["window_size"] == 2
    assert summary["feature_order"] == list(FEATURE_ORDER)
    assert metrics["validation"]["row_count"] == 2
    assert metrics["test"]["row_count"] == 2


def _assert_common_artifacts(run_dir: Path) -> None:
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "history.csv").exists()
    assert (run_dir / "predictions_validation.parquet").exists()
    assert (run_dir / "predictions_test.parquet").exists()


def _write_window_dataset(tmp_path: Path) -> Path:
    output_root = tmp_path / "processed"
    dataset_dir = output_root / "model_windows" / "window_size=2"
    config_path = tmp_path / "pipeline.toml"
    train_rows = [
        ("train-neg-1", _negative_window(), 0),
        ("train-neg-2", _negative_window(offset=0.1), 0),
        ("train-pos-1", _positive_window(), 1),
        ("train-pos-2", _positive_window(offset=0.1), 1),
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
    return config_path


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
