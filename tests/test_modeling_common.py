from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from modeling_common.artifacts import build_summary_payload, prepare_run_directory
from modeling_common.config import load_window_data_config
from modeling_common.dataset import iter_split_batches, summarize_dataset
from modeling_common.metrics import binary_classification_metrics, roc_auc_score


def test_load_window_data_config_reads_output_and_window_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.toml"
    config_path.write_text(
        "\n".join(
            [
                "[output]",
                f'root = "{tmp_path / "processed"}"',
                'model_window_dirname = "model_windows"',
                "",
                "[model_windows]",
                "length = 12",
                'feature_order = ["price_yes", "signed_token_amount"]',
            ]
        )
        + "\n"
    )

    data_config = load_window_data_config(config_path)

    assert data_config.output_root == tmp_path / "processed"
    assert data_config.window_size == 12
    assert data_config.feature_order == ("price_yes", "signed_token_amount")
    assert data_config.dataset_dir == tmp_path / "processed" / "model_windows" / "window_size=12"


def test_dataset_iteration_and_summary_preserve_shapes(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "processed" / "model_windows" / "window_size=2"
    _write_split(
        dataset_dir / "split=train" / "part-0.parquet",
        [
            ("train-1", [[[0.1, 1.0], [0.2, 2.0]]], [1]),
            ("train-2", [[[0.3, 3.0], [0.4, 4.0]]], [0]),
        ],
    )
    _write_split(
        dataset_dir / "split=validation" / "part-0.parquet",
        [("val-1", [[[0.5, 5.0], [0.6, 6.0]]], [1])],
    )
    _write_split(
        dataset_dir / "split=test" / "part-0.parquet",
        [("test-1", [[[0.7, 7.0], [0.8, 8.0]]], [0])],
    )

    train_batches = list(iter_split_batches(dataset_dir, "train", batch_size=8))
    flattened_batches = list(iter_split_batches(dataset_dir, "train", batch_size=8, flatten=True))
    split_summaries = summarize_dataset(dataset_dir)

    assert len(train_batches) == 1
    assert train_batches[0]["window_id"] == ["train-1", "train-2"]
    assert np.asarray(train_batches[0]["features"]).shape == (2, 2, 2)
    assert np.asarray(flattened_batches[0]["features"]).shape == (2, 4)
    assert split_summaries["train"].row_count == 2
    assert split_summaries["train"].positive_rows == 1
    assert split_summaries["validation"].row_count == 1
    assert split_summaries["test"].negative_rows == 1


def test_metrics_and_summary_payload_include_auc_and_counts(tmp_path: Path) -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.1, 0.4, 0.6, 0.9], dtype=np.float64)

    metrics = binary_classification_metrics(labels, probabilities)
    run_dir = prepare_run_directory(tmp_path, model_name="lr", window_size=50)
    payload = build_summary_payload(
        model_name="lr",
        dataset_dir=tmp_path / "dataset",
        window_size=50,
        feature_order=("price_yes",),
        split_summaries=[{"split": "train", "row_count": 10}],
        class_weights={"negative": 1.0, "positive": 1.5},
        train_batch_size=512,
    )

    assert np.isclose(roc_auc_score(labels, probabilities), 1.0)
    assert metrics["auc_roc"] == 1.0
    assert metrics["positive_rows"] == 2
    assert run_dir.exists()
    assert payload["window_size"] == 50
    assert payload["class_weights"] == {"negative": 1.0, "positive": 1.5}


def _write_split(path: Path, rows: list[tuple[str, list[list[list[float]]], list[int]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    window_ids: list[str] = []
    features: list[list[list[float]]] = []
    labels: list[int] = []
    for window_id, feature_rows, label_rows in rows:
        window_ids.append(window_id)
        features.extend(feature_rows)
        labels.extend(label_rows)
    table = pa.table({"window_id": window_ids, "features": features, "label": labels})
    pq.write_table(table, path, compression="zstd")
