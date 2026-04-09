from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from modeling_common.artifacts import build_summary_payload, prepare_run_directory
from modeling_common.config import load_window_data_config
from modeling_common.diagnostics import build_debug_diagnostics
from modeling_common.dataset import iter_split_batches, summarize_dataset
from modeling_common.metrics import binary_classification_metrics, brier_score_loss, pr_auc_score, roc_auc_score
from modeling_common.sequence_scaling import (
    SequenceStandardizer,
    apply_sequence_standardization,
    build_default_clip_enabled,
    build_default_scale_enabled,
    build_default_transform_kinds,
)
from modeling_common.tabular_scaling import TabularStandardizer, apply_tabular_standardization
from modeling_common.window_features import (
    build_window_summary_feature_names,
    compute_window_summary_features,
)


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


def test_sequence_standardizer_uses_train_only_stats() -> None:
    train_features = np.asarray(
        [
            [[1.0, 10.0], [3.0, 14.0]],
            [[5.0, 18.0], [7.0, 22.0]],
        ],
        dtype=np.float32,
    )
    validation_features = np.asarray([[[101.0, 1000.0], [103.0, 1004.0]]], dtype=np.float32)

    standardizer = SequenceStandardizer(("price_yes", "custom_continuous_feature"))
    standardizer.update(train_features)
    stats = standardizer.finalize()

    transformed_train = apply_sequence_standardization(train_features, stats)
    transformed_validation = apply_sequence_standardization(validation_features, stats)

    assert stats.row_count == 4
    assert np.allclose(stats.mean, np.asarray([4.0, 16.0], dtype=np.float32))
    assert np.allclose(np.round(stats.scale, 6), np.asarray([2.236068, 4.472136], dtype=np.float32))
    assert np.allclose(transformed_train.reshape(-1, 2).mean(axis=0), np.zeros(2), atol=1e-6)
    assert transformed_validation[0, 0, 0] > 40.0


def test_sequence_standardizer_supports_default_transforms_clipping_and_unscaled_binary_features() -> None:
    features = np.asarray(
        [
            [[2.0, 0.0, 10.0], [-2.0, 1.0, 100.0]],
            [[4.0, 0.0, 1000.0], [-4.0, 1.0, 10000.0]],
        ],
        dtype=np.float32,
    )
    feature_names = ("signed_token_amount", "role_is_maker", "time_delta_seconds")
    standardizer = SequenceStandardizer(
        feature_names,
        transform_kinds=build_default_transform_kinds(feature_names),
        clip_lower=np.asarray([-1.0, -np.inf, 0.0], dtype=np.float32),
        clip_upper=np.asarray([1.0, np.inf, 3.0], dtype=np.float32),
        scale_enabled=build_default_scale_enabled(feature_names),
        clip_enabled=build_default_clip_enabled(feature_names),
        clip_percentiles=(1.0, 99.0),
        clip_sample_rows=8,
    )
    standardizer.update(features)
    stats = standardizer.finalize()

    transformed = apply_sequence_standardization(features, stats)

    assert stats.transform_kinds == ("signed_log1p", "identity", "log1p")
    assert stats.scale_enabled.tolist() == [True, False, True]
    assert stats.clip_enabled.tolist() == [True, False, True]
    assert np.allclose(np.unique(transformed[..., 1]), np.asarray([0.0, 1.0], dtype=np.float32))
    assert np.isfinite(transformed).all()
    assert float(np.abs(transformed[..., 0]).max()) <= 2.0
    assert float(np.abs(transformed[..., 2]).max()) <= 2.0


def test_debug_diagnostics_include_threshold_sweep_and_split_summaries() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.1, 0.4, 0.6, 0.9], dtype=np.float64)
    logits = np.log(probabilities / (1.0 - probabilities))

    diagnostics = build_debug_diagnostics(
        split_payloads={
            "train": {"labels": labels, "probabilities": probabilities, "logits": logits},
            "validation": {"labels": labels, "probabilities": probabilities, "logits": logits},
            "test": {"labels": labels, "probabilities": probabilities, "logits": logits},
        },
        reference_thresholds={"default_0_5": 0.5, "class_weight_adjusted": 0.4},
    )

    assert diagnostics["reference_thresholds"]["default_0_5"] == 0.5
    assert diagnostics["splits"]["validation"]["pr_auc"] == pr_auc_score(labels, probabilities)
    assert diagnostics["splits"]["validation"]["brier_score"] == brier_score_loss(labels, probabilities)
    assert diagnostics["splits"]["validation"]["metrics_at_reference_thresholds"]["default_0_5"]["f1"] > 0.0
    assert diagnostics["validation_threshold_sweep"]["best_by_metric"]["accuracy"]["threshold"] >= 0.0
    assert diagnostics["calibration"]["validation"]["num_bins"] == 10
    assert "validation_best_f1" in diagnostics["split_metrics_at_validation_best_thresholds"]["test"]


def test_window_summary_features_and_names_are_consistent() -> None:
    features = np.asarray(
        [
            [[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]],
            [[2.0, 20.0], [4.0, 24.0], [6.0, 28.0]],
        ],
        dtype=np.float32,
    )

    summary_features = compute_window_summary_features(features)
    summary_names = build_window_summary_feature_names(("price_yes", "usd_amount"))

    assert summary_features.shape == (2, 16)
    assert len(summary_names) == 16
    assert summary_names[:4] == [
        "price_yes__summary_first",
        "usd_amount__summary_first",
        "price_yes__summary_last",
        "usd_amount__summary_last",
    ]
    assert np.allclose(summary_features[0, :4], np.asarray([1.0, 10.0, 5.0, 18.0], dtype=np.float32))


def test_tabular_standardizer_uses_train_only_stats() -> None:
    train_features = np.asarray(
        [
            [1.0, 10.0],
            [3.0, 14.0],
            [5.0, 18.0],
        ],
        dtype=np.float32,
    )
    validation_features = np.asarray([[101.0, 1000.0]], dtype=np.float32)

    standardizer = TabularStandardizer(("feature_a", "feature_b"))
    standardizer.update(train_features)
    stats = standardizer.finalize()

    transformed_train = apply_tabular_standardization(train_features, stats)
    transformed_validation = apply_tabular_standardization(validation_features, stats)

    assert stats.row_count == 3
    assert np.allclose(stats.mean, np.asarray([3.0, 14.0], dtype=np.float32))
    assert np.allclose(transformed_train.mean(axis=0), np.zeros(2), atol=1e-6)
    assert transformed_validation[0, 0] > 40.0


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
