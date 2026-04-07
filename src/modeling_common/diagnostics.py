from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np

from modeling_common.metrics import binary_classification_metrics, brier_score_loss, pr_auc_score


def summarize_array(values: np.ndarray) -> Dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "row_count": 0,
            "min": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
        }
    return {
        "row_count": int(array.size),
        "min": float(array.min()),
        "p01": float(np.quantile(array, 0.01)),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def threshold_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> Dict[str, float | int]:
    metrics = binary_classification_metrics(labels, probabilities, threshold=threshold)
    return {
        "threshold": float(threshold),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "predicted_positive_rate": float(metrics["predicted_positive_rate"]),
        "true_positive": int(metrics["true_positive"]),
        "true_negative": int(metrics["true_negative"]),
        "false_positive": int(metrics["false_positive"]),
        "false_negative": int(metrics["false_negative"]),
    }


def default_threshold_grid(*reference_thresholds: float) -> list[float]:
    grid = np.linspace(0.05, 0.95, num=19)
    all_thresholds = np.concatenate((grid, np.asarray(reference_thresholds, dtype=np.float64)))
    return sorted({float(np.clip(threshold, 0.0, 1.0)) for threshold in all_thresholds.tolist()})


def split_diagnostics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    logits: np.ndarray,
    *,
    reference_thresholds: Mapping[str, float],
) -> Dict[str, object]:
    return {
        "probability_summary": summarize_array(probabilities),
        "logit_summary": summarize_array(logits),
        "pr_auc": float(pr_auc_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "metrics_at_reference_thresholds": {
            name: threshold_metrics(labels, probabilities, threshold)
            for name, threshold in reference_thresholds.items()
        },
    }


def validation_threshold_sweep(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    reference_thresholds: Mapping[str, float],
) -> Dict[str, object]:
    thresholds = default_threshold_grid(*reference_thresholds.values())
    sweep = [threshold_metrics(labels, probabilities, threshold) for threshold in thresholds]
    return {
        "reference_thresholds": {name: float(value) for name, value in reference_thresholds.items()},
        "sweep": sweep,
        "best_by_metric": {
            metric_name: max(sweep, key=lambda row: (float(row[metric_name]), -abs(float(row["threshold"]) - 0.5)))
            for metric_name in ("accuracy", "precision", "recall", "f1")
        },
    }


def build_debug_diagnostics(
    *,
    split_payloads: Mapping[str, Mapping[str, np.ndarray]],
    reference_thresholds: Mapping[str, float],
) -> Dict[str, object]:
    return {
        "reference_thresholds": {name: float(value) for name, value in reference_thresholds.items()},
        "splits": {
            split: split_diagnostics(
                np.asarray(payload["labels"], dtype=np.int64),
                np.asarray(payload["probabilities"], dtype=np.float64),
                np.asarray(payload["logits"], dtype=np.float64),
                reference_thresholds=reference_thresholds,
            )
            for split, payload in split_payloads.items()
        },
        "validation_threshold_sweep": validation_threshold_sweep(
            np.asarray(split_payloads["validation"]["labels"], dtype=np.int64),
            np.asarray(split_payloads["validation"]["probabilities"], dtype=np.float64),
            reference_thresholds=reference_thresholds,
        ),
    }
