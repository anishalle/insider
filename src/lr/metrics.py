from __future__ import annotations

from typing import Dict

import numpy as np


def roc_auc_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    positives = int(y_true.sum())
    negatives = int(y_true.shape[0] - positives)
    if positives == 0 or negatives == 0:
        return 0.5

    order = np.argsort(y_prob, kind="mergesort")
    sorted_labels = y_true[order]
    sorted_scores = y_prob[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)

    start = 0
    while start < sorted_scores.shape[0]:
        end = start + 1
        while end < sorted_scores.shape[0] and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0 + 1.0
        ranks[start:end] = average_rank
        start = end

    positive_rank_sum = ranks[sorted_labels == 1].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, *, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    predictions = (y_prob >= threshold).astype(np.int64)

    tp = int(((predictions == 1) & (y_true == 1)).sum())
    tn = int(((predictions == 0) & (y_true == 0)).sum())
    fp = int(((predictions == 1) & (y_true == 0)).sum())
    fn = int(((predictions == 0) & (y_true == 1)).sum())

    precision = tp / float(tp + fp) if (tp + fp) else 0.0
    recall = tp / float(tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / float(y_true.shape[0]) if y_true.shape[0] else 0.0
    f1 = 2.0 * precision * recall / float(precision + recall) if (precision + recall) else 0.0

    row_count = int(y_true.shape[0])
    positive_rows = int(y_true.sum())
    negative_rows = int(row_count - positive_rows)

    return {
        "row_count": row_count,
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "positive_rate": float(y_true.mean()) if y_true.shape[0] else 0.0,
        "predicted_positive_rate": float(predictions.mean()) if predictions.shape[0] else 0.0,
        "auc_roc": roc_auc_score(y_true, y_prob),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }
