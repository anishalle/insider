from __future__ import annotations

from typing import Dict

import numpy as np


def roc_auc_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    if positives == 0 or negatives == 0:
        return 0.5

    order = np.argsort(y_prob, kind="mergesort")
    sorted_labels = y_true[order]
    sorted_scores = y_prob[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)

    index = 0
    while index < len(sorted_scores):
        next_index = index + 1
        while next_index < len(sorted_scores) and sorted_scores[next_index] == sorted_scores[index]:
            next_index += 1
        average_rank = (index + next_index - 1) / 2.0 + 1.0
        ranks[index:next_index] = average_rank
        index = next_index

    positive_rank_sum = ranks[sorted_labels == 1].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> Dict[str, object]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    if y_true.ndim != 1 or y_prob.ndim != 1 or y_true.shape[0] != y_prob.shape[0]:
        raise ValueError("labels and probabilities must be 1D arrays of equal length")

    predictions = (y_prob >= threshold).astype(np.int64)
    true_positive = int(((predictions == 1) & (y_true == 1)).sum())
    true_negative = int(((predictions == 0) & (y_true == 0)).sum())
    false_positive = int(((predictions == 1) & (y_true == 0)).sum())
    false_negative = int(((predictions == 0) & (y_true == 1)).sum())

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    accuracy = (true_positive + true_negative) / len(y_true) if len(y_true) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "row_count": int(len(y_true)),
        "positive_rows": int(y_true.sum()),
        "negative_rows": int(len(y_true) - int(y_true.sum())),
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "predicted_positive_rate": float(predictions.mean()) if len(predictions) else 0.0,
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positive": int(true_positive),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
    }
