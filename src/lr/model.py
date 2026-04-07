from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class StandardizerState:
    mean: np.ndarray
    scale: np.ndarray


class Standardizer:
    def __init__(self, n_features: int) -> None:
        self.n_features = n_features
        self.count = 0
        self.sum = np.zeros(n_features, dtype=np.float64)
        self.sum_sq = np.zeros(n_features, dtype=np.float64)

    def update(self, features: np.ndarray) -> None:
        x = np.asarray(features, dtype=np.float64)
        self.count += int(x.shape[0])
        self.sum += x.sum(axis=0)
        self.sum_sq += np.square(x).sum(axis=0)

    def finalize(self) -> StandardizerState:
        if self.count == 0:
            raise ValueError("Cannot finalize standardizer without observations.")
        mean = self.sum / float(self.count)
        variance = np.maximum(self.sum_sq / float(self.count) - np.square(mean), 1e-12)
        scale = np.sqrt(variance)
        scale[scale < 1e-6] = 1.0
        return StandardizerState(mean=mean.astype(np.float32), scale=scale.astype(np.float32))


def apply_standardization(features: np.ndarray, state: StandardizerState) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    return (x - state.mean) / state.scale


class LogisticRegressionModel:
    def __init__(self, n_features: int) -> None:
        self.weights = np.zeros(n_features, dtype=np.float32)
        self.bias = 0.0

    def logits(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=np.float32).dot(self.weights) + self.bias

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        return sigmoid(self.logits(features))

    def state_dict(self) -> Dict[str, object]:
        return {
            "weights": self.weights.copy(),
            "bias": float(self.bias),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.weights = np.asarray(state["weights"], dtype=np.float32)
        self.bias = float(state["bias"])


class AdamState:
    def __init__(self, n_features: int) -> None:
        self.t = 0
        self.m_w = np.zeros(n_features, dtype=np.float32)
        self.v_w = np.zeros(n_features, dtype=np.float32)
        self.m_b = 0.0
        self.v_b = 0.0


def sigmoid(values: np.ndarray) -> np.ndarray:
    z = np.asarray(values, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out.astype(np.float32)


def logistic_loss_and_gradients(
    features: np.ndarray,
    labels: np.ndarray,
    model: LogisticRegressionModel,
    *,
    positive_weight: float,
    negative_weight: float = 1.0,
    l2: float = 0.0,
) -> Tuple[float, np.ndarray, float]:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    logits = model.logits(x).astype(np.float64)
    probs = sigmoid(logits).astype(np.float64)
    sample_weights = np.where(y > 0.5, positive_weight, negative_weight).astype(np.float64)
    normalizer = float(sample_weights.sum()) if sample_weights.size else 1.0
    normalizer = max(normalizer, 1.0)

    loss_terms = np.logaddexp(0.0, logits) - y * logits
    weighted_loss = float(np.sum(sample_weights * loss_terms) / normalizer)
    weighted_loss += 0.5 * l2 * float(np.sum(np.square(model.weights)))

    error = (probs - y) * sample_weights / normalizer
    grad_w = x.T.dot(error).astype(np.float32) + (l2 * model.weights)
    grad_b = float(np.sum(error))
    return weighted_loss, grad_w, grad_b


def adam_step(
    model: LogisticRegressionModel,
    optimizer: AdamState,
    grad_w: np.ndarray,
    grad_b: float,
    *,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> None:
    optimizer.t += 1
    optimizer.m_w = beta1 * optimizer.m_w + (1.0 - beta1) * grad_w
    optimizer.v_w = beta2 * optimizer.v_w + (1.0 - beta2) * np.square(grad_w)
    optimizer.m_b = beta1 * optimizer.m_b + (1.0 - beta1) * grad_b
    optimizer.v_b = beta2 * optimizer.v_b + (1.0 - beta2) * (grad_b * grad_b)

    bias_correction1 = 1.0 - beta1 ** optimizer.t
    bias_correction2 = 1.0 - beta2 ** optimizer.t
    step_w = optimizer.m_w / bias_correction1
    step_v = optimizer.v_w / bias_correction2
    step_b = optimizer.m_b / bias_correction1
    step_b_v = optimizer.v_b / bias_correction2

    model.weights -= learning_rate * step_w / (np.sqrt(step_v) + epsilon)
    model.bias -= learning_rate * step_b / (np.sqrt(step_b_v) + epsilon)

