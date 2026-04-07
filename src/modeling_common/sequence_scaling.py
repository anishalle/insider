from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class SequenceStandardizationStats:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    row_count: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "enabled": True,
            "row_count": int(self.row_count),
            "features": [
                {
                    "name": feature_name,
                    "mean": float(self.mean[index]),
                    "scale": float(self.scale[index]),
                }
                for index, feature_name in enumerate(self.feature_names)
            ],
        }


class SequenceStandardizer:
    def __init__(self, feature_names: Sequence[str]) -> None:
        self.feature_names = tuple(feature_names)
        if not self.feature_names:
            raise ValueError("feature_names must not be empty.")
        self._row_count = 0
        self._sum = np.zeros(len(self.feature_names), dtype=np.float64)
        self._sum_squares = np.zeros(len(self.feature_names), dtype=np.float64)

    def update(self, features: np.ndarray) -> None:
        array = np.asarray(features, dtype=np.float64)
        if array.ndim != 3 or array.shape[2] != len(self.feature_names):
            raise ValueError(
                "Expected sequence features with shape (batch, window, %d), got %s"
                % (len(self.feature_names), array.shape)
            )
        flattened = array.reshape(-1, array.shape[2])
        self._row_count += int(flattened.shape[0])
        self._sum += flattened.sum(axis=0)
        self._sum_squares += np.square(flattened).sum(axis=0)

    def finalize(self) -> SequenceStandardizationStats:
        if self._row_count == 0:
            raise ValueError("Cannot finalize sequence standardizer with zero rows.")
        mean = self._sum / self._row_count
        variance = np.maximum(self._sum_squares / self._row_count - np.square(mean), 0.0)
        scale = np.sqrt(variance)
        scale[scale < 1e-12] = 1.0
        return SequenceStandardizationStats(
            feature_names=self.feature_names,
            mean=mean.astype(np.float32),
            scale=scale.astype(np.float32),
            row_count=self._row_count,
        )


def apply_sequence_standardization(
    features: np.ndarray,
    stats: SequenceStandardizationStats,
) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != len(stats.feature_names):
        raise ValueError(
            "Expected sequence features with shape (batch, window, %d), got %s"
            % (len(stats.feature_names), array.shape)
        )
    mean = stats.mean.reshape(1, 1, -1)
    scale = stats.scale.reshape(1, 1, -1)
    return ((array - mean) / scale).astype(np.float32, copy=False)
