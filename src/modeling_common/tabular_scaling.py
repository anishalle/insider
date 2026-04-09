from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class TabularStandardizationStats:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    row_count: int

    def to_dict(self) -> Dict[str, object]:
        return {
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


class TabularStandardizer:
    def __init__(self, feature_names: Sequence[str]) -> None:
        self.feature_names = tuple(feature_names)
        if not self.feature_names:
            raise ValueError("feature_names must not be empty.")
        self._row_count = 0
        self._sum = np.zeros(len(self.feature_names), dtype=np.float64)
        self._sum_squares = np.zeros(len(self.feature_names), dtype=np.float64)

    def update(self, features: np.ndarray) -> None:
        array = np.asarray(features, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != len(self.feature_names):
            raise ValueError(
                "Expected tabular features with shape (batch, %d), got %s"
                % (len(self.feature_names), array.shape)
            )
        array64 = np.asarray(array, dtype=np.float64)
        self._row_count += int(array.shape[0])
        self._sum += array64.sum(axis=0)
        self._sum_squares += np.square(array64).sum(axis=0)

    def finalize(self) -> TabularStandardizationStats:
        if self._row_count == 0:
            raise ValueError("Cannot finalize tabular standardizer with zero rows.")
        mean = self._sum / self._row_count
        variance = np.maximum(self._sum_squares / self._row_count - np.square(mean), 0.0)
        scale = np.sqrt(variance)
        scale[scale < 1e-12] = 1.0
        return TabularStandardizationStats(
            feature_names=self.feature_names,
            mean=mean.astype(np.float32),
            scale=scale.astype(np.float32),
            row_count=int(self._row_count),
        )


def apply_tabular_standardization(
    features: np.ndarray,
    stats: TabularStandardizationStats,
) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != len(stats.feature_names):
        raise ValueError(
            "Expected tabular features with shape (batch, %d), got %s"
            % (len(stats.feature_names), array.shape)
        )
    return ((array - stats.mean.reshape(1, -1)) / stats.scale.reshape(1, -1)).astype(np.float32, copy=False)
