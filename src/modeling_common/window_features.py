from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


SUMMARY_SUFFIXES: Tuple[str, ...] = (
    "first",
    "last",
    "mean",
    "std",
    "min",
    "max",
    "delta",
    "last5_mean",
)


def build_window_summary_feature_names(feature_names: Sequence[str]) -> List[str]:
    names: List[str] = []
    for suffix in SUMMARY_SUFFIXES:
        for feature_name in feature_names:
            names.append(f"{feature_name}__summary_{suffix}")
    return names


def compute_window_summary_features(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("Expected sequence features with rank 3, got %s" % (array.shape,))

    first = array[:, 0, :]
    last = array[:, -1, :]
    mean = array.mean(axis=1)
    std = array.std(axis=1)
    minimum = array.min(axis=1)
    maximum = array.max(axis=1)
    delta = last - first
    recent = array[:, -min(5, array.shape[1]) :, :].mean(axis=1)

    return np.concatenate(
        (
            first,
            last,
            mean,
            std,
            minimum,
            maximum,
            delta,
            recent,
        ),
        axis=1,
    ).astype(np.float32, copy=False)
