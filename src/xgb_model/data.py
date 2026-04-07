from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np

from modeling_common.dataset import iter_split_batches, summarize_split


@dataclass(frozen=True)
class SplitArrays:
    window_ids: List[str]
    features: np.ndarray
    labels: np.ndarray


def flatten_window_features(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("Expected sequence features with rank 3, got %s" % (array.shape,))
    return np.ascontiguousarray(array.reshape(array.shape[0], array.shape[1] * array.shape[2]))


def build_flattened_feature_names(feature_order: Sequence[str], window_size: int) -> List[str]:
    return [f"{feature_name}__step_{step:02d}" for step in range(window_size) for feature_name in feature_order]


def iter_flattened_split_batches(
    dataset_dir: Path,
    split: str,
    *,
    batch_size: int,
) -> Iterator[SplitArrays]:
    for batch in iter_split_batches(dataset_dir, split, batch_size=batch_size, flatten=False):
        features = flatten_window_features(batch["features"])
        labels = np.asarray(batch["labels"], dtype=np.int64)
        window_ids = list(batch["window_id"])
        yield SplitArrays(window_ids=window_ids, features=features, labels=labels)


def load_split_arrays(
    dataset_dir: Path,
    split: str,
    *,
    batch_size: int,
    row_count: int | None = None,
) -> SplitArrays:
    total_rows = int(row_count if row_count is not None else summarize_split(dataset_dir, split, batch_size=batch_size).row_count)
    feature_batches: List[np.ndarray] = []
    label_batches: List[np.ndarray] = []
    window_ids: List[str] = []
    for batch in iter_flattened_split_batches(dataset_dir, split, batch_size=batch_size):
        feature_batches.append(batch.features)
        label_batches.append(batch.labels)
        window_ids.extend(batch.window_ids)

    if feature_batches:
        features = np.concatenate(feature_batches, axis=0)
        labels = np.concatenate(label_batches, axis=0)
    else:
        features = np.zeros((0, 0), dtype=np.float32)
        labels = np.zeros((0,), dtype=np.int64)

    if total_rows != features.shape[0]:
        raise ValueError(
            "Expected %d rows for split=%s, loaded %d rows." % (total_rows, split, features.shape[0])
        )
    return SplitArrays(window_ids=window_ids, features=features, labels=labels)
