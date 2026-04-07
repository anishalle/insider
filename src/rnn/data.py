from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq


SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitSummary:
    split: str
    row_count: int
    positive_rows: int
    negative_rows: int
    positive_rate: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "split": self.split,
            "row_count": self.row_count,
            "positive_rows": self.positive_rows,
            "negative_rows": self.negative_rows,
            "positive_rate": self.positive_rate,
        }


@dataclass(frozen=True)
class SplitBatch:
    window_ids: List[str]
    features: np.ndarray
    labels: np.ndarray


def split_dir(dataset_dir: Path, split: str) -> Path:
    return dataset_dir / ("split=%s" % split)


def list_split_files(dataset_dir: Path, split: str) -> List[Path]:
    files = sorted(split_dir(dataset_dir, split).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError("No parquet files found for split=%s under %s" % (split, dataset_dir))
    return files


def iter_split_batches(
    dataset_dir: Path,
    split: str,
    *,
    batch_size: int,
    shuffle_files: bool = False,
    shuffle_rows: bool = False,
    seed: Optional[int] = None,
) -> Iterator[SplitBatch]:
    file_paths = list_split_files(dataset_dir, split)
    rng = random.Random(seed)
    if shuffle_files:
        rng.shuffle(file_paths)

    for file_path in file_paths:
        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=["window_id", "features", "label"],
            use_threads=True,
        ):
            window_ids = batch.column(0).to_pylist()
            features = np.asarray(batch.column(1).to_pylist(), dtype=np.float32)
            labels = np.asarray(batch.column(2).to_pylist(), dtype=np.int64)
            if features.ndim != 3:
                raise ValueError("Expected features to have rank 3, got shape %s" % (features.shape,))
            if shuffle_rows and len(labels) > 1:
                permutation = np.random.default_rng(seed=rng.randrange(1 << 30)).permutation(len(labels))
                window_ids = [window_ids[index] for index in permutation.tolist()]
                features = features[permutation]
                labels = labels[permutation]
            yield SplitBatch(window_ids=window_ids, features=features, labels=labels)


def summarize_split(dataset_dir: Path, split: str, *, batch_size: int = 4096) -> SplitSummary:
    row_count = 0
    positive_rows = 0
    for batch in iter_split_batches(dataset_dir, split, batch_size=batch_size):
        row_count += int(batch.labels.shape[0])
        positive_rows += int(batch.labels.sum())
    negative_rows = row_count - positive_rows
    positive_rate = float(positive_rows / row_count) if row_count else 0.0
    return SplitSummary(
        split=split,
        row_count=row_count,
        positive_rows=positive_rows,
        negative_rows=negative_rows,
        positive_rate=positive_rate,
    )


def summarize_dataset(dataset_dir: Path, *, batch_size: int = 4096) -> Dict[str, SplitSummary]:
    return {split: summarize_split(dataset_dir, split, batch_size=batch_size) for split in SPLITS}


def load_split_arrays(
    dataset_dir: Path,
    split: str,
    *,
    batch_size: int = 4096,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    window_ids: List[str] = []
    feature_batches: List[np.ndarray] = []
    label_batches: List[np.ndarray] = []
    for batch in iter_split_batches(dataset_dir, split, batch_size=batch_size):
        window_ids.extend(batch.window_ids)
        feature_batches.append(batch.features)
        label_batches.append(batch.labels)
    if feature_batches:
        features = np.concatenate(feature_batches, axis=0)
        labels = np.concatenate(label_batches, axis=0)
    else:
        features = np.zeros((0, 0, 0), dtype=np.float32)
        labels = np.zeros((0,), dtype=np.int64)
    return window_ids, features, labels

