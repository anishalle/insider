from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, Iterator, List, Optional, Sequence, Union

import numpy as np
import pyarrow.parquet as pq


SPLITS: Sequence[str] = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitSummary:
    split: str
    row_count: int
    positive_rows: int
    negative_rows: int
    positive_rate: float

    def to_dict(self) -> Dict[str, Union[float, int, str]]:
        return {
            "split": self.split,
            "row_count": self.row_count,
            "positive_rows": self.positive_rows,
            "negative_rows": self.negative_rows,
            "positive_rate": self.positive_rate,
        }


def split_dir(dataset_dir: Path, split: str) -> Path:
    return dataset_dir / f"split={split}"


def list_split_files(dataset_dir: Path, split: str) -> List[Path]:
    resolved_dir = split_dir(dataset_dir, split)
    files = sorted(resolved_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found for split={split}: {resolved_dir}")
    return files


def iter_split_batches(
    dataset_dir: Path,
    split: str,
    *,
    batch_size: int,
    shuffle_files: bool = False,
    shuffle_rows: bool = False,
    seed: Optional[int] = None,
    flatten: bool = False,
) -> Iterator[Dict[str, np.ndarray | List[str]]]:
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
            ids = batch.column(0).to_pylist()
            features = np.asarray(batch.column(1).to_pylist(), dtype=np.float32)
            labels = np.asarray(batch.column(2).to_pylist(), dtype=np.int64)
            if features.ndim != 3:
                raise ValueError(
                    f"Expected feature batch with shape (batch, window, feature), got {features.shape}"
                )
            if flatten:
                features = features.reshape(features.shape[0], features.shape[1] * features.shape[2])
            if shuffle_rows and len(labels) > 1:
                permutation = np.random.default_rng(seed=rng.randrange(1 << 30)).permutation(len(labels))
                features = features[permutation]
                labels = labels[permutation]
                ids = [ids[index] for index in permutation.tolist()]
            yield {
                "window_id": ids,
                "features": features,
                "labels": labels,
            }


def summarize_split(dataset_dir: Path, split: str, *, batch_size: int = 4096) -> SplitSummary:
    row_count = 0
    positive_rows = 0
    for batch in iter_split_batches(dataset_dir, split, batch_size=batch_size):
        labels = batch["labels"]
        assert isinstance(labels, np.ndarray)
        row_count += int(labels.shape[0])
        positive_rows += int(labels.sum())
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
