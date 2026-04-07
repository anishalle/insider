from __future__ import annotations

from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def prepare_run_directory(
    output_root: Path,
    *,
    model_name: str,
    window_size: int,
    output_dir: Optional[Path] = None,
) -> Path:
    if output_dir is not None:
        run_dir = output_dir
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = output_root / "models" / model_name / f"window_size={window_size}" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def write_history_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_predictions_parquet(
    path: Path,
    *,
    window_ids: Sequence[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    split: str,
    threshold: float = 0.5,
) -> None:
    predicted_labels = (probabilities >= threshold).astype(np.int64)
    table = pa.table(
        {
            "window_id": list(window_ids),
            "label": labels.astype(np.int64),
            "probability": probabilities.astype(np.float64),
            "prediction": predicted_labels,
            "split": [split] * len(window_ids),
        }
    )
    pq.write_table(table, path, compression="zstd")


def build_summary_payload(
    *,
    model_name: str,
    dataset_dir: Path,
    window_size: int,
    feature_order: Sequence[str],
    split_summaries: Iterable[Dict[str, object]],
    class_weights: Dict[str, float],
    train_batch_size: int,
) -> Dict[str, object]:
    return {
        "model_name": model_name,
        "dataset_dir": str(dataset_dir),
        "window_size": window_size,
        "feature_order": list(feature_order),
        "train_batch_size": train_batch_size,
        "split_summaries": list(split_summaries),
        "class_weights": class_weights,
    }
