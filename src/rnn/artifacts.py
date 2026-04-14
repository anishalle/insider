from __future__ import annotations

from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from modeling_common.provenance import build_run_provenance


def make_run_dir(output_root: Path, model_name: str, window_size: int, output_dir: Optional[Path] = None) -> Path:
    if output_dir is not None:
        run_dir = output_dir
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = output_root / "models" / model_name / ("window_size=%d" % window_size) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
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
    predictions = (np.asarray(probabilities) >= threshold).astype(np.int64)
    table = pa.table(
        {
            "window_id": list(window_ids),
            "label": np.asarray(labels, dtype=np.int64),
            "probability": np.asarray(probabilities, dtype=np.float64),
            "prediction": predictions,
            "split": [split] * len(window_ids),
        }
    )
    pq.write_table(table, path, compression="zstd")


def build_summary(
    *,
    model_name: str,
    config_path: Path | None,
    dataset_dir: Path,
    output_root: Path,
    manifest_dirname: str,
    window_size: int,
    feature_order: Sequence[str],
    split_summaries: Iterable[Dict[str, object]],
    class_weights: Dict[str, float],
    hyperparameters: Dict[str, object],
) -> Dict[str, object]:
    provenance = build_run_provenance(
        config_path=config_path,
        output_root=output_root,
        dataset_dir=dataset_dir,
        manifest_dirname=manifest_dirname,
    )
    return {
        "model_name": model_name,
        "created_at_utc": provenance["generated_at_utc"],
        "dataset_dir": str(dataset_dir),
        "output_root": str(output_root),
        "window_size": window_size,
        "feature_count": len(feature_order),
        "feature_order": list(feature_order),
        "split_summaries": list(split_summaries),
        "class_weights": class_weights,
        "hyperparameters": hyperparameters,
        "provenance": provenance,
    }
