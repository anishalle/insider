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


def prepare_run_directory(
    output_root: Path,
    *,
    window_size: int,
    output_dir: Optional[Path] = None,
) -> Path:
    if output_dir is not None:
        run_dir = output_dir
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = output_root / "models" / "logistic_regression" / ("window_size=%d" % window_size) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


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
    predicted = (probabilities >= threshold).astype(np.int64)
    table = pa.table(
        {
            "window_id": list(window_ids),
            "label": labels.astype(np.int64),
            "probability": probabilities.astype(np.float64),
            "prediction": predicted,
            "split": [split] * len(window_ids),
        }
    )
    pq.write_table(table, str(path), compression="zstd")


def write_model_json(
    path: Path,
    *,
    weights: np.ndarray,
    bias: float,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    feature_order: Sequence[str],
    window_size: int,
    class_weights: Dict[str, float],
    hyperparameters: Dict[str, object],
) -> None:
    payload = {
        "model_type": "logistic_regression",
        "window_size": window_size,
        "feature_order": list(feature_order),
        "weights": weights.astype(np.float64).tolist(),
        "bias": float(bias),
        "feature_mean": feature_mean.astype(np.float64).tolist(),
        "feature_scale": feature_scale.astype(np.float64).tolist(),
        "class_weights": class_weights,
        "hyperparameters": hyperparameters,
    }
    write_json(path, payload)


def build_summary_payload(
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
    training_config: Dict[str, object],
) -> Dict[str, object]:
    split_rows = list(split_summaries)
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
        "feature_count": len(feature_order) * window_size,
        "feature_width": len(feature_order),
        "feature_order": list(feature_order),
        "split_summaries": split_rows,
        "class_weights": class_weights,
        "training_config": training_config,
        "dataset_balance": {
            "train_positive_rows": class_weights.get("train_positive_rows"),
            "train_negative_rows": class_weights.get("train_negative_rows"),
            "train_positive_rate": class_weights.get("train_positive_rate"),
        },
        "provenance": provenance,
    }
