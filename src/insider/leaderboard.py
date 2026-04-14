from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from modeling_common.provenance import metric_at


FIELDNAMES = (
    "model_name",
    "created_at_utc",
    "run_dir",
    "window_size",
    "feature_width",
    "feature_count",
    "seed",
    "pooling",
    "include_summary_features",
    "selector_name",
    "selector_value",
    "validation_auc_roc",
    "validation_pr_auc",
    "validation_brier_score",
    "test_auc_roc",
    "test_pr_auc",
    "test_brier_score",
    "dataset_dir",
    "dataset_manifest_path",
    "config_path",
    "config_hash",
    "git_sha",
)


def build_model_leaderboard(output_root: Path, *, output_path: Path | None = None) -> Path:
    models_dir = output_root / "models"
    resolved_output_path = output_path or (output_root / "reports" / "model_leaderboard.csv")
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if models_dir.exists():
        for summary_path in sorted(models_dir.rglob("summary.json")):
            metrics_path = summary_path.parent / "metrics.json"
            if not metrics_path.exists():
                continue
            summary = _read_json(summary_path)
            metrics = _read_json(metrics_path)
            if not isinstance(summary, Mapping) or not isinstance(metrics, Mapping):
                continue
            rows.append(_build_row(summary_path.parent, summary, metrics))

    rows.sort(
        key=lambda row: (
            _sort_key(row["validation_auc_roc"]),
            _sort_key(row["validation_pr_auc"]),
            str(row["created_at_utc"] or ""),
        ),
        reverse=True,
    )

    with resolved_output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDNAMES))
        writer.writeheader()
        writer.writerows(rows)
    return resolved_output_path


def _build_row(run_dir: Path, summary: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    provenance = summary.get("provenance", {})
    if not isinstance(provenance, Mapping):
        provenance = {}
    model_config = _first_mapping(summary, "model_config", "training_config", "hyperparameters")
    selector_name, selector_value = _extract_selector(metrics)
    return {
        "model_name": summary.get("model_name"),
        "created_at_utc": summary.get("created_at_utc"),
        "run_dir": str(run_dir),
        "window_size": summary.get("window_size"),
        "feature_width": summary.get("feature_width", _feature_width(summary)),
        "feature_count": summary.get("feature_count"),
        "seed": model_config.get("seed"),
        "pooling": model_config.get("pooling"),
        "include_summary_features": model_config.get("include_summary_features"),
        "selector_name": selector_name,
        "selector_value": selector_value,
        "validation_auc_roc": metric_at(metrics, "validation", "auc_roc"),
        "validation_pr_auc": metric_at(metrics, "validation", "pr_auc"),
        "validation_brier_score": metric_at(metrics, "validation", "brier_score"),
        "test_auc_roc": metric_at(metrics, "test", "auc_roc"),
        "test_pr_auc": metric_at(metrics, "test", "pr_auc"),
        "test_brier_score": metric_at(metrics, "test", "brier_score"),
        "dataset_dir": summary.get("dataset_dir"),
        "dataset_manifest_path": provenance.get("dataset_manifest_path"),
        "config_path": provenance.get("config_path"),
        "config_hash": provenance.get("config_hash"),
        "git_sha": provenance.get("git_sha"),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _first_mapping(payload: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _feature_width(summary: Mapping[str, Any]) -> int | None:
    feature_order = summary.get("feature_order")
    if isinstance(feature_order, list):
        return len(feature_order)
    return None


def _extract_selector(metrics: Mapping[str, Any]) -> tuple[str | None, Any]:
    for field in ("best_epoch", "best_iteration", "best_iteration_1based"):
        if field in metrics:
            return field, metrics.get(field)
    return None, None


def _sort_key(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("-inf")
