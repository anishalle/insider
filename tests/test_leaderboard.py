from __future__ import annotations

import csv
import json
from pathlib import Path

from insider.audit import build_model_audit
from insider.leaderboard import build_model_leaderboard


def test_build_model_leaderboard_collects_run_summaries(tmp_path: Path) -> None:
    output_root = tmp_path / "processed"
    _write_run(
        output_root=output_root,
        model_name="xgboost",
        stamp="20260101T000000Z",
        summary={
            "model_name": "xgboost",
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            "dataset_dir": str(output_root / "model_windows" / "window_size=2"),
            "window_size": 2,
            "feature_count": 14,
            "feature_order": ["price_yes"] * 7,
            "training_config": {"seed": 7},
            "provenance": {
                "config_path": str((tmp_path / "pipeline.toml").resolve()),
                "config_hash": "abc123",
                "git_sha": "deadbeef",
                "git_dirty": False,
                "dataset_manifest_path": str(output_root / "manifests" / "build_model_windows_2.json"),
            },
        },
        metrics={
            "best_iteration": 11,
            "validation": {"auc_roc": 0.71, "pr_auc": 0.62, "brier_score": 0.22},
            "test": {"auc_roc": 0.7, "pr_auc": 0.61, "brier_score": 0.23},
        },
        diagnostics={
            "validation_threshold_sweep": {
                "best_by_metric": {
                    "accuracy": {"threshold": 0.6},
                    "f1": {"threshold": 0.55},
                }
            },
            "split_metrics_at_validation_best_thresholds": {
                "validation": {
                    "validation_best_accuracy": {"accuracy": 0.64},
                    "validation_best_f1": {"f1": 0.58},
                },
                "test": {
                    "validation_best_accuracy": {
                        "accuracy": 0.63,
                        "precision": 0.68,
                        "recall": 0.42,
                    },
                    "validation_best_f1": {"f1": 0.57},
                },
            },
        },
    )
    _write_run(
        output_root=output_root,
        model_name="lstm",
        stamp="20260102T000000Z",
        summary={
            "model_name": "lstm",
            "created_at_utc": "2026-01-02T00:00:00+00:00",
            "dataset_dir": str(output_root / "model_windows" / "window_size=2"),
            "window_size": 2,
            "feature_count": 14,
            "feature_order": ["price_yes"] * 7,
            "model_config": {"seed": 42, "pooling": "attention"},
            "provenance": {
                "config_path": str((tmp_path / "pipeline.toml").resolve()),
                "config_hash": "def456",
                "git_sha": "beadfeed",
                "git_dirty": True,
                "dataset_manifest_path": str(output_root / "manifests" / "build_model_windows_2.json"),
            },
        },
        metrics={
            "best_epoch": 3,
            "validation": {"auc_roc": 0.69, "pr_auc": 0.65, "brier_score": 0.21},
            "test": {"auc_roc": 0.68, "pr_auc": 0.64, "brier_score": 0.22},
        },
        diagnostics={
            "validation_threshold_sweep": {
                "best_by_metric": {
                    "accuracy": {"threshold": 0.58},
                    "f1": {"threshold": 0.52},
                }
            },
            "split_metrics_at_validation_best_thresholds": {
                "validation": {
                    "validation_best_accuracy": {"accuracy": 0.66},
                    "validation_best_f1": {"f1": 0.61},
                },
                "test": {
                    "validation_best_accuracy": {
                        "accuracy": 0.65,
                        "precision": 0.67,
                        "recall": 0.49,
                    },
                    "validation_best_f1": {"f1": 0.6},
                },
            },
        },
    )

    output_path = build_model_leaderboard(output_root)
    audit_path = build_model_audit(output_root)

    with output_path.open() as handle:
        rows = list(csv.DictReader(handle))
    with audit_path.open() as handle:
        audit_rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0]["model_name"] == "lstm"
    assert rows[0]["selector_name"] == "best_epoch"
    assert rows[0]["selector_value"] == "3"
    assert rows[0]["validation_best_accuracy_threshold"] == "0.58"
    assert rows[0]["test_accuracy_at_validation_best_accuracy"] == "0.65"
    assert rows[0]["test_precision_at_validation_best_accuracy"] == "0.67"
    assert rows[0]["test_recall_at_validation_best_accuracy"] == "0.49"
    assert rows[0]["git_dirty"] == "True"
    assert rows[1]["config_hash"] == "abc123"
    assert audit_rows == rows


def _write_run(
    *,
    output_root: Path,
    model_name: str,
    stamp: str,
    summary: dict[str, object],
    metrics: dict[str, object],
    diagnostics: dict[str, object],
) -> None:
    run_dir = output_root / "models" / model_name / "window_size=2" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(metrics) + "\n")
    (run_dir / "diagnostics.json").write_text(json.dumps(diagnostics) + "\n")
