from __future__ import annotations

import csv
import json
from pathlib import Path

from insider.leaderboard import build_model_leaderboard


def test_build_model_leaderboard_collects_run_summaries(tmp_path: Path) -> None:
    output_root = tmp_path / "processed"
    run_dir = output_root / "models" / "logistic_regression" / "window_size=2" / "20260101T000000Z"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model_name": "logistic_regression",
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
            "dataset_manifest_path": str(output_root / "manifests" / "build_model_windows_2.json"),
        },
    }
    metrics = {
        "best_epoch": 3,
        "validation": {"auc_roc": 0.61, "pr_auc": 0.62, "brier_score": 0.2},
        "test": {"auc_roc": 0.6, "pr_auc": 0.61, "brier_score": 0.21},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(metrics) + "\n")

    output_path = build_model_leaderboard(output_root)

    with output_path.open() as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["model_name"] == "logistic_regression"
    assert rows[0]["selector_name"] == "best_epoch"
    assert rows[0]["selector_value"] == "3"
    assert rows[0]["config_hash"] == "abc123"
