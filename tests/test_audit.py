from __future__ import annotations

import json
from pathlib import Path

from insider.audit import run_audit_task, submit_audit_manifest, write_audit_manifest
from insider.config import load_config


def test_write_audit_manifest_emits_expected_task_groups(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    manifest_path = tmp_path / "audit-manifest.json"

    result = write_audit_manifest(
        config,
        config_path=config_path,
        output_path=manifest_path,
        phase="coarse",
        max_parallel_jobs=4,
    )

    assert result == manifest_path
    payload = json.loads(manifest_path.read_text())
    assert payload["phase"] == "coarse"
    assert payload["max_parallel_jobs"] == 4
    assert len(payload["task_groups"]["build_model_windows"]) == 6
    assert len(payload["task_groups"]["train_xgboost"]) == 25
    assert len(payload["task_groups"]["train_lstm"]) == 13
    assert len(payload["task_groups"]["aggregate_reports"]) == 2

    first_window_task = payload["task_groups"]["build_model_windows"][0]
    assert "--output-dir" in first_window_task["command"]
    first_xgboost_task = payload["task_groups"]["train_xgboost"][0]
    assert "--dataset-dir" in first_xgboost_task["command"]
    assert "--output-dir" in first_xgboost_task["command"]
    first_lstm_task = payload["task_groups"]["train_lstm"][0]
    assert "--debug-metrics" in first_lstm_task["command"]
    aggregate_tasks = payload["task_groups"]["aggregate_reports"]
    assert aggregate_tasks[0]["command"][-1] == str(tmp_path / "processed" / "reports" / "model-leaderboard-coarse.csv")
    assert aggregate_tasks[1]["command"][-1] == str(tmp_path / "processed" / "reports" / "model-audit-coarse.csv")


def test_run_audit_task_executes_manifest_command(tmp_path: Path) -> None:
    manifest_path = tmp_path / "audit-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "task_groups": {
                    "aggregate_reports": [
                        {
                            "task_id": "ping",
                            "command": ["python3", "-c", "print('audit-task-ok')"],
                        }
                    ]
                }
            }
        )
        + "\n"
    )

    command = run_audit_task(
        manifest_path=manifest_path,
        group_name="aggregate_reports",
        group_index=0,
    )

    assert command == "python3 -c print('audit-task-ok')"


def test_submit_audit_manifest_dry_run_emits_grouped_sbatch_commands(tmp_path: Path) -> None:
    manifest_path = tmp_path / "audit-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "max_parallel_jobs": 4,
                "task_groups": {
                    "build_model_windows": [{"task_id": "build-1", "command": ["python3", "-c", "print(1)"]}],
                    "train_xgboost": [{"task_id": "xgb-1", "command": ["python3", "-c", "print(2)"]}],
                    "train_lstm": [{"task_id": "lstm-1", "command": ["python3", "-c", "print(3)"]}],
                    "aggregate_reports": [{"task_id": "agg-1", "command": ["python3", "-c", "print(4)"]}],
                },
            }
        )
        + "\n"
    )

    commands = submit_audit_manifest(manifest_path=manifest_path, dry_run=True)

    assert len(commands) == 4
    assert "AUDIT_GROUP_NAME=build_model_windows" in commands[0]
    assert "AUDIT_GROUP_NAME=train_xgboost" in commands[1]
    assert "AUDIT_GROUP_NAME=train_lstm" in commands[2]
    assert "AUDIT_GROUP_NAME=aggregate_reports" in commands[3]


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "pipeline.toml"
    config_path.write_text(
        "\n".join(
            [
                "[inputs]",
                f'trades_path = "{tmp_path / "trades.parquet"}"',
                "",
                "[output]",
                f'root = "{tmp_path / "processed"}"',
                "",
                "[runtime]",
                "threads = 32",
                'memory_limit = "192GB"',
                f'temp_directory = "{tmp_path / "tmp"}"',
                "",
                "[model_windows]",
                "length = 50",
                "stride = 16",
                'feature_order = ["price_yes", "signed_token_amount", "usd_amount", "side", "role_is_maker", "time_delta_seconds", "market_age_seconds", "market_trade_count_1h", "market_volume_1h", "market_price_mean_1h", "market_price_std_1h", "market_price_return_1h", "user_trade_count_1h", "user_market_trade_count_1h", "user_signed_flow_1h", "user_usd_volume_1h"]',
                "train_ratio = 0.8",
                "validation_ratio = 0.1",
                "test_ratio = 0.1",
            ]
        )
        + "\n"
    )
    return config_path
