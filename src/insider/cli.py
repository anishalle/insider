from __future__ import annotations

import argparse
from pathlib import Path

from insider.audit import build_model_audit, run_audit_task, submit_audit_manifest, write_audit_manifest
from insider.analysis import visualize_signals
from insider.config import load_config
from insider.leaderboard import build_model_leaderboard
from insider.pipeline import (
    build_model_windows,
    build_sequences,
    label_user_trades,
    prepare_trades,
    run_pipeline,
    smoke_test,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="insider", description="Preprocess Polymarket trading data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in (
        "prepare-trades",
        "label-user-trades",
        "build-sequences",
        "build-model-windows",
        "run-pipeline",
        "smoke-test",
        "visualize-signals",
        "build-model-leaderboard",
        "build-model-audit",
        "write-audit-manifest",
        "run-audit-task",
        "submit-audit-manifest",
    ):
        command_parser = subparsers.add_parser(command)
        if command not in {"run-audit-task", "submit-audit-manifest"}:
            command_parser.add_argument("--config", type=Path, required=True, help="Path to the pipeline TOML config.")
        if command not in {"visualize-signals", "build-model-leaderboard", "build-model-audit", "write-audit-manifest"}:
            command_parser.add_argument("--overwrite", action="store_true", help="Replace existing stage output.")
        if command in {"prepare-trades", "run-pipeline"}:
            command_parser.add_argument("--start", help="Optional inclusive UTC ISO timestamp filter.")
            command_parser.add_argument("--end", help="Optional exclusive UTC ISO timestamp filter.")
        if command == "build-model-windows":
            command_parser.add_argument(
                "--window-size",
                type=int,
                help="Optional model-window length override. Defaults to [model_windows].length.",
            )
            command_parser.add_argument(
                "--stride",
                type=int,
                help="Optional model-window stride override. Defaults to [model_windows].stride.",
            )
            command_parser.add_argument(
                "--output-dir",
                type=Path,
                help="Optional output directory override for the emitted model-window dataset.",
            )
        if command == "visualize-signals":
            command_parser.add_argument(
                "--output-dir",
                type=Path,
                help="Directory where analysis summaries and plots are written. Defaults to <output.root>/reports/signal-review.",
            )
            command_parser.add_argument(
                "--examples-per-class",
                type=int,
                default=4,
                help="How many strong true and false examples to plot.",
            )
            command_parser.add_argument(
                "--ambiguous-examples",
                type=int,
                default=2,
                help="How many near-zero markout examples to include for calibration.",
            )
            command_parser.add_argument(
                "--lookback-minutes",
                type=int,
                default=60,
                help="Minutes of context before the selected trade to include in each plot.",
            )
            command_parser.add_argument(
                "--lookahead-minutes",
                type=int,
                default=30,
                help="Minutes of context after the selected trade to include in each plot.",
            )
            command_parser.add_argument(
                "--model-window-size",
                type=int,
                default=50,
                help="Model-window size used for training-readiness estimates and optional summary reads.",
            )
        if command == "build-model-leaderboard":
            command_parser.add_argument(
                "--output-path",
                type=Path,
                help="Optional CSV path for the generated leaderboard. Defaults to <output.root>/reports/model_leaderboard.csv.",
            )
        if command == "build-model-audit":
            command_parser.add_argument(
                "--output-path",
                type=Path,
                help="Optional CSV path for the generated audit report. Defaults to <output.root>/reports/model_audit.csv.",
            )
        if command == "write-audit-manifest":
            command_parser.add_argument("--output-path", type=Path, required=True, help="Path to the audit manifest JSON.")
            command_parser.add_argument(
                "--phase",
                choices=("coarse", "focused"),
                default="coarse",
                help="Whether to emit the coarse search manifest or the focused follow-up manifest.",
            )
            command_parser.add_argument(
                "--audit-path",
                type=Path,
                help="Existing audit CSV used to seed focused follow-up manifests.",
            )
            command_parser.add_argument(
                "--max-parallel-jobs",
                type=int,
                default=4,
                help="Maximum concurrent jobs the manifest should assume.",
            )
        if command == "run-audit-task":
            command_parser.add_argument("--manifest-path", type=Path, required=True, help="Path to the audit manifest JSON.")
            command_parser.add_argument("--group-name", required=True, help="Audit task group name.")
            command_parser.add_argument("--group-index", type=int, required=True, help="Zero-based task index within the group.")
        if command == "submit-audit-manifest":
            command_parser.add_argument("--manifest-path", type=Path, required=True, help="Path to the audit manifest JSON.")
            command_parser.add_argument("--dry-run", action="store_true", help="Print the sbatch commands instead of submitting them.")

    args = parser.parse_args()
    config = load_config(args.config) if hasattr(args, "config") else None

    if args.command == "prepare-trades":
        assert config is not None
        result = prepare_trades(
            config,
            start=_parse_optional_datetime(args.start),
            end=_parse_optional_datetime(args.end),
            overwrite=args.overwrite,
        )
    elif args.command == "label-user-trades":
        assert config is not None
        result = label_user_trades(config, overwrite=args.overwrite)
    elif args.command == "build-sequences":
        assert config is not None
        result = build_sequences(config, overwrite=args.overwrite)
    elif args.command == "build-model-windows":
        assert config is not None
        result = build_model_windows(
            config,
            window_size=args.window_size,
            stride=args.stride,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    elif args.command == "run-pipeline":
        assert config is not None
        result = run_pipeline(
            config,
            start=_parse_optional_datetime(args.start),
            end=_parse_optional_datetime(args.end),
            overwrite=args.overwrite,
        )
    elif args.command == "build-model-leaderboard":
        assert config is not None
        result = build_model_leaderboard(config.output.root, output_path=args.output_path)
    elif args.command == "build-model-audit":
        assert config is not None
        result = build_model_audit(config.output.root, output_path=args.output_path)
    elif args.command == "write-audit-manifest":
        assert config is not None
        result = write_audit_manifest(
            config,
            config_path=args.config,
            output_path=args.output_path,
            phase=args.phase,
            audit_path=args.audit_path,
            max_parallel_jobs=args.max_parallel_jobs,
        )
    elif args.command == "run-audit-task":
        result = run_audit_task(
            manifest_path=args.manifest_path,
            group_name=args.group_name,
            group_index=args.group_index,
        )
    elif args.command == "submit-audit-manifest":
        result = submit_audit_manifest(manifest_path=args.manifest_path, dry_run=args.dry_run)
    else:
        assert config is not None
        if args.command == "smoke-test":
            result = smoke_test(config, overwrite=args.overwrite)
        else:
            result = visualize_signals(
                config,
                output_dir=args.output_dir,
                config_path=args.config,
                examples_per_class=args.examples_per_class,
                ambiguous_examples=args.ambiguous_examples,
                lookback_minutes=args.lookback_minutes,
                lookahead_minutes=args.lookahead_minutes,
                model_window_size=args.model_window_size,
            )

    if isinstance(result, list):
        for item in result:
            print(item)
    else:
        print(result)


def _parse_optional_datetime(value: str | None):
    if value in (None, ""):
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    from datetime import datetime

    return datetime.fromisoformat(value)
