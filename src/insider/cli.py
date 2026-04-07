from __future__ import annotations

import argparse
from pathlib import Path

from insider.config import load_config
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

    for command in ("prepare-trades", "label-user-trades", "build-sequences", "build-model-windows", "run-pipeline", "smoke-test"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--config", type=Path, required=True, help="Path to the pipeline TOML config.")
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

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "prepare-trades":
        result = prepare_trades(
            config,
            start=_parse_optional_datetime(args.start),
            end=_parse_optional_datetime(args.end),
            overwrite=args.overwrite,
        )
    elif args.command == "label-user-trades":
        result = label_user_trades(config, overwrite=args.overwrite)
    elif args.command == "build-sequences":
        result = build_sequences(config, overwrite=args.overwrite)
    elif args.command == "build-model-windows":
        result = build_model_windows(
            config,
            window_size=args.window_size,
            stride=args.stride,
            overwrite=args.overwrite,
        )
    elif args.command == "run-pipeline":
        result = run_pipeline(
            config,
            start=_parse_optional_datetime(args.start),
            end=_parse_optional_datetime(args.end),
            overwrite=args.overwrite,
        )
    else:
        result = smoke_test(config, overwrite=args.overwrite)

    print(result)


def _parse_optional_datetime(value: str | None):
    if value in (None, ""):
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    from datetime import datetime

    return datetime.fromisoformat(value)
