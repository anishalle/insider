from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from insider.config import PipelineConfig
from insider.leaderboard import collect_model_report_rows, write_model_report
from modeling_common.provenance import find_repo_root, utc_now_iso


def build_model_audit(output_root: Path, *, output_path: Path | None = None) -> Path:
    resolved_output_path = output_path or (output_root / "reports" / "model_audit.csv")
    rows = collect_model_report_rows(output_root)
    write_model_report(rows, resolved_output_path)
    return resolved_output_path


def write_audit_manifest(
    config: PipelineConfig,
    *,
    config_path: Path,
    output_path: Path,
    phase: str = "coarse",
    audit_path: Path | None = None,
    max_parallel_jobs: int = 4,
) -> Path:
    resolved_phase = phase.strip().lower()
    if resolved_phase not in {"coarse", "focused"}:
        raise ValueError("phase must be 'coarse' or 'focused'.")

    task_groups = (
        _build_coarse_task_groups(config=config, config_path=config_path)
        if resolved_phase == "coarse"
        else _build_focused_task_groups(config=config, config_path=config_path, audit_path=audit_path)
    )
    payload = {
        "generated_at_utc": utc_now_iso(),
        "phase": resolved_phase,
        "config_path": str(config_path.resolve()),
        "output_root": str(config.output.root),
        "max_parallel_jobs": max(1, min(int(max_parallel_jobs), 4)),
        "task_groups": task_groups,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return output_path


def run_audit_task(*, manifest_path: Path, group_name: str, group_index: int) -> str:
    manifest = _load_manifest(manifest_path)
    groups = manifest.get("task_groups", {})
    if not isinstance(groups, Mapping) or group_name not in groups:
        raise KeyError(f"Unknown audit task group: {group_name}")
    tasks = groups[group_name]
    if not isinstance(tasks, list):
        raise ValueError(f"Audit task group is malformed: {group_name}")
    index = int(group_index)
    if index < 0 or index >= len(tasks):
        raise IndexError(f"Audit task index {index} is out of range for {group_name}.")
    task = tasks[index]
    if not isinstance(task, Mapping):
        raise ValueError("Audit task entry is malformed.")
    command = task.get("command")
    if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
        raise ValueError("Audit task command is malformed.")
    subprocess.run(command, check=True)
    return " ".join(command)


def submit_audit_manifest(*, manifest_path: Path, dry_run: bool = False) -> list[str]:
    manifest = _load_manifest(manifest_path)
    task_groups = manifest.get("task_groups", {})
    if not isinstance(task_groups, Mapping):
        raise ValueError("Audit manifest is missing task_groups.")

    max_parallel_jobs = max(1, min(int(manifest.get("max_parallel_jobs", 4)), 4))
    build_tasks = _group_tasks(task_groups, "build_model_windows")
    xgboost_tasks = _group_tasks(task_groups, "train_xgboost")
    lstm_tasks = _group_tasks(task_groups, "train_lstm")
    aggregate_tasks = _group_tasks(task_groups, "aggregate_reports")

    commands: list[str] = []
    build_job_id = None
    if build_tasks:
        build_job_id, build_cmd = _submit_group(
            script_path=Path("jobs/audit/run-window-build-dev.sbatch"),
            manifest_path=manifest_path,
            group_name="build_model_windows",
            task_count=len(build_tasks),
            concurrency=min(2, max_parallel_jobs),
            dry_run=dry_run,
        )
        commands.append(build_cmd)

    remaining_parallel = max_parallel_jobs
    xgboost_parallel = 0
    lstm_parallel = 0
    if xgboost_tasks and lstm_tasks:
        xgboost_parallel = max(1, max_parallel_jobs // 2)
        lstm_parallel = max(1, max_parallel_jobs - xgboost_parallel)
    elif xgboost_tasks:
        xgboost_parallel = max_parallel_jobs
    elif lstm_tasks:
        lstm_parallel = max_parallel_jobs

    train_dependencies = [job_id for job_id in (build_job_id,) if job_id]
    xgboost_job_id = None
    lstm_job_id = None
    if xgboost_tasks:
        xgboost_job_id, xgboost_cmd = _submit_group(
            script_path=Path("jobs/audit/run-xgboost-sweep-normal.sbatch"),
            manifest_path=manifest_path,
            group_name="train_xgboost",
            task_count=len(xgboost_tasks),
            concurrency=xgboost_parallel,
            dependency_job_ids=train_dependencies,
            dry_run=dry_run,
        )
        commands.append(xgboost_cmd)
    if lstm_tasks:
        lstm_job_id, lstm_cmd = _submit_group(
            script_path=Path("jobs/audit/run-lstm-sweep-a30-2.12gb.sbatch"),
            manifest_path=manifest_path,
            group_name="train_lstm",
            task_count=len(lstm_tasks),
            concurrency=lstm_parallel,
            dependency_job_ids=train_dependencies,
            dry_run=dry_run,
        )
        commands.append(lstm_cmd)

    aggregate_dependencies = [job_id for job_id in (xgboost_job_id, lstm_job_id, build_job_id) if job_id]
    if aggregate_tasks:
        _, aggregate_cmd = _submit_group(
            script_path=Path("jobs/audit/run-aggregate-dev.sbatch"),
            manifest_path=manifest_path,
            group_name="aggregate_reports",
            task_count=len(aggregate_tasks),
            concurrency=1,
            dependency_job_ids=aggregate_dependencies,
            dry_run=dry_run,
        )
        commands.append(aggregate_cmd)

    return commands


def _build_coarse_task_groups(*, config: PipelineConfig, config_path: Path) -> dict[str, list[dict[str, Any]]]:
    dataset_specs = {
        (50, 16),
        (32, 8),
        (32, 16),
        (50, 8),
        (64, 8),
        (64, 16),
    }
    build_tasks = [
        _build_window_task(
            config=config,
            config_path=config_path,
            phase="coarse",
            window_size=window_size,
            stride=stride,
        )
        for window_size, stride in sorted(dataset_specs)
    ]
    xgboost_tasks = _dedupe_tasks(
        [
            _build_xgboost_task(
                config=config,
                config_path=config_path,
                phase="coarse",
                task_label="baseline",
                window_size=50,
                stride=16,
                include_summary_features=True,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=1.0,
                gamma=0.0,
                reg_lambda=1.0,
                reg_alpha=0.0,
            )
        ]
        + list(_sample_xgboost_coarse_tasks(config=config, config_path=config_path))
    )
    lstm_tasks = _dedupe_tasks(
        [
            _build_lstm_task(
                config=config,
                config_path=config_path,
                phase="coarse",
                task_label="baseline",
                window_size=50,
                stride=16,
                pooling="attention",
                include_summary_features=False,
                hidden_size=128,
                num_layers=1,
                dropout=0.1,
                learning_rate=0.0003,
                weight_decay=0.0001,
            )
        ]
        + list(_sample_lstm_coarse_tasks(config=config, config_path=config_path))
    )
    aggregate_tasks = _build_aggregate_tasks(config=config, config_path=config_path, phase="coarse")
    return {
        "build_model_windows": build_tasks,
        "train_xgboost": xgboost_tasks,
        "train_lstm": lstm_tasks,
        "aggregate_reports": aggregate_tasks,
    }


def _build_focused_task_groups(
    *,
    config: PipelineConfig,
    config_path: Path,
    audit_path: Path | None,
) -> dict[str, list[dict[str, Any]]]:
    if audit_path is None:
        raise ValueError("Focused audit manifest generation requires --audit-path.")
    xgboost_rows = _top_rows_by_model(audit_path, model_name="xgboost", limit=3)
    lstm_rows = _top_rows_by_model(audit_path, model_name="lstm", limit=2)
    if not xgboost_rows and not lstm_rows:
        raise ValueError("Focused audit manifest generation found no eligible runs in the audit CSV.")

    xgboost_tasks: list[dict[str, Any]] = []
    for rank, row in enumerate(xgboost_rows, start=1):
        summary = _read_run_summary(Path(row["run_dir"]))
        training_config = _read_mapping(summary, "training_config")
        dataset_dir = Path(str(summary["dataset_dir"]))
        window_size = int(summary["window_size"])
        xgboost_tasks.append(
            _build_xgboost_task(
                config=config,
                config_path=config_path,
                phase="focused",
                task_label=f"top{rank}-lr",
                window_size=window_size,
                stride=_stride_from_dataset_dir(dataset_dir),
                dataset_dir=dataset_dir,
                include_summary_features=bool(training_config.get("include_summary_features", True)),
                learning_rate=max(0.01, float(training_config.get("learning_rate", 0.05)) * 0.6),
                max_depth=int(training_config.get("max_depth", 8)),
                subsample=min(1.0, float(training_config.get("subsample", 0.8)) + 0.1),
                colsample_bytree=min(1.0, float(training_config.get("colsample_bytree", 0.8)) + 0.1),
                min_child_weight=float(training_config.get("min_child_weight", 1.0)),
                gamma=float(training_config.get("gamma", 0.0)),
                reg_lambda=max(5.0, float(training_config.get("reg_lambda", 1.0))),
                reg_alpha=float(training_config.get("reg_alpha", 0.0)),
            )
        )
        xgboost_tasks.append(
            _build_xgboost_task(
                config=config,
                config_path=config_path,
                phase="focused",
                task_label=f"top{rank}-depth",
                window_size=window_size,
                stride=_stride_from_dataset_dir(dataset_dir),
                dataset_dir=dataset_dir,
                include_summary_features=bool(training_config.get("include_summary_features", True)),
                learning_rate=float(training_config.get("learning_rate", 0.05)),
                max_depth=min(12, int(training_config.get("max_depth", 8)) + 1),
                subsample=float(training_config.get("subsample", 0.8)),
                colsample_bytree=float(training_config.get("colsample_bytree", 0.8)),
                min_child_weight=max(1.0, float(training_config.get("min_child_weight", 1.0)) * 2.0),
                gamma=max(1.0, float(training_config.get("gamma", 0.0))),
                reg_lambda=float(training_config.get("reg_lambda", 1.0)),
                reg_alpha=max(0.5, float(training_config.get("reg_alpha", 0.0))),
            )
        )

    lstm_tasks: list[dict[str, Any]] = []
    for rank, row in enumerate(lstm_rows, start=1):
        summary = _read_run_summary(Path(row["run_dir"]))
        model_config = _read_mapping(summary, "model_config")
        dataset_dir = Path(str(summary["dataset_dir"]))
        window_size = int(summary["window_size"])
        lstm_tasks.append(
            _build_lstm_task(
                config=config,
                config_path=config_path,
                phase="focused",
                task_label=f"top{rank}-wider",
                window_size=window_size,
                stride=_stride_from_dataset_dir(dataset_dir),
                dataset_dir=dataset_dir,
                pooling=str(model_config.get("pooling", "attention")),
                include_summary_features=bool(model_config.get("include_summary_features", False)),
                hidden_size=min(256, int(model_config.get("hidden_size", 128)) * 2),
                num_layers=int(model_config.get("num_layers", 1)),
                dropout=min(0.3, float(model_config.get("dropout", 0.1)) + 0.1),
                learning_rate=min(float(model_config.get("learning_rate", 0.001)), 0.0003),
                weight_decay=max(0.0001, float(model_config.get("weight_decay", 0.0001))),
                epochs=max(20, int(model_config.get("epochs", 12)) + 8),
                patience=max(5, int(model_config.get("patience", 4))),
            )
        )
        lstm_tasks.append(
            _build_lstm_task(
                config=config,
                config_path=config_path,
                phase="focused",
                task_label=f"top{rank}-features",
                window_size=window_size,
                stride=_stride_from_dataset_dir(dataset_dir),
                dataset_dir=dataset_dir,
                pooling=str(model_config.get("pooling", "attention")),
                include_summary_features=not bool(model_config.get("include_summary_features", False)),
                hidden_size=int(model_config.get("hidden_size", 128)),
                num_layers=int(model_config.get("num_layers", 1)),
                dropout=float(model_config.get("dropout", 0.1)),
                learning_rate=float(model_config.get("learning_rate", 0.001)),
                weight_decay=max(0.001, float(model_config.get("weight_decay", 0.0001))),
                epochs=max(20, int(model_config.get("epochs", 12))),
                patience=max(5, int(model_config.get("patience", 4))),
            )
        )

    aggregate_tasks = _build_aggregate_tasks(config=config, config_path=config_path, phase="focused")
    return {
        "build_model_windows": [],
        "train_xgboost": _dedupe_tasks(xgboost_tasks),
        "train_lstm": _dedupe_tasks(lstm_tasks),
        "aggregate_reports": aggregate_tasks,
    }


def _sample_xgboost_coarse_tasks(*, config: PipelineConfig, config_path: Path) -> Iterable[dict[str, Any]]:
    window_sizes = [32, 50, 64]
    strides = [8, 16]
    summary_toggles = [True, False]
    learning_rates = [0.03, 0.05, 0.1]
    max_depths = [6, 8, 10]
    subsamples = [0.7, 0.85, 1.0]
    colsamples = [0.6, 0.8, 1.0]
    min_child_weights = [1.0, 5.0, 10.0]
    gammas = [0.0, 1.0, 5.0]
    reg_lambdas = [1.0, 5.0, 10.0]
    reg_alphas = [0.0, 0.5, 1.0]

    candidates: list[tuple[Any, ...]] = []
    for window_size in window_sizes:
        for stride in strides:
            for include_summary_features in summary_toggles:
                for learning_rate in learning_rates:
                    for max_depth in max_depths:
                        for subsample in subsamples:
                            for colsample_bytree in colsamples:
                                for min_child_weight in min_child_weights:
                                    for gamma in gammas:
                                        for reg_lambda in reg_lambdas:
                                            for reg_alpha in reg_alphas:
                                                candidates.append(
                                                    (
                                                        window_size,
                                                        stride,
                                                        include_summary_features,
                                                        learning_rate,
                                                        max_depth,
                                                        subsample,
                                                        colsample_bytree,
                                                        min_child_weight,
                                                        gamma,
                                                        reg_lambda,
                                                        reg_alpha,
                                                    )
                                                )

    for index in _sample_indices(len(candidates), sample_count=24):
        (
            window_size,
            stride,
            include_summary_features,
            learning_rate,
            max_depth,
            subsample,
            colsample_bytree,
            min_child_weight,
            gamma,
            reg_lambda,
            reg_alpha,
        ) = candidates[index]
        yield _build_xgboost_task(
            config=config,
            config_path=config_path,
            phase="coarse",
            task_label=f"sample{index:05d}",
            window_size=window_size,
            stride=stride,
            include_summary_features=include_summary_features,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
        )


def _sample_lstm_coarse_tasks(*, config: PipelineConfig, config_path: Path) -> Iterable[dict[str, Any]]:
    window_sizes = [50, 64]
    strides = [8, 16]
    poolings = ["attention", "mean_last", "max"]
    summary_toggles = [True, False]
    hidden_sizes = [128, 256]
    num_layers_options = [1, 2]
    dropouts = [0.1, 0.2, 0.3]
    learning_rates = [0.001, 0.0003]
    weight_decays = [0.0001, 0.001]

    candidates: list[tuple[Any, ...]] = []
    for window_size in window_sizes:
        for stride in strides:
            for pooling in poolings:
                for include_summary_features in summary_toggles:
                    for hidden_size in hidden_sizes:
                        for num_layers in num_layers_options:
                            for dropout in dropouts:
                                for learning_rate in learning_rates:
                                    for weight_decay in weight_decays:
                                        candidates.append(
                                            (
                                                window_size,
                                                stride,
                                                pooling,
                                                include_summary_features,
                                                hidden_size,
                                                num_layers,
                                                dropout,
                                                learning_rate,
                                                weight_decay,
                                            )
                                        )

    for index in _sample_indices(len(candidates), sample_count=12):
        (
            window_size,
            stride,
            pooling,
            include_summary_features,
            hidden_size,
            num_layers,
            dropout,
            learning_rate,
            weight_decay,
        ) = candidates[index]
        yield _build_lstm_task(
            config=config,
            config_path=config_path,
            phase="coarse",
            task_label=f"sample{index:04d}",
            window_size=window_size,
            stride=stride,
            pooling=pooling,
            include_summary_features=include_summary_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )


def _build_window_task(
    *,
    config: PipelineConfig,
    config_path: Path,
    phase: str,
    window_size: int,
    stride: int,
) -> dict[str, Any]:
    dataset_dir = _audit_dataset_dir(config, phase=phase, window_size=window_size, stride=stride)
    command_config_path = _command_config_path(config_path)
    task_id = f"windows-{phase}-w{window_size}-s{stride}"
    command = [
        "python",
        "-m",
        "insider",
        "build-model-windows",
        "--config",
        command_config_path,
        "--window-size",
        str(window_size),
        "--stride",
        str(stride),
        "--output-dir",
        str(dataset_dir),
    ]
    return {
        "task_id": task_id,
        "task_kind": "build_model_windows",
        "description": f"Build model windows for window_size={window_size}, stride={stride}",
        "command": command,
        "dataset_dir": str(dataset_dir),
    }


def _build_xgboost_task(
    *,
    config: PipelineConfig,
    config_path: Path,
    phase: str,
    task_label: str,
    window_size: int,
    stride: int,
    include_summary_features: bool,
    learning_rate: float,
    max_depth: int,
    subsample: float,
    colsample_bytree: float,
    min_child_weight: float,
    gamma: float,
    reg_lambda: float,
    reg_alpha: float,
    dataset_dir: Path | None = None,
) -> dict[str, Any]:
    resolved_dataset_dir = dataset_dir or _audit_dataset_dir(config, phase=phase, window_size=window_size, stride=stride)
    command_config_path = _command_config_path(config_path)
    include_summary_label = "sf1" if include_summary_features else "sf0"
    task_id = _slug(
        "xgb",
        phase,
        task_label,
        f"w{window_size}",
        f"s{stride}",
        include_summary_label,
        f"lr{learning_rate}",
        f"d{max_depth}",
    )
    output_dir = config.output.root / "models" / "xgboost" / f"window_size={window_size}" / f"audit-{task_id}"
    command = [
        "python",
        "-m",
        "xgb_model.train",
        "--config",
        command_config_path,
        "--window-size",
        str(window_size),
        "--dataset-dir",
        str(resolved_dataset_dir),
        "--output-dir",
        str(output_dir),
        "--num-round",
        "1000",
        "--early-stopping-rounds",
        "75",
        "--learning-rate",
        _format_float(learning_rate),
        "--max-depth",
        str(int(max_depth)),
        "--subsample",
        _format_float(subsample),
        "--colsample-bytree",
        _format_float(colsample_bytree),
        "--min-child-weight",
        _format_float(min_child_weight),
        "--gamma",
        _format_float(gamma),
        "--reg-lambda",
        _format_float(reg_lambda),
        "--reg-alpha",
        _format_float(reg_alpha),
        "--max-bin",
        "256",
        "--tree-method",
        "hist",
        "--nthread",
        str(config.runtime.threads),
    ]
    if not include_summary_features:
        command.append("--disable-summary-features")
    return {
        "task_id": task_id,
        "task_kind": "train_xgboost",
        "description": f"Train XGBoost audit run {task_id}",
        "command": command,
        "dataset_dir": str(resolved_dataset_dir),
        "output_dir": str(output_dir),
    }


def _build_lstm_task(
    *,
    config: PipelineConfig,
    config_path: Path,
    phase: str,
    task_label: str,
    window_size: int,
    stride: int,
    pooling: str,
    include_summary_features: bool,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    epochs: int = 20,
    patience: int = 5,
    dataset_dir: Path | None = None,
) -> dict[str, Any]:
    resolved_dataset_dir = dataset_dir or _audit_dataset_dir(config, phase=phase, window_size=window_size, stride=stride)
    command_config_path = _command_config_path(config_path)
    include_summary_label = "sf1" if include_summary_features else "sf0"
    task_id = _slug(
        "lstm",
        phase,
        task_label,
        f"w{window_size}",
        f"s{stride}",
        pooling,
        include_summary_label,
        f"h{hidden_size}",
        f"nl{num_layers}",
    )
    output_dir = config.output.root / "models" / "lstm" / f"window_size={window_size}" / f"audit-{task_id}"
    command = [
        "python",
        "-m",
        "lstm.train",
        "--config",
        command_config_path,
        "--window-size",
        str(window_size),
        "--dataset-dir",
        str(resolved_dataset_dir),
        "--output-dir",
        str(output_dir),
        "--device",
        "cuda",
        "--hidden-size",
        str(int(hidden_size)),
        "--num-layers",
        str(int(num_layers)),
        "--dropout",
        _format_float(dropout),
        "--pooling",
        pooling,
        "--epochs",
        str(int(epochs)),
        "--batch-size",
        "256",
        "--eval-batch-size",
        "512",
        "--learning-rate",
        _format_float(learning_rate),
        "--weight-decay",
        _format_float(weight_decay),
        "--patience",
        str(int(patience)),
        "--debug-metrics",
    ]
    if not include_summary_features:
        command.append("--disable-summary-features")
    return {
        "task_id": task_id,
        "task_kind": "train_lstm",
        "description": f"Train LSTM audit run {task_id}",
        "command": command,
        "dataset_dir": str(resolved_dataset_dir),
        "output_dir": str(output_dir),
    }


def _build_aggregate_tasks(
    *,
    config: PipelineConfig,
    config_path: Path,
    phase: str,
) -> list[dict[str, Any]]:
    command_config_path = _command_config_path(config_path)
    return [
        {
            "task_id": f"leaderboard-{phase}",
            "task_kind": "aggregate_reports",
            "description": f"Build model leaderboard for {phase} audit runs",
            "command": [
                "python",
                "-m",
                "insider",
                "build-model-leaderboard",
                "--config",
                command_config_path,
                "--output-path",
                str(_report_output_path(config.output.root, f"model-leaderboard-{phase}.csv")),
            ],
        },
        {
            "task_id": f"audit-{phase}",
            "task_kind": "aggregate_reports",
            "description": f"Build model audit report for {phase} audit runs",
            "command": [
                "python",
                "-m",
                "insider",
                "build-model-audit",
                "--config",
                command_config_path,
                "--output-path",
                str(_report_output_path(config.output.root, f"model-audit-{phase}.csv")),
            ],
        },
    ]


def _submit_group(
    *,
    script_path: Path,
    manifest_path: Path,
    group_name: str,
    task_count: int,
    concurrency: int,
    dependency_job_ids: Sequence[str] = (),
    dry_run: bool,
) -> tuple[str | None, str]:
    if task_count < 1:
        raise ValueError("task_count must be positive when submitting a group.")
    command = [
        "sbatch",
        f"--export=ALL,AUDIT_MANIFEST_PATH={manifest_path},AUDIT_GROUP_NAME={group_name}",
        f"--array=0-{task_count - 1}%{max(1, int(concurrency))}",
    ]
    if dependency_job_ids:
        command.append(f"--dependency=afterok:{':'.join(dependency_job_ids)}")
    command.append(str(script_path))
    command_str = " ".join(command)
    if dry_run:
        return None, command_str
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"Unable to parse sbatch output: {result.stdout.strip()}")
    return match.group(1), command_str


def _group_tasks(task_groups: Mapping[str, Any], group_name: str) -> list[Mapping[str, Any]]:
    tasks = task_groups.get(group_name, [])
    if not isinstance(tasks, list):
        raise ValueError(f"Task group {group_name} is malformed.")
    return [task for task in tasks if isinstance(task, Mapping)]


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _report_output_path(output_root: Path, filename: str) -> Path:
    return output_root / "reports" / filename


def _audit_dataset_dir(config: PipelineConfig, *, phase: str, window_size: int, stride: int) -> Path:
    return config.output.root / "audit_datasets" / phase / f"window_size={window_size}" / f"stride={stride}"


def _sample_indices(total_count: int, *, sample_count: int) -> list[int]:
    if sample_count >= total_count:
        return list(range(total_count))
    step = total_count / float(sample_count)
    indices = {min(total_count - 1, int(index * step)) for index in range(sample_count)}
    if len(indices) < sample_count:
        indices.update(range(sample_count - len(indices)))
    return sorted(indices)[:sample_count]


def _slug(*parts: object) -> str:
    text = "-".join(str(part) for part in parts)
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return normalized


def _format_float(value: float) -> str:
    return f"{float(value):g}"


def _dedupe_tasks(tasks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task in tasks:
        task_id = str(task.get("task_id"))
        if task_id in seen:
            continue
        seen.add(task_id)
        deduped.append(task)
    return deduped


def _top_rows_by_model(audit_path: Path, *, model_name: str, limit: int) -> list[dict[str, str]]:
    with audit_path.open() as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("model_name") == model_name]
    return rows[:limit]


def _read_run_summary(run_dir: Path) -> Mapping[str, Any]:
    payload = json.loads((run_dir / "summary.json").read_text())
    if not isinstance(payload, Mapping):
        raise ValueError(f"Run summary is malformed: {run_dir}")
    return payload


def _read_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _stride_from_dataset_dir(dataset_dir: Path) -> int:
    for part in dataset_dir.parts:
        if part.startswith("stride="):
            return int(part.split("=", 1)[1])
    return 16


def _command_config_path(config_path: Path) -> str:
    repo_root = find_repo_root(config_path)
    if repo_root is None:
        return str(config_path)
    try:
        return str(config_path.resolve().relative_to(repo_root))
    except Exception:
        return str(config_path)
