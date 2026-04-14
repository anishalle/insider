from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def config_hash(config_path: Optional[Path]) -> Optional[str]:
    if config_path is None:
        return None
    resolved = config_path.resolve()
    if not resolved.exists():
        return None
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def find_repo_root(start_path: Optional[Path]) -> Optional[Path]:
    if start_path is None:
        return None
    current = start_path.resolve()
    if current.is_file():
        current = current.parent
    while True:
        if (current / ".git").exists():
            return current
        if current.parent == current:
            return None
        current = current.parent


def git_sha(start_path: Optional[Path]) -> Optional[str]:
    repo_root = find_repo_root(start_path)
    if repo_root is None:
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def find_dataset_manifest_path(
    output_root: Path,
    dataset_dir: Path,
    *,
    manifest_dirname: str = "manifests",
) -> Optional[Path]:
    manifest_dir = output_root / manifest_dirname
    if not manifest_dir.exists():
        return None
    target = str(dataset_dir.resolve())
    for manifest_path in sorted(manifest_dir.glob("*.json")):
        try:
            payload = json.loads(manifest_path.read_text())
        except Exception:
            continue
        candidate = payload.get("dataset_dir")
        if candidate is None:
            continue
        try:
            candidate_path = str(Path(str(candidate)).resolve())
        except Exception:
            continue
        if candidate_path == target:
            return manifest_path
    return None


def load_manifest_metadata(manifest_path: Optional[Path]) -> dict[str, Any]:
    if manifest_path is None or not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_run_provenance(
    *,
    config_path: Optional[Path],
    output_root: Path,
    dataset_dir: Optional[Path] = None,
    manifest_dirname: str = "manifests",
    generated_at_utc: Optional[str] = None,
) -> dict[str, Any]:
    manifest_path = (
        find_dataset_manifest_path(output_root, dataset_dir, manifest_dirname=manifest_dirname)
        if dataset_dir is not None
        else None
    )
    manifest_payload = load_manifest_metadata(manifest_path)
    return {
        "generated_at_utc": generated_at_utc or utc_now_iso(),
        "git_sha": git_sha(config_path or output_root),
        "config_path": str(config_path.resolve()) if config_path is not None else None,
        "config_hash": config_hash(config_path),
        "dataset_dir": str(dataset_dir) if dataset_dir is not None else None,
        "dataset_manifest_path": str(manifest_path) if manifest_path is not None else None,
        "dataset_manifest_stage": manifest_payload.get("stage"),
        "dataset_manifest_row_count": _maybe_int(manifest_payload.get("row_count")),
        "dataset_manifest_min_trade_time": manifest_payload.get("min_trade_time"),
        "dataset_manifest_max_trade_time": manifest_payload.get("max_trade_time"),
    }


def build_stage_provenance(
    *,
    config_path: Optional[Path],
    output_root: Path,
    generated_at_utc: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "generated_at_utc": generated_at_utc or utc_now_iso(),
        "git_sha": git_sha(config_path or output_root),
        "config_path": str(config_path.resolve()) if config_path is not None else None,
        "config_hash": config_hash(config_path),
    }


def metric_at(metrics: Mapping[str, Any], section: str, field: str) -> Any:
    section_payload = metrics.get(section, {})
    if not isinstance(section_payload, Mapping):
        return None
    return section_payload.get(field)


def _maybe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None
