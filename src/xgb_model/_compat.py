from __future__ import annotations

import importlib
from importlib import metadata
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

_EXTERNAL_XGBOOST: Optional[ModuleType] = None


def load_external_xgboost() -> ModuleType:
    global _EXTERNAL_XGBOOST
    if _EXTERNAL_XGBOOST is not None:
        return _EXTERNAL_XGBOOST

    try:
        metadata.distribution("xgboost")
    except metadata.PackageNotFoundError as exc:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "The third-party xgboost package is not installed in the active environment."
        ) from exc

    repo_src = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[2]
    original_sys_path = list(sys.path)
    local_package = sys.modules.get("xgboost")
    try:
        sys.path = [entry for entry in sys.path if not _is_local_repo_path(entry, repo_root, repo_src)]
        sys.modules.pop("xgboost", None)
        module = importlib.import_module("xgboost")
    finally:
        sys.path = original_sys_path
        if local_package is not None:
            sys.modules["xgboost"] = local_package
    _EXTERNAL_XGBOOST = module
    return module


def _is_local_repo_path(entry: str, repo_root: Path, repo_src: Path) -> bool:
    if entry == "":
        return True
    try:
        resolved = Path(entry).resolve()
    except Exception:
        return False
    return resolved == repo_root or resolved == repo_src
