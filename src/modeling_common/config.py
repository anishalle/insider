from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.9 in Juno
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass(frozen=True)
class WindowDataConfig:
    config_path: Path
    output_root: Path
    model_window_dirname: str
    feature_order: Tuple[str, ...]
    window_size: int

    @property
    def dataset_dir(self) -> Path:
        return self.output_root / self.model_window_dirname / f"window_size={self.window_size}"


def load_window_data_config(config_path: Path, window_size: Optional[int] = None) -> WindowDataConfig:
    raw = _load_toml(config_path)
    output = raw.get("output", {})
    model_windows = raw.get("model_windows", {})
    resolved_window_size = int(model_windows.get("length", 50) if window_size is None else window_size)
    feature_order = tuple(model_windows.get("feature_order", ()))
    return WindowDataConfig(
        config_path=config_path,
        output_root=Path(output.get("root", "outputs/default")),
        model_window_dirname=str(output.get("model_window_dirname", "model_windows")),
        feature_order=feature_order,
        window_size=resolved_window_size,
    )


def _load_toml(path: Path) -> Dict[str, Any]:
    return tomllib.loads(path.read_text())

