from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class WindowConfig:
    config_path: Path
    output_root: Path
    model_window_dirname: str
    manifest_dirname: str
    window_size: int
    feature_order: Tuple[str, ...]

    @property
    def dataset_dir(self) -> Path:
        return self.output_root / self.model_window_dirname / ("window_size=%d" % self.window_size)


def load_window_config(config_path: Path, window_size: Optional[int] = None) -> WindowConfig:
    text = config_path.read_text()
    output_root = _extract_path(text, "output", "root", default="outputs/default")
    model_window_dirname = _extract_scalar(text, "output", "model_window_dirname", default="model_windows")
    config_window_size = _extract_int(text, "model_windows", "length", default=50)
    resolved_window_size = int(window_size if window_size is not None else config_window_size)
    feature_order = tuple(_extract_string_list(text, "model_windows", "feature_order"))
    if not feature_order:
        raise ValueError("Could not parse model_windows.feature_order from %s" % config_path)
    return WindowConfig(
        config_path=config_path.resolve(),
        output_root=Path(output_root),
        model_window_dirname=model_window_dirname,
        manifest_dirname=_extract_scalar(text, "output", "manifest_dirname", default="manifests"),
        window_size=resolved_window_size,
        feature_order=feature_order,
    )


def _extract_path(text: str, section: str, key: str, default: str) -> str:
    return _extract_scalar(text, section, key, default=default)


def _extract_scalar(text: str, section: str, key: str, default: str) -> str:
    block = _section_block(text, section)
    pattern = re.compile(r"^\s*%s\s*=\s*([\"'])(.*?)\1\s*$" % re.escape(key), re.M)
    match = pattern.search(block)
    if match:
        return match.group(2)
    int_match = re.compile(r"^\s*%s\s*=\s*([0-9]+)\s*$" % re.escape(key), re.M).search(block)
    if int_match:
        return int_match.group(1)
    return default


def _extract_int(text: str, section: str, key: str, default: int) -> int:
    value = _extract_scalar(text, section, key, default=str(default))
    try:
        return int(value)
    except ValueError:
        return default


def _extract_string_list(text: str, section: str, key: str) -> List[str]:
    block = _section_block(text, section)
    pattern = re.compile(r"^\s*%s\s*=\s*\[(.*?)\]" % re.escape(key), re.S | re.M)
    match = pattern.search(block)
    if not match:
        return []
    raw_items = re.findall(r"['\"]([^'\"]+)['\"]", match.group(1))
    return [item.strip() for item in raw_items if item.strip()]


def _section_block(text: str, section: str) -> str:
    marker = "[%s]" % section
    start = text.find(marker)
    if start < 0:
        return ""
    start = text.find("\n", start)
    if start < 0:
        return ""
    next_section = text.find("\n[", start + 1)
    if next_section < 0:
        return text[start + 1 :]
    return text[start + 1 : next_section + 1]
