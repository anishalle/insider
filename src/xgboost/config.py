from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import ast
import re


DEFAULT_FEATURE_ORDER = (
    "price_yes",
    "signed_token_amount",
    "usd_amount",
    "side",
    "role_is_maker",
    "time_delta_seconds",
    "market_age_seconds",
)


@dataclass(frozen=True)
class RuntimeConfig:
    threads: int = 32
    memory_limit: str = "192GB"
    temp_directory: Path | None = None


@dataclass(frozen=True)
class WindowConfig:
    config_path: Path
    output_root: Path
    model_window_dirname: str
    window_size: int
    feature_order: Tuple[str, ...]
    runtime: RuntimeConfig

    @property
    def dataset_dir(self) -> Path:
        return self.output_root / self.model_window_dirname / ("window_size=%d" % self.window_size)


def load_window_config(config_path: Path, window_size: Optional[int] = None) -> WindowConfig:
    raw = _load_pipeline_toml(config_path)
    output_raw = raw.get("output", {})
    runtime_raw = raw.get("runtime", {})
    model_raw = raw.get("model_windows", {})

    resolved_window_size = int(model_raw.get("length", 50) if window_size is None else window_size)
    feature_order = tuple(model_raw.get("feature_order", DEFAULT_FEATURE_ORDER))
    if not feature_order:
        raise ValueError("model_windows.feature_order is required in the config.")

    runtime = RuntimeConfig(
        threads=int(runtime_raw.get("threads", 32)),
        memory_limit=str(runtime_raw.get("memory_limit", "192GB")),
        temp_directory=Path(runtime_raw["temp_directory"]) if runtime_raw.get("temp_directory") else None,
    )

    return WindowConfig(
        config_path=config_path,
        output_root=Path(output_raw.get("root", "outputs/default")),
        model_window_dirname=str(output_raw.get("model_window_dirname", "model_windows")),
        window_size=resolved_window_size,
        feature_order=feature_order,
        runtime=runtime,
    )


def _load_pipeline_toml(path: Path) -> Dict[str, Dict[str, Any]]:
    text = path.read_text()
    for loader in (_load_tomllib, _load_tomli, _load_minimal_toml):
        try:
            return loader(text)
        except Exception:
            continue
    raise ValueError("Unable to parse pipeline TOML configuration.")


def _load_tomllib(text: str) -> Dict[str, Dict[str, Any]]:
    import tomllib  # type: ignore

    return tomllib.loads(text)


def _load_tomli(text: str) -> Dict[str, Dict[str, Any]]:
    import tomli  # type: ignore

    return tomli.loads(text)


def _load_minimal_toml(text: str) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    current_section: Optional[str] = None
    for raw_line in text.splitlines():
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].strip()
            result.setdefault(current_section, {})
            continue
        if current_section is None:
            raise ValueError("Encountered key/value pair before any TOML section.")
        if "=" not in line:
            raise ValueError("Invalid TOML line: %s" % line)
        key, value_text = line.split("=", 1)
        result[current_section][key.strip()] = _parse_toml_value(value_text.strip())
    return result


def _strip_comment(line: str) -> str:
    in_string = False
    escaped = False
    for index, char in enumerate(line):
        if char == "\\" and in_string:
            escaped = not escaped
            continue
        if char == '"' and not escaped:
            in_string = not in_string
        if char == "#" and not in_string:
            return line[:index]
        escaped = False
    return line


def _parse_toml_value(value_text: str) -> Any:
    lowered = re.sub(r"\btrue\b", "True", value_text, flags=re.IGNORECASE)
    lowered = re.sub(r"\bfalse\b", "False", lowered, flags=re.IGNORECASE)
    lowered = re.sub(r"\bnull\b", "None", lowered, flags=re.IGNORECASE)
    try:
        return ast.literal_eval(lowered)
    except Exception:
        return value_text.strip('"')

