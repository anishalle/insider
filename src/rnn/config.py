from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
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
class OutputConfig:
    root: Path
    model_window_dirname: str = "model_windows"


@dataclass(frozen=True)
class ModelWindowConfig:
    length: int = 50
    stride: int = 16
    feature_order: Tuple[str, ...] = DEFAULT_FEATURE_ORDER
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    config_path: Path
    output: OutputConfig
    model_windows: ModelWindowConfig

    @property
    def dataset_dir(self) -> Path:
        return self.output.root / self.output.model_window_dirname / ("window_size=%d" % self.model_windows.length)


def load_config(config_path: Path, window_size: Optional[int] = None) -> TrainingConfig:
    raw = _load_pipeline_toml(config_path)
    output_raw = raw.get("output", {})
    model_raw = raw.get("model_windows", {})

    resolved_window_size = int(model_raw.get("length", 50) if window_size is None else window_size)
    feature_order = tuple(model_raw.get("feature_order", DEFAULT_FEATURE_ORDER))
    model_windows = ModelWindowConfig(
        length=resolved_window_size,
        stride=int(model_raw.get("stride", 16)),
        feature_order=feature_order,
        train_ratio=float(model_raw.get("train_ratio", 0.8)),
        validation_ratio=float(model_raw.get("validation_ratio", 0.1)),
        test_ratio=float(model_raw.get("test_ratio", 0.1)),
    )

    ratio_total = round(
        model_windows.train_ratio + model_windows.validation_ratio + model_windows.test_ratio,
        10,
    )
    if ratio_total != 1.0:
        raise ValueError("Model window split ratios must sum to 1.0.")
    if model_windows.length < 1 or model_windows.stride < 1:
        raise ValueError("Model window length and stride must be positive integers.")

    return TrainingConfig(
        config_path=config_path,
        output=OutputConfig(
            root=Path(output_raw.get("root", "outputs/default")),
            model_window_dirname=str(output_raw.get("model_window_dirname", "model_windows")),
        ),
        model_windows=model_windows,
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

