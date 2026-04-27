from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
import hashlib
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from insider.features import DEFAULT_CONTRACT_ADDRESSES, DEFAULT_FEATURE_ORDER


def _parse_optional_datetime(value: str | None) -> datetime | None:
    if value in (None, ""):
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


@dataclass(frozen=True)
class InputConfig:
    trades_path: Path
    markets_path: Path | None = None


@dataclass(frozen=True)
class OutputConfig:
    root: Path
    prepared_dirname: str = "prepared_trades"
    labeled_dirname: str = "labeled_user_trades"
    sequence_dirname: str = "sequence_dataset"
    model_window_dirname: str = "model_windows"
    manifest_dirname: str = "manifests"


@dataclass(frozen=True)
class RuntimeConfig:
    threads: int = 32
    memory_limit: str = "96GB"
    temp_directory: Path = Path("tmp/duckdb")
    preserve_insertion_order: bool = False


@dataclass(frozen=True)
class LabelConfig:
    horizon_minutes: int = 5
    contract_addresses: tuple[str, ...] = DEFAULT_CONTRACT_ADDRESSES
    usd_decimals: int = 6
    token_decimals: int = 6
    max_future_lag_seconds: int | None = None
    min_abs_markout_bps: float = 0.0


@dataclass(frozen=True)
class SequenceConfig:
    length: int = 64
    stride: int = 16
    feature_order: tuple[str, ...] = DEFAULT_FEATURE_ORDER
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    purge_minutes: int = 0
    embargo_minutes: int = 0


@dataclass(frozen=True)
class ModelWindowConfig:
    length: int = 50
    stride: int = 16
    feature_order: tuple[str, ...] = DEFAULT_FEATURE_ORDER
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    purge_minutes: int = 0
    embargo_minutes: int = 0


@dataclass(frozen=True)
class SmokeConfig:
    start: datetime | None = None
    end: datetime | None = None


@dataclass(frozen=True)
class PipelineConfig:
    inputs: InputConfig
    output: OutputConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    model_windows: ModelWindowConfig = field(default_factory=ModelWindowConfig)
    smoke: SmokeConfig = field(default_factory=SmokeConfig)
    source_config_path: Path | None = None
    source_config_hash: str | None = None

    def with_output_root(self, output_root: Path) -> "PipelineConfig":
        return replace(self, output=replace(self.output, root=output_root))


def load_config(path: Path) -> PipelineConfig:
    text = path.read_text()
    raw = tomllib.loads(text)

    inputs_raw = raw.get("inputs", {})
    output_raw = raw.get("output", {})
    runtime_raw = raw.get("runtime", {})
    label_raw = raw.get("label", {})
    sequence_raw = raw.get("sequence", {})
    model_windows_raw = raw.get("model_windows", {})
    smoke_raw = raw.get("smoke", {})

    sequence = SequenceConfig(
        length=int(sequence_raw.get("length", 64)),
        stride=int(sequence_raw.get("stride", 16)),
        feature_order=tuple(sequence_raw.get("feature_order", DEFAULT_FEATURE_ORDER)),
        train_ratio=float(sequence_raw.get("train_ratio", 0.8)),
        validation_ratio=float(sequence_raw.get("validation_ratio", 0.1)),
        test_ratio=float(sequence_raw.get("test_ratio", 0.1)),
        purge_minutes=int(sequence_raw.get("purge_minutes", 0)),
        embargo_minutes=int(sequence_raw.get("embargo_minutes", 0)),
    )

    ratio_total = round(sequence.train_ratio + sequence.validation_ratio + sequence.test_ratio, 10)
    if ratio_total != 1.0:
        raise ValueError("Sequence split ratios must sum to 1.0.")
    if sequence.length < 1 or sequence.stride < 1:
        raise ValueError("Sequence length and stride must be positive integers.")
    if sequence.purge_minutes < 0 or sequence.embargo_minutes < 0:
        raise ValueError("Sequence purge and embargo minutes must be non-negative.")

    model_windows = ModelWindowConfig(
        length=int(model_windows_raw.get("length", 50)),
        stride=int(model_windows_raw.get("stride", 16)),
        feature_order=tuple(model_windows_raw.get("feature_order", DEFAULT_FEATURE_ORDER)),
        train_ratio=float(model_windows_raw.get("train_ratio", 0.8)),
        validation_ratio=float(model_windows_raw.get("validation_ratio", 0.1)),
        test_ratio=float(model_windows_raw.get("test_ratio", 0.1)),
        purge_minutes=int(model_windows_raw.get("purge_minutes", 0)),
        embargo_minutes=int(model_windows_raw.get("embargo_minutes", 0)),
    )

    model_ratio_total = round(
        model_windows.train_ratio + model_windows.validation_ratio + model_windows.test_ratio,
        10,
    )
    if model_ratio_total != 1.0:
        raise ValueError("Model window split ratios must sum to 1.0.")
    if model_windows.length < 1 or model_windows.stride < 1:
        raise ValueError("Model window length and stride must be positive integers.")
    if model_windows.purge_minutes < 0 or model_windows.embargo_minutes < 0:
        raise ValueError("Model window purge and embargo minutes must be non-negative.")

    label = LabelConfig(
        horizon_minutes=int(label_raw.get("horizon_minutes", 5)),
        contract_addresses=tuple(label_raw.get("contract_addresses", DEFAULT_CONTRACT_ADDRESSES)),
        usd_decimals=int(label_raw.get("usd_decimals", 6)),
        token_decimals=int(label_raw.get("token_decimals", 6)),
        max_future_lag_seconds=(
            int(label_raw["max_future_lag_seconds"])
            if label_raw.get("max_future_lag_seconds") is not None
            else None
        ),
        min_abs_markout_bps=float(label_raw.get("min_abs_markout_bps", 0.0)),
    )
    if label.max_future_lag_seconds is not None and label.max_future_lag_seconds < 0:
        raise ValueError("label.max_future_lag_seconds must be non-negative when provided.")
    if label.min_abs_markout_bps < 0.0:
        raise ValueError("label.min_abs_markout_bps must be non-negative.")

    return PipelineConfig(
        inputs=InputConfig(
            trades_path=Path(inputs_raw["trades_path"]),
            markets_path=Path(inputs_raw["markets_path"]) if inputs_raw.get("markets_path") else None,
        ),
        output=OutputConfig(
            root=Path(output_raw.get("root", "outputs/default")),
            prepared_dirname=output_raw.get("prepared_dirname", "prepared_trades"),
            labeled_dirname=output_raw.get("labeled_dirname", "labeled_user_trades"),
            sequence_dirname=output_raw.get("sequence_dirname", "sequence_dataset"),
            model_window_dirname=output_raw.get("model_window_dirname", "model_windows"),
            manifest_dirname=output_raw.get("manifest_dirname", "manifests"),
        ),
        runtime=RuntimeConfig(
            threads=int(runtime_raw.get("threads", 32)),
            memory_limit=str(runtime_raw.get("memory_limit", "96GB")),
            temp_directory=Path(runtime_raw.get("temp_directory", "tmp/duckdb")),
            preserve_insertion_order=bool(runtime_raw.get("preserve_insertion_order", False)),
        ),
        label=label,
        sequence=sequence,
        model_windows=model_windows,
        smoke=SmokeConfig(
            start=_parse_optional_datetime(smoke_raw.get("start")),
            end=_parse_optional_datetime(smoke_raw.get("end")),
        ),
        source_config_path=path.resolve(),
        source_config_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
