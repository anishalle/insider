from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
import tomllib

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


@dataclass(frozen=True)
class SequenceConfig:
    length: int = 64
    stride: int = 16
    feature_order: tuple[str, ...] = DEFAULT_FEATURE_ORDER
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1


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
    smoke: SmokeConfig = field(default_factory=SmokeConfig)

    def with_output_root(self, output_root: Path) -> "PipelineConfig":
        return replace(self, output=replace(self.output, root=output_root))


def load_config(path: Path) -> PipelineConfig:
    raw = tomllib.loads(path.read_text())

    inputs_raw = raw.get("inputs", {})
    output_raw = raw.get("output", {})
    runtime_raw = raw.get("runtime", {})
    label_raw = raw.get("label", {})
    sequence_raw = raw.get("sequence", {})
    smoke_raw = raw.get("smoke", {})

    sequence = SequenceConfig(
        length=int(sequence_raw.get("length", 64)),
        stride=int(sequence_raw.get("stride", 16)),
        feature_order=tuple(sequence_raw.get("feature_order", DEFAULT_FEATURE_ORDER)),
        train_ratio=float(sequence_raw.get("train_ratio", 0.8)),
        validation_ratio=float(sequence_raw.get("validation_ratio", 0.1)),
        test_ratio=float(sequence_raw.get("test_ratio", 0.1)),
    )

    ratio_total = round(sequence.train_ratio + sequence.validation_ratio + sequence.test_ratio, 10)
    if ratio_total != 1.0:
        raise ValueError("Sequence split ratios must sum to 1.0.")
    if sequence.length < 1 or sequence.stride < 1:
        raise ValueError("Sequence length and stride must be positive integers.")

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
            manifest_dirname=output_raw.get("manifest_dirname", "manifests"),
        ),
        runtime=RuntimeConfig(
            threads=int(runtime_raw.get("threads", 32)),
            memory_limit=str(runtime_raw.get("memory_limit", "96GB")),
            temp_directory=Path(runtime_raw.get("temp_directory", "tmp/duckdb")),
            preserve_insertion_order=bool(runtime_raw.get("preserve_insertion_order", False)),
        ),
        label=LabelConfig(
            horizon_minutes=int(label_raw.get("horizon_minutes", 5)),
            contract_addresses=tuple(label_raw.get("contract_addresses", DEFAULT_CONTRACT_ADDRESSES)),
            usd_decimals=int(label_raw.get("usd_decimals", 6)),
            token_decimals=int(label_raw.get("token_decimals", 6)),
        ),
        sequence=sequence,
        smoke=SmokeConfig(
            start=_parse_optional_datetime(smoke_raw.get("start")),
            end=_parse_optional_datetime(smoke_raw.get("end")),
        ),
    )
