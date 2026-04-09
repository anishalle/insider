from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
from modeling_common.diagnostics import build_debug_diagnostics
from modeling_common.metrics import binary_classification_metrics
from modeling_common.sequence_scaling import (
    SequenceStandardizationStats,
    SequenceStandardizer,
    apply_sequence_standardization,
    build_default_clip_enabled,
    build_default_scale_enabled,
    build_default_transform_kinds,
    transform_sequence_features,
)
from modeling_common.tabular_scaling import (
    TabularStandardizationStats,
    TabularStandardizer,
    apply_tabular_standardization,
)
from modeling_common.window_features import (
    build_window_summary_feature_names,
    compute_window_summary_features,
)

try:
    import tomllib as toml_parser  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python 3.9 on Juno
    try:
        import tomli as toml_parser  # type: ignore[no-redef]
    except ModuleNotFoundError:  # pragma: no cover - fallback parser below
        toml_parser = None


SPLITS: Tuple[str, str, str] = ("train", "validation", "test")
DEFAULT_CLIP_PERCENTILES: Tuple[float, float] = (0.5, 99.5)
DEFAULT_CLIP_SAMPLE_ROWS = 250000


@dataclass(frozen=True)
class PipelineWindowConfig:
    output_root: Path
    model_window_dirname: str
    feature_order: Tuple[str, ...]
    window_size: int
    dataset_dir: Path


@dataclass(frozen=True)
class SplitStats:
    split: str
    row_count: int
    positive_rows: int
    negative_rows: int
    positive_rate: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "split": self.split,
            "row_count": self.row_count,
            "positive_rows": self.positive_rows,
            "negative_rows": self.negative_rows,
            "positive_rate": self.positive_rate,
        }


@dataclass(frozen=True)
class EvalResult:
    ids: List[str]
    labels: np.ndarray
    probabilities: np.ndarray
    logits: np.ndarray
    metrics: Dict[str, object]


@dataclass(frozen=True)
class TrainResult:
    best_epoch: int
    best_validation_auc_roc: float
    history: List[Dict[str, object]]
    train_eval: EvalResult
    validation_eval: EvalResult
    test_eval: EvalResult
    class_weight_pos: float
    class_weight_neg: float
    feature_standardization: SequenceStandardizationStats
    summary_feature_standardization: TabularStandardizationStats


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        pooling: str,
        summary_feature_size: int,
    ) -> None:
        super(LSTMClassifier, self).__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        if pooling not in {"last", "mean", "max", "mean_last"}:
            raise ValueError("Unsupported pooling mode: %s" % pooling)
        self.pooling = pooling
        pooled_size = hidden_size * 2 if pooling == "mean_last" else hidden_size
        self.summary_feature_size = int(summary_feature_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(pooled_size + self.summary_feature_size, 1)

    def forward(self, inputs: torch.Tensor, summary_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs, _ = self.lstm(inputs)
        if self.pooling == "last":
            pooled = outputs[:, -1, :]
        elif self.pooling == "mean":
            pooled = outputs.mean(dim=1)
        elif self.pooling == "max":
            pooled = outputs.max(dim=1).values
        else:
            pooled = torch.cat((outputs[:, -1, :], outputs.mean(dim=1)), dim=1)
        if self.summary_feature_size > 0:
            if summary_features is None:
                raise ValueError("summary_features are required when summary_feature_size > 0.")
            pooled = torch.cat((pooled, summary_features), dim=1)
        logits = self.head(self.dropout(pooled))
        return logits.squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser(prog="lstm.train", description="Train an LSTM on model windows.")
    parser.add_argument("--config", type=Path, required=True, help="Path to configs/pipeline.toml.")
    parser.add_argument("--window-size", type=int, default=None, help="Override model-window size.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for model artifacts.")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden size for the LSTM.")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout used in the model.")
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean_last",
        choices=("last", "mean", "max", "mean_last"),
        help="How to pool the LSTM outputs before the classifier head.",
    )
    parser.add_argument("--epochs", type=int, default=12, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=2048, help="Evaluation batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=1.0,
        help="Clip gradient norms to this value. Disable with 0.",
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau factor applied after validation AUC stalls.",
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=1,
        help="ReduceLROnPlateau patience in epochs.",
    )
    parser.add_argument(
        "--scheduler-min-lr",
        type=float,
        default=1e-5,
        help="Minimum learning rate for ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--positive-class-weight",
        type=float,
        default=None,
        help="Optional positive-class weight override. Defaults to the train-split imbalance ratio.",
    )
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience on validation AUC.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override. Defaults to cuda when available, else cpu.",
    )
    parser.add_argument(
        "--debug-metrics",
        action="store_true",
        help="Write diagnostics.json with score summaries and validation threshold sweeps.",
    )
    args = parser.parse_args()

    config = load_window_config(args.config, args.window_size)
    run_dir = prepare_run_dir(config, args.output_dir)
    device = choose_device(args.device)
    set_seed(args.seed)

    result = train_lstm(
        config=config,
        run_dir=run_dir,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        scheduler_min_lr=args.scheduler_min_lr,
        positive_class_weight=args.positive_class_weight,
        patience=args.patience,
        seed=args.seed,
        device=device,
        pooling=args.pooling,
    )

    write_run_artifacts(config, run_dir, args, result)
    print(json.dumps({"run_dir": str(run_dir), "best_epoch": result.best_epoch, "best_validation_auc_roc": result.best_validation_auc_roc}, indent=2))


def load_window_config(config_path: Path, window_size: Optional[int]) -> PipelineWindowConfig:
    raw = parse_simple_toml(config_path)
    output = raw.get("output", {})
    model_windows = raw.get("model_windows", {})
    resolved_window_size = int(model_windows.get("length", 50) if window_size is None else window_size)
    feature_order = tuple(model_windows.get("feature_order", []))
    if not feature_order:
        raise ValueError("model_windows.feature_order is required in the config.")
    output_root = Path(output.get("root", "outputs/default"))
    model_window_dirname = str(output.get("model_window_dirname", "model_windows"))
    dataset_dir = output_root / model_window_dirname / ("window_size=%d" % resolved_window_size)
    if not dataset_dir.exists():
        raise FileNotFoundError("Model-window dataset not found: %s" % dataset_dir)
    return PipelineWindowConfig(
        output_root=output_root,
        model_window_dirname=model_window_dirname,
        feature_order=feature_order,
        window_size=resolved_window_size,
        dataset_dir=dataset_dir,
    )


def prepare_run_dir(config: PipelineWindowConfig, output_dir: Optional[Path]) -> Path:
    if output_dir is not None:
        run_dir = output_dir
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = (
            config.output_root
            / "models"
            / "lstm"
            / ("window_size=%d" % config.window_size)
            / timestamp
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def parse_simple_toml(path: Path) -> Dict[str, Dict[str, object]]:
    if toml_parser is not None:
        return toml_parser.loads(path.read_text())

    result: Dict[str, Dict[str, object]] = {}
    current_section: Optional[str] = None
    pending_key: Optional[str] = None
    pending_value_parts: List[str] = []
    bracket_depth = 0
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if pending_key is not None:
            pending_value_parts.append(line)
            bracket_depth += line.count("[") - line.count("]")
            if bracket_depth <= 0:
                result[current_section][pending_key] = parse_toml_value(" ".join(pending_value_parts))
                pending_key = None
                pending_value_parts = []
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].strip()
            result.setdefault(current_section, {})
            continue
        if current_section is None or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if value.startswith("[") and not value.endswith("]"):
            pending_key = key.strip()
            pending_value_parts = [value]
            bracket_depth = value.count("[") - value.count("]")
            continue
        result[current_section][key.strip()] = parse_toml_value(value)
    return result


def parse_toml_value(value: str) -> object:
    if value.startswith("#"):
        return ""
    try:
        return ast.literal_eval(value)
    except Exception:
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return value.strip('"')


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_name: Optional[str]) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_split_files(dataset_dir: Path, split: str) -> List[Path]:
    split_dir = dataset_dir / ("split=%s" % split)
    files = sorted(split_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError("No parquet files found for split=%s in %s" % (split, split_dir))
    return files


def iter_split_batches(
    dataset_dir: Path,
    split: str,
    batch_size: int,
    *,
    shuffle_files: bool = False,
    shuffle_rows: bool = False,
    seed: Optional[int] = None,
) -> Iterator[Tuple[List[str], np.ndarray, np.ndarray]]:
    file_paths = list_split_files(dataset_dir, split)
    rng = random.Random(seed)
    if shuffle_files:
        rng.shuffle(file_paths)

    for file_path in file_paths:
        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=["window_id", "features", "label"],
            use_threads=True,
        ):
            ids = batch.column(0).to_pylist()
            features = np.asarray(batch.column(1).to_pylist(), dtype=np.float32)
            labels = np.asarray(batch.column(2).to_pylist(), dtype=np.int64)
            if features.ndim != 3:
                raise ValueError("Expected feature tensor of rank 3, got %s" % (features.shape,))
            if shuffle_rows and len(labels) > 1:
                permutation = np.random.default_rng(seed=rng.randrange(1 << 30)).permutation(len(labels))
                features = features[permutation]
                labels = labels[permutation]
                ids = [ids[index] for index in permutation.tolist()]
            yield ids, features, labels


def summarize_split(dataset_dir: Path, split: str, batch_size: int) -> SplitStats:
    row_count = 0
    positive_rows = 0
    for _, _, labels in iter_split_batches(dataset_dir, split, batch_size):
        row_count += int(labels.shape[0])
        positive_rows += int(labels.sum())
    negative_rows = row_count - positive_rows
    positive_rate = float(positive_rows / row_count) if row_count else 0.0
    return SplitStats(
        split=split,
        row_count=row_count,
        positive_rows=positive_rows,
        negative_rows=negative_rows,
        positive_rate=positive_rate,
    )


def summarize_dataset(dataset_dir: Path, batch_size: int) -> Dict[str, SplitStats]:
    return {split: summarize_split(dataset_dir, split, batch_size) for split in SPLITS}


def count_train_class_weights(dataset_dir: Path, batch_size: int) -> Tuple[int, int, float, float]:
    train_stats = summarize_split(dataset_dir, "train", batch_size)
    if train_stats.positive_rows == 0:
        raise ValueError("Train split contains no positive labels.")
    if train_stats.negative_rows == 0:
        raise ValueError("Train split contains no negative labels.")
    class_weight_pos = float(train_stats.negative_rows) / float(train_stats.positive_rows)
    class_weight_neg = 1.0
    return train_stats.positive_rows, train_stats.negative_rows, class_weight_pos, class_weight_neg


def fit_train_standardizer(
    dataset_dir: Path,
    feature_order: Sequence[str],
    batch_size: int,
) -> SequenceStandardizationStats:
    transform_kinds = build_default_transform_kinds(feature_order)
    clip_enabled = build_default_clip_enabled(feature_order)
    scale_enabled = build_default_scale_enabled(feature_order)
    clip_lower, clip_upper, clip_sample_rows = estimate_train_clip_bounds(
        dataset_dir,
        feature_order,
        batch_size=batch_size,
        transform_kinds=transform_kinds,
        clip_enabled=clip_enabled,
        clip_percentiles=DEFAULT_CLIP_PERCENTILES,
        sample_rows=DEFAULT_CLIP_SAMPLE_ROWS,
    )
    standardizer = SequenceStandardizer(
        feature_order,
        transform_kinds=transform_kinds,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
        scale_enabled=scale_enabled,
        clip_enabled=clip_enabled,
        clip_percentiles=DEFAULT_CLIP_PERCENTILES,
        clip_sample_rows=clip_sample_rows,
    )
    for _, features, _ in iter_split_batches(dataset_dir, "train", batch_size):
        standardizer.update(features)
    return standardizer.finalize()


def fit_train_summary_standardizer(
    dataset_dir: Path,
    feature_order: Sequence[str],
    batch_size: int,
) -> TabularStandardizationStats:
    summary_feature_names = build_window_summary_feature_names(feature_order)
    standardizer = TabularStandardizer(summary_feature_names)
    for _, features, _ in iter_split_batches(dataset_dir, "train", batch_size):
        standardizer.update(compute_window_summary_features(features))
    return standardizer.finalize()


def estimate_train_clip_bounds(
    dataset_dir: Path,
    feature_order: Sequence[str],
    *,
    batch_size: int,
    transform_kinds: Sequence[str],
    clip_enabled: np.ndarray,
    clip_percentiles: Tuple[float, float],
    sample_rows: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    feature_count = len(feature_order)
    enabled = np.asarray(clip_enabled, dtype=bool)
    if sample_rows <= 0 or not bool(enabled.any()):
        return (
            np.full(feature_count, -np.inf, dtype=np.float32),
            np.full(feature_count, np.inf, dtype=np.float32),
            0,
        )

    samples: List[np.ndarray] = []
    rows_collected = 0
    for _, features, _ in iter_split_batches(
        dataset_dir,
        "train",
        batch_size,
        shuffle_files=True,
        shuffle_rows=True,
        seed=0,
    ):
        transformed = transform_sequence_features(features, transform_kinds).reshape(-1, feature_count)
        remaining = sample_rows - rows_collected
        if remaining <= 0:
            break
        take_count = min(remaining, transformed.shape[0])
        samples.append(np.asarray(transformed[:take_count], dtype=np.float32))
        rows_collected += int(take_count)
        if rows_collected >= sample_rows:
            break

    if not samples:
        return (
            np.full(feature_count, -np.inf, dtype=np.float32),
            np.full(feature_count, np.inf, dtype=np.float32),
            0,
        )

    sample_matrix = np.concatenate(samples, axis=0)
    lower = np.full(feature_count, -np.inf, dtype=np.float32)
    upper = np.full(feature_count, np.inf, dtype=np.float32)
    for index, should_clip in enumerate(enabled.tolist()):
        if not should_clip:
            continue
        lower[index] = float(np.quantile(sample_matrix[:, index], clip_percentiles[0] / 100.0))
        upper[index] = float(np.quantile(sample_matrix[:, index], clip_percentiles[1] / 100.0))
    return lower, upper, int(sample_matrix.shape[0])


def train_lstm(
    *,
    config: PipelineWindowConfig,
    run_dir: Path,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    scheduler_factor: float,
    scheduler_patience: int,
    scheduler_min_lr: float,
    positive_class_weight: Optional[float],
    patience: int,
    seed: int,
    device: torch.device,
    pooling: str,
) -> TrainResult:
    _, _, inferred_class_weight_pos, class_weight_neg = count_train_class_weights(config.dataset_dir, batch_size)
    class_weight_pos = (
        inferred_class_weight_pos if positive_class_weight is None else float(positive_class_weight)
    )
    feature_standardization = fit_train_standardizer(config.dataset_dir, config.feature_order, eval_batch_size)
    summary_feature_standardization = fit_train_summary_standardizer(
        config.dataset_dir,
        config.feature_order,
        eval_batch_size,
    )
    model = LSTMClassifier(
        input_size=len(config.feature_order),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        pooling=pooling,
        summary_feature_size=len(summary_feature_standardization.feature_names),
    )
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([class_weight_pos], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=scheduler_min_lr,
    )

    best_state = None
    best_validation_auc = float("-inf")
    best_epoch = 0
    history: List[Dict[str, object]] = []
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_rows = 0
        for _, features, labels in iter_split_batches(
            config.dataset_dir,
            "train",
            batch_size,
            shuffle_files=True,
            shuffle_rows=True,
            seed=seed + epoch,
        ):
            features_tensor = torch.as_tensor(
                apply_sequence_standardization(features, feature_standardization),
                dtype=torch.float32,
                device=device,
            )
            summary_tensor = torch.as_tensor(
                apply_tabular_standardization(
                    compute_window_summary_features(features),
                    summary_feature_standardization,
                ),
                dtype=torch.float32,
                device=device,
            )
            labels_tensor = torch.as_tensor(labels, dtype=torch.float32, device=device)
            optimizer.zero_grad()
            logits = model(features_tensor, summary_tensor)
            loss = criterion(logits, labels_tensor)
            loss.backward()
            if gradient_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            train_loss_total += float(loss.detach().cpu().item()) * int(labels_tensor.shape[0])
            train_rows += int(labels_tensor.shape[0])

        validation_eval = evaluate_split(
            model,
            config.dataset_dir,
            "validation",
            eval_batch_size,
            device,
            feature_standardization=feature_standardization,
            summary_feature_standardization=summary_feature_standardization,
        )
        average_train_loss = train_loss_total / train_rows if train_rows else 0.0
        current_learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": average_train_loss,
                "validation_auc_roc": float(validation_eval.metrics["auc_roc"]),
                "validation_loss": float(validation_eval.metrics["loss"]),
                "validation_accuracy": float(validation_eval.metrics["accuracy"]),
                "validation_positive_rate": float(validation_eval.metrics["positive_rate"]),
                "learning_rate": current_learning_rate,
            }
        )

        current_validation_auc = float(validation_eval.metrics["auc_roc"])
        scheduler.step(current_validation_auc)
        if current_validation_auc > best_validation_auc:
            best_validation_auc = current_validation_auc
            best_epoch = epoch
            best_state = {
                "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "model_config": {
                    "input_size": len(config.feature_order),
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "pooling": pooling,
                    "summary_feature_size": len(summary_feature_standardization.feature_names),
                },
                "feature_order": list(config.feature_order),
                "summary_feature_order": list(summary_feature_standardization.feature_names),
                "window_size": config.window_size,
                "best_epoch": epoch,
                "best_validation_auc_roc": current_validation_auc,
                "feature_standardization": feature_standardization.to_dict(),
                "summary_feature_standardization": summary_feature_standardization.to_dict(),
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(best_state, checkpoint_path)
    model.load_state_dict(best_state["model_state_dict"])

    train_eval = evaluate_split(
        model,
        config.dataset_dir,
        "train",
        eval_batch_size,
        device,
        feature_standardization=feature_standardization,
        summary_feature_standardization=summary_feature_standardization,
    )
    validation_eval = evaluate_split(
        model,
        config.dataset_dir,
        "validation",
        eval_batch_size,
        device,
        feature_standardization=feature_standardization,
        summary_feature_standardization=summary_feature_standardization,
    )
    test_eval = evaluate_split(
        model,
        config.dataset_dir,
        "test",
        eval_batch_size,
        device,
        feature_standardization=feature_standardization,
        summary_feature_standardization=summary_feature_standardization,
    )
    return TrainResult(
        best_epoch=best_epoch,
        best_validation_auc_roc=best_validation_auc,
        history=history,
        train_eval=train_eval,
        validation_eval=validation_eval,
        test_eval=test_eval,
        class_weight_pos=class_weight_pos,
        class_weight_neg=class_weight_neg,
        feature_standardization=feature_standardization,
        summary_feature_standardization=summary_feature_standardization,
    )


def evaluate_split(
    model: nn.Module,
    dataset_dir: Path,
    split: str,
    batch_size: int,
    device: torch.device,
    *,
    feature_standardization: SequenceStandardizationStats,
    summary_feature_standardization: TabularStandardizationStats,
) -> EvalResult:
    model.eval()
    all_ids: List[str] = []
    all_labels: List[np.ndarray] = []
    all_probabilities: List[np.ndarray] = []
    all_logits: List[np.ndarray] = []
    total_loss = 0.0
    total_rows = 0
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    with torch.no_grad():
        for ids, features, labels in iter_split_batches(dataset_dir, split, batch_size):
            features_tensor = torch.as_tensor(
                apply_sequence_standardization(features, feature_standardization),
                dtype=torch.float32,
                device=device,
            )
            summary_tensor = torch.as_tensor(
                apply_tabular_standardization(
                    compute_window_summary_features(features),
                    summary_feature_standardization,
                ),
                dtype=torch.float32,
                device=device,
            )
            labels_tensor = torch.as_tensor(labels, dtype=torch.float32, device=device)
            logits = model(features_tensor, summary_tensor)
            loss = criterion(logits, labels_tensor)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            all_ids.extend(ids)
            all_labels.append(labels)
            all_probabilities.append(probabilities)
            all_logits.append(logits.detach().cpu().numpy())
            total_loss += float(loss.detach().cpu().item())
            total_rows += int(labels.shape[0])

    labels_array = np.concatenate(all_labels, axis=0) if all_labels else np.zeros(0, dtype=np.int64)
    probabilities_array = (
        np.concatenate(all_probabilities, axis=0) if all_probabilities else np.zeros(0, dtype=np.float64)
    )
    logits_array = np.concatenate(all_logits, axis=0) if all_logits else np.zeros(0, dtype=np.float64)
    metrics = binary_classification_metrics(labels_array, probabilities_array)
    metrics["loss"] = float(total_loss / total_rows) if total_rows else 0.0
    return EvalResult(
        ids=all_ids,
        labels=labels_array,
        probabilities=probabilities_array,
        logits=logits_array,
        metrics=metrics,
    )


def write_run_artifacts(config: PipelineWindowConfig, run_dir: Path, args: argparse.Namespace, result: TrainResult) -> None:
    split_summaries = {split: summarize_split(config.dataset_dir, split, args.eval_batch_size).to_dict() for split in SPLITS}
    feature_shift_report = build_feature_shift_report(
        config.dataset_dir,
        feature_order=config.feature_order,
        batch_size=args.eval_batch_size,
        feature_standardization=result.feature_standardization,
        split_summaries=split_summaries,
    )
    summary = {
        "model_name": "lstm",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(config.dataset_dir),
        "output_root": str(config.output_root),
        "window_size": config.window_size,
        "feature_order": list(config.feature_order),
        "model_config": {
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "pooling": args.pooling,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "optimizer": "AdamW",
            "gradient_clip_norm": args.gradient_clip_norm,
            "scheduler_factor": args.scheduler_factor,
            "scheduler_patience": args.scheduler_patience,
            "scheduler_min_lr": args.scheduler_min_lr,
            "positive_class_weight": args.positive_class_weight,
            "patience": args.patience,
            "seed": args.seed,
            "device": str(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")),
            "debug_metrics": bool(args.debug_metrics),
        },
        "split_summaries": split_summaries,
        "class_weights": {
            "positive": result.class_weight_pos,
            "negative": result.class_weight_neg,
        },
        "feature_standardization": result.feature_standardization.to_dict(),
        "summary_feature_standardization": result.summary_feature_standardization.to_dict(),
        "best_epoch": result.best_epoch,
        "best_validation_auc_roc": result.best_validation_auc_roc,
        "row_counts": {
            "train": int(result.train_eval.metrics["row_count"]),
            "validation": int(result.validation_eval.metrics["row_count"]),
            "test": int(result.test_eval.metrics["row_count"]),
        },
        "feature_shift_report_path": "feature_shift.json",
    }
    metrics = {
        "train": result.train_eval.metrics,
        "validation": result.validation_eval.metrics,
        "test": result.test_eval.metrics,
        "best_epoch": result.best_epoch,
        "best_validation_auc_roc": result.best_validation_auc_roc,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    (run_dir / "feature_shift.json").write_text(json.dumps(feature_shift_report, indent=2, default=str) + "\n")
    if args.debug_metrics:
        reference_thresholds = {
            "default_0_5": 0.5,
            "class_weight_adjusted": result.class_weight_neg / (result.class_weight_neg + result.class_weight_pos),
        }
        (run_dir / "diagnostics.json").write_text(
            json.dumps(
                build_debug_diagnostics(
                    split_payloads={
                        "train": {
                            "labels": result.train_eval.labels,
                            "probabilities": result.train_eval.probabilities,
                            "logits": result.train_eval.logits,
                        },
                        "validation": {
                            "labels": result.validation_eval.labels,
                            "probabilities": result.validation_eval.probabilities,
                            "logits": result.validation_eval.logits,
                        },
                        "test": {
                            "labels": result.test_eval.labels,
                            "probabilities": result.test_eval.probabilities,
                            "logits": result.test_eval.logits,
                        },
                    },
                    reference_thresholds=reference_thresholds,
                ),
                indent=2,
                default=str,
            )
            + "\n"
        )
    write_history_csv(run_dir / "history.csv", result.history)
    write_predictions_parquet(
        run_dir / "predictions_validation.parquet",
        ids=result.validation_eval.ids,
        labels=result.validation_eval.labels,
        probabilities=result.validation_eval.probabilities,
        split="validation",
    )
    write_predictions_parquet(
        run_dir / "predictions_test.parquet",
        ids=result.test_eval.ids,
        labels=result.test_eval.labels,
        probabilities=result.test_eval.probabilities,
        split="test",
    )


def write_history_csv(path: Path, history: Sequence[Dict[str, object]]) -> None:
    if not history:
        return
    fieldnames = list(history[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def write_predictions_parquet(
    path: Path,
    *,
    ids: Sequence[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    split: str,
) -> None:
    prediction = (probabilities >= 0.5).astype(np.int64)
    table = pa.table(
        {
            "window_id": list(ids),
            "label": labels.astype(np.int64),
            "probability": probabilities.astype(np.float64),
            "prediction": prediction,
            "split": [split] * len(ids),
        }
    )
    pq.write_table(table, path, compression="zstd")


def build_feature_shift_report(
    dataset_dir: Path,
    *,
    feature_order: Sequence[str],
    batch_size: int,
    feature_standardization: SequenceStandardizationStats,
    split_summaries: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    split_payloads = {
        split: summarize_feature_shift_split(
            dataset_dir,
            split,
            batch_size=batch_size,
            feature_order=feature_order,
            feature_standardization=feature_standardization,
            positive_rate=float(split_summaries[split]["positive_rate"]),
        )
        for split in SPLITS
    }

    train_payload = split_payloads["train"]
    drift_vs_train: Dict[str, object] = {}
    train_raw_features = {row["name"]: row for row in train_payload["raw_feature_summary"]}
    train_preprocessed_features = {row["name"]: row for row in train_payload["preprocessed_feature_summary"]}
    for split in ("validation", "test"):
        current = split_payloads[split]
        current_raw = {row["name"]: row for row in current["raw_feature_summary"]}
        current_preprocessed = {row["name"]: row for row in current["preprocessed_feature_summary"]}
        drift_vs_train[split] = {
            "positive_rate_delta": float(current["positive_rate"]) - float(train_payload["positive_rate"]),
            "features": [
                {
                    "name": feature_name,
                    "raw_mean_delta": float(current_raw[feature_name]["mean"]) - float(train_raw_features[feature_name]["mean"]),
                    "preprocessed_mean_delta": float(current_preprocessed[feature_name]["mean"])
                    - float(train_preprocessed_features[feature_name]["mean"]),
                }
                for feature_name in feature_order
            ],
        }

    return {
        "feature_order": list(feature_order),
        "window_size": int(train_payload["window_size"]),
        "split_positive_rates": {split: float(split_payloads[split]["positive_rate"]) for split in SPLITS},
        "splits": split_payloads,
        "drift_vs_train": drift_vs_train,
    }


def summarize_feature_shift_split(
    dataset_dir: Path,
    split: str,
    *,
    batch_size: int,
    feature_order: Sequence[str],
    feature_standardization: SequenceStandardizationStats,
    positive_rate: float,
) -> Dict[str, object]:
    feature_count = len(feature_order)
    position_count: Optional[int] = None

    raw_sum = np.zeros(feature_count, dtype=np.float64)
    raw_sum_squares = np.zeros(feature_count, dtype=np.float64)
    raw_min = np.full(feature_count, np.inf, dtype=np.float64)
    raw_max = np.full(feature_count, -np.inf, dtype=np.float64)
    raw_position_sum: Optional[np.ndarray] = None
    raw_position_sum_squares: Optional[np.ndarray] = None

    pre_sum = np.zeros(feature_count, dtype=np.float64)
    pre_sum_squares = np.zeros(feature_count, dtype=np.float64)
    pre_min = np.full(feature_count, np.inf, dtype=np.float64)
    pre_max = np.full(feature_count, -np.inf, dtype=np.float64)

    row_count = 0
    for _, features, _ in iter_split_batches(dataset_dir, split, batch_size):
        if position_count is None:
            position_count = int(features.shape[1])
            raw_position_sum = np.zeros((position_count, feature_count), dtype=np.float64)
            raw_position_sum_squares = np.zeros((position_count, feature_count), dtype=np.float64)
        raw_flat = np.asarray(features, dtype=np.float64).reshape(-1, feature_count)
        raw_sum += raw_flat.sum(axis=0)
        raw_sum_squares += np.square(raw_flat).sum(axis=0)
        raw_min = np.minimum(raw_min, raw_flat.min(axis=0))
        raw_max = np.maximum(raw_max, raw_flat.max(axis=0))
        raw_position_sum += np.asarray(features, dtype=np.float64).sum(axis=0)
        raw_position_sum_squares += np.square(np.asarray(features, dtype=np.float64)).sum(axis=0)

        preprocessed = apply_sequence_standardization(features, feature_standardization)
        pre_flat = np.asarray(preprocessed, dtype=np.float64).reshape(-1, feature_count)
        pre_sum += pre_flat.sum(axis=0)
        pre_sum_squares += np.square(pre_flat).sum(axis=0)
        pre_min = np.minimum(pre_min, pre_flat.min(axis=0))
        pre_max = np.maximum(pre_max, pre_flat.max(axis=0))
        row_count += int(features.shape[0])

    if position_count is None or raw_position_sum is None or raw_position_sum_squares is None:
        raise ValueError("Unable to summarize feature shift for split=%s" % split)

    total_steps = max(row_count * position_count, 1)
    raw_feature_summary = []
    preprocessed_feature_summary = []
    for index, feature_name in enumerate(feature_order):
        raw_mean = raw_sum[index] / total_steps
        raw_variance = max(raw_sum_squares[index] / total_steps - raw_mean * raw_mean, 0.0)
        pre_mean = pre_sum[index] / total_steps
        pre_variance = max(pre_sum_squares[index] / total_steps - pre_mean * pre_mean, 0.0)
        raw_feature_summary.append(
            {
                "name": feature_name,
                "mean": float(raw_mean),
                "std": float(np.sqrt(raw_variance)),
                "min": float(raw_min[index]),
                "max": float(raw_max[index]),
                "position_mean": (raw_position_sum[:, index] / max(row_count, 1)).astype(np.float64).tolist(),
                "position_std": np.sqrt(
                    np.maximum(raw_position_sum_squares[:, index] / max(row_count, 1) - np.square(raw_position_sum[:, index] / max(row_count, 1)), 0.0)
                ).astype(np.float64).tolist(),
            }
        )
        preprocessed_feature_summary.append(
            {
                "name": feature_name,
                "mean": float(pre_mean),
                "std": float(np.sqrt(pre_variance)),
                "min": float(pre_min[index]),
                "max": float(pre_max[index]),
            }
        )

    return {
        "split": split,
        "row_count": int(row_count),
        "positive_rate": float(positive_rate),
        "window_size": int(position_count),
        "raw_feature_summary": raw_feature_summary,
        "preprocessed_feature_summary": preprocessed_feature_summary,
    }


if __name__ == "__main__":
    main()
