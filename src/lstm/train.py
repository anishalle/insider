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

try:
    import tomllib as toml_parser  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python 3.9 on Juno
    try:
        import tomli as toml_parser  # type: ignore[no-redef]
    except ModuleNotFoundError:  # pragma: no cover - fallback parser below
        toml_parser = None


SPLITS: Tuple[str, str, str] = ("train", "validation", "test")


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


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
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
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(inputs)
        last_step = outputs[:, -1, :]
        logits = self.head(self.dropout(last_step))
        return logits.squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser(prog="lstm.train", description="Train an LSTM on model windows.")
    parser.add_argument("--config", type=Path, required=True, help="Path to configs/pipeline.toml.")
    parser.add_argument("--window-size", type=int, default=None, help="Override model-window size.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for model artifacts.")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden size for the LSTM.")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout used in the model.")
    parser.add_argument("--epochs", type=int, default=12, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=2048, help="Evaluation batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay.")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience on validation AUC.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override. Defaults to cuda when available, else cpu.",
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
        patience=args.patience,
        seed=args.seed,
        device=device,
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
    patience: int,
    seed: int,
    device: torch.device,
) -> TrainResult:
    _, _, class_weight_pos, class_weight_neg = count_train_class_weights(config.dataset_dir, batch_size)
    model = LSTMClassifier(
        input_size=len(config.feature_order),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([class_weight_pos], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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
            features_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
            labels_tensor = torch.as_tensor(labels, dtype=torch.float32, device=device)
            optimizer.zero_grad()
            logits = model(features_tensor)
            loss = criterion(logits, labels_tensor)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.detach().cpu().item()) * int(labels_tensor.shape[0])
            train_rows += int(labels_tensor.shape[0])

        validation_eval = evaluate_split(model, config.dataset_dir, "validation", eval_batch_size, device)
        average_train_loss = train_loss_total / train_rows if train_rows else 0.0
        history.append(
            {
                "epoch": epoch,
                "train_loss": average_train_loss,
                "validation_auc_roc": float(validation_eval.metrics["auc_roc"]),
                "validation_loss": float(validation_eval.metrics["loss"]),
                "validation_accuracy": float(validation_eval.metrics["accuracy"]),
                "validation_positive_rate": float(validation_eval.metrics["positive_rate"]),
            }
        )

        current_validation_auc = float(validation_eval.metrics["auc_roc"])
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
                },
                "feature_order": list(config.feature_order),
                "window_size": config.window_size,
                "best_epoch": epoch,
                "best_validation_auc_roc": current_validation_auc,
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

    train_eval = evaluate_split(model, config.dataset_dir, "train", eval_batch_size, device)
    validation_eval = evaluate_split(model, config.dataset_dir, "validation", eval_batch_size, device)
    test_eval = evaluate_split(model, config.dataset_dir, "test", eval_batch_size, device)
    return TrainResult(
        best_epoch=best_epoch,
        best_validation_auc_roc=best_validation_auc,
        history=history,
        train_eval=train_eval,
        validation_eval=validation_eval,
        test_eval=test_eval,
        class_weight_pos=class_weight_pos,
        class_weight_neg=class_weight_neg,
    )


def evaluate_split(
    model: nn.Module,
    dataset_dir: Path,
    split: str,
    batch_size: int,
    device: torch.device,
) -> EvalResult:
    model.eval()
    all_ids: List[str] = []
    all_labels: List[np.ndarray] = []
    all_probabilities: List[np.ndarray] = []
    total_loss = 0.0
    total_rows = 0
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    with torch.no_grad():
        for ids, features, labels in iter_split_batches(dataset_dir, split, batch_size):
            features_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
            labels_tensor = torch.as_tensor(labels, dtype=torch.float32, device=device)
            logits = model(features_tensor)
            loss = criterion(logits, labels_tensor)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            all_ids.extend(ids)
            all_labels.append(labels)
            all_probabilities.append(probabilities)
            total_loss += float(loss.detach().cpu().item())
            total_rows += int(labels.shape[0])

    labels_array = np.concatenate(all_labels, axis=0) if all_labels else np.zeros(0, dtype=np.int64)
    probabilities_array = (
        np.concatenate(all_probabilities, axis=0) if all_probabilities else np.zeros(0, dtype=np.float64)
    )
    metrics = classification_metrics(labels_array, probabilities_array)
    metrics["loss"] = float(total_loss / total_rows) if total_rows else 0.0
    return EvalResult(ids=all_ids, labels=labels_array, probabilities=probabilities_array, metrics=metrics)


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, object]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    if y_true.ndim != 1 or y_prob.ndim != 1 or y_true.shape[0] != y_prob.shape[0]:
        raise ValueError("labels and probabilities must be one-dimensional arrays of equal length.")
    predictions = (y_prob >= 0.5).astype(np.int64)
    true_positive = int(((predictions == 1) & (y_true == 1)).sum())
    true_negative = int(((predictions == 0) & (y_true == 0)).sum())
    false_positive = int(((predictions == 1) & (y_true == 0)).sum())
    false_negative = int(((predictions == 0) & (y_true == 1)).sum())
    precision = true_positive / float(true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / float(true_positive + false_negative) if (true_positive + false_negative) else 0.0
    accuracy = float(true_positive + true_negative) / float(len(y_true)) if len(y_true) else 0.0
    f1 = 2.0 * precision * recall / float(precision + recall) if (precision + recall) else 0.0
    return {
        "row_count": int(len(y_true)),
        "positive_rows": int(y_true.sum()),
        "negative_rows": int(len(y_true) - int(y_true.sum())),
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "predicted_positive_rate": float(predictions.mean()) if len(predictions) else 0.0,
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def roc_auc_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    if positives == 0 or negatives == 0:
        return 0.5
    order = np.argsort(y_prob, kind="mergesort")
    sorted_scores = y_prob[order]
    sorted_labels = y_true[order]
    ranks = np.empty(len(sorted_scores), dtype=np.float64)
    index = 0
    while index < len(sorted_scores):
        next_index = index + 1
        while next_index < len(sorted_scores) and sorted_scores[next_index] == sorted_scores[index]:
            next_index += 1
        average_rank = (index + next_index - 1) / 2.0 + 1.0
        ranks[index:next_index] = average_rank
        index = next_index
    positive_rank_sum = float(ranks[sorted_labels == 1].sum())
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / float(positives * negatives)
    return float(auc)


def write_run_artifacts(config: PipelineWindowConfig, run_dir: Path, args: argparse.Namespace, result: TrainResult) -> None:
    split_summaries = {split: summarize_split(config.dataset_dir, split, args.eval_batch_size).to_dict() for split in SPLITS}
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
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seed": args.seed,
            "device": str(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")),
        },
        "split_summaries": split_summaries,
        "class_weights": {
            "positive": result.class_weight_pos,
            "negative": result.class_weight_neg,
        },
        "best_epoch": result.best_epoch,
        "best_validation_auc_roc": result.best_validation_auc_roc,
        "row_counts": {
            "train": int(result.train_eval.metrics["row_count"]),
            "validation": int(result.validation_eval.metrics["row_count"]),
            "test": int(result.test_eval.metrics["row_count"]),
        },
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


if __name__ == "__main__":
    main()
