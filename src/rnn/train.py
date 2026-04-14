from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from rnn.artifacts import build_summary, make_run_dir, write_csv, write_json, write_predictions_parquet
from rnn.config import load_config
from rnn.data import iter_split_batches, summarize_dataset
from rnn.metrics import classification_metrics
from rnn.model import RNNBinaryClassifier


def main() -> None:
    args = _parse_args()
    config = load_config(args.config, window_size=args.window_size)
    dataset_dir = config.dataset_dir
    if not dataset_dir.exists():
        raise FileNotFoundError("Model-window dataset not found: %s" % dataset_dir)

    run_dir = make_run_dir(config.output.root, "rnn", config.model_windows.length, output_dir=args.output_dir)
    device = _resolve_device(args.device)

    split_summaries = summarize_dataset(dataset_dir)
    train_summary = split_summaries["train"]
    if train_summary.positive_rows == 0:
        raise ValueError("Training split contains no positive labels.")
    if train_summary.negative_rows == 0:
        raise ValueError("Training split contains no negative labels.")

    positive_class_weight = float(train_summary.negative_rows) / float(train_summary.positive_rows)
    summary_payload = build_summary(
        model_name="rnn",
        config_path=config.config_path,
        dataset_dir=dataset_dir,
        output_root=config.output.root,
        manifest_dirname=config.output.manifest_dirname,
        window_size=config.model_windows.length,
        feature_order=config.model_windows.feature_order,
        split_summaries=[summary.to_dict() for summary in split_summaries.values()],
        class_weights={
            "positive_class_weight": positive_class_weight,
            "negative_class_weight": 1.0,
        },
        hyperparameters={
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
            "device": str(device),
        },
    )
    write_json(run_dir / "summary.json", summary_payload)

    model = RNNBinaryClassifier(
        input_size=len(config.model_windows.feature_order),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    best_result = _train_model(
        model=model,
        dataset_dir=dataset_dir,
        device=device,
        positive_class_weight=positive_class_weight,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
    )

    validation_ids, validation_labels, validation_probabilities = _predict_split(
        best_result["model"], dataset_dir, "validation", device, args.eval_batch_size
    )
    test_ids, test_labels, test_probabilities = _predict_split(
        best_result["model"], dataset_dir, "test", device, args.eval_batch_size
    )

    validation_metrics = classification_metrics(validation_labels, validation_probabilities)
    test_metrics = classification_metrics(test_labels, test_probabilities)
    train_metrics = classification_metrics(best_result["train_labels"], best_result["train_probabilities"])

    metrics_payload = {
        "best_epoch": best_result["best_epoch"],
        "best_validation_auc_roc": best_result["best_validation_auc_roc"],
        "train": train_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    write_json(run_dir / "metrics.json", metrics_payload)
    write_csv(run_dir / "history.csv", best_result["history"])
    write_predictions_parquet(
        run_dir / "predictions_validation.parquet",
        window_ids=validation_ids,
        labels=validation_labels,
        probabilities=validation_probabilities,
        split="validation",
    )
    write_predictions_parquet(
        run_dir / "predictions_test.parquet",
        window_ids=test_ids,
        labels=test_labels,
        probabilities=test_probabilities,
        split="test",
    )
    torch.save(
        {
            "model_state_dict": best_result["model"].state_dict(),
            "model_name": "rnn",
            "best_epoch": best_result["best_epoch"],
            "hyperparameters": summary_payload["hyperparameters"],
            "feature_order": list(config.model_windows.feature_order),
            "window_size": config.model_windows.length,
        },
        run_dir / "checkpoint.pt",
    )

    print(run_dir)


def _train_model(
    *,
    model: nn.Module,
    dataset_dir: Path,
    device: torch.device,
    positive_class_weight: float,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    seed: int,
) -> Dict[str, object]:
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_class_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_validation_auc = float("-inf")
    best_state = None
    best_epoch = 0
    history: List[Dict[str, object]] = []
    no_improvement = 0
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_rows = 0
        for batch in _iter_training_batches(dataset_dir, batch_size=batch_size, seed=seed + epoch):
            features = torch.as_tensor(batch.features, dtype=torch.float32, device=device)
            labels = torch.as_tensor(batch.labels, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.detach().cpu().item()) * int(labels.shape[0])
            train_rows += int(labels.shape[0])

        _, validation_labels, validation_probabilities = _predict_split(
            model, dataset_dir, "validation", device, eval_batch_size
        )
        validation_metrics = classification_metrics(validation_labels, validation_probabilities)
        validation_auc = float(validation_metrics["auc_roc"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_total / train_rows if train_rows else 0.0,
                "validation_auc_roc": validation_auc,
                "validation_accuracy": validation_metrics["accuracy"],
            }
        )
        if validation_auc > best_validation_auc:
            best_validation_auc = validation_auc
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not complete any epochs.")
    model.load_state_dict(best_state)
    train_ids, train_labels, train_probabilities = _predict_split(
        model, dataset_dir, "train", device, eval_batch_size
    )

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_auc_roc": best_validation_auc,
        "train_ids": train_ids,
        "train_labels": train_labels,
        "train_probabilities": train_probabilities,
    }


def _predict_split(
    model: nn.Module,
    dataset_dir: Path,
    split: str,
    device: torch.device,
    batch_size: int,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    model.eval()
    window_ids: List[str] = []
    label_batches: List[np.ndarray] = []
    probability_batches: List[np.ndarray] = []
    with torch.no_grad():
        for batch in iter_split_batches(dataset_dir, split, batch_size=batch_size):
            features = torch.as_tensor(batch.features, dtype=torch.float32, device=device)
            logits = model(features)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            window_ids.extend(batch.window_ids)
            label_batches.append(batch.labels)
            probability_batches.append(probabilities)
    labels = np.concatenate(label_batches, axis=0) if label_batches else np.zeros((0,), dtype=np.int64)
    probabilities = (
        np.concatenate(probability_batches, axis=0) if probability_batches else np.zeros((0,), dtype=np.float64)
    )
    return window_ids, labels, probabilities


def _iter_training_batches(dataset_dir: Path, batch_size: int, seed: int):
    for batch in iter_split_batches(
        dataset_dir,
        "train",
        batch_size=batch_size,
        shuffle_files=True,
        shuffle_rows=True,
        seed=seed,
    ):
        yield batch


def _resolve_device(device_name: Optional[str]) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="rnn", description="Train a vanilla RNN on windowed Polymarket data.")
    parser.add_argument("--config", type=Path, required=True, help="Path to configs/pipeline.toml.")
    parser.add_argument("--window-size", type=int, default=50, help="Model-window size to train on.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional explicit run directory.")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Train batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=2048, help="Evaluation batch size.")
    parser.add_argument("--hidden-size", type=int, default=64, help="RNN hidden width.")
    parser.add_argument("--num-layers", type=int, default=1, help="RNN layer count.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Inter-layer dropout for multi-layer RNNs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay.")
    parser.add_argument("--patience", type=int, default=5, help="Validation AUC early-stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Optional torch device override.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
