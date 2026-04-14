from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from lr.artifacts import (
    build_summary_payload,
    prepare_run_directory,
    write_history_csv,
    write_json,
    write_model_json,
    write_predictions_parquet,
)
from lr.config import WindowConfig, load_window_config
from lr.data import count_train_class_balance, iter_split_batches, summarize_dataset, summarize_split
from lr.metrics import binary_metrics, roc_auc_score
from lr.model import (
    AdamState,
    LogisticRegressionModel,
    Standardizer,
    StandardizerState,
    adam_step,
    apply_standardization,
    logistic_loss_and_gradients,
)


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    history: List[Dict[str, object]]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    training_metrics: Dict[str, float]
    model_state: Dict[str, object]
    standardizer_state: StandardizerState


def main() -> None:
    parser = argparse.ArgumentParser(prog="lr.train", description="Train logistic regression on model windows.")
    parser.add_argument("--config", type=Path, required=True, help="Path to configs/pipeline.toml")
    parser.add_argument("--window-size", type=int, default=None, help="Model window size to train on.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional run directory override.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = load_window_config(args.config, window_size=args.window_size)
    result = train_logistic_regression(
        config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        l2=args.l2,
        patience=args.patience,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(result)


def train_logistic_regression(
    config: WindowConfig,
    *,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    l2: float,
    patience: int,
    seed: int,
    output_dir: Optional[Path],
) -> Dict[str, object]:
    dataset_dir = config.dataset_dir
    if not dataset_dir.exists():
        raise FileNotFoundError("Model window dataset not found: %s" % dataset_dir)

    split_summaries = summarize_dataset(dataset_dir)
    train_balance = count_train_class_balance(dataset_dir)
    feature_width = len(config.feature_order)
    if feature_width < 1:
        raise ValueError("feature_order must contain at least one feature.")
    n_features = config.window_size * feature_width

    standardizer = Standardizer(n_features)
    for batch in iter_split_batches(
        dataset_dir,
        "train",
        batch_size=batch_size,
        flatten=False,
        shuffle_files=False,
        shuffle_rows=False,
    ):
        features = reshape_batch_features(batch["features"], config.window_size, feature_width)
        standardizer.update(features)
    standardizer_state = standardizer.finalize()

    model = LogisticRegressionModel(n_features)
    optimizer = AdamState(n_features)
    best_state = model.state_dict()
    best_validation_auc = float("-inf")
    best_epoch = 0
    history: List[Dict[str, object]] = []
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss_total = 0.0
        train_row_count = 0
        for batch in iter_split_batches(
            dataset_dir,
            "train",
            batch_size=batch_size,
            flatten=False,
            shuffle_files=True,
            shuffle_rows=True,
            seed=seed + epoch,
        ):
            features = apply_standardization(
                reshape_batch_features(batch["features"], config.window_size, feature_width),
                standardizer_state,
            )
            labels = np.asarray(batch["labels"], dtype=np.int64)
            loss, grad_w, grad_b = logistic_loss_and_gradients(
                features,
                labels,
                model,
                positive_weight=float(train_balance["positive_weight"]),
                negative_weight=1.0,
                l2=l2,
            )
            adam_step(model, optimizer, grad_w, grad_b, learning_rate=learning_rate)
            train_loss_total += float(loss) * int(labels.shape[0])
            train_row_count += int(labels.shape[0])

        validation_metrics = evaluate_split_metrics(
            dataset_dir,
            "validation",
            model,
            standardizer_state,
            batch_size=eval_batch_size,
            window_size=config.window_size,
            feature_width=feature_width,
        )
        train_average_loss = train_loss_total / float(train_row_count) if train_row_count else 0.0
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_average_loss,
                "validation_auc_roc": validation_metrics["auc_roc"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_positive_rate": validation_metrics["positive_rate"],
            }
        )

        if validation_metrics["auc_roc"] > best_validation_auc:
            best_validation_auc = float(validation_metrics["auc_roc"])
            best_epoch = epoch
            best_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    training_metrics = evaluate_split_metrics(
        dataset_dir,
        "train",
        model,
        standardizer_state,
        batch_size=eval_batch_size,
        window_size=config.window_size,
        feature_width=feature_width,
    )
    validation_metrics, validation_predictions = predict_split(
        dataset_dir,
        "validation",
        model,
        standardizer_state,
        batch_size=eval_batch_size,
        window_size=config.window_size,
        feature_width=feature_width,
    )
    test_metrics, test_predictions = predict_split(
        dataset_dir,
        "test",
        model,
        standardizer_state,
        batch_size=eval_batch_size,
        window_size=config.window_size,
        feature_width=feature_width,
    )

    run_dir = prepare_run_directory(config.output_root, window_size=config.window_size, output_dir=output_dir)
    feature_order = list(config.feature_order)
    training_config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "learning_rate": learning_rate,
        "l2": l2,
        "patience": patience,
        "seed": seed,
    }
    summary_payload = build_summary_payload(
        model_name="logistic_regression",
        config_path=config.config_path,
        dataset_dir=dataset_dir,
        output_root=config.output_root,
        manifest_dirname=config.manifest_dirname,
        window_size=config.window_size,
        feature_order=feature_order,
        split_summaries=[split_summaries[split].to_dict() for split in ("train", "validation", "test")],
        class_weights=train_balance,
        training_config=training_config,
    )
    metrics_payload = {
        "best_epoch": best_epoch,
        "best_validation_auc_roc": best_validation_auc,
        "training": training_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
    }

    write_json(run_dir / "summary.json", summary_payload)
    write_json(run_dir / "metrics.json", metrics_payload)
    write_history_csv(run_dir / "history.csv", history)
    write_predictions_parquet(
        run_dir / "predictions_validation.parquet",
        window_ids=validation_predictions["window_id"],
        labels=validation_predictions["labels"],
        probabilities=validation_predictions["probabilities"],
        split="validation",
    )
    write_predictions_parquet(
        run_dir / "predictions_test.parquet",
        window_ids=test_predictions["window_id"],
        labels=test_predictions["labels"],
        probabilities=test_predictions["probabilities"],
        split="test",
    )
    write_model_json(
        run_dir / "model.json",
        weights=model.weights,
        bias=model.bias,
        feature_mean=standardizer_state.mean,
        feature_scale=standardizer_state.scale,
        feature_order=feature_order,
        window_size=config.window_size,
        class_weights=train_balance,
        hyperparameters=training_config,
    )

    return {
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_validation_auc_roc": best_validation_auc,
    }


def evaluate_split_metrics(
    dataset_dir: Path,
    split: str,
    model: LogisticRegressionModel,
    standardizer_state: StandardizerState,
    *,
    batch_size: int,
    window_size: int,
    feature_width: int,
) -> Dict[str, float]:
    labels_chunks: List[np.ndarray] = []
    probability_chunks: List[np.ndarray] = []

    for batch in iter_split_batches(dataset_dir, split, batch_size=batch_size, flatten=False):
        features = apply_standardization(
            reshape_batch_features(batch["features"], window_size, feature_width),
            standardizer_state,
        )
        probabilities = model.probabilities(features)
        labels = np.asarray(batch["labels"], dtype=np.int64)
        labels_chunks.append(labels)
        probability_chunks.append(probabilities)

    labels_all = np.concatenate(labels_chunks, axis=0) if labels_chunks else np.zeros(0, dtype=np.int64)
    probabilities_all = (
        np.concatenate(probability_chunks, axis=0) if probability_chunks else np.zeros(0, dtype=np.float32)
    )
    return binary_metrics(labels_all, probabilities_all)


def predict_split(
    dataset_dir: Path,
    split: str,
    model: LogisticRegressionModel,
    standardizer_state: StandardizerState,
    *,
    batch_size: int,
    window_size: int,
    feature_width: int,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    labels_chunks: List[np.ndarray] = []
    probability_chunks: List[np.ndarray] = []
    window_ids: List[str] = []

    for batch in iter_split_batches(dataset_dir, split, batch_size=batch_size, flatten=False):
        features = apply_standardization(
            reshape_batch_features(batch["features"], window_size, feature_width),
            standardizer_state,
        )
        probabilities = model.probabilities(features)
        labels = np.asarray(batch["labels"], dtype=np.int64)
        labels_chunks.append(labels)
        probability_chunks.append(probabilities)
        window_ids.extend(batch["window_id"])

    labels_all = np.concatenate(labels_chunks, axis=0) if labels_chunks else np.zeros(0, dtype=np.int64)
    probabilities_all = (
        np.concatenate(probability_chunks, axis=0) if probability_chunks else np.zeros(0, dtype=np.float32)
    )
    return binary_metrics(labels_all, probabilities_all), {
        "window_id": window_ids,
        "labels": labels_all,
        "probabilities": probabilities_all,
    }


def reshape_batch_features(features: np.ndarray, window_size: int, feature_width: int) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("Expected feature batch with shape (batch, window, feature), got %s" % (array.shape,))
    if array.shape[1] != window_size or array.shape[2] != feature_width:
        raise ValueError(
            "Expected feature batch with shape (batch, %d, %d), got %s"
            % (window_size, feature_width, array.shape)
        )
    return array.reshape(array.shape[0], window_size * feature_width)


if __name__ == "__main__":
    main()
