from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

from modeling_common.dataset import iter_split_batches
from modeling_common.metrics import binary_classification_metrics


@dataclass
class TorchTrainingResult:
    best_epoch: int
    history: List[Dict[str, Union[float, int]]]
    validation_metrics: Dict[str, Union[float, int]]
    test_metrics: Dict[str, Union[float, int]]
    validation_ids: Sequence[str]
    validation_labels: np.ndarray
    validation_probabilities: np.ndarray
    test_ids: Sequence[str]
    test_labels: np.ndarray
    test_probabilities: np.ndarray


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_sequence_model(
    *,
    model: nn.Module,
    dataset_dir,
    train_batch_size: int,
    eval_batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    positive_class_weight: float,
    early_stopping_patience: int,
    device: torch.device,
    seed: int,
    optimizer_factory: Optional[Callable[[Sequence[nn.Parameter]], torch.optim.Optimizer]] = None,
) -> TorchTrainingResult:
    set_global_seed(seed)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_class_weight], dtype=torch.float32, device=device)
    )
    optimizer = (
        optimizer_factory(model.parameters())
        if optimizer_factory is not None
        else torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    )

    best_validation_auc = float("-inf")
    best_epoch = 0
    best_state = None
    history: List[Dict[str, float | int]] = []
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_rows = 0
        for batch in iter_split_batches(
            dataset_dir,
            "train",
            batch_size=train_batch_size,
            shuffle_files=True,
            shuffle_rows=True,
            seed=seed + epoch,
        ):
            features = torch.as_tensor(batch["features"], dtype=torch.float32, device=device)
            labels = torch.as_tensor(batch["labels"], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features).reshape(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.detach().cpu().item()) * int(labels.shape[0])
            train_rows += int(labels.shape[0])

        validation_ids, validation_labels, validation_probabilities = predict_sequence_model(
            model=model,
            dataset_dir=dataset_dir,
            split="validation",
            batch_size=eval_batch_size,
            device=device,
        )
        validation_metrics = binary_classification_metrics(validation_labels, validation_probabilities)
        average_train_loss = train_loss_total / train_rows if train_rows else 0.0
        history.append(
            {
                "epoch": epoch,
                "train_loss": average_train_loss,
                "validation_auc_roc": float(validation_metrics["auc_roc"]),
                "validation_accuracy": float(validation_metrics["accuracy"]),
                "validation_positive_rate": float(validation_metrics["positive_rate"]),
            }
        )

        if float(validation_metrics["auc_roc"]) > best_validation_auc:
            best_validation_auc = float(validation_metrics["auc_roc"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                break

    if best_state is None:
        raise RuntimeError("No training epochs completed.")

    model.load_state_dict(best_state)
    validation_ids, validation_labels, validation_probabilities = predict_sequence_model(
        model=model,
        dataset_dir=dataset_dir,
        split="validation",
        batch_size=eval_batch_size,
        device=device,
    )
    test_ids, test_labels, test_probabilities = predict_sequence_model(
        model=model,
        dataset_dir=dataset_dir,
        split="test",
        batch_size=eval_batch_size,
        device=device,
    )
    return TorchTrainingResult(
        best_epoch=best_epoch,
        history=history,
        validation_metrics=binary_classification_metrics(validation_labels, validation_probabilities),
        test_metrics=binary_classification_metrics(test_labels, test_probabilities),
        validation_ids=validation_ids,
        validation_labels=validation_labels,
        validation_probabilities=validation_probabilities,
        test_ids=test_ids,
        test_labels=test_labels,
        test_probabilities=test_probabilities,
    )


def predict_sequence_model(
    *,
    model: nn.Module,
    dataset_dir,
    split: str,
    batch_size: int,
    device: torch.device,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    model.eval()
    all_ids: List[str] = []
    label_batches: List[np.ndarray] = []
    probability_batches: List[np.ndarray] = []
    with torch.no_grad():
        for batch in iter_split_batches(dataset_dir, split, batch_size=batch_size):
            features = torch.as_tensor(batch["features"], dtype=torch.float32, device=device)
            logits = model(features).reshape(-1)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            labels = np.asarray(batch["labels"], dtype=np.int64)
            all_ids.extend(batch["window_id"])
            label_batches.append(labels)
            probability_batches.append(probabilities)
    labels = np.concatenate(label_batches, axis=0) if label_batches else np.zeros(0, dtype=np.int64)
    probabilities = (
        np.concatenate(probability_batches, axis=0) if probability_batches else np.zeros(0, dtype=np.float64)
    )
    return all_ids, labels, probabilities
