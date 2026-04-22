from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from modeling_common.artifacts import write_predictions_parquet
from modeling_common.diagnostics import build_debug_diagnostics
from modeling_common.metrics import binary_classification_metrics
from modeling_common.dataset import summarize_dataset

from xgb_model._compat import load_external_xgboost
from xgb_model.artifacts import (
    build_summary_payload,
    make_run_dir,
    write_history_csv,
    write_json,
    write_model_json,
)
from xgb_model.config import WindowConfig, load_window_config
from xgb_model.data import build_augmented_feature_names, load_split_arrays


@dataclass(frozen=True)
class EvalResult:
    ids: List[str]
    labels: np.ndarray
    probabilities: np.ndarray
    logits: np.ndarray
    metrics: Dict[str, object]


@dataclass(frozen=True)
class TrainResult:
    run_dir: Path
    best_iteration: int
    best_validation_auc_roc: float
    history: List[Dict[str, object]]
    train_eval: EvalResult
    validation_eval: EvalResult
    test_eval: EvalResult
    booster: Any


def main() -> None:
    parser = argparse.ArgumentParser(prog="xgb_model.train", description="Train XGBoost on flattened model windows.")
    parser.add_argument("--config", type=Path, required=True, help="Path to configs/pipeline.toml")
    parser.add_argument("--window-size", type=int, default=None, help="Model window size to train on.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Optional override for the model-window dataset directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional run directory override.")
    parser.add_argument("--batch-size", type=int, default=8192, help="Batch size used while materializing splits.")
    parser.add_argument("--eval-batch-size", type=int, default=16384, help="Batch size used for evaluation loads.")
    parser.add_argument("--num-round", type=int, default=300, help="Maximum boosting rounds.")
    parser.add_argument("--early-stopping-rounds", type=int, default=30, help="Early stopping patience.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Boosting learning rate.")
    parser.add_argument("--max-depth", type=int, default=8, help="Tree depth.")
    parser.add_argument("--subsample", type=float, default=0.8, help="Row subsampling rate.")
    parser.add_argument("--colsample-bytree", type=float, default=0.8, help="Column subsampling rate.")
    parser.add_argument("--min-child-weight", type=float, default=1.0, help="Minimum child weight.")
    parser.add_argument("--gamma", type=float, default=0.0, help="Minimum loss reduction.")
    parser.add_argument("--reg-lambda", type=float, default=1.0, help="L2 regularization.")
    parser.add_argument("--reg-alpha", type=float, default=0.0, help="L1 regularization.")
    parser.add_argument("--max-bin", type=int, default=256, help="Histogram bin count.")
    parser.add_argument("--tree-method", type=str, default="hist", help="XGBoost tree method.")
    parser.add_argument("--objective", type=str, default="binary:logistic", help="Training objective.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--nthread", type=int, default=None, help="Thread count override.")
    parser.add_argument(
        "--disable-summary-features",
        action="store_true",
        help="Disable derived window summary features and train on the flattened raw window only.",
    )
    args = parser.parse_args()

    config = load_window_config(args.config, window_size=args.window_size, dataset_dir=args.dataset_dir)
    result = train_xgboost(
        config=config,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_round=args.num_round,
        early_stopping_rounds=args.early_stopping_rounds,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        max_bin=args.max_bin,
        tree_method=args.tree_method,
        objective=args.objective,
        seed=args.seed,
        nthread=args.nthread,
        include_summary_features=not args.disable_summary_features,
    )

    print(
        json.dumps(
            {
                "run_dir": str(result.run_dir),
                "best_iteration": result.best_iteration,
                "best_validation_auc_roc": result.best_validation_auc_roc,
            },
            indent=2,
        )
    )


def train_xgboost(
    *,
    config: WindowConfig,
    output_dir: Optional[Path],
    batch_size: int,
    eval_batch_size: int,
    num_round: int,
    early_stopping_rounds: int,
    learning_rate: float,
    max_depth: int,
    subsample: float,
    colsample_bytree: float,
    min_child_weight: float,
    gamma: float,
    reg_lambda: float,
    reg_alpha: float,
    max_bin: int,
    tree_method: str,
    objective: str,
    seed: int,
    nthread: Optional[int],
    include_summary_features: bool,
) -> TrainResult:
    dataset_dir = config.dataset_dir
    if not dataset_dir.exists():
        raise FileNotFoundError("Model-window dataset not found: %s" % dataset_dir)

    split_summaries = summarize_dataset(dataset_dir, batch_size=batch_size)
    train_summary = split_summaries["train"]
    if train_summary.positive_rows == 0:
        raise ValueError("Training split contains no positive labels.")
    if train_summary.negative_rows == 0:
        raise ValueError("Training split contains no negative labels.")

    feature_order = list(config.feature_order)
    flat_feature_names = build_augmented_feature_names(
        feature_order,
        config.window_size,
        include_summary_features=include_summary_features,
    )
    feature_count = len(flat_feature_names)

    train_arrays = load_split_arrays(
        dataset_dir,
        "train",
        batch_size=batch_size,
        row_count=train_summary.row_count,
        include_summary_features=include_summary_features,
    )
    validation_arrays = load_split_arrays(
        dataset_dir,
        "validation",
        batch_size=eval_batch_size,
        row_count=split_summaries["validation"].row_count,
        include_summary_features=include_summary_features,
    )
    test_arrays = load_split_arrays(
        dataset_dir,
        "test",
        batch_size=eval_batch_size,
        row_count=split_summaries["test"].row_count,
        include_summary_features=include_summary_features,
    )

    xgb = load_external_xgboost()
    dtrain = _make_dmatrix(xgb, train_arrays.features, train_arrays.labels, flat_feature_names)
    dvalidation = _make_dmatrix(xgb, validation_arrays.features, validation_arrays.labels, flat_feature_names)
    dtest = _make_dmatrix(xgb, test_arrays.features, test_arrays.labels, flat_feature_names)

    scale_pos_weight = float(train_summary.negative_rows) / float(train_summary.positive_rows)
    params = {
        "objective": objective,
        # XGBoost early stopping tracks the last metric in eval_metric.
        # Keep validation AUC last so best_iteration/best_score align with the repo's selection metric.
        "eval_metric": ["aucpr", "logloss", "auc"],
        "eta": learning_rate,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "min_child_weight": min_child_weight,
        "gamma": gamma,
        "lambda": reg_lambda,
        "alpha": reg_alpha,
        "max_bin": max_bin,
        "tree_method": tree_method,
        "scale_pos_weight": scale_pos_weight,
        "seed": seed,
        "verbosity": 1,
    }
    if nthread is not None:
        params["nthread"] = int(nthread)
    elif config.runtime.threads > 0:
        params["nthread"] = int(config.runtime.threads)

    evals_result: Dict[str, Dict[str, List[float]]] = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_round,
        evals=[(dtrain, "train"), (dvalidation, "validation")],
        evals_result=evals_result,
        early_stopping_rounds=early_stopping_rounds,
        maximize=True,
        verbose_eval=False,
    )

    best_iteration = _resolve_best_iteration(booster, evals_result)
    if best_iteration < 0:
        best_iteration = 0
    best_validation_auc = _resolve_best_score(booster, evals_result)

    history = _build_history_rows(evals_result)

    train_eval = _evaluate_booster(
        booster,
        dtrain,
        train_arrays.window_ids,
        train_arrays.labels,
        best_iteration=best_iteration,
    )
    validation_eval = _evaluate_booster(
        booster,
        dvalidation,
        validation_arrays.window_ids,
        validation_arrays.labels,
        best_iteration=best_iteration,
    )
    test_eval = _evaluate_booster(
        booster,
        dtest,
        test_arrays.window_ids,
        test_arrays.labels,
        best_iteration=best_iteration,
    )

    run_dir = make_run_dir(config.output_root, "xgboost", config.window_size, output_dir=output_dir)
    training_config = {
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "num_round": num_round,
        "early_stopping_rounds": early_stopping_rounds,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "min_child_weight": min_child_weight,
        "gamma": gamma,
        "reg_lambda": reg_lambda,
        "reg_alpha": reg_alpha,
        "max_bin": max_bin,
        "tree_method": tree_method,
        "objective": objective,
        "seed": seed,
        "nthread": params.get("nthread"),
        "scale_pos_weight": scale_pos_weight,
        "include_summary_features": bool(include_summary_features),
    }

    summary_payload = build_summary_payload(
        model_name="xgboost",
        config_path=config.config_path,
        dataset_dir=dataset_dir,
        output_root=config.output_root,
        manifest_dirname=config.manifest_dirname,
        window_size=config.window_size,
        feature_order=feature_order,
        split_summaries=[split_summaries[split].to_dict() for split in ("train", "validation", "test")],
        class_weights={
            "positive": scale_pos_weight,
            "negative": 1.0,
        },
        training_config=training_config,
        feature_count=feature_count,
        feature_names=flat_feature_names,
    )
    metrics_payload = {
        "best_iteration": best_iteration,
        "best_iteration_1based": best_iteration + 1,
        "best_validation_auc_roc": best_validation_auc,
        "train": train_eval.metrics,
        "validation": validation_eval.metrics,
        "test": test_eval.metrics,
    }

    write_json(run_dir / "summary.json", summary_payload)
    write_json(run_dir / "metrics.json", metrics_payload)
    write_json(
        run_dir / "diagnostics.json",
        build_debug_diagnostics(
            split_payloads={
                "train": {
                    "labels": train_eval.labels,
                    "probabilities": train_eval.probabilities,
                    "logits": train_eval.logits,
                },
                "validation": {
                    "labels": validation_eval.labels,
                    "probabilities": validation_eval.probabilities,
                    "logits": validation_eval.logits,
                },
                "test": {
                    "labels": test_eval.labels,
                    "probabilities": test_eval.probabilities,
                    "logits": test_eval.logits,
                },
            },
            reference_thresholds={
                "default_0_5": 0.5,
                "class_weight_adjusted": 1.0 / (1.0 + scale_pos_weight),
            },
        ),
    )
    write_history_csv(run_dir / "history.csv", history)
    write_predictions_parquet(
        run_dir / "predictions_validation.parquet",
        window_ids=validation_eval.ids,
        labels=validation_eval.labels,
        probabilities=validation_eval.probabilities,
        split="validation",
    )
    write_predictions_parquet(
        run_dir / "predictions_test.parquet",
        window_ids=test_eval.ids,
        labels=test_eval.labels,
        probabilities=test_eval.probabilities,
        split="test",
    )
    write_model_json(run_dir / "model.json", booster)

    return TrainResult(
        run_dir=run_dir,
        best_iteration=best_iteration,
        best_validation_auc_roc=best_validation_auc,
        history=history,
        train_eval=train_eval,
        validation_eval=validation_eval,
        test_eval=test_eval,
        booster=booster,
    )


def _make_dmatrix(xgb: Any, features: np.ndarray, labels: np.ndarray, feature_names: Sequence[str]) -> Any:
    matrix_factory = getattr(xgb, "QuantileDMatrix", None)
    if matrix_factory is None:
        matrix_factory = xgb.DMatrix
    return matrix_factory(
        np.asarray(features, dtype=np.float32),
        label=np.asarray(labels, dtype=np.float32),
        feature_names=list(feature_names),
    )


def _evaluate_booster(
    booster: Any,
    dmatrix: Any,
    window_ids: Sequence[str],
    labels: np.ndarray,
    *,
    best_iteration: int,
) -> EvalResult:
    probabilities = _predict_probabilities(booster, dmatrix, best_iteration=best_iteration)
    metrics = binary_classification_metrics(np.asarray(labels, dtype=np.int64), probabilities)
    logits = _safe_logit(probabilities)
    return EvalResult(
        ids=list(window_ids),
        labels=np.asarray(labels, dtype=np.int64),
        probabilities=probabilities,
        logits=logits,
        metrics=metrics,
    )


def _predict_probabilities(booster: Any, dmatrix: Any, *, best_iteration: int) -> np.ndarray:
    iteration_range = (0, best_iteration + 1)
    try:
        probabilities = booster.predict(dmatrix, iteration_range=iteration_range)
    except TypeError:  # pragma: no cover - older xgboost compatibility
        probabilities = booster.predict(dmatrix, ntree_limit=best_iteration + 1)
    return np.asarray(probabilities, dtype=np.float64)


def _build_history_rows(evals_result: Dict[str, Dict[str, List[float]]]) -> List[Dict[str, object]]:
    if not evals_result:
        return []
    iteration_count = max(len(metric_values) for dataset_values in evals_result.values() for metric_values in dataset_values.values())
    history: List[Dict[str, object]] = []
    for index in range(iteration_count):
        row: Dict[str, object] = {"iteration": index + 1}
        for dataset_name, dataset_values in evals_result.items():
            for metric_name, metric_values in dataset_values.items():
                if index < len(metric_values):
                    row[f"{dataset_name}_{metric_name}"] = float(metric_values[index])
        history.append(row)
    return history


def _safe_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _best_validation_score(evals_result: Dict[str, Dict[str, List[float]]]) -> float:
    validation_values = evals_result.get("validation", {})
    auc_values = validation_values.get("auc", [])
    if not auc_values:
        return 0.0
    return float(max(auc_values))


def _resolve_best_iteration(booster: Any, evals_result: Dict[str, Dict[str, List[float]]]) -> int:
    best_iteration = getattr(booster, "best_iteration", None)
    if best_iteration is not None:
        return int(best_iteration)
    validation_values = evals_result.get("validation", {})
    auc_values = validation_values.get("auc", [])
    if not auc_values:
        return 0
    return int(np.argmax(np.asarray(auc_values, dtype=np.float64)))


def _resolve_best_score(booster: Any, evals_result: Dict[str, Dict[str, List[float]]]) -> float:
    best_score = getattr(booster, "best_score", None)
    if best_score is not None:
        try:
            return float(best_score)
        except (TypeError, ValueError):
            pass
    return _best_validation_score(evals_result)


if __name__ == "__main__":
    main()
