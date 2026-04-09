from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

DEFAULT_LOG1P_FEATURES: Tuple[str, ...] = (
    "usd_amount",
    "time_delta_seconds",
    "market_age_seconds",
    "market_trade_count_1h",
    "market_volume_1h",
    "user_trade_count_1h",
    "user_market_trade_count_1h",
    "user_usd_volume_1h",
)
DEFAULT_SIGNED_LOG1P_FEATURES: Tuple[str, ...] = (
    "signed_token_amount",
    "user_signed_flow_1h",
)
DEFAULT_UNSCALED_FEATURES: Tuple[str, ...] = ("side", "role_is_maker")
DEFAULT_CLIPPED_FEATURES: Tuple[str, ...] = (
    "signed_token_amount",
    "usd_amount",
    "time_delta_seconds",
    "market_age_seconds",
    "market_trade_count_1h",
    "market_volume_1h",
    "user_trade_count_1h",
    "user_market_trade_count_1h",
    "user_signed_flow_1h",
    "user_usd_volume_1h",
)


@dataclass(frozen=True)
class SequenceStandardizationStats:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    row_count: int
    transform_kinds: tuple[str, ...]
    clip_lower: np.ndarray
    clip_upper: np.ndarray
    scale_enabled: np.ndarray
    clip_enabled: np.ndarray
    clip_percentiles: tuple[float, float]
    clip_sample_rows: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "enabled": True,
            "row_count": int(self.row_count),
            "clip_percentiles": [float(self.clip_percentiles[0]), float(self.clip_percentiles[1])],
            "clip_sample_rows": int(self.clip_sample_rows),
            "features": [
                {
                    "name": feature_name,
                    "transform": self.transform_kinds[index],
                    "mean": float(self.mean[index]),
                    "scale": float(self.scale[index]),
                    "scale_enabled": bool(self.scale_enabled[index]),
                    "clip_enabled": bool(self.clip_enabled[index]),
                    "clip_lower": float(self.clip_lower[index]),
                    "clip_upper": float(self.clip_upper[index]),
                }
                for index, feature_name in enumerate(self.feature_names)
            ],
        }


class SequenceStandardizer:
    def __init__(
        self,
        feature_names: Sequence[str],
        *,
        transform_kinds: Optional[Sequence[str]] = None,
        clip_lower: Optional[np.ndarray] = None,
        clip_upper: Optional[np.ndarray] = None,
        scale_enabled: Optional[np.ndarray] = None,
        clip_enabled: Optional[np.ndarray] = None,
        clip_percentiles: Tuple[float, float] = (0.5, 99.5),
        clip_sample_rows: int = 0,
    ) -> None:
        self.feature_names = tuple(feature_names)
        if not self.feature_names:
            raise ValueError("feature_names must not be empty.")
        resolved_transform_kinds = (
            tuple(transform_kinds)
            if transform_kinds is not None
            else build_default_transform_kinds(self.feature_names)
        )
        if len(resolved_transform_kinds) != len(self.feature_names):
            raise ValueError("transform_kinds must match feature_names length.")
        self.transform_kinds = resolved_transform_kinds
        self.scale_enabled = (
            np.asarray(scale_enabled, dtype=bool)
            if scale_enabled is not None
            else build_default_scale_enabled(self.feature_names)
        )
        self.clip_enabled = (
            np.asarray(clip_enabled, dtype=bool)
            if clip_enabled is not None
            else build_default_clip_enabled(self.feature_names)
        )
        if self.scale_enabled.shape != (len(self.feature_names),):
            raise ValueError("scale_enabled must have one value per feature.")
        if self.clip_enabled.shape != (len(self.feature_names),):
            raise ValueError("clip_enabled must have one value per feature.")
        self.clip_lower = (
            np.asarray(clip_lower, dtype=np.float32)
            if clip_lower is not None
            else np.full(len(self.feature_names), -np.inf, dtype=np.float32)
        )
        self.clip_upper = (
            np.asarray(clip_upper, dtype=np.float32)
            if clip_upper is not None
            else np.full(len(self.feature_names), np.inf, dtype=np.float32)
        )
        if self.clip_lower.shape != (len(self.feature_names),):
            raise ValueError("clip_lower must have one value per feature.")
        if self.clip_upper.shape != (len(self.feature_names),):
            raise ValueError("clip_upper must have one value per feature.")
        self.clip_percentiles = (float(clip_percentiles[0]), float(clip_percentiles[1]))
        self.clip_sample_rows = int(clip_sample_rows)
        self._row_count = 0
        self._sum = np.zeros(len(self.feature_names), dtype=np.float64)
        self._sum_squares = np.zeros(len(self.feature_names), dtype=np.float64)

    def update(self, features: np.ndarray) -> None:
        array = np.asarray(features, dtype=np.float32)
        if array.ndim != 3 or array.shape[2] != len(self.feature_names):
            raise ValueError(
                "Expected sequence features with shape (batch, window, %d), got %s"
                % (len(self.feature_names), array.shape)
            )
        transformed = transform_sequence_features(array, self.transform_kinds)
        clipped = clip_sequence_features(
            transformed,
            clip_lower=self.clip_lower,
            clip_upper=self.clip_upper,
            clip_enabled=self.clip_enabled,
        )
        flattened = np.asarray(clipped, dtype=np.float64).reshape(-1, array.shape[2])
        self._row_count += int(flattened.shape[0])
        self._sum += flattened.sum(axis=0)
        self._sum_squares += np.square(flattened).sum(axis=0)

    def finalize(self) -> SequenceStandardizationStats:
        if self._row_count == 0:
            raise ValueError("Cannot finalize sequence standardizer with zero rows.")
        mean = self._sum / self._row_count
        variance = np.maximum(self._sum_squares / self._row_count - np.square(mean), 0.0)
        scale = np.sqrt(variance)
        scale[scale < 1e-12] = 1.0
        return SequenceStandardizationStats(
            feature_names=self.feature_names,
            mean=mean.astype(np.float32),
            scale=scale.astype(np.float32),
            row_count=self._row_count,
            transform_kinds=self.transform_kinds,
            clip_lower=self.clip_lower.astype(np.float32),
            clip_upper=self.clip_upper.astype(np.float32),
            scale_enabled=self.scale_enabled.astype(bool),
            clip_enabled=self.clip_enabled.astype(bool),
            clip_percentiles=self.clip_percentiles,
            clip_sample_rows=self.clip_sample_rows,
        )


def build_default_transform_kinds(feature_names: Sequence[str]) -> tuple[str, ...]:
    return tuple(_default_transform_kind(name) for name in feature_names)


def build_default_scale_enabled(feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray([name not in DEFAULT_UNSCALED_FEATURES for name in feature_names], dtype=bool)


def build_default_clip_enabled(feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray([name in DEFAULT_CLIPPED_FEATURES for name in feature_names], dtype=bool)


def transform_sequence_features(
    features: np.ndarray,
    transform_kinds: Sequence[str],
) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != len(transform_kinds):
        raise ValueError(
            "Expected sequence features with shape (batch, window, %d), got %s"
            % (len(transform_kinds), array.shape)
        )
    transformed = np.array(array, dtype=np.float32, copy=True)
    for index, transform_kind in enumerate(transform_kinds):
        if transform_kind == "identity":
            continue
        if transform_kind == "log1p":
            transformed[..., index] = np.log1p(np.maximum(transformed[..., index], 0.0))
            continue
        if transform_kind == "signed_log1p":
            values = transformed[..., index]
            transformed[..., index] = np.sign(values) * np.log1p(np.abs(values))
            continue
        raise ValueError("Unsupported transform_kind: %s" % transform_kind)
    return transformed


def clip_sequence_features(
    features: np.ndarray,
    *,
    clip_lower: np.ndarray,
    clip_upper: np.ndarray,
    clip_enabled: np.ndarray,
) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != int(clip_enabled.shape[0]):
        raise ValueError(
            "Expected sequence features with shape (batch, window, %d), got %s"
            % (int(clip_enabled.shape[0]), array.shape)
        )
    clipped = np.array(array, dtype=np.float32, copy=True)
    for index, enabled in enumerate(np.asarray(clip_enabled, dtype=bool).tolist()):
        if not enabled:
            continue
        clipped[..., index] = np.clip(clipped[..., index], clip_lower[index], clip_upper[index])
    return clipped


def apply_sequence_standardization(
    features: np.ndarray,
    stats: SequenceStandardizationStats,
) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != len(stats.feature_names):
        raise ValueError(
            "Expected sequence features with shape (batch, window, %d), got %s"
            % (len(stats.feature_names), array.shape)
        )
    transformed = transform_sequence_features(array, stats.transform_kinds)
    clipped = clip_sequence_features(
        transformed,
        clip_lower=stats.clip_lower,
        clip_upper=stats.clip_upper,
        clip_enabled=stats.clip_enabled,
    )
    mean = stats.mean.reshape(1, 1, -1)
    scale = stats.scale.reshape(1, 1, -1)
    standardized = np.array(clipped, dtype=np.float32, copy=True)
    for index, enabled in enumerate(stats.scale_enabled.tolist()):
        if not enabled:
            continue
        standardized[..., index] = (standardized[..., index] - mean[..., index]) / scale[..., index]
    return standardized.astype(np.float32, copy=False)


def _default_transform_kind(feature_name: str) -> str:
    if feature_name in DEFAULT_LOG1P_FEATURES:
        return "log1p"
    if feature_name in DEFAULT_SIGNED_LOG1P_FEATURES:
        return "signed_log1p"
    return "identity"
