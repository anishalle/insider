from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable


DEFAULT_CONTRACT_ADDRESSES = (
    "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e",
    "0xc5d563a36ae78145c45a50134d48a1215220f80a",
)

DEFAULT_FEATURE_ORDER = (
    "price_yes",
    "signed_token_amount",
    "usd_amount",
    "side",
    "role_is_maker",
    "time_delta_seconds",
    "market_age_seconds",
)


def normalize_yes_price(price: float | None, nonusdc_side: str | None) -> float | None:
    if price is None:
        return None
    if (nonusdc_side or "").strip().lower() == "token2":
        return 1.0 - float(price)
    return float(price)


def direction_to_side(direction: str | None) -> int | None:
    normalized = (direction or "").strip().lower()
    if not normalized:
        return None
    if normalized in {"buy", "bid", "b", "long"}:
        return 1
    if normalized in {"sell", "ask", "s", "short"}:
        return -1
    if "buy" in normalized or "bid" in normalized or normalized.startswith("+"):
        return 1
    if "sell" in normalized or "ask" in normalized or normalized.startswith("-"):
        return -1
    return None


def encode_role(role: str | None) -> float:
    return 1.0 if (role or "").strip().lower() == "maker" else 0.0


def compute_markout(price_yes: float | None, future_price_yes: float | None, side: int | None) -> float | None:
    if price_yes is None or future_price_yes is None or side is None:
        return None
    return int(side) * (float(future_price_yes) - float(price_yes))


def compute_markout_bps(price_yes: float | None, future_price_yes: float | None, side: int | None) -> float | None:
    if price_yes in (None, 0) or future_price_yes is None or side is None:
        return None
    return int(side) * ((float(future_price_yes) / float(price_yes)) - 1.0) * 10_000.0


def to_utc_epoch_seconds(value: datetime) -> float:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


@dataclass(frozen=True)
class SplitCutoffs:
    train_end_epoch: float
    validation_end_epoch: float

    def assign(self, trade_time: datetime) -> str:
        epoch = to_utc_epoch_seconds(trade_time)
        if epoch <= self.train_end_epoch:
            return "train"
        if epoch <= self.validation_end_epoch:
            return "validation"
        return "test"


def compute_ratio_cutoffs(
    ordered_timestamps: Iterable[datetime],
    train_ratio: float,
    validation_ratio: float,
) -> SplitCutoffs:
    timestamps = sorted(to_utc_epoch_seconds(ts) for ts in ordered_timestamps)
    if not timestamps:
        raise ValueError("Cannot compute split cutoffs without timestamps.")
    train_index = min(len(timestamps) - 1, max(0, int(len(timestamps) * train_ratio) - 1))
    validation_index = min(
        len(timestamps) - 1,
        max(train_index, int(len(timestamps) * (train_ratio + validation_ratio)) - 1),
    )
    return SplitCutoffs(
        train_end_epoch=timestamps[train_index],
        validation_end_epoch=timestamps[validation_index],
    )
