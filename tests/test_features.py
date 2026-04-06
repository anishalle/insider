from datetime import datetime, timezone

from insider.features import (
    SplitCutoffs,
    compute_markout,
    compute_markout_bps,
    compute_ratio_cutoffs,
    direction_to_side,
    encode_role,
    normalize_yes_price,
)


def test_normalize_yes_price_flips_token2() -> None:
    assert normalize_yes_price(0.25, "token2") == 0.75
    assert normalize_yes_price(0.25, "token1") == 0.25


def test_direction_to_side_handles_common_variants() -> None:
    assert direction_to_side("BUY") == 1
    assert direction_to_side("sell") == -1
    assert direction_to_side("+maker") == 1
    assert direction_to_side("ask-side") == -1
    assert direction_to_side("unknown") is None


def test_markout_math_matches_notebook_definition() -> None:
    assert round(compute_markout(0.40, 0.55, 1), 10) == 0.15
    assert round(compute_markout_bps(0.40, 0.55, 1), 2) == 3750.0
    assert round(compute_markout_bps(0.40, 0.55, -1), 2) == -3750.0
    assert encode_role("maker") == 1.0
    assert encode_role("taker") == 0.0


def test_ratio_cutoffs_assign_time_ordered_splits() -> None:
    timestamps = [
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 2, tzinfo=timezone.utc),
        datetime(2024, 1, 3, tzinfo=timezone.utc),
        datetime(2024, 1, 4, tzinfo=timezone.utc),
        datetime(2024, 1, 5, tzinfo=timezone.utc),
    ]
    cutoffs = compute_ratio_cutoffs(timestamps, 0.6, 0.2)
    assert isinstance(cutoffs, SplitCutoffs)
    assert cutoffs.assign(datetime(2024, 1, 2, tzinfo=timezone.utc)) == "train"
    assert cutoffs.assign(datetime(2024, 1, 4, tzinfo=timezone.utc)) == "validation"
    assert cutoffs.assign(datetime(2024, 1, 5, tzinfo=timezone.utc)) == "test"
