from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from insider.config import PipelineConfig
from insider.duckdb_utils import connect


@dataclass(frozen=True)
class SignalCandidate:
    selection_reason: str
    label: int
    user: str
    market_id: str
    asset_id: str
    role: str
    side: int
    trade_time: datetime
    target_time: datetime
    future_trade_time: datetime
    price_yes: float
    future_price_yes: float
    markout: float
    markout_bps: float
    usd_amount: float
    token_amount: float
    transaction_hash: str
    order_hash: str


@dataclass(frozen=True)
class ContextRow:
    trade_time: datetime
    price_yes: float
    usd_amount: float
    transaction_hash: str | None


def visualize_signals(
    config: PipelineConfig,
    *,
    output_dir: Path | None = None,
    examples_per_class: int = 4,
    ambiguous_examples: int = 2,
    lookback_minutes: int = 60,
    lookahead_minutes: int = 30,
    model_window_size: int = 50,
) -> Path:
    if examples_per_class < 1:
        raise ValueError("examples_per_class must be at least 1.")
    if ambiguous_examples < 0:
        raise ValueError("ambiguous_examples cannot be negative.")
    if lookback_minutes < 1 or lookahead_minutes < 1:
        raise ValueError("lookback_minutes and lookahead_minutes must be positive.")

    resolved_output_dir = output_dir or Path("/home/axa230262/work/001 research/insider/analysis/signal-review")
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = resolved_output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    labeled_dir = config.output.root / config.output.labeled_dirname
    if not labeled_dir.exists():
        raise FileNotFoundError(f"Labeled dataset not found: {labeled_dir}")

    labeled_glob = _dataset_glob(labeled_dir)
    prepared_dir = config.output.root / config.output.prepared_dirname
    prepared_glob = _dataset_glob(prepared_dir) if prepared_dir.exists() else None
    window_dir = config.output.root / config.output.model_window_dirname / f"window_size={model_window_size}"
    window_glob = _dataset_glob(window_dir) if window_dir.exists() else None

    connection = connect(config.runtime, config.output.root)
    try:
        summary = _compute_summary(connection, labeled_glob)
        monthly_balance = _fetch_monthly_balance(connection, labeled_glob)
        candidates = _select_candidates(
            connection,
            labeled_glob,
            examples_per_class=examples_per_class,
            ambiguous_examples=ambiguous_examples,
        )
        feasibility = _compute_training_feasibility(
            connection,
            labeled_glob,
            window_glob=window_glob,
            model_window_size=model_window_size,
            stride=config.model_windows.stride,
        )

        for candidate in candidates:
            context_rows = _fetch_context_rows(
                connection,
                prepared_glob=prepared_glob,
                labeled_glob=labeled_glob,
                candidate=candidate,
                lookback_minutes=lookback_minutes,
                lookahead_minutes=lookahead_minutes,
            )
            plot_path = plots_dir / _candidate_filename(candidate)
            _render_candidate_plot(candidate, context_rows, plot_path)

    finally:
        connection.close()

    _write_json(resolved_output_dir / "summary.json", summary)
    _write_json(resolved_output_dir / "training_feasibility.json", feasibility)
    _write_csv(resolved_output_dir / "class_balance_over_time.csv", monthly_balance)
    _write_csv(
        resolved_output_dir / "candidate_signals.csv",
        [_candidate_to_row(candidate) for candidate in candidates],
    )

    return resolved_output_dir


def _compute_summary(connection: Any, labeled_glob: str) -> dict[str, Any]:
    totals = _fetch_one(
        connection,
        f"""
        SELECT
            COUNT(*) AS total_rows,
            SUM(label) AS positive_rows,
            COUNT(*) - SUM(label) AS negative_rows,
            AVG(CAST(label AS DOUBLE)) AS positive_rate,
            COUNT(DISTINCT user) AS distinct_users,
            COUNT(DISTINCT market_id) AS distinct_markets,
            COUNT(DISTINCT asset_id) AS distinct_assets,
            MIN(trade_time) AS min_trade_time,
            MAX(trade_time) AS max_trade_time,
            AVG(ABS(markout_bps)) AS avg_abs_markout_bps,
            AVG(usd_amount) AS avg_usd_amount
        FROM read_parquet('{_sql_escape(labeled_glob)}')
        """,
    )
    if not totals:
        raise ValueError("Labeled dataset is empty.")

    positive_rows = int(totals["positive_rows"])
    negative_rows = int(totals["negative_rows"])
    minority_count = min(positive_rows, negative_rows)
    majority_count = max(positive_rows, negative_rows)

    return {
        "dataset": "labeled_user_trades",
        "totals": {
            "rows": int(totals["total_rows"]),
            "positive_rows": positive_rows,
            "negative_rows": negative_rows,
            "positive_rate": float(totals["positive_rate"]),
            "negative_rate": float(1.0 - float(totals["positive_rate"])),
            "positive_to_negative_ratio": (positive_rows / negative_rows) if negative_rows else None,
            "majority_to_minority_ratio": (majority_count / minority_count) if minority_count else None,
            "distinct_users": int(totals["distinct_users"]),
            "distinct_markets": int(totals["distinct_markets"]),
            "distinct_assets": int(totals["distinct_assets"]),
            "min_trade_time": totals["min_trade_time"],
            "max_trade_time": totals["max_trade_time"],
            "avg_abs_markout_bps": float(totals["avg_abs_markout_bps"]),
            "avg_usd_amount": float(totals["avg_usd_amount"]),
        },
        "markout_percentiles": {
            "markout": _fetch_percentiles(connection, labeled_glob, "markout"),
            "markout_bps": _fetch_percentiles(connection, labeled_glob, "markout_bps"),
            "usd_amount": _fetch_percentiles(connection, labeled_glob, "usd_amount"),
        },
    }


def _fetch_percentiles(connection: Any, labeled_glob: str, column: str) -> dict[str, float]:
    row = _fetch_one(
        connection,
        f"""
        SELECT quantile_cont({column}, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]) AS values
        FROM read_parquet('{_sql_escape(labeled_glob)}')
        """,
    )
    values = row["values"]
    percentiles = ("p01", "p05", "p25", "p50", "p75", "p95", "p99")
    return {name: float(value) for name, value in zip(percentiles, values)}


def _fetch_monthly_balance(connection: Any, labeled_glob: str) -> list[dict[str, Any]]:
    return _fetch_all(
        connection,
        f"""
        SELECT
            strftime(trade_time, '%Y-%m') AS month,
            COUNT(*) AS row_count,
            SUM(label) AS positive_rows,
            COUNT(*) - SUM(label) AS negative_rows,
            AVG(CAST(label AS DOUBLE)) AS positive_rate,
            AVG(ABS(markout_bps)) AS avg_abs_markout_bps,
            AVG(usd_amount) AS avg_usd_amount
        FROM read_parquet('{_sql_escape(labeled_glob)}')
        GROUP BY 1
        ORDER BY 1
        """,
    )


def _select_candidates(
    connection: Any,
    labeled_glob: str,
    *,
    examples_per_class: int,
    ambiguous_examples: int,
) -> list[SignalCandidate]:
    queries = [
        (
            "true_signal_top_markout_bps",
            f"""
            SELECT
                label,
                user,
                market_id,
                asset_id,
                role,
                side,
                trade_time,
                target_time,
                future_trade_time,
                price_yes,
                future_price_yes,
                markout,
                markout_bps,
                usd_amount,
                token_amount,
                transaction_hash,
                order_hash
            FROM read_parquet('{_sql_escape(labeled_glob)}')
            WHERE label = 1
            ORDER BY markout_bps DESC, usd_amount DESC, trade_time ASC
            LIMIT {examples_per_class}
            """,
        ),
        (
            "false_signal_top_markout_bps",
            f"""
            SELECT
                label,
                user,
                market_id,
                asset_id,
                role,
                side,
                trade_time,
                target_time,
                future_trade_time,
                price_yes,
                future_price_yes,
                markout,
                markout_bps,
                usd_amount,
                token_amount,
                transaction_hash,
                order_hash
            FROM read_parquet('{_sql_escape(labeled_glob)}')
            WHERE label = 0
            ORDER BY markout_bps ASC, usd_amount DESC, trade_time ASC
            LIMIT {examples_per_class}
            """,
        ),
    ]
    if ambiguous_examples:
        queries.append(
            (
                "near_zero_markout",
                f"""
                SELECT
                    label,
                    user,
                    market_id,
                    asset_id,
                    role,
                    side,
                    trade_time,
                    target_time,
                    future_trade_time,
                    price_yes,
                    future_price_yes,
                    markout,
                    markout_bps,
                    usd_amount,
                    token_amount,
                    transaction_hash,
                    order_hash
                FROM read_parquet('{_sql_escape(labeled_glob)}')
                ORDER BY ABS(markout_bps) ASC, usd_amount DESC, trade_time ASC
                LIMIT {ambiguous_examples}
                """,
            )
        )

    deduped: dict[tuple[str, str, str], SignalCandidate] = {}
    for selection_reason, query in queries:
        for row in _fetch_all(connection, query):
            key = (
                str(row["transaction_hash"]),
                str(row["user"]),
                str(row["role"]),
            )
            if key in deduped:
                continue
            deduped[key] = SignalCandidate(
                selection_reason=selection_reason,
                label=int(row["label"]),
                user=str(row["user"]),
                market_id=str(row["market_id"]),
                asset_id=str(row["asset_id"]),
                role=str(row["role"]),
                side=int(row["side"]),
                trade_time=row["trade_time"],
                target_time=row["target_time"],
                future_trade_time=row["future_trade_time"],
                price_yes=float(row["price_yes"]),
                future_price_yes=float(row["future_price_yes"]),
                markout=float(row["markout"]),
                markout_bps=float(row["markout_bps"]),
                usd_amount=float(row["usd_amount"]),
                token_amount=float(row["token_amount"]),
                transaction_hash=str(row["transaction_hash"]),
                order_hash=str(row["order_hash"]),
            )
    return list(deduped.values())


def _compute_training_feasibility(
    connection: Any,
    labeled_glob: str,
    *,
    window_glob: str | None,
    model_window_size: int,
    stride: int,
) -> dict[str, Any]:
    monthly_rates = _fetch_all(
        connection,
        f"""
        SELECT
            strftime(trade_time, '%Y-%m') AS month,
            AVG(CAST(label AS DOUBLE)) AS positive_rate
        FROM read_parquet('{_sql_escape(labeled_glob)}')
        GROUP BY 1
        ORDER BY 1
        """,
    )
    monthly_rate_values = [float(row["positive_rate"]) for row in monthly_rates]
    monthly_rate_range = (max(monthly_rate_values) - min(monthly_rate_values)) if monthly_rate_values else 0.0

    user_concentration = _fetch_one(
        connection,
        f"""
        WITH user_totals AS (
            SELECT
                user,
                COUNT(*) AS row_count,
                SUM(usd_amount) AS usd_total
            FROM read_parquet('{_sql_escape(labeled_glob)}')
            GROUP BY 1
        ),
        ranked AS (
            SELECT
                *,
                row_number() OVER (ORDER BY row_count DESC, user ASC) AS rn_rows,
                row_number() OVER (ORDER BY usd_total DESC, user ASC) AS rn_usd,
                SUM(row_count) OVER () AS total_rows,
                SUM(usd_total) OVER () AS total_usd
            FROM user_totals
        )
        SELECT
            SUM(CASE WHEN rn_rows <= 100 THEN row_count ELSE 0 END) / MAX(total_rows) AS top_100_row_share,
            SUM(CASE WHEN rn_usd <= 100 THEN usd_total ELSE 0 END) / MAX(total_usd) AS top_100_usd_share
        FROM ranked
        """,
    )
    market_concentration = _fetch_one(
        connection,
        f"""
        WITH market_totals AS (
            SELECT
                market_id,
                COUNT(*) AS row_count,
                SUM(usd_amount) AS usd_total
            FROM read_parquet('{_sql_escape(labeled_glob)}')
            GROUP BY 1
        ),
        ranked AS (
            SELECT
                *,
                row_number() OVER (ORDER BY row_count DESC, market_id ASC) AS rn_rows,
                row_number() OVER (ORDER BY usd_total DESC, market_id ASC) AS rn_usd,
                SUM(row_count) OVER () AS total_rows,
                SUM(usd_total) OVER () AS total_usd
            FROM market_totals
        )
        SELECT
            SUM(CASE WHEN rn_rows <= 100 THEN row_count ELSE 0 END) / MAX(total_rows) AS top_100_row_share,
            SUM(CASE WHEN rn_usd <= 100 THEN usd_total ELSE 0 END) / MAX(total_usd) AS top_100_usd_share
        FROM ranked
        """,
    )
    stream_lengths = _fetch_all(
        connection,
        f"""
        WITH stream_lengths AS (
            SELECT
                user,
                market_id,
                COUNT(*) AS stream_len
            FROM read_parquet('{_sql_escape(labeled_glob)}')
            GROUP BY 1, 2
        )
        SELECT
            CASE
                WHEN stream_len < 10 THEN '001-009'
                WHEN stream_len < 25 THEN '010-024'
                WHEN stream_len < 50 THEN '025-049'
                WHEN stream_len < 100 THEN '050-099'
                ELSE '100+'
            END AS bucket,
            COUNT(*) AS stream_count,
            AVG(stream_len) AS avg_stream_len
        FROM stream_lengths
        GROUP BY 1
        ORDER BY 1
        """,
    )
    window_estimates = _fetch_one(
        connection,
        f"""
        WITH stream_lengths AS (
            SELECT
                COUNT(*) AS stream_len
            FROM read_parquet('{_sql_escape(labeled_glob)}')
            GROUP BY user, market_id
        )
        SELECT
            COUNT(*) AS stream_count,
            SUM(CASE WHEN stream_len >= {model_window_size} THEN 1 ELSE 0 END) AS eligible_stream_count,
            SUM(
                CASE
                    WHEN stream_len >= {model_window_size} THEN stream_len - {model_window_size} + 1
                    ELSE 0
                END
            ) AS dense_window_count,
            SUM(
                CASE
                    WHEN stream_len >= {model_window_size}
                    THEN CAST(floor((stream_len - {model_window_size})::DOUBLE / {stride}) AS BIGINT) + 1
                    ELSE 0
                END
            ) AS stride_window_count
        FROM stream_lengths
        """,
    )

    actual_window_summary = None
    if window_glob is not None:
        actual_window_summary = {
            "splits": _fetch_all(
                connection,
                f"""
                SELECT
                    split,
                    COUNT(*) AS row_count,
                    AVG(CAST(label AS DOUBLE)) AS positive_rate
                FROM read_parquet('{_sql_escape(window_glob)}')
                GROUP BY 1
                ORDER BY 1
                """,
            ),
            "shape": _fetch_one(
                connection,
                f"""
                SELECT
                    MIN(window_size) AS min_window_size,
                    MAX(window_size) AS max_window_size,
                    MIN(length(features)) AS min_feature_rows,
                    MAX(length(features)) AS max_feature_rows,
                    MIN(length(features[1])) AS min_feature_width,
                    MAX(length(features[1])) AS max_feature_width
                FROM read_parquet('{_sql_escape(window_glob)}')
                """,
            ),
        }

    eligible_stream_count = int(window_estimates["eligible_stream_count"])
    stride_window_count = int(window_estimates["stride_window_count"])
    concentration_risk = (
        float(user_concentration["top_100_row_share"]) > 0.35
        or float(market_concentration["top_100_row_share"]) > 0.35
    )
    drift_risk = monthly_rate_range > 0.15
    imbalance_ratio = _fetch_class_imbalance_ratio(connection, labeled_glob)

    if eligible_stream_count < 100 or stride_window_count < 1_000:
        verdict = "blocked_for_training"
    elif imbalance_ratio > 2.0 or concentration_risk or drift_risk:
        verdict = "feasible_with_weighting_or_sampling"
    else:
        verdict = "feasible_now"

    return {
        "verdict": verdict,
        "class_imbalance": {
            "majority_to_minority_ratio": imbalance_ratio,
            "monthly_positive_rate_range": monthly_rate_range,
        },
        "concentration": {
            "users": user_concentration,
            "markets": market_concentration,
        },
        "stream_length_buckets": stream_lengths,
        "window_readiness": {
            "model_window_size": model_window_size,
            "stride": stride,
            "estimated_from_labeled_trades": {
                "stream_count": int(window_estimates["stream_count"]),
                "eligible_stream_count": eligible_stream_count,
                "dense_window_count": int(window_estimates["dense_window_count"]),
                "stride_window_count": stride_window_count,
            },
            "actual_model_window_summary": actual_window_summary,
        },
    }


def _fetch_class_imbalance_ratio(connection: Any, labeled_glob: str) -> float:
    row = _fetch_one(
        connection,
        f"""
        SELECT
            SUM(label) AS positive_rows,
            COUNT(*) - SUM(label) AS negative_rows
        FROM read_parquet('{_sql_escape(labeled_glob)}')
        """,
    )
    positive_rows = int(row["positive_rows"])
    negative_rows = int(row["negative_rows"])
    minority = min(positive_rows, negative_rows)
    majority = max(positive_rows, negative_rows)
    return float(majority / minority) if minority else float("inf")


def _fetch_context_rows(
    connection: Any,
    *,
    prepared_glob: str | None,
    labeled_glob: str,
    candidate: SignalCandidate,
    lookback_minutes: int,
    lookahead_minutes: int,
) -> list[ContextRow]:
    window_start = candidate.trade_time - timedelta(minutes=lookback_minutes)
    window_end = candidate.trade_time + timedelta(minutes=lookahead_minutes)

    context_rows: list[ContextRow] = []
    if prepared_glob is not None:
        context_rows = [
            ContextRow(
                trade_time=row["trade_time"],
                price_yes=float(row["price_yes"]),
                usd_amount=float(row["usd_amount"]),
                transaction_hash=str(row["transaction_hash"]) if row["transaction_hash"] is not None else None,
            )
            for row in _fetch_all(
                connection,
                f"""
                SELECT
                    trade_time,
                    price_yes,
                    usd_amount,
                    transaction_hash
                FROM read_parquet('{_sql_escape(prepared_glob)}')
                WHERE asset_id = '{_sql_escape(candidate.asset_id)}'
                  AND trade_time BETWEEN TIMESTAMP '{_format_timestamp(window_start)}'
                                     AND TIMESTAMP '{_format_timestamp(window_end)}'
                ORDER BY trade_time
                """,
            )
        ]
    if context_rows:
        return context_rows

    return [
        ContextRow(
            trade_time=row["trade_time"],
            price_yes=float(row["price_yes"]),
            usd_amount=float(row["usd_amount"]),
            transaction_hash=str(row["transaction_hash"]) if row["transaction_hash"] is not None else None,
        )
        for row in _fetch_all(
            connection,
            f"""
            SELECT DISTINCT
                trade_time,
                price_yes,
                usd_amount,
                transaction_hash
            FROM read_parquet('{_sql_escape(labeled_glob)}')
            WHERE asset_id = '{_sql_escape(candidate.asset_id)}'
              AND trade_time BETWEEN TIMESTAMP '{_format_timestamp(window_start)}'
                                 AND TIMESTAMP '{_format_timestamp(window_end)}'
            ORDER BY trade_time
            """,
        )
    ]


def _render_candidate_plot(candidate: SignalCandidate, context_rows: list[ContextRow], output_path: Path) -> None:
    cache_dir = Path(tempfile.gettempdir()) / "insider-matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    if not context_rows:
        context_rows = [
            ContextRow(
                trade_time=candidate.trade_time,
                price_yes=candidate.price_yes,
                usd_amount=candidate.usd_amount,
                transaction_hash=candidate.transaction_hash,
            )
        ]

    times = [row.trade_time for row in context_rows]
    prices = [row.price_yes for row in context_rows]
    notionals = [max(row.usd_amount, 1e-9) for row in context_rows]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5]},
    )
    signal_color = "#1f7a1f" if candidate.label == 1 else "#b22222"

    axes[0].plot(times, prices, color="#355070", linewidth=1.6, marker="o", markersize=3, alpha=0.85)
    axes[0].scatter([candidate.trade_time], [candidate.price_yes], color=signal_color, s=70, zorder=3)
    axes[0].scatter([candidate.future_trade_time], [candidate.future_price_yes], color="#111111", marker="x", s=75, zorder=3)
    axes[0].axvline(candidate.trade_time, color=signal_color, linestyle="--", linewidth=1.2)
    axes[0].axvline(candidate.target_time, color="#666666", linestyle=":", linewidth=1.0)
    axes[0].set_ylabel("YES price")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(alpha=0.2)

    axes[1].vlines(times, [0.0], notionals, color="#6d597a", linewidth=1.2, alpha=0.75)
    axes[1].scatter([candidate.trade_time], [max(candidate.usd_amount, 1e-9)], color=signal_color, s=70, zorder=3)
    axes[1].set_ylabel("USD notional")
    axes[1].set_xlabel("Trade time")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.2)

    title_prefix = "True signal" if candidate.label == 1 else "False signal"
    axes[0].set_title(f"{title_prefix} | {candidate.selection_reason} | market={candidate.market_id}")
    details = "\n".join(
        [
            f"user={candidate.user}",
            f"role={candidate.role} side={candidate.side:+d}",
            f"markout={candidate.markout:.4f}",
            f"markout_bps={candidate.markout_bps:.2f}",
            f"usd={candidate.usd_amount:.2f} token={candidate.token_amount:.4f}",
        ]
    )
    axes[0].text(
        0.01,
        0.99,
        details,
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _candidate_filename(candidate: SignalCandidate) -> str:
    label_name = "true" if candidate.label == 1 else "false"
    timestamp = candidate.trade_time.strftime("%Y%m%d%H%M%S")
    return f"{label_name}_{candidate.selection_reason}_{timestamp}_{_slugify(candidate.market_id)}.png"


def _candidate_to_row(candidate: SignalCandidate) -> dict[str, Any]:
    row = asdict(candidate)
    row["trade_time"] = candidate.trade_time
    row["target_time"] = candidate.target_time
    row["future_trade_time"] = candidate.future_trade_time
    return row


def _fetch_one(connection: Any, query: str) -> dict[str, Any]:
    rows = _fetch_all(connection, query)
    if not rows:
        raise ValueError("Expected query to return one row.")
    return rows[0]


def _fetch_all(connection: Any, query: str) -> list[dict[str, Any]]:
    cursor = connection.execute(query)
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _dataset_glob(dataset_dir: Path) -> str:
    return str(dataset_dir / "**" / "*.parquet")


def _format_timestamp(value: datetime) -> str:
    return value.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _csv_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _slugify(value: str) -> str:
    allowed = []
    for char in value.lower():
        if char.isalnum():
            allowed.append(char)
        elif char in {"-", "_"}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_")[:60] or "market"


def _sql_escape(value: str | Path) -> str:
    return str(value).replace("'", "''")
