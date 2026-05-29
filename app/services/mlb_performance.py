"""CLV, grading, and backtest reporting for MLB edges.

Includes the research-grade analytics suite used by the dashboard's
Performance / CLV tab — score-band segmentation, over/under split,
projection calibration, projection buckets, timing buckets, and
factor attribution. All new helpers are additive and treat missing
data (no closing line, no final score, etc.) as None rather than
crashing.
"""

from __future__ import annotations

from datetime import date as date_cls, datetime, timedelta
from statistics import mean, pstdev
from typing import Any, Callable, Iterable

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import MlbEdge, MlbFinalScore, MlbGame
from app.services.card_date import TZ_ARIZONA


# ---------------------------------------------------------------------------
# Timezone helpers — Arizona/MST is the source-of-truth date for grading.
# ---------------------------------------------------------------------------

def arizona_today() -> str:
    """Today's date in Arizona/MST (ISO YYYY-MM-DD)."""
    return datetime.now(TZ_ARIZONA).date().isoformat()


def arizona_yesterday() -> str:
    """Yesterday's date in Arizona/MST."""
    return (datetime.now(TZ_ARIZONA).date() - timedelta(days=1)).isoformat()


def arizona_window(days: int, *, today: str | None = None) -> tuple[str, str]:
    """Inclusive (start, end) ISO date window of `days` ending on `today` AZ."""
    end_iso = today or arizona_today()
    end = date_cls.fromisoformat(end_iso)
    start = end - timedelta(days=max(int(days) - 1, 0))
    return start.isoformat(), end.isoformat()


# ---------------------------------------------------------------------------
# Score-band segmentation. Five bands replace the previous coarse <65 bucket.
# A meaningful research sample is required before any band's ROI/win-rate
# should be acted on; sample_size_label() exposes that confidence tier.
# ---------------------------------------------------------------------------

# Display order is intentional: weak → high conviction.
SCORE_BANDS = ("<55", "55-64", "65-74", "75-84", "85+")
SCORE_BAND_LABELS = {
    "<55": "weak",
    "55-64": "watchlist only",
    "65-74": "playable / low conviction",
    "75-84": "strong candidate",
    "85+": "high conviction",
}

# Projection buckets are based on model_projected_total (or, when that's
# missing, the market entry total as a proxy). They expose whether the
# model's 10+ projections are systematically losing.
PROJECTION_BUCKETS = ("<7.5", "7.5-8.5", "8.5-9.5", "9.5-10.5", "10.5+")

# Timing buckets describe how far before first pitch the signal was emitted.
TIMING_BUCKETS = (">12h", "6-12h", "1-6h", "<1h", "after start / invalid")


def score_band(score: float | None) -> str:
    """Map a numeric score to a fixed band string. NaN/None → ``<55``."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "<55"
    if s < 55:
        return "<55"
    if s < 65:
        return "55-64"
    if s < 75:
        return "65-74"
    if s < 85:
        return "75-84"
    return "85+"


def projection_bucket(value: float | None) -> str | None:
    """Bucket a projected total into the five fixed bands. None → None."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v < 7.5:
        return "<7.5"
    if v < 8.5:
        return "7.5-8.5"
    if v < 9.5:
        return "8.5-9.5"
    if v < 10.5:
        return "9.5-10.5"
    return "10.5+"


def sample_size_label(n: int) -> dict[str, str]:
    """Confidence tier for a graded sample. Used to render the warning panel."""
    if n < 50:
        return {"tier": "exploratory", "label": "exploratory only"}
    if n < 200:
        return {"tier": "early", "label": "early signal"}
    if n < 500:
        return {"tier": "moderate", "label": "moderate confidence"}
    return {"tier": "strong", "label": "stronger research sample"}


def grade_edge(
    edge: MlbEdge,
    *,
    result: str | None = None,
    win_loss_push: str | None = None,
    closing_line: float | None = None,
    closing_price: float | None = None,
    current_line: float | None = None,
    opening_line: float | None = None,
    actual_total: float | None = None,
    model_projected_total: float | None = None,
) -> MlbEdge:
    if opening_line is not None:
        edge.opening_line = opening_line
    if current_line is not None:
        edge.current_line = current_line
    if closing_line is not None:
        edge.closing_line = closing_line
    if closing_price is not None:
        edge.closing_price = closing_price
    if result is not None:
        edge.result = result
    if win_loss_push is not None:
        edge.win_loss_push = win_loss_push.lower()
    if actual_total is not None:
        edge.actual_total = float(actual_total)
    if model_projected_total is not None:
        edge.model_projected_total = float(model_projected_total)

    edge.recommended_line = edge.recommended_line if edge.recommended_line is not None else edge.line
    edge.implied_probability_at_entry = implied_probability(edge.best_price)
    edge.implied_probability_at_close = implied_probability(edge.closing_price)
    edge.clv_points = clv_points(edge)
    edge.clv_percent = clv_percent(edge)
    edge.roi_units = roi_units(edge.win_loss_push, edge.best_price)
    edge.graded_at = datetime.utcnow()
    return edge


def update_closing_line_fields(
    edge: MlbEdge,
    *,
    closing_line: float | None = None,
    closing_price: float | None = None,
) -> MlbEdge:
    if closing_line is not None:
        edge.closing_line = closing_line
    if closing_price is not None:
        edge.closing_price = closing_price
        edge.implied_probability_at_close = implied_probability(edge.closing_price)
    if closing_line is not None or closing_price is not None:
        edge.closing_snapshot_at = datetime.utcnow()
    edge.clv_points = clv_points(edge)
    edge.clv_percent = clv_percent(edge)
    return edge


def performance_summary(
    db: Session,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    edges = _graded_edges(db, start_date=start_date, end_date=end_date)
    wins = sum(1 for e in edges if e.win_loss_push == "win")
    losses = sum(1 for e in edges if e.win_loss_push == "loss")
    pushes = sum(1 for e in edges if e.win_loss_push == "push")
    decided = wins + losses
    summary: dict[str, Any] = {
        "graded_edges": len(edges),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": round(wins / decided, 4) if decided else None,
        "roi_units": round(sum(e.roi_units or 0.0 for e in edges), 4),
        "average_clv_points": _avg(e.clv_points for e in edges),
        "average_clv_percent": _avg(e.clv_percent for e in edges),
        "average_prediction_score": _avg(e.prediction_score for e in edges),
        "average_execution_score": _avg(e.execution_score for e in edges),
        "average_legacy_score": _avg(
            e.legacy_score if e.legacy_score is not None else e.score for e in edges
        ),
    }
    if start_date or end_date:
        summary["start_date"] = start_date
        summary["end_date"] = end_date
    return summary


def performance_by_market(
    db: Session,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    return _grouped_report(
        _graded_edges(db, start_date=start_date, end_date=end_date),
        lambda e: e.edge_type,
        "edge_type",
    )


def performance_by_score_band(
    db: Session,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    """Per-band rollup. Bands with zero graded edges are still emitted so the
    dashboard can show an honest empty cell instead of silently hiding them.
    """
    edges = _graded_edges(db, start_date=start_date, end_date=end_date)
    groups: dict[str, list[MlbEdge]] = {band: [] for band in SCORE_BANDS}
    for edge in edges:
        groups[score_band(edge.score)].append(edge)
    rows: list[dict[str, Any]] = []
    for band in SCORE_BANDS:
        items = groups[band]
        overs = [e for e in items if (e.side or "").lower() == "over"]
        unders = [e for e in items if (e.side or "").lower() == "under"]
        row = {
            "score_band": band,
            "band_label": SCORE_BAND_LABELS[band],
            **_summary_for_edges(items),
            "over_count": len(overs),
            "under_count": len(unders),
            "over_win_rate": _win_rate(overs),
            "under_win_rate": _win_rate(unders),
        }
        # Honesty signal: < 30 graded edges in a band → don't trust the ROI.
        row["stable"] = row["graded_edges"] >= 30
        rows.append(row)
    return rows


def performance_by_score_axis(
    db: Session,
    *,
    axis: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    """Per-band rollup for one score axis.

    ``axis`` may be ``legacy``, ``prediction``, or ``execution``. Legacy
    preserves the old single-score report; prediction/execution let the
    dashboard evaluate model correctness and price execution separately.
    """
    axis = _normalize_score_axis(axis)
    edges = _graded_edges(db, start_date=start_date, end_date=end_date)
    groups: dict[str, list[MlbEdge]] = {band: [] for band in SCORE_BANDS}
    for edge in edges:
        groups[score_band(_score_for_axis(edge, axis))].append(edge)

    rows: list[dict[str, Any]] = []
    for band in SCORE_BANDS:
        items = groups[band]
        row = {
            "score_axis": axis,
            "score_band": band,
            "band_label": SCORE_BAND_LABELS[band],
            **_summary_for_edges(items),
            "average_axis_score": _avg(_score_for_axis(e, axis) for e in items),
            "missing_axis_score_count": sum(
                1 for e in items if _score_for_axis(e, axis) is None
            ),
        }
        row["stable"] = row["graded_edges"] >= 30
        rows.append(row)
    return rows


def lookup_edge_score_band(
    db: Session,
    *,
    edge_type: str | None,
    score: float | None,
    score_axis: str = "legacy",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Historical record of *similar* graded edges — same ``edge_type`` and
    score band — for the card's history + calibration sections.

    Returns win_rate / roi / avg_clv / sample_size plus a Laplace-smoothed
    ``calibrated_probability`` (MLB-edge-native analog of the Falcon-signal
    ``lookup_calibrated_probability``). All fields are ``None`` / 0 when no
    comparable graded edges exist yet, so the card can show an honest
    "not enough graded history" state rather than a fabricated number.
    """
    score_axis = _normalize_score_axis(score_axis)
    band = score_band(float(score)) if score is not None else None
    empty = {
        "edge_type": edge_type,
        "score_axis": score_axis,
        "score_band": band,
        "sample_size": 0,
        "wins": 0,
        "losses": 0,
        "pushes": 0,
        "win_rate": None,
        "roi_units": None,
        "avg_clv_points": None,
        "calibrated_probability": None,
    }
    if band is None:
        return empty
    edges = [
        e
        for e in _graded_edges(db, start_date=start_date, end_date=end_date)
        if (edge_type is None or e.edge_type == edge_type)
        and score_band(_score_for_axis(e, score_axis)) == band
    ]
    if not edges:
        return empty
    wins = sum(1 for e in edges if e.win_loss_push == "win")
    losses = sum(1 for e in edges if e.win_loss_push == "loss")
    pushes = sum(1 for e in edges if e.win_loss_push == "push")
    decided = wins + losses
    return {
        "edge_type": edge_type,
        "score_axis": score_axis,
        "score_band": band,
        "sample_size": len(edges),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": round(wins / decided, 4) if decided else None,
        "roi_units": round(sum(e.roi_units or 0.0 for e in edges) / len(edges), 4),
        "avg_clv_points": _avg(e.clv_points for e in edges),
        # Laplace smoothing keeps small samples honest.
        "calibrated_probability": round((wins + 1) / (decided + 2), 4) if decided else None,
    }


def clv_report(
    db: Session,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    edges = _graded_edges(db, start_date=start_date, end_date=end_date)
    with_clv = [e for e in edges if e.clv_points is not None]
    positive = [e for e in with_clv if (e.clv_points or 0.0) > 0]
    negative = [e for e in with_clv if (e.clv_points or 0.0) < 0]
    by_side = {
        "over": _clv_breakdown([e for e in edges if (e.side or "").lower() == "over"]),
        "under": _clv_breakdown([e for e in edges if (e.side or "").lower() == "under"]),
    }
    by_band = {
        band: _clv_breakdown([e for e in edges if score_band(e.score) == band])
        for band in SCORE_BANDS
    }
    by_prediction_band = {
        band: _clv_breakdown([
            e for e in edges if score_band(e.prediction_score) == band
        ])
        for band in SCORE_BANDS
    }
    by_execution_band = {
        band: _clv_breakdown([
            e for e in edges if score_band(e.execution_score) == band
        ])
        for band in SCORE_BANDS
    }
    by_edge_type: dict[str, dict[str, Any]] = {}
    for edge in edges:
        by_edge_type.setdefault(edge.edge_type or "unknown", []).append(edge)  # type: ignore[arg-type]
    by_edge_type = {k: _clv_breakdown(v) for k, v in by_edge_type.items()}  # type: ignore[arg-type]
    return {
        "edges_with_clv": len(with_clv),
        "average_clv_points": _avg(e.clv_points for e in edges),
        "average_clv_percent": _avg(e.clv_percent for e in edges),
        # Positive-CLV rate is computed only over edges where CLV could be
        # calculated; otherwise a missing closing line would silently drag
        # the rate toward zero.
        "positive_clv_rate": round(len(positive) / len(with_clv), 4) if with_clv else None,
        "negative_clv_rate": round(len(negative) / len(with_clv), 4) if with_clv else None,
        "missing_clv_count": len(edges) - len(with_clv),
        "by_side": by_side,
        "by_score_band": by_band,
        "by_prediction_score_band": by_prediction_band,
        "by_execution_score_band": by_execution_band,
        "by_edge_type": by_edge_type,
        "top_positive": [_edge_perf(e) for e in sorted(with_clv, key=lambda e: e.clv_points or -999, reverse=True)[:10]],
        "top_negative": [_edge_perf(e) for e in sorted(with_clv, key=lambda e: e.clv_points or 999)[:10]],
    }


def research_health(
    db: Session,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Headline research-health panel. CLV-first ordering by design — ROI and
    win rate sit at the bottom because they're high-variance lagging metrics.
    """
    edges = _graded_edges(db, start_date=start_date, end_date=end_date)
    with_clv = [e for e in edges if e.clv_points is not None]
    positive = [e for e in with_clv if (e.clv_points or 0.0) > 0]
    decided = [e for e in edges if e.win_loss_push in {"win", "loss"}]
    wins = sum(1 for e in decided if e.win_loss_push == "win")
    sample = sample_size_label(len(edges))
    return {
        "positive_clv_rate": round(len(positive) / len(with_clv), 4) if with_clv else None,
        "average_clv_points": _avg(e.clv_points for e in edges),
        "average_clv_percent": _avg(e.clv_percent for e in edges),
        "roi_units": round(sum(e.roi_units or 0.0 for e in edges), 4),
        "win_rate": round(wins / len(decided), 4) if decided else None,
        "graded_sample_size": len(edges),
        "edges_with_clv": len(with_clv),
        "sample_size_tier": sample["tier"],
        "sample_size_label": sample["label"],
    }


def performance_by_side(
    db: Session,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    edge_type: str = "game_total",
) -> dict[str, Any]:
    """Over vs Under split. Defaults to game_total since side is only
    meaningful there. Emits a directional-bias warning if one side dominates
    candidate generation (>65%).
    """
    edges = [
        e
        for e in _graded_edges(db, start_date=start_date, end_date=end_date)
        if (edge_type is None or e.edge_type == edge_type)
    ]
    overs = [e for e in edges if (e.side or "").lower() == "over"]
    unders = [e for e in edges if (e.side or "").lower() == "under"]
    total = len(overs) + len(unders)
    bias_warning: str | None = None
    if total >= 10:
        over_share = len(overs) / total if total else 0.0
        if over_share > 0.65:
            bias_warning = "Model may be directionally biased toward over."
        elif over_share < 0.35:
            bias_warning = "Model may be directionally biased toward under."
    return {
        "edge_type": edge_type,
        "over": _side_metrics(overs),
        "under": _side_metrics(unders),
        "total_graded": total,
        "over_share": round(len(overs) / total, 4) if total else None,
        "under_share": round(len(unders) / total, 4) if total else None,
        "directional_bias_warning": bias_warning,
    }


def projection_calibration(
    db: Session,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Diagnose whether the model is systematically inflating game totals.

    For each graded game_total edge we report avg model projection vs
    avg market entry, avg closing line, and avg actual total. Warnings
    fire on absolute miss > 0.75 runs in either direction.
    """
    edges = [
        e
        for e in _graded_edges(db, start_date=start_date, end_date=end_date)
        if e.edge_type == "game_total"
    ]
    # Build per-edge calibration rows; missing values just don't contribute
    # to the average. We never silently substitute defaults.
    rows: list[dict[str, Any]] = []
    actuals = _actual_totals_by_game(db)
    for e in edges:
        proj = e.model_projected_total
        entry = e.recommended_line if e.recommended_line is not None else e.line
        close = e.closing_line
        actual = e.actual_total if e.actual_total is not None else actuals.get(e.game_pk)
        rows.append(
            {
                "id": e.id,
                "game_pk": e.game_pk,
                "side": e.side,
                "score": e.score,
                "prediction_score": e.prediction_score,
                "execution_score": e.execution_score,
                "legacy_score": e.legacy_score if e.legacy_score is not None else e.score,
                "model_projected_total": proj,
                "market_entry_total": entry,
                "closing_total": close,
                "actual_total": actual,
                "model_vs_entry": _delta(proj, entry),
                "model_vs_close": _delta(proj, close),
                "projection_error": _delta(proj, actual),
                "absolute_projection_error": (
                    abs(_delta(proj, actual)) if _delta(proj, actual) is not None else None
                ),
            }
        )
    avg_proj = _avg(r["model_projected_total"] for r in rows)
    avg_entry = _avg(r["market_entry_total"] for r in rows)
    avg_close = _avg(r["closing_total"] for r in rows)
    avg_actual = _avg(r["actual_total"] for r in rows)
    avg_err = _avg(r["projection_error"] for r in rows)
    avg_abs_err = _avg(r["absolute_projection_error"] for r in rows)
    warnings: list[str] = []
    if avg_proj is not None and avg_close is not None and (avg_proj - avg_close) > 0.75:
        warnings.append("Model totals may be inflated versus market close.")
    if avg_err is not None and avg_err > 0.75:
        warnings.append("Model is over-projecting actual runs.")
    if avg_err is not None and avg_err < -0.75:
        warnings.append("Model is under-projecting actual runs.")
    return {
        "graded_game_totals": len(rows),
        "rows_with_projection": sum(1 for r in rows if r["model_projected_total"] is not None),
        "rows_with_actual": sum(1 for r in rows if r["actual_total"] is not None),
        "avg_model_projected_total": avg_proj,
        "avg_market_entry_total": avg_entry,
        "avg_closing_total": avg_close,
        "avg_actual_total": avg_actual,
        "avg_projection_error": avg_err,
        "avg_absolute_projection_error": avg_abs_err,
        "warnings": warnings,
    }


def performance_by_projection_bucket(
    db: Session,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    """Bucket graded game_total edges by projected total. When the model
    didn't store a projection we fall back to the market entry line so the
    bucket still has signal — flagged via ``projection_source`` per row.
    """
    edges = [
        e
        for e in _graded_edges(db, start_date=start_date, end_date=end_date)
        if e.edge_type == "game_total"
    ]
    actuals = _actual_totals_by_game(db)
    buckets: dict[str, list[tuple[MlbEdge, float | None, float | None, str]]] = {
        b: [] for b in PROJECTION_BUCKETS
    }
    for e in edges:
        proj = e.model_projected_total
        source = "model"
        if proj is None:
            proj = e.recommended_line if e.recommended_line is not None else e.line
            source = "market_entry"
        bucket = projection_bucket(proj)
        if bucket is None:
            continue
        actual = e.actual_total if e.actual_total is not None else actuals.get(e.game_pk)
        err = _delta(proj, actual)
        buckets[bucket].append((e, err, actual, source))
    rows: list[dict[str, Any]] = []
    for bucket in PROJECTION_BUCKETS:
        items = buckets[bucket]
        edges_only = [t[0] for t in items]
        errors = [t[1] for t in items if t[1] is not None]
        overs = sum(1 for e in edges_only if (e.side or "").lower() == "over")
        unders = sum(1 for e in edges_only if (e.side or "").lower() == "under")
        rows.append(
            {
                "projection_bucket": bucket,
                **_summary_for_edges(edges_only),
                "avg_projection_error": _avg(errors) if errors else None,
                "avg_absolute_projection_error": (
                    _avg(abs(x) for x in errors) if errors else None
                ),
                "over_count": overs,
                "under_count": unders,
                "projection_source": (
                    "model" if all(t[3] == "model" for t in items)
                    else "mixed" if items
                    else None
                ),
            }
        )
    return rows


def performance_by_timing(
    db: Session,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    """Bucket edges by how far before first pitch the signal was created.

    Requires ``MlbGame.start_time`` to be set; rows missing it land in the
    ``after start / invalid`` bucket so they're still counted, not dropped.
    """
    rows = _graded_edges_with_games(db, start_date=start_date, end_date=end_date)
    buckets: dict[str, list[MlbEdge]] = {b: [] for b in TIMING_BUCKETS}
    for edge, game in rows:
        start = game.start_time if game is not None else None
        created = edge.created_at
        bucket = _timing_bucket(created, start)
        buckets[bucket].append(edge)
    out: list[dict[str, Any]] = []
    for bucket in TIMING_BUCKETS:
        items = buckets[bucket]
        out.append({"timing_bucket": bucket, **_summary_for_edges(items)})
    return out


def factor_attribution(
    db: Session,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    side: str | None = None,
) -> list[dict[str, Any]]:
    """Per-factor win/loss split + correlation with CLV and ROI.

    Factors with sample <50 are flagged ``unstable=True`` so the dashboard
    can render them in a muted style instead of implying they're actionable.
    """
    edges = _graded_edges(db, start_date=start_date, end_date=end_date)
    if side is not None:
        side_l = side.lower()
        edges = [e for e in edges if (e.side or "").lower() == side_l]
    aggregated: dict[str, dict[str, list[float]]] = {}
    for e in edges:
        for factor, value in (e.factors or {}).items():
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            bucket = aggregated.setdefault(
                factor,
                {"wins": [], "losses": [], "all": [], "clv": [], "roi": []},
            )
            outcome = (e.win_loss_push or "").lower()
            if outcome == "win":
                bucket["wins"].append(v)
            elif outcome == "loss":
                bucket["losses"].append(v)
            bucket["all"].append(v)
            if e.clv_points is not None:
                bucket["clv"].append((v, float(e.clv_points)))  # type: ignore[arg-type]
            if e.roi_units is not None:
                bucket["roi"].append((v, float(e.roi_units)))  # type: ignore[arg-type]
    out: list[dict[str, Any]] = []
    for factor, vals in aggregated.items():
        wins, losses, all_vals = vals["wins"], vals["losses"], vals["all"]
        avg_win = _avg(wins) if wins else None
        avg_loss = _avg(losses) if losses else None
        delta = (
            round(avg_win - avg_loss, 4)
            if avg_win is not None and avg_loss is not None
            else None
        )
        out.append(
            {
                "factor": factor,
                "sample_size": len(all_vals),
                "avg_score_on_wins": avg_win,
                "avg_score_on_losses": avg_loss,
                "delta_win_minus_loss": delta,
                "correlation_with_clv": _pearson(vals["clv"]),  # type: ignore[arg-type]
                "correlation_with_roi": _pearson(vals["roi"]),  # type: ignore[arg-type]
                "unstable": len(all_vals) < 50,
                "side": side,
            }
        )
    # Sort by absolute correlation with CLV so the most predictive factors
    # for line movement bubble to the top.
    out.sort(
        key=lambda r: abs(r["correlation_with_clv"] or 0.0),
        reverse=True,
    )
    return out


# ---------------------------------------------------------------------------
# Factor distribution audit
#
# Diagnoses whether a factor actually carries signal or is silently parked
# at the neutral 50.0 sentinel because its upstream producer is a stub.
# Surfaces variance, the % of values stuck at 50, and a no-information flag
# (low variance + weak CLV correlation) so the dashboard can show the
# operator which weight slices are dead weight.
# ---------------------------------------------------------------------------

# A factor sample within ±NEUTRAL_TOLERANCE of 50.0 counts as "stuck at
# neutral" — small numeric drift from a stub producer that *adds* a tiny
# weather or book-count adjustment should still register as stuck.
NEUTRAL_TOLERANCE = 0.5
# Population stdev below this is treated as "no spread" — the producer is
# returning effectively the same value for every edge.
LOW_VARIANCE_STDEV = 2.5
# Min |Pearson(value, clv)| to consider a factor informative when its
# variance does carry some spread. Below this AND with low variance →
# no_information.
MIN_INFORMATIVE_CORRELATION = 0.05
# Sample-size floor for the no_information flag — small samples can look
# uninformative purely from variance, so we only call a factor dead weight
# once we have enough graded edges to trust the verdict.
MIN_NO_INFO_SAMPLE = 30


def factor_distribution(
    db: Session,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    edge_type: str | None = None,
) -> dict[str, Any]:
    """Audit per-factor distributions against graded edges in the window.

    For each factor that appears in any edge's ``factors`` dict we report
    the spread (mean / stdev / variance / min / max / p10/p50/p90), the
    rate of samples parked at the neutral 50 sentinel, and a
    ``no_information`` flag set when both the variance is near-zero *and*
    the factor's value carries no detectable CLV/ROI signal. The output is
    ordered by score impact (avg |contribution|) so dead-weight factors
    bubble to the top of the operator's attention.
    """
    edges = _graded_edges(db, start_date=start_date, end_date=end_date)
    if edge_type is not None:
        edges = [e for e in edges if e.edge_type == edge_type]

    # Gather raw values + CLV/ROI pairs per factor in a single pass.
    aggregated: dict[str, dict[str, Any]] = {}
    for edge in edges:
        contributions = edge.score_contributions or {}
        for factor, value in (edge.factors or {}).items():
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            bucket = aggregated.setdefault(
                factor,
                {
                    "values": [],
                    "contributions": [],
                    "clv_pairs": [],
                    "roi_pairs": [],
                },
            )
            bucket["values"].append(v)
            try:
                contribution = float(contributions.get(factor, 0.0))
            except (TypeError, ValueError):
                contribution = 0.0
            bucket["contributions"].append(contribution)
            if edge.clv_points is not None:
                bucket["clv_pairs"].append((v, float(edge.clv_points)))
            if edge.roi_units is not None:
                bucket["roi_pairs"].append((v, float(edge.roi_units)))

    if not aggregated:
        return {
            "edge_type": edge_type,
            "graded_sample_size": len(edges),
            "factors": [],
            "summary": {
                "stuck_at_neutral_factors": [],
                "no_information_factors": [],
            },
        }

    # Compute totals so we can express each factor's mean |contribution|
    # as a share of the score's overall absolute movement.
    total_abs_contribution = sum(
        abs(c) for bucket in aggregated.values() for c in bucket["contributions"]
    )
    rows: list[dict[str, Any]] = []
    for factor, bucket in aggregated.items():
        values: list[float] = bucket["values"]
        contributions: list[float] = bucket["contributions"]
        n = len(values)
        mean_v = mean(values) if values else None
        stdev_v = pstdev(values) if n > 1 else 0.0
        stuck = sum(1 for v in values if abs(v - 50.0) <= NEUTRAL_TOLERANCE)
        stuck_rate = round(stuck / n, 4) if n else None
        sorted_vals = sorted(values)
        avg_abs_contrib = mean(abs(c) for c in contributions) if contributions else 0.0
        contribution_share = (
            round(sum(abs(c) for c in contributions) / total_abs_contribution, 4)
            if total_abs_contribution
            else None
        )
        corr_clv = _pearson(bucket["clv_pairs"])
        corr_roi = _pearson(bucket["roi_pairs"])
        low_variance = stdev_v < LOW_VARIANCE_STDEV
        uninformative_corr = (
            corr_clv is None
            or abs(corr_clv) < MIN_INFORMATIVE_CORRELATION
        )
        no_information = (
            n >= MIN_NO_INFO_SAMPLE and low_variance and uninformative_corr
        )
        rows.append(
            {
                "factor": factor,
                "sample_size": n,
                "mean": round(mean_v, 4) if mean_v is not None else None,
                "stdev": round(stdev_v, 4),
                "variance": round(stdev_v * stdev_v, 4),
                "min": round(min(sorted_vals), 4) if sorted_vals else None,
                "max": round(max(sorted_vals), 4) if sorted_vals else None,
                "p10": round(_percentile(sorted_vals, 0.10), 4) if sorted_vals else None,
                "p50": round(_percentile(sorted_vals, 0.50), 4) if sorted_vals else None,
                "p90": round(_percentile(sorted_vals, 0.90), 4) if sorted_vals else None,
                "stuck_at_neutral_rate": stuck_rate,
                "stuck_at_neutral_count": stuck,
                "low_variance": low_variance,
                "correlation_with_clv": corr_clv,
                "correlation_with_roi": corr_roi,
                "avg_abs_contribution_points": round(avg_abs_contrib, 4),
                "contribution_share": contribution_share,
                "no_information": no_information,
            }
        )

    # Surface the worst offenders separately so the dashboard / export can
    # render a one-line "these factors are dead weight" callout without
    # forcing the operator to scan the full table.
    stuck_offenders = sorted(
        [r for r in rows if (r["stuck_at_neutral_rate"] or 0) >= 0.95],
        key=lambda r: r["stuck_at_neutral_rate"] or 0.0,
        reverse=True,
    )
    no_info_offenders = sorted(
        [r for r in rows if r["no_information"]],
        key=lambda r: r["avg_abs_contribution_points"],
        reverse=True,
    )
    rows.sort(key=lambda r: r["avg_abs_contribution_points"], reverse=True)
    return {
        "edge_type": edge_type,
        "graded_sample_size": len(edges),
        "factors": rows,
        "summary": {
            "stuck_at_neutral_factors": [
                {"factor": r["factor"], "rate": r["stuck_at_neutral_rate"]}
                for r in stuck_offenders
            ],
            "no_information_factors": [
                {"factor": r["factor"], "stdev": r["stdev"], "corr_clv": r["correlation_with_clv"]}
                for r in no_info_offenders
            ],
        },
    }


def score_attribution(
    db: Session,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    edge_type: str | None = None,
) -> dict[str, Any]:
    """Per-factor score attribution: how many points each factor actually
    moves the final score on average, and how that movement correlates
    with downstream outcomes.

    Unlike ``factor_attribution`` (which works on factor *values*), this
    operates on the additive *contributions* — i.e. weight × (value − 50).
    A factor with high variance but a tiny weight will show as low
    impact here; a factor with modest variance but a fat weight will
    show as high impact. That's what the operator needs to understand
    "where is my score actually coming from."
    """
    edges = _graded_edges(db, start_date=start_date, end_date=end_date)
    if edge_type is not None:
        edges = [e for e in edges if e.edge_type == edge_type]

    aggregated: dict[str, dict[str, list[float]]] = {}
    for edge in edges:
        contributions = edge.score_contributions or {}
        for factor, contrib in contributions.items():
            try:
                c = float(contrib)
            except (TypeError, ValueError):
                continue
            bucket = aggregated.setdefault(
                factor,
                {
                    "all": [],
                    "wins": [],
                    "losses": [],
                    "clv_pairs": [],
                    "roi_pairs": [],
                },
            )
            bucket["all"].append(c)
            outcome = (edge.win_loss_push or "").lower()
            if outcome == "win":
                bucket["wins"].append(c)
            elif outcome == "loss":
                bucket["losses"].append(c)
            if edge.clv_points is not None:
                bucket["clv_pairs"].append((c, float(edge.clv_points)))  # type: ignore[arg-type]
            if edge.roi_units is not None:
                bucket["roi_pairs"].append((c, float(edge.roi_units)))  # type: ignore[arg-type]

    total_abs = sum(abs(v) for bucket in aggregated.values() for v in bucket["all"])
    rows: list[dict[str, Any]] = []
    for factor, vals in aggregated.items():
        all_vals: list[float] = vals["all"]
        n = len(all_vals)
        avg = mean(all_vals) if all_vals else 0.0
        abs_sum = sum(abs(v) for v in all_vals)
        avg_abs = abs_sum / n if n else 0.0
        sd = pstdev(all_vals) if n > 1 else 0.0
        share = round(abs_sum / total_abs, 4) if total_abs else None
        avg_win = mean(vals["wins"]) if vals["wins"] else None
        avg_loss = mean(vals["losses"]) if vals["losses"] else None
        delta = (
            round(avg_win - avg_loss, 4)
            if avg_win is not None and avg_loss is not None
            else None
        )
        rows.append(
            {
                "factor": factor,
                "sample_size": n,
                "avg_contribution_points": round(avg, 4),
                "avg_abs_contribution_points": round(avg_abs, 4),
                "stdev_contribution": round(sd, 4),
                "contribution_share": share,
                "avg_contribution_on_wins": (
                    round(avg_win, 4) if avg_win is not None else None
                ),
                "avg_contribution_on_losses": (
                    round(avg_loss, 4) if avg_loss is not None else None
                ),
                "delta_win_minus_loss": delta,
                "correlation_with_clv": _pearson(vals["clv_pairs"]),  # type: ignore[arg-type]
                "correlation_with_roi": _pearson(vals["roi_pairs"]),  # type: ignore[arg-type]
                "unstable": n < 50,
            }
        )
    rows.sort(key=lambda r: r["contribution_share"] or 0.0, reverse=True)
    return {
        "edge_type": edge_type,
        "graded_sample_size": len(edges),
        "total_absolute_contribution_points": round(total_abs, 4),
        "factors": rows,
    }


# Underperforming-side rolling window: how many days back the scoring
# engine averages over when deciding whether to penalize a side at scan
# time. Long enough to avoid pure noise, short enough to react when the
# model's directional bias breaks.
SIDE_PERF_ROLLING_DAYS = 14
# A side needs at least this many graded edges before its underperformance
# can drive a penalty. Without this floor a 0/2 cold streak would slash
# every over-recommendation.
SIDE_PERF_MIN_SAMPLE = 20
# Win-rate threshold below which a side is considered "underperforming".
# 0.45 sits just under coin-flip — we don't want to penalize a side that
# is merely variance-noisy around 50%.
SIDE_PERF_LOSING_WIN_RATE = 0.45
# Max points we'll subtract from a 0-100 score for the worst possible
# side underperformance. Bounded so the penalty can't single-handedly
# flip a high-conviction candidate into Pass.
SIDE_PERF_MAX_PENALTY_POINTS = 6.0


def recent_side_performance(
    db: Session,
    *,
    today: str | None = None,
    days: int = SIDE_PERF_ROLLING_DAYS,
    edge_type: str = "game_total",
) -> dict[str, Any]:
    """Rolling per-side performance used by the scoring engine to penalize
    edges on a recently-losing side. Pure read — no scoring side effects.

    Returns the per-side ``win_rate``, ``roi_units`` and ``sample_size``
    over the lookback window, plus a derived ``penalty_points`` that the
    engine can subtract from the candidate's final score. ``penalty_points``
    is ``0.0`` for any side with too small a sample or a win rate above
    ``SIDE_PERF_LOSING_WIN_RATE``.
    """
    start, end = arizona_window(days, today=today)
    edges = [
        e
        for e in _graded_edges(db, start_date=start, end_date=end)
        if e.edge_type == edge_type
    ]
    out: dict[str, Any] = {
        "edge_type": edge_type,
        "lookback_days": days,
        "window_start": start,
        "window_end": end,
        "sides": {},
    }
    for side in ("over", "under"):
        side_edges = [e for e in edges if (e.side or "").lower() == side]
        wins = sum(1 for e in side_edges if e.win_loss_push == "win")
        losses = sum(1 for e in side_edges if e.win_loss_push == "loss")
        decided = wins + losses
        win_rate = wins / decided if decided else None
        roi = round(sum(e.roi_units or 0.0 for e in side_edges), 4)
        # Penalty is proportional to how far below the losing threshold
        # the rolling win rate has fallen, scaled to the configured cap.
        if (
            win_rate is not None
            and decided >= SIDE_PERF_MIN_SAMPLE
            and win_rate < SIDE_PERF_LOSING_WIN_RATE
        ):
            shortfall = SIDE_PERF_LOSING_WIN_RATE - win_rate
            penalty = min(
                SIDE_PERF_MAX_PENALTY_POINTS,
                shortfall * (SIDE_PERF_MAX_PENALTY_POINTS / SIDE_PERF_LOSING_WIN_RATE),
            )
            penalty = round(penalty, 2)
        else:
            penalty = 0.0
        out["sides"][side] = {
            "sample_size": len(side_edges),
            "decided": decided,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 4) if win_rate is not None else None,
            "roi_units": roi,
            "penalty_points": penalty,
        }
    return out


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolated percentile. Caller guarantees non-empty + sorted."""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def top_factors_by_performance(
    db: Session,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    edges = _graded_edges(db, start_date=start_date, end_date=end_date)
    values: dict[str, list[float]] = {}
    for edge in edges:
        for factor, value in (edge.factors or {}).items():
            try:
                values.setdefault(factor, []).append((edge.roi_units or 0.0) * float(value) / 100)
            except (TypeError, ValueError):
                continue
    return [
        {"factor": factor, "performance_score": round(mean(vals), 4), "sample": len(vals)}
        for factor, vals in sorted(values.items(), key=lambda item: mean(item[1]), reverse=True)
    ]


def performance_diagnostics(
    db: Session,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Lightweight visibility counts for the dashboard debug panel.

    Tells the user *why* the Performance tab is empty: missing edge snapshots,
    missing closing lines, no final scores, or no graded results yet.
    """

    snapshot_q = select(func.count()).select_from(MlbEdge).where(MlbEdge.is_valid.is_(True))
    live_final_q = select(func.count()).select_from(MlbGame).where(
        func.lower(func.coalesce(MlbGame.game_status, "")).like("%final%")
    )
    persisted_final_q = select(func.count()).select_from(MlbFinalScore)
    closing_q = select(func.count()).select_from(MlbEdge).where(MlbEdge.closing_line.is_not(None))
    graded_q = (
        select(func.count())
        .select_from(MlbEdge)
        .where(MlbEdge.win_loss_push.is_not(None))
        .where(MlbEdge.is_valid.is_(True))
    )
    last_graded_q = select(func.max(MlbEdge.graded_at))

    if start_date or end_date:
        snapshot_q = _apply_edge_date_window(snapshot_q, start_date, end_date)
        graded_q = _apply_edge_date_window(graded_q, start_date, end_date)
        last_graded_q = _apply_edge_date_window(last_graded_q, start_date, end_date)
        closing_q = _apply_edge_date_window(closing_q, start_date, end_date)
        if start_date:
            live_final_q = live_final_q.where(MlbGame.game_date >= start_date)
            persisted_final_q = persisted_final_q.where(MlbFinalScore.generated_for_date >= start_date)
        if end_date:
            live_final_q = live_final_q.where(MlbGame.game_date <= end_date)
            persisted_final_q = persisted_final_q.where(MlbFinalScore.generated_for_date <= end_date)

    last_graded_at = db.execute(last_graded_q).scalar()
    persisted_finals = int(db.execute(persisted_final_q).scalar() or 0)
    live_finals = int(db.execute(live_final_q).scalar() or 0)
    diagnostics: dict[str, Any] = {
        "start_date": start_date,
        "end_date": end_date,
        "snapshot_count": int(db.execute(snapshot_q).scalar() or 0),
        # ``final_score_count`` is the union view the dashboard treats as
        # "do we have any finals to grade from?" — persisted rows count
        # toward this because they're the source of truth for grading.
        "final_score_count": max(persisted_finals, live_finals),
        "persisted_final_score_count": persisted_finals,
        "live_final_count": live_finals,
        "closing_line_count": int(db.execute(closing_q).scalar() or 0),
        "graded_edge_count": int(db.execute(graded_q).scalar() or 0),
        "last_graded_at": last_graded_at.isoformat() if last_graded_at else None,
    }
    reasons: list[str] = []
    if diagnostics["snapshot_count"] == 0:
        label = f" for {start_date}" if start_date and start_date == end_date else ""
        reasons.append(
            f"No saved edge snapshots{label}. Run an MLB edge scan during game day to enable grading."
        )
    elif diagnostics["final_score_count"] == 0:
        reasons.append("Edge snapshots exist but no final game scores have been ingested yet.")
    elif diagnostics["graded_edge_count"] == 0:
        reasons.append(
            "Final scores and snapshots exist but no edges have been graded yet — click "
            "'Grade MLB results'."
        )
    diagnostics["reason"] = reasons[0] if reasons else None
    return diagnostics


def implied_probability(price: Any) -> float | None:
    try:
        decimal_price = float(price)
    except (TypeError, ValueError):
        return None
    if decimal_price <= 1:
        return None
    return round(1 / decimal_price, 4)


def clv_points(edge: MlbEdge) -> float | None:
    """Directional CLV in line points.

    For overs: positive when the line moved up (market climbed past entry).
    For unders: positive when the line moved down (market dropped past entry).
    """
    if edge.recommended_line is None or edge.closing_line is None:
        return None
    side = (edge.side or "").lower()
    direction = 1 if side == "over" else -1
    return round((float(edge.closing_line) - float(edge.recommended_line)) * direction, 4)


def clv_percent(edge: MlbEdge) -> float | None:
    """CLV expressed as a fraction of entry total.

    Defined as ``clv_points / entry_total`` per the research spec — line-based
    rather than probability-based. Falls back to the probability shift when
    no line CLV is available so price-only markets still get a CLV signal.
    """
    pts = clv_points(edge)
    entry = edge.recommended_line if edge.recommended_line is not None else edge.line
    if pts is not None and entry not in (None, 0):
        return round(pts / float(entry), 4)
    # Price-only fallback: probability shift between entry and close.
    entry_prob = edge.implied_probability_at_entry
    close_prob = edge.implied_probability_at_close
    if entry_prob is None or close_prob is None or entry_prob == 0:
        return None
    return round((close_prob - entry_prob) / entry_prob, 4)


def roi_units(win_loss_push: str | None, price: Any) -> float | None:
    outcome = (win_loss_push or "").lower()
    if outcome == "push":
        return 0.0
    if outcome == "loss":
        return -1.0
    if outcome != "win":
        return None
    try:
        decimal_price = float(price)
    except (TypeError, ValueError):
        return 1.0
    return round(max(decimal_price - 1, 0.0), 4)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _graded_edges(
    db: Session,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[MlbEdge]:
    query = (
        select(MlbEdge)
        .where(MlbEdge.win_loss_push.is_not(None))
        .where(MlbEdge.is_valid.is_(True))
    )
    query = _apply_edge_date_window(query, start_date, end_date)
    return list(db.scalars(query))


def _graded_edges_with_games(
    db: Session,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[tuple[MlbEdge, MlbGame | None]]:
    """Join graded edges with their MlbGame row (None when missing).

    Uses an outer join so an orphaned edge still appears in timing buckets
    rather than silently dropping out of the sample.
    """
    query = (
        select(MlbEdge, MlbGame)
        .outerjoin(MlbGame, MlbGame.game_pk == MlbEdge.game_pk)
        .where(MlbEdge.win_loss_push.is_not(None))
        .where(MlbEdge.is_valid.is_(True))
    )
    query = _apply_edge_date_window(query, start_date, end_date)
    return [(edge, game) for edge, game in db.execute(query).all()]


def _actual_totals_by_game(db: Session) -> dict[int, float]:
    """Lookup of game_pk → total runs from the persisted finals table."""
    rows = db.execute(select(MlbFinalScore.game_pk, MlbFinalScore.total_runs)).all()
    return {int(gp): float(tr) for gp, tr in rows if tr is not None}


def _apply_edge_date_window(query, start_date: str | None, end_date: str | None):
    """Filter MlbEdge queries by ``generated_for_date``.

    We use ``generated_for_date`` (the Arizona card date the edge was emitted
    for) instead of ``created_at`` so timezone rollovers don't drop edges
    captured just after UTC midnight.
    """
    if start_date:
        query = query.where(MlbEdge.generated_for_date >= start_date)
    if end_date:
        query = query.where(MlbEdge.generated_for_date <= end_date)
    return query


def _grouped_report(edges: list[MlbEdge], key_fn: Callable[[MlbEdge], Any], key_name: str) -> list[dict[str, Any]]:
    groups: dict[str, list[MlbEdge]] = {}
    for edge in edges:
        groups.setdefault(str(key_fn(edge)), []).append(edge)
    return [
        {key_name: key, **_summary_for_edges(items)}
        for key, items in sorted(groups.items())
    ]


def _normalize_score_axis(axis: str | None) -> str:
    axis_l = str(axis or "legacy").strip().lower()
    if axis_l in {"prediction", "prediction_score"}:
        return "prediction"
    if axis_l in {"execution", "execution_score"}:
        return "execution"
    if axis_l in {"legacy", "legacy_score", "score"}:
        return "legacy"
    raise ValueError(f"unsupported score axis: {axis}")


def _score_for_axis(edge: MlbEdge, axis: str) -> float | None:
    axis = _normalize_score_axis(axis)
    if axis == "prediction":
        return edge.prediction_score
    if axis == "execution":
        return edge.execution_score
    return edge.legacy_score if edge.legacy_score is not None else edge.score


def _summary_for_edges(edges: list[MlbEdge]) -> dict[str, Any]:
    wins = sum(1 for e in edges if e.win_loss_push == "win")
    losses = sum(1 for e in edges if e.win_loss_push == "loss")
    pushes = sum(1 for e in edges if e.win_loss_push == "push")
    decided = wins + losses
    return {
        "graded_edges": len(edges),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": round(wins / decided, 4) if decided else None,
        "roi_units": round(sum(e.roi_units or 0.0 for e in edges), 4),
        "average_clv_points": _avg(e.clv_points for e in edges),
        "average_clv_percent": _avg(e.clv_percent for e in edges),
        "average_score": _avg(e.score for e in edges),
        "average_prediction_score": _avg(e.prediction_score for e in edges),
        "average_execution_score": _avg(e.execution_score for e in edges),
        "average_legacy_score": _avg(
            e.legacy_score if e.legacy_score is not None else e.score for e in edges
        ),
        "positive_clv_rate": _positive_clv_rate(edges),
    }


def _side_metrics(edges: list[MlbEdge]) -> dict[str, Any]:
    wins = sum(1 for e in edges if e.win_loss_push == "win")
    losses = sum(1 for e in edges if e.win_loss_push == "loss")
    decided = wins + losses
    return {
        "count": len(edges),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / decided, 4) if decided else None,
        "roi_units": round(sum(e.roi_units or 0.0 for e in edges), 4),
        "average_score": _avg(e.score for e in edges),
        "average_prediction_score": _avg(e.prediction_score for e in edges),
        "average_execution_score": _avg(e.execution_score for e in edges),
        "average_legacy_score": _avg(
            e.legacy_score if e.legacy_score is not None else e.score for e in edges
        ),
        "average_clv_points": _avg(e.clv_points for e in edges),
        "average_clv_percent": _avg(e.clv_percent for e in edges),
        "positive_clv_rate": _positive_clv_rate(edges),
    }


def _clv_breakdown(edges: list[MlbEdge]) -> dict[str, Any]:
    with_clv = [e for e in edges if e.clv_points is not None]
    positive = [e for e in with_clv if (e.clv_points or 0.0) > 0]
    return {
        "count": len(edges),
        "edges_with_clv": len(with_clv),
        "average_clv_points": _avg(e.clv_points for e in edges),
        "average_clv_percent": _avg(e.clv_percent for e in edges),
        "positive_clv_rate": round(len(positive) / len(with_clv), 4) if with_clv else None,
    }


def _positive_clv_rate(edges: list[MlbEdge]) -> float | None:
    with_clv = [e for e in edges if e.clv_points is not None]
    if not with_clv:
        return None
    positive = sum(1 for e in with_clv if (e.clv_points or 0.0) > 0)
    return round(positive / len(with_clv), 4)


def _win_rate(edges: list[MlbEdge]) -> float | None:
    wins = sum(1 for e in edges if e.win_loss_push == "win")
    losses = sum(1 for e in edges if e.win_loss_push == "loss")
    decided = wins + losses
    return round(wins / decided, 4) if decided else None


def _edge_perf(edge: MlbEdge) -> dict[str, Any]:
    return {
        "id": edge.id,
        "market": edge.market,
        "side": edge.side,
        "score": edge.score,
        "prediction_score": edge.prediction_score,
        "execution_score": edge.execution_score,
        "legacy_score": edge.legacy_score if edge.legacy_score is not None else edge.score,
        "win_loss_push": edge.win_loss_push,
        "clv_points": edge.clv_points,
        "clv_percent": edge.clv_percent,
        "roi_units": edge.roi_units,
    }


def _avg(values: Iterable[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(mean(vals), 4) if vals else None


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    try:
        return round(float(a) - float(b), 4)
    except (TypeError, ValueError):
        return None


def _timing_bucket(created: datetime | None, start: datetime | None) -> str:
    """Pick a timing bucket from edge.created_at vs game start_time.

    Missing data lands in the ``after start / invalid`` bucket so it's
    explicitly visible rather than silently dropped.
    """
    if created is None or start is None:
        return "after start / invalid"
    try:
        # Both timestamps live in UTC in this codebase; if either is tz-aware
        # we strip the tzinfo so subtraction can't raise.
        c = created.replace(tzinfo=None) if created.tzinfo else created
        s = start.replace(tzinfo=None) if start.tzinfo else start
        delta = (s - c).total_seconds() / 3600.0
    except Exception:  # noqa: BLE001 — bad data should land in invalid bucket
        return "after start / invalid"
    if delta <= 0:
        return "after start / invalid"
    if delta > 12:
        return ">12h"
    if delta > 6:
        return "6-12h"
    if delta > 1:
        return "1-6h"
    return "<1h"


def _pearson(pairs: list[tuple[float, float]]) -> float | None:
    """Defensive Pearson correlation. Returns None when the sample is too
    small (<3 pairs) or one variable is constant (zero variance).
    """
    if not pairs or len(pairs) < 3:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    try:
        sx = pstdev(xs)
        sy = pstdev(ys)
    except Exception:  # noqa: BLE001
        return None
    if not sx or not sy:
        return None
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in pairs) / len(pairs)
    return round(cov / (sx * sy), 4)
