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


def lookup_edge_score_band(
    db: Session,
    *,
    edge_type: str | None,
    score: float | None,
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
    band = score_band(float(score)) if score is not None else None
    empty = {
        "edge_type": edge_type,
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
        if (edge_type is None or e.edge_type == edge_type) and score_band(e.score) == band
    ]
    if not edges:
        return empty
    wins = sum(1 for e in edges if e.win_loss_push == "win")
    losses = sum(1 for e in edges if e.win_loss_push == "loss")
    pushes = sum(1 for e in edges if e.win_loss_push == "push")
    decided = wins + losses
    return {
        "edge_type": edge_type,
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
