"""CLV, grading, and backtest reporting for MLB edges."""

from __future__ import annotations

from datetime import date as date_cls, datetime, timedelta
from statistics import mean
from typing import Any

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


def grade_edge(
    edge: MlbEdge,
    *,
    result: str | None = None,
    win_loss_push: str | None = None,
    closing_line: float | None = None,
    closing_price: float | None = None,
    current_line: float | None = None,
    opening_line: float | None = None,
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
    return _grouped_report(
        _graded_edges(db, start_date=start_date, end_date=end_date),
        lambda e: score_band(e.score),
        "score_band",
    )


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
    positive = [e for e in edges if (e.clv_percent or 0.0) > 0]
    negative = [e for e in edges if (e.clv_percent or 0.0) < 0]
    return {
        "edges_with_clv": sum(1 for e in edges if e.clv_percent is not None or e.clv_points is not None),
        "average_clv_points": _avg(e.clv_points for e in edges),
        "average_clv_percent": _avg(e.clv_percent for e in edges),
        "positive_clv_rate": round(len(positive) / len(edges), 4) if edges else None,
        "negative_clv_rate": round(len(negative) / len(edges), 4) if edges else None,
        "top_positive": [_edge_perf(e) for e in sorted(edges, key=lambda e: e.clv_percent or -999, reverse=True)[:10]],
        "top_negative": [_edge_perf(e) for e in sorted(edges, key=lambda e: e.clv_percent or 999)[:10]],
    }


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


def score_band(score: float) -> str:
    if score < 65:
        return "<65"
    if score < 75:
        return "65-74"
    if score < 85:
        return "75-84"
    return "85+"


def implied_probability(price: Any) -> float | None:
    try:
        decimal_price = float(price)
    except (TypeError, ValueError):
        return None
    if decimal_price <= 1:
        return None
    return round(1 / decimal_price, 4)


def clv_points(edge: MlbEdge) -> float | None:
    if edge.recommended_line is None or edge.closing_line is None:
        return None
    direction = 1 if edge.side.lower() == "over" else -1
    return round((edge.closing_line - edge.recommended_line) * direction, 4)


def clv_percent(edge: MlbEdge) -> float | None:
    entry = edge.implied_probability_at_entry
    close = edge.implied_probability_at_close
    if entry is None or close is None:
        return None
    return round((close - entry) / entry, 4)


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


def _grouped_report(edges: list[MlbEdge], key_fn: Any, key_name: str) -> list[dict[str, Any]]:
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
    decided = wins + losses
    return {
        "graded_edges": len(edges),
        "win_rate": round(wins / decided, 4) if decided else None,
        "roi_units": round(sum(e.roi_units or 0.0 for e in edges), 4),
        "average_clv_points": _avg(e.clv_points for e in edges),
        "average_clv_percent": _avg(e.clv_percent for e in edges),
    }


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


def _avg(values: Any) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(mean(vals), 4) if vals else None
