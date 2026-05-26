"""CLV, grading, and backtest reporting for MLB edges."""

from __future__ import annotations

from datetime import datetime
from statistics import mean
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import MlbEdge


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


def performance_summary(db: Session) -> dict[str, Any]:
    edges = _graded_edges(db)
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
    }


def performance_by_market(db: Session) -> list[dict[str, Any]]:
    return _grouped_report(_graded_edges(db), lambda e: e.edge_type, "edge_type")


def performance_by_score_band(db: Session) -> list[dict[str, Any]]:
    return _grouped_report(_graded_edges(db), lambda e: score_band(e.score), "score_band")


def clv_report(db: Session) -> dict[str, Any]:
    edges = _graded_edges(db)
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


def top_factors_by_performance(db: Session) -> list[dict[str, Any]]:
    edges = _graded_edges(db)
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


def _graded_edges(db: Session) -> list[MlbEdge]:
    return list(db.scalars(select(MlbEdge).where(MlbEdge.win_loss_push.is_not(None))))


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
