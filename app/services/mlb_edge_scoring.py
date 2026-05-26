"""Unified MLB edge scoring and presentation rules."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


TOTAL_WEIGHTS = {
    "odds_edge": 0.30,
    "movement": 0.20,
    "environment": 0.20,
    "pitcher_matchup": 0.15,
    "smart_money": 0.10,
    "data_quality": 0.05,
}

PITCHER_K_WEIGHTS = {
    "sportsbook_price_edge": 0.25,
    "pitcher_recent_form": 0.25,
    "matchup_k_profile": 0.20,
    "line_movement": 0.15,
    "environment": 0.10,
    "data_quality": 0.05,
}


def weighted_score(factors: dict[str, float], weights: dict[str, float]) -> float:
    raw = sum(_clamp(factors.get(name, 50.0)) * weight for name, weight in weights.items())
    if raw > 95:
        logger.warning("Edge score capped: raw=%.2f", raw)
    return round(min(raw, 95.0), 2)


def classify_edge(score: float, warnings: list[str] | None = None) -> dict[str, str]:
    warnings = warnings or []
    if score < 65:
        action = "Pass"
    elif score < 75:
        action = "Watch"
    elif score < 85:
        action = "Bettable only at price"
    else:
        action = "Strong candidate"

    if len(warnings) >= 3:
        confidence = "low"
    elif score >= 82 and len(warnings) <= 1:
        confidence = "high"
    elif score >= 70 and len(warnings) <= 3:
        confidence = "medium"
    else:
        confidence = "low"
    return {"action": action, "confidence": confidence}


def data_quality_score(*, book_count: int, weather_ok: bool, statcast_ok: bool, odds_ok: bool) -> float:
    score = 45.0
    if odds_ok:
        score += 20
    if book_count >= 2:
        score += 15
    if weather_ok:
        score += 10
    if statcast_ok:
        score += 10
    return _clamp(score)


def chase_risk(*, movement_score: float, line_disagreement: float, score: float) -> str:
    if movement_score >= 72 or line_disagreement >= 1.0:
        return "high"
    if movement_score >= 58 or score >= 85:
        return "medium"
    return "low"


def edge_to_dict(edge: Any) -> dict[str, Any]:
    return {
        "id": edge.id,
        "game_pk": edge.game_pk,
        "edge_type": edge.edge_type,
        "market": edge.market,
        "normalized_market_name": edge.normalized_market_name,
        "market_scope": edge.market_scope,
        "is_valid": edge.is_valid,
        "validation_reason": edge.validation_reason,
        "side": edge.side,
        "line": edge.line,
        "best_book": edge.best_book,
        "best_price": edge.best_price,
        "consensus_price": edge.consensus_price,
        "score": edge.score,
        "confidence": edge.confidence,
        "action": edge.action,
        "chase_risk": edge.chase_risk,
        "reasons": edge.reasons or [],
        "warnings": edge.warnings or [],
        "data_sources_used": edge.data_sources_used or [],
        "factors": edge.factors or {},
        "generated_for_date": edge.generated_for_date,
        "opening_line": edge.opening_line,
        "current_line": edge.current_line,
        "recommended_line": edge.recommended_line,
        "closing_line": edge.closing_line,
        "closing_price": edge.closing_price,
        "result": edge.result,
        "win_loss_push": edge.win_loss_push,
        "implied_probability_at_entry": edge.implied_probability_at_entry,
        "implied_probability_at_close": edge.implied_probability_at_close,
        "clv_points": edge.clv_points,
        "clv_percent": edge.clv_percent,
        "roi_units": edge.roi_units,
        "graded_at": edge.graded_at.isoformat() if edge.graded_at else None,
        "created_at": edge.created_at.isoformat() if edge.created_at else None,
    }


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))
