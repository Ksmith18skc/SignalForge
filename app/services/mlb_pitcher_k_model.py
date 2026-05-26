"""Pitcher strikeout prop scoring."""

from __future__ import annotations

from typing import Any

from app.services.mlb_edge_scoring import (
    PITCHER_K_WEIGHTS,
    chase_risk,
    classify_edge,
    data_quality_score,
    weighted_score,
)
from app.services.mlb_odds_analysis import movement_score, odds_edge_score


def pitcher_k_edges(
    *,
    game: dict[str, Any],
    pitcher: dict[str, Any],
    prop_analysis: dict[str, Any],
    statcast_context: dict[str, Any],
    environment: dict[str, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for side in ("over", "under"):
        out.append(_pitcher_edge(game, pitcher, prop_analysis, statcast_context, environment, side))
    return out


def _pitcher_edge(
    game: dict[str, Any],
    pitcher: dict[str, Any],
    prop: dict[str, Any],
    statcast: dict[str, Any],
    environment: dict[str, Any],
    side: str,
) -> dict[str, Any]:
    warnings = list(prop.get("warnings") or []) + list(statcast.get("warnings") or [])
    summary = statcast.get("summary") or {}
    k_per_start = _num(summary.get("strikeouts_per_start"))
    line = _num(prop.get("line"))
    recent_form = 50.0
    if k_per_start is not None and line is not None:
        recent_form = 50 + (k_per_start - line) * 8
        if side == "under":
            recent_form = 100 - recent_form
    elif k_per_start is None:
        warnings.append("Recent Statcast K summary missing")

    matchup_k_profile = 50.0  # Placeholder until opponent K tendency is wired.
    env_score = environment.get("k_environment_score") or 50.0
    if side == "over":
        env_score = 100 - float(env_score) * 0.35
    factors = {
        "sportsbook_price_edge": odds_edge_score(prop, side),
        "pitcher_recent_form": _clamp(recent_form),
        "matchup_k_profile": matchup_k_profile,
        "line_movement": movement_score(prop, side),
        "environment": _clamp(float(env_score)),
        "data_quality": data_quality_score(
            book_count=int(prop.get("book_count") or 0),
            weather_ok=not any("Weather missing" in w for w in environment.get("warnings", [])),
            statcast_ok=bool(summary),
            odds_ok=bool(prop.get("rows")),
        ),
    }
    score = weighted_score(factors, PITCHER_K_WEIGHTS)
    warnings.extend(environment.get("warnings") or [])
    cls = classify_edge(score, warnings)
    reasons = _reasons(side, pitcher, prop, summary, environment, factors)
    return {
        "edge_type": "pitcher_strikeouts",
        "game_pk": game["game_pk"],
        "market": f"{pitcher.get('name') or 'Pitcher'} {side.title()} {line if line is not None else '?'} Ks",
        "side": side,
        "line": line,
        "best_book": prop.get(f"best_{side}_book"),
        "best_price": prop.get(f"best_{side}_price"),
        "consensus_price": prop.get("consensus_price"),
        "score": score,
        "confidence": cls["confidence"],
        "action": cls["action"],
        "chase_risk": chase_risk(
            movement_score=factors["line_movement"],
            line_disagreement=float(prop.get("line_disagreement") or 0.0),
            score=score,
        ),
        "reasons": reasons,
        "warnings": _dedupe(warnings),
        "data_sources_used": ["MLB StatsAPI", "WeatherAPI", "Cached Statcast", "Odds-API.io"],
        "factors": factors,
    }


def _reasons(
    side: str,
    pitcher: dict[str, Any],
    prop: dict[str, Any],
    summary: dict[str, Any],
    environment: dict[str, Any],
    factors: dict[str, float],
) -> list[str]:
    reasons = [
        f"{pitcher.get('name') or 'Pitcher'} recent form score: {factors['pitcher_recent_form']:.1f}",
        f"Best {side} price from {prop.get(f'best_{side}_book') or 'no book available'}",
        f"K environment score: {environment.get('k_environment_score', 50)}",
    ]
    if summary.get("strikeouts_per_start") is not None:
        reasons.insert(0, f"Recent strikeouts/start: {summary['strikeouts_per_start']}")
    return reasons[:4]


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(v for v in values if v))
