"""Pitcher strikeout prop scoring."""

from __future__ import annotations

from typing import Any

from app.services.mlb_edge_scoring import (
    PITCHER_K_WEIGHTS,
    additive_contributions,
    chase_risk,
    classify_edge,
    data_quality_score,
    weighted_score,
)
from app.services.mlb_odds_analysis import movement_score, odds_edge_score
from app.services.mlb_market_validation import MarketSubtype, normalized_prop_name


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
    line = _num(prop.get(f"best_{side}_line")) or _num(prop.get("line"))
    projected_k = _num(
        prop.get("ballparkpal_projected_k")
        or prop.get("projected_k")
        or prop.get("projected_strikeouts")
    )
    recent_form = 50.0
    if projected_k is not None and line is not None:
        recent_form = 50 + (projected_k - line) * 8
        if side == "under":
            recent_form = 100 - recent_form
    elif k_per_start is not None and line is not None:
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
    if projected_k is not None:
        factors["projected_k"] = projected_k
    score = weighted_score(factors, PITCHER_K_WEIGHTS)
    warnings.extend(environment.get("warnings") or [])
    cls = classify_edge(score, warnings)
    reasons = _reasons(side, pitcher, prop, summary, environment, factors)
    market_scope = prop.get("market_scope") or MarketSubtype.PLAYER_PROP.value
    try:
        MarketSubtype(market_scope)
    except ValueError:
        market_scope = MarketSubtype.PLAYER_PROP.value
    normalized_name = normalized_prop_name(
        player=pitcher.get("name"),
        side=side,
        line=line,
    )
    return {
        "edge_type": "pitcher_strikeouts",
        "game_pk": game["game_pk"],
        "market": normalized_name,
        "normalized_market_name": normalized_name,
        "market_scope": market_scope,
        "is_valid": bool(prop.get("is_valid", True)),
        "validation_reason": prop.get("validation_reason") or "",
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
        "data_sources_used": _data_sources(prop),
        "factors": factors,
        "score_contributions": additive_contributions(factors, PITCHER_K_WEIGHTS),
        "projected_strikeouts": projected_k,
        "source": prop.get("source"),
        "execution_source": prop.get("execution_source"),
        "market_type": prop.get("market_type"),
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
    projected_k = prop.get("ballparkpal_projected_k") or prop.get("projected_k")
    if projected_k is not None:
        reasons.insert(0, f"BallparkPal projected Ks: {projected_k}")
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


def _data_sources(prop: dict[str, Any]) -> list[str]:
    if str(prop.get("source") or "") == "ballparkpal_csv":
        return ["MLB StatsAPI", "WeatherAPI", "Cached Statcast", "ballparkpal_csv"]
    return ["MLB StatsAPI", "WeatherAPI", "Cached Statcast", "Odds-API.io"]
