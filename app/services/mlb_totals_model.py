"""Full-game MLB totals model."""

from __future__ import annotations

from typing import Any

from app.services.mlb_edge_scoring import (
    TOTAL_WEIGHTS,
    additive_contributions,
    chase_risk,
    classify_edge,
    data_quality_score,
    weighted_score,
)
from app.services.mlb_odds_analysis import movement_score, odds_edge_score
from app.services.mlb_market_validation import MarketSubtype, normalized_total_name


def total_edges(
    *,
    game: dict[str, Any],
    odds_analysis: dict[str, Any],
    environment: dict[str, Any],
    pitcher_matchup_score: float = 50.0,
    smart_money_score: float = 50.0,
) -> list[dict[str, Any]]:
    return [
        _total_edge(game, odds_analysis, environment, "over", pitcher_matchup_score, smart_money_score),
        _total_edge(game, odds_analysis, environment, "under", pitcher_matchup_score, smart_money_score),
    ]


def _total_edge(
    game: dict[str, Any],
    odds: dict[str, Any],
    environment: dict[str, Any],
    side: str,
    pitcher_matchup_score: float,
    smart_money_score: float,
) -> dict[str, Any]:
    env = float(
        environment.get("run_environment_score" if side == "over" else "under_environment_score")
        or 50.0
    )
    factors = {
        "odds_edge": odds_edge_score(odds, side),
        "movement": movement_score(odds, side),
        "environment": env,
        "pitcher_matchup": pitcher_matchup_score if side == "over" else 100 - pitcher_matchup_score,
        "smart_money": smart_money_score,
        "data_quality": data_quality_score(
            book_count=int(odds.get("book_count") or 0),
            weather_ok=not any("Weather missing" in w for w in environment.get("warnings", [])),
            statcast_ok=True,
            odds_ok=bool(odds.get("rows")),
        ),
    }
    score = weighted_score(factors, TOTAL_WEIGHTS)
    warnings = list(odds.get("warnings") or []) + list(environment.get("warnings") or [])
    cls = classify_edge(score, warnings)
    line = odds.get("consensus_total_line")
    market_scope = odds.get("market_scope") or MarketSubtype.FULL_GAME_TOTAL.value
    try:
        scope_enum = MarketSubtype(market_scope)
    except ValueError:
        scope_enum = MarketSubtype.FULL_GAME_TOTAL
    normalized_name = normalized_total_name(
        scope=scope_enum,
        side=side,
        line=line,
        home=game.get("home_team"),
        away=game.get("away_team"),
    )
    return {
        "edge_type": "game_total",
        "game_pk": game["game_pk"],
        "market": normalized_name,
        "normalized_market_name": normalized_name,
        "market_scope": market_scope,
        "is_valid": bool(odds.get("is_valid", True)),
        "validation_reason": odds.get("validation_reason") or "",
        "side": side,
        "line": line,
        "best_book": odds.get(f"best_{side}_book"),
        "best_price": odds.get(f"best_{side}_price"),
        "consensus_price": odds.get("consensus_price"),
        "score": score,
        "confidence": cls["confidence"],
        "action": cls["action"],
        "chase_risk": chase_risk(
            movement_score=factors["movement"],
            line_disagreement=float(odds.get("line_disagreement") or 0.0),
            score=score,
        ),
        "reasons": _reasons(side, odds, environment, factors),
        "warnings": list(dict.fromkeys(warnings)),
        "data_sources_used": ["MLB StatsAPI", "WeatherAPI", "Odds-API.io", "SignalForge smart money"],
        "factors": factors,
        "score_contributions": additive_contributions(factors, TOTAL_WEIGHTS),
    }


def _reasons(side: str, odds: dict[str, Any], environment: dict[str, Any], factors: dict[str, float]) -> list[str]:
    env_key = "run_environment_score" if side == "over" else "under_environment_score"
    return [
        f"Environment supports {side}: {environment.get(env_key, 50)}",
        f"Odds edge score: {factors['odds_edge']:.1f}",
        f"Book count: {odds.get('book_count') or 0}",
        "Bullpen fatigue and umpire impact are placeholders in this version",
    ]
