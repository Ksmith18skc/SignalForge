"""Full-game MLB totals model.

Post-refactor design: every game-total edge now carries TWO independent
scores plus the legacy score. ``prediction_score`` rates how likely the
side is to be correct (projection / wallets / matchup / environment /
model confidence — NO sportsbook price edge). ``execution_score`` rates
how cheap / tradable the price is (sportsbook price edge, line
movement, CLV signal, market quality — NO model conviction).
``legacy_score`` mirrors the original TOTAL_WEIGHTS weighted score and
is preserved for the Pricing Edge tab + backwards compatibility.
"""

from __future__ import annotations

from typing import Any

from app.services.mlb_edge_scoring import (
    EXECUTION_WEIGHTS,
    PREDICTION_WEIGHTS,
    TOTAL_WEIGHTS,
    additive_contributions,
    chase_risk,
    classify_edge,
    clv_signal_score,
    compute_execution_score,
    compute_prediction_score,
    data_quality_score,
    is_cheap_price_trap,
    market_quality_score,
    wallet_alignment_score,
    weighted_score,
)
from app.services.mlb_odds_analysis import movement_score, odds_edge_score
from app.services.mlb_market_validation import MarketSubtype, normalized_total_name
from app.services.mlb_projection import (
    model_projected_total,
    projection_confidence_score,
    projection_edge_score,
)

# Projection-confidence penalty knobs (applied to prediction_score only).
# A confidence of 100 leaves the score untouched; a confidence of 0
# pulls the score halfway to the neutral 50 baseline. Linear in between.
PROJECTION_CONFIDENCE_FLOOR_BLEND = 0.5


def total_edges(
    *,
    game: dict[str, Any],
    odds_analysis: dict[str, Any],
    environment: dict[str, Any],
    pitcher_matchup_score: float = 50.0,
    smart_money_score: float = 50.0,
    side_penalty_points: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    return [
        _total_edge(
            game, odds_analysis, environment, "over",
            pitcher_matchup_score, smart_money_score, side_penalty_points,
        ),
        _total_edge(
            game, odds_analysis, environment, "under",
            pitcher_matchup_score, smart_money_score, side_penalty_points,
        ),
    ]


def _total_edge(
    game: dict[str, Any],
    odds: dict[str, Any],
    environment: dict[str, Any],
    side: str,
    pitcher_matchup_score: float,
    smart_money_score: float,
    side_penalty_points: dict[str, float] | None,
) -> dict[str, Any]:
    env_score_for_side = float(
        environment.get("run_environment_score" if side == "over" else "under_environment_score")
        or 50.0
    )
    consensus_line = odds.get("consensus_total_line")
    projected_total = model_projected_total(
        consensus_line=consensus_line, environment=environment,
    )
    price_edge = odds_edge_score(odds, side)
    proj_edge = projection_edge_score(
        side=side,
        consensus_line=consensus_line,
        projected_total=projected_total,
    )
    proj_confidence = projection_confidence_score(
        environment=environment,
        book_count=int(odds.get("book_count") or 0),
        statcast_ok=True,
    )
    line_move = movement_score(odds, side)
    clv_sig = clv_signal_score(odds, side)
    book_count = int(odds.get("book_count") or 0)
    line_disagreement = float(odds.get("line_disagreement") or 0.0)
    weather_ok = not any(
        "Weather missing" in w for w in environment.get("warnings") or []
    )
    market_q = market_quality_score(
        book_count=book_count,
        odds_ok=bool(odds.get("rows")),
        line_disagreement=line_disagreement,
        weather_ok=weather_ok,
    )
    dq = data_quality_score(
        book_count=book_count,
        weather_ok=weather_ok,
        statcast_ok=True,
        odds_ok=bool(odds.get("rows")),
    )
    pitcher_factor = (
        pitcher_matchup_score if side == "over" else 100 - pitcher_matchup_score
    )

    # ---- Legacy factors (the original TOTAL_WEIGHTS inputs, unchanged).
    # odds_edge is now PURE price-edge again — projection_edge has been
    # promoted to its own first-class factor and lives in prediction_score.
    legacy_factors = {
        "odds_edge": price_edge,
        "movement": line_move,
        "environment": env_score_for_side,
        "pitcher_matchup": pitcher_factor,
        "smart_money": smart_money_score,
        "data_quality": dq,
    }
    legacy_score_raw = weighted_score(legacy_factors, TOTAL_WEIGHTS)
    legacy_breakdown = additive_contributions(legacy_factors, TOTAL_WEIGHTS)

    # ---- Prediction inputs. NO sportsbook price edge. Wallet alignment
    # starts at the neutral 50 sentinel here and gets recomputed inside
    # _apply_wallet_flow once the wallet_context has been built.
    prediction_factors = {
        "projection_edge": proj_edge,
        "wallet_alignment": 50.0,
        "pitcher_matchup": pitcher_factor,
        "environment": env_score_for_side,
        "model_confidence": proj_confidence,
    }
    prediction_raw, prediction_breakdown = compute_prediction_score(prediction_factors)
    prediction_penalties = _apply_prediction_penalties(
        prediction_raw,
        side=side,
        projection_confidence=proj_confidence,
        side_penalty_points=side_penalty_points,
    )
    prediction_score = prediction_penalties["adjusted_score"]

    # ---- Execution inputs. ONLY pricing/market info.
    execution_factors = {
        "sportsbook_price_edge": price_edge,
        "line_movement": line_move,
        "clv_signal": clv_sig,
        "market_quality": market_q,
    }
    execution_score, execution_breakdown = compute_execution_score(execution_factors)

    cheap_trap = is_cheap_price_trap(
        prediction_score=prediction_score, execution_score=execution_score,
    )

    # Combine every factor for downstream diagnostics (factor_distribution
    # audit + the existing card score chips). Per-axis breakdowns sit
    # alongside as ``prediction_breakdown`` / ``execution_breakdown``.
    factors = {
        **legacy_factors,
        # First-class diagnostic factors used by the audit dashboards and
        # the new prediction/execution scoring. Stored here with zero
        # TOTAL_WEIGHTS so they don't double-count against legacy_score.
        "price_edge": price_edge,
        "sportsbook_price_edge": price_edge,
        "projection_edge": proj_edge,
        "projection_confidence": proj_confidence,
        "wallet_alignment": 50.0,  # filled in by _apply_wallet_flow
        "model_confidence": proj_confidence,
        "line_movement": line_move,
        "clv_signal": clv_sig,
        "market_quality": market_q,
    }

    warnings = list(odds.get("warnings") or []) + list(environment.get("warnings") or [])
    warnings.extend(prediction_penalties["warnings"])
    if cheap_trap:
        warnings.append(
            f"Cheap Price Trap: execution_score={execution_score:.1f} but "
            f"prediction_score={prediction_score:.1f} (<65)."
        )

    # Card classification is driven by prediction_score — "is this likely
    # to be the right side?" If a high-execution / low-prediction trap
    # shows up, the dashboard surfaces it via the badge but the recommended
    # action stays Pass/Watch based on conviction.
    cls = classify_edge(prediction_score, warnings)

    market_scope = odds.get("market_scope") or MarketSubtype.FULL_GAME_TOTAL.value
    try:
        scope_enum = MarketSubtype(market_scope)
    except ValueError:
        scope_enum = MarketSubtype.FULL_GAME_TOTAL
    normalized_name = normalized_total_name(
        scope=scope_enum,
        side=side,
        line=consensus_line,
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
        "line": consensus_line,
        "best_book": odds.get(f"best_{side}_book"),
        "best_price": odds.get(f"best_{side}_price"),
        "consensus_price": odds.get("consensus_price"),
        # Backwards-compat: ``score`` stays the legacy weighted score so
        # any external consumer (the Pricing Edge tab, old templates,
        # exports) keeps working unchanged. The two new axes live in
        # their own fields.
        "score": legacy_score_raw,
        "legacy_score": legacy_score_raw,
        "prediction_score": prediction_score,
        "execution_score": execution_score,
        "cheap_price_trap": cheap_trap,
        "confidence": cls["confidence"],
        "action": cls["action"],
        "chase_risk": chase_risk(
            movement_score=line_move,
            line_disagreement=line_disagreement,
            score=prediction_score,
        ),
        "reasons": _reasons(side, odds, environment, factors, projected_total),
        "warnings": list(dict.fromkeys(warnings)),
        "data_sources_used": ["MLB StatsAPI", "WeatherAPI", "Odds-API.io", "SignalForge smart money"],
        "factors": factors,
        # ``score_contributions`` keeps decomposing the legacy weighted
        # score so the existing factor_distribution audit + score_attribution
        # report continue to render against the same baseline. The two
        # new breakdowns live in their own keys.
        "score_contributions": legacy_breakdown,
        "prediction_breakdown": prediction_breakdown,
        "execution_breakdown": execution_breakdown,
        "model_projected_total": projected_total,
        "score_penalties": prediction_penalties["breakdown"],
    }


def _apply_prediction_penalties(
    raw_score: float,
    *,
    side: str,
    projection_confidence: float,
    side_penalty_points: dict[str, float] | None,
) -> dict[str, Any]:
    """Apply prediction-axis penalties: low projection confidence + recent
    side underperformance. Applied AFTER the prediction weighted score
    so the operator can see the raw model conviction separately from
    the portfolio-management adjustments.
    """
    warnings: list[str] = []
    breakdown: dict[str, float] = {}

    conf_blend = PROJECTION_CONFIDENCE_FLOOR_BLEND + (
        (1.0 - PROJECTION_CONFIDENCE_FLOOR_BLEND)
        * (max(0.0, min(100.0, projection_confidence)) / 100.0)
    )
    after_confidence = 50.0 + (raw_score - 50.0) * conf_blend
    confidence_delta = round(after_confidence - raw_score, 2)
    breakdown["projection_confidence"] = confidence_delta
    if abs(confidence_delta) >= 1.0:
        warnings.append(
            f"Prediction reduced by {abs(confidence_delta):.1f} points "
            f"for low projection confidence ({projection_confidence:.0f}/100)."
        )

    side_points = 0.0
    if side_penalty_points:
        try:
            side_points = float(side_penalty_points.get(side.lower(), 0.0) or 0.0)
        except (TypeError, ValueError):
            side_points = 0.0
    after_side = after_confidence - side_points
    side_delta = round(after_side - after_confidence, 2)
    breakdown["side_underperformance"] = side_delta
    if side_points >= 1.0:
        warnings.append(
            f"Prediction reduced by {side_points:.1f} points for recent "
            f"underperformance on {side} side."
        )

    adjusted = max(0.0, min(95.0, after_side))
    return {
        "adjusted_score": round(adjusted, 2),
        "raw_score": raw_score,
        "breakdown": breakdown,
        "warnings": warnings,
    }


def recompute_with_wallet_alignment(
    payload: dict[str, Any], *, wallet_context: dict[str, Any] | None,
) -> None:
    """Recompute the wallet-aware piece of prediction_score in place.

    Called by the edge engine after ``build_wallet_context`` attaches
    ``wallet_context``. The other prediction factors are wallet-agnostic
    so we only need to re-weight prediction_score and re-classify.
    """
    factors = payload.setdefault("factors", {})
    new_alignment = wallet_alignment_score(wallet_context)
    factors["wallet_alignment"] = new_alignment

    prediction_factors = {
        "projection_edge": float(factors.get("projection_edge") or 50.0),
        "wallet_alignment": new_alignment,
        "pitcher_matchup": float(factors.get("pitcher_matchup") or 50.0),
        "environment": float(factors.get("environment") or 50.0),
        "model_confidence": float(factors.get("model_confidence") or 50.0),
    }
    raw, breakdown = compute_prediction_score(prediction_factors)
    # Re-apply the same penalties that were captured at scan time so the
    # wallet bump can't accidentally erase a low-confidence pull.
    penalties = payload.get("score_penalties") or {}
    blend_delta = float(penalties.get("projection_confidence") or 0.0)
    side_delta = float(penalties.get("side_underperformance") or 0.0)
    adjusted = max(0.0, min(95.0, raw + blend_delta + side_delta))
    payload["prediction_score"] = round(adjusted, 2)
    payload["prediction_breakdown"] = breakdown
    # Cheap-price-trap depends on prediction_score, so refresh the flag.
    payload["cheap_price_trap"] = is_cheap_price_trap(
        prediction_score=payload["prediction_score"],
        execution_score=float(payload.get("execution_score") or 0.0),
    )
    warnings = [
        str(w) for w in (payload.get("warnings") or [])
        if not str(w).startswith("Cheap Price Trap:")
    ]
    if payload["cheap_price_trap"]:
        warnings.append(
            f"Cheap Price Trap: execution_score={float(payload.get('execution_score') or 0.0):.1f} "
            f"but prediction_score={payload['prediction_score']:.1f} (<65)."
        )
    payload["warnings"] = warnings
    # Re-classify off the updated prediction_score.
    cls = classify_edge(payload["prediction_score"], warnings)
    payload["confidence"] = cls["confidence"]
    payload["action"] = cls["action"]


def _reasons(
    side: str,
    odds: dict[str, Any],
    environment: dict[str, Any],
    factors: dict[str, float],
    projected_total: float | None,
) -> list[str]:
    env_key = "run_environment_score" if side == "over" else "under_environment_score"
    reasons = [
        f"Environment supports {side}: {environment.get(env_key, 50)}",
        f"Projection edge: {factors.get('projection_edge', 50):.1f}",
        f"Line movement: {factors.get('line_movement', 50):.1f}",
        f"CLV signal: {factors.get('clv_signal', 50):.1f}",
        f"Market quality: {factors.get('market_quality', 50):.1f}",
        f"Projection confidence: {factors.get('projection_confidence', 50):.1f}",
        f"Book count: {odds.get('book_count') or 0}",
    ]
    if projected_total is not None:
        reasons.append(f"Model projected total: {projected_total:.2f}")
    return reasons
