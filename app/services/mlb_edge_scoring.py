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


# ---------------------------------------------------------------------------
# Prediction / Execution split (the post-audit scoring refactor).
#
# The legacy score collapsed "is this side likely to win" (a prediction
# problem) and "is this side priced well right now" (an execution problem)
# into a single number. That meant a tradeably-mispriced candidate could
# sort above a model+wallet-aligned pick simply because of sportsbook
# noise. We now compute two independent scores:
#
#   prediction_score — model conviction. Inputs: projection edge, wallet
#                       alignment, pitcher matchup, environment, model
#                       confidence. NO sportsbook price edge.
#   execution_score  — pricing / market quality. Inputs: sportsbook
#                       price edge, line movement, CLV signal, market
#                       quality. NO model conviction.
#
# Watchlist ranking sorts by (prediction_score, wallet_alignment_score,
# execution_score) so the terminal surfaces "most likely correct +
# wallet-confirmed" above "cheap price." ``score`` is preserved on the
# edge payload as a legacy alias for backwards compatibility, with the
# same value as ``legacy_score`` (the original weighted score from
# TOTAL_WEIGHTS) so older consumers keep working.
# ---------------------------------------------------------------------------

PREDICTION_WEIGHTS = {
    "projection_edge": 0.35,
    "wallet_alignment": 0.25,
    "pitcher_matchup": 0.15,
    "environment": 0.15,
    "model_confidence": 0.10,
}

EXECUTION_WEIGHTS = {
    "sportsbook_price_edge": 0.50,
    "line_movement": 0.20,
    "clv_signal": 0.15,
    "market_quality": 0.15,
}

# Cheap Price Trap: a candidate where the market is cheap but the model
# isn't behind it. Surfaces as a badge on the card. Thresholds picked to
# make the trap fire on the obvious case (price-only outlier with a weak
# prediction) without grabbing every borderline edge.
CHEAP_PRICE_TRAP_EXECUTION_FLOOR = 70.0
CHEAP_PRICE_TRAP_PREDICTION_CEILING = 65.0


def weighted_score(factors: dict[str, float], weights: dict[str, float]) -> float:
    raw = sum(_clamp(factors.get(name, 50.0)) * weight for name, weight in weights.items())
    if raw > 95:
        logger.warning("Edge score capped: raw=%.2f", raw)
    return round(min(raw, 95.0), 2)


def additive_contributions(
    factors: dict[str, float],
    weights: dict[str, float],
    *,
    baseline: float = 50.0,
) -> dict[str, float]:
    """Decompose a weighted score into additive per-factor point contributions.

    Each factor contributes ``(value - baseline) * weight`` so the contributions
    sum to ``score - baseline`` — i.e. how many points above/below a neutral 50
    each factor pushed the edge. Lets the card show "+12 sportsbook edge / −6
    crowded consensus" instead of opaque 0–100 bars.
    """
    return {
        name: round((_clamp(factors.get(name, baseline)) - baseline) * weight, 2)
        for name, weight in weights.items()
    }


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
        "prediction_score": edge.prediction_score,
        "execution_score": edge.execution_score,
        "legacy_score": edge.legacy_score if edge.legacy_score is not None else edge.score,
        "prediction_breakdown": edge.prediction_breakdown,
        "execution_breakdown": edge.execution_breakdown,
        "cheap_price_trap": bool(edge.cheap_price_trap) if edge.cheap_price_trap is not None else None,
        "confidence": edge.confidence,
        "action": edge.action,
        "chase_risk": edge.chase_risk,
        "reasons": edge.reasons or [],
        "warnings": edge.warnings or [],
        "data_sources_used": edge.data_sources_used or [],
        "factors": edge.factors or {},
        "wallet_context": edge.wallet_context or None,
        "score_contributions": edge.score_contributions or None,
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
        "model_projected_total": edge.model_projected_total,
        "graded_at": edge.graded_at.isoformat() if edge.graded_at else None,
        "created_at": edge.created_at.isoformat() if edge.created_at else None,
    }


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Dual-score (prediction + execution) helpers.
# ---------------------------------------------------------------------------

# Bayesian-style consensus midpoint that pulls a small wallet sample toward
# the neutral 50 sentinel. Without this, a single elite trader on either
# side would either max the alignment score to 100 or floor it at 0.
WALLET_ALIGNMENT_PRIOR_STRENGTH = 4.0
# Cap per-elite bumps so consensus_pct stays the primary driver.
ELITE_WALLET_BUMP_POINTS = 5.0
ELITE_WALLET_BUMP_CAP = 15.0


def wallet_alignment_score(wallet_context: dict[str, Any] | None) -> float:
    """0-100 score for how strongly tracked wallets endorse this side.

    No wallets / no info → 50.0 (neutral). With a wallet sample the score
    is anchored on ``consensus_pct`` (what fraction of tracked exposure
    sits on this side) and bumped by elite agreement / penalized by elite
    disagreement. A 4-wallet Bayesian prior pulls tiny samples toward the
    neutral midpoint so a single contrarian wallet can't flip the score.
    """
    if not isinstance(wallet_context, dict):
        return 50.0
    consensus = wallet_context.get("consensus_pct")
    tracked = float(wallet_context.get("tracked_wallet_count") or 0)
    elite_agree = float(wallet_context.get("elite_wallet_agreement") or 0)
    elite_disagree = float(wallet_context.get("elite_wallet_disagreement") or 0)
    if consensus is None or tracked <= 0:
        return 50.0
    try:
        consensus_pct = float(consensus)
    except (TypeError, ValueError):
        return 50.0
    # Bayesian smoothing toward 50 — the prior carries (50, prior_strength)
    # weight so a 1-wallet sample stays close to neutral.
    smoothed = (
        consensus_pct * tracked + 50.0 * WALLET_ALIGNMENT_PRIOR_STRENGTH
    ) / (tracked + WALLET_ALIGNMENT_PRIOR_STRENGTH)
    elite_bonus = min(elite_agree * ELITE_WALLET_BUMP_POINTS, ELITE_WALLET_BUMP_CAP)
    elite_penalty = min(elite_disagree * ELITE_WALLET_BUMP_POINTS, ELITE_WALLET_BUMP_CAP)
    return round(_clamp(smoothed + elite_bonus - elite_penalty), 2)


def clv_signal_score(odds_analysis: dict[str, Any], side: str) -> float:
    """0-100 score predicting CLV potential at scan time.

    Built from forward-looking proxies — not actual CLV, which we only
    know after close. We reward dispersion (multiple books disagreeing
    means there's a cheap book to fade later) and steam moving in our
    direction, and we penalize markets that already have very tight
    consensus (no CLV room left). Returns 50 when none of those
    components are present.
    """
    direction = str(odds_analysis.get("movement_direction") or "").lower()
    steam = float(odds_analysis.get("steam_velocity") or 0.0)
    line_disagreement = float(odds_analysis.get("line_disagreement") or 0.0)
    book_count = int(odds_analysis.get("book_count") or 0)
    raw = 50.0
    # Dispersion: each 0.25 runs of line disagreement = +4 pts of CLV
    # potential. Caps at +20 so a 1.25-run disagreement doesn't pin the
    # score; thicker disagreement is usually a stale-book artifact.
    raw += min(line_disagreement * 16, 20.0)
    if direction:
        if side.lower() in direction:
            raw += min(steam * 2.5, 15.0)
        else:
            raw -= min(steam * 2.5, 15.0)
    # Thin books → fewer opportunities to capture CLV via line shopping.
    if book_count < 3:
        raw -= 5.0
    return round(_clamp(raw), 2)


def market_quality_score(
    *,
    book_count: int,
    odds_ok: bool,
    line_disagreement: float,
    weather_ok: bool,
) -> float:
    """0-100 score for how trustworthy the execution market is.

    Differs from ``data_quality_score`` (which rates the *model's*
    inputs) — this rates the *market's* inputs: do we have enough books
    to trust the consensus line, is the book disagreement small enough
    to imply real competition, are the odds rows present at all.
    """
    score = 40.0
    if odds_ok:
        score += 20.0
    if book_count >= 5:
        score += 25.0
    elif book_count >= 3:
        score += 18.0
    elif book_count >= 2:
        score += 10.0
    # Very high disagreement = something is wrong with one of the books.
    if line_disagreement >= 1.5:
        score -= 10.0
    elif line_disagreement >= 1.0:
        score -= 5.0
    if weather_ok:
        score += 5.0
    return round(_clamp(score), 2)


def compute_prediction_score(
    factors: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """Run weighted_score + additive_contributions for prediction inputs.

    Returns ``(score, breakdown)`` where ``breakdown`` is the per-input
    contribution in points above the neutral 50 baseline.
    """
    score = weighted_score(factors, PREDICTION_WEIGHTS)
    breakdown = additive_contributions(factors, PREDICTION_WEIGHTS)
    return score, breakdown


def compute_execution_score(
    factors: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """Run weighted_score + additive_contributions for execution inputs."""
    score = weighted_score(factors, EXECUTION_WEIGHTS)
    breakdown = additive_contributions(factors, EXECUTION_WEIGHTS)
    return score, breakdown


def is_cheap_price_trap(
    *, prediction_score: float, execution_score: float,
) -> bool:
    """A cheap price trap is a high execution score with a weak model.

    Returns True when execution is at-or-above the trap floor and
    prediction is below the trap ceiling, i.e. the market thinks the
    price is good but our model doesn't think the side is right.
    """
    return (
        execution_score >= CHEAP_PRICE_TRAP_EXECUTION_FLOOR
        and prediction_score < CHEAP_PRICE_TRAP_PREDICTION_CEILING
    )


def watchlist_sort_key(payload: dict[str, Any]) -> tuple[float, float, float]:
    """Composite sort key used by the MLB terminal / watchlist.

    Sorts by prediction_score first, wallet_alignment_score second,
    execution_score third. Used as ``key=watchlist_sort_key, reverse=True``.
    Missing scores fall to 0 so a partially-scored edge sinks below a
    fully-scored one.
    """
    factors = payload.get("factors") or {}
    return (
        float(payload.get("prediction_score") or 0.0),
        float(factors.get("wallet_alignment") or 0.0),
        float(payload.get("execution_score") or 0.0),
    )
