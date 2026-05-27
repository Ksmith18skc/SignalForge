"""Per-signal explainability payload.

Aggregates every learning row tied to a Signal (factors, contributing
wallets, regime, calibration, archetype tags, tier history) into a single
JSON-serialisable dict the dashboard's "Falcon Intelligence" expander can
render directly.

This is a pure read function. It does not call Falcon, mutate state, or
recompute weights — that work belongs to ``falcon_learning`` and
``falcon_retraining``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.models import (
    AdaptiveFactorWeight,
    Signal,
    SignalFactorAttribution,
    SignalLearningSnapshot,
    SignalRegimeFeatures,
    SignalRegimeSnapshot,
    SignalWalletContribution,
    WalletBehaviorProfile,
    WalletLearningStats,
    WalletTierHistory,
)
from app.services.falcon_retraining import (
    lookup_calibrated_probability,
    lookup_regime_stats,
)

logger = logging.getLogger(__name__)


def _factor_with_history(
    db: Session,
    *,
    factor_name: str,
    sport: str | None,
    market_type: str | None,
    factor_value: float,
    factor_weight: float,
) -> dict[str, Any]:
    """Per-factor block: current value/weight + adaptive context."""
    adaptive_row = db.scalar(
        select(AdaptiveFactorWeight).where(
            AdaptiveFactorWeight.factor_name == factor_name,
            AdaptiveFactorWeight.sport == (sport or "*"),
            AdaptiveFactorWeight.market_type == (market_type or "*"),
        )
    )
    if adaptive_row is None:
        adaptive_row = db.scalar(
            select(AdaptiveFactorWeight).where(
                AdaptiveFactorWeight.factor_name == factor_name,
                AdaptiveFactorWeight.sport == "*",
                AdaptiveFactorWeight.market_type == "*",
            )
        )
    return {
        "factor_name": factor_name,
        "value": float(factor_value),
        "weight": float(factor_weight),
        "adaptive": {
            "current_weight": adaptive_row.current_weight if adaptive_row else None,
            "rolling_roi": adaptive_row.rolling_roi if adaptive_row else None,
            "rolling_clv": adaptive_row.rolling_clv if adaptive_row else None,
            "predictive_power": adaptive_row.predictive_power if adaptive_row else None,
            "sample_size": adaptive_row.sample_size if adaptive_row else 0,
            "confidence": adaptive_row.confidence if adaptive_row else 0.0,
        } if adaptive_row else None,
    }


def _wallet_block(
    db: Session,
    *,
    wallet_address: str,
    contribution_weight: float,
    side: str | None,
    size_usd: float | None,
    entry_price: float | None,
) -> dict[str, Any]:
    stats = db.get(WalletLearningStats, wallet_address)
    latest_tier = db.scalar(
        select(WalletTierHistory)
        .where(WalletTierHistory.wallet_address == wallet_address)
        .where(WalletTierHistory.sport.is_(None))
        .order_by(desc(WalletTierHistory.captured_at))
        .limit(1)
    )
    archetypes = list(
        db.scalars(
            select(WalletBehaviorProfile)
            .where(WalletBehaviorProfile.wallet_address == wallet_address)
            .order_by(desc(WalletBehaviorProfile.score))
        )
    )
    return {
        "wallet_address": wallet_address,
        "side": side,
        "size_usd": size_usd,
        "entry_price": entry_price,
        "contribution_weight": contribution_weight,
        "tier": latest_tier.tier if latest_tier else None,
        "tier_reason": latest_tier.reason if latest_tier else None,
        "roi": stats.roi if stats else None,
        "avg_clv": stats.avg_clv if stats else None,
        "win_rate": (
            stats.wins / (stats.wins + stats.losses)
            if stats and (stats.wins + stats.losses) > 0 else None
        ),
        "sample_size": stats.sample_size if stats else 0,
        "confidence_weight": stats.confidence_weight if stats else None,
        "archetypes": [
            {"archetype": a.archetype, "score": a.score} for a in archetypes
        ],
    }


def explain_signal(db: Session, signal_id: int) -> dict[str, Any] | None:
    """Build the explainability payload for one signal.

    Returns ``None`` only when the signal itself doesn't exist; partial
    learning data (no factors, no regime) still returns a valid payload
    with the missing sections marked ``available: False``.
    """
    signal = db.get(Signal, signal_id)
    if signal is None:
        return None
    snapshot = db.get(SignalLearningSnapshot, signal_id)
    factor_rows = list(
        db.scalars(
            select(SignalFactorAttribution)
            .where(SignalFactorAttribution.signal_id == signal_id)
            .order_by(SignalFactorAttribution.factor_name)
        )
    )
    wallet_rows = list(
        db.scalars(
            select(SignalWalletContribution)
            .where(SignalWalletContribution.signal_id == signal_id)
        )
    )
    regime_row = db.get(SignalRegimeFeatures, signal_id)
    snapshot_row = db.get(SignalRegimeSnapshot, signal_id)

    sport = snapshot.sport if snapshot else None
    market_type = snapshot.market_type if snapshot else None

    factors = [
        _factor_with_history(
            db,
            factor_name=row.factor_name,
            sport=sport,
            market_type=market_type,
            factor_value=row.factor_value,
            factor_weight=row.factor_weight,
        )
        for row in factor_rows
    ]
    wallets = [
        _wallet_block(
            db,
            wallet_address=row.wallet_address,
            contribution_weight=row.contribution_weight,
            side=row.side,
            size_usd=row.size_usd,
            entry_price=row.entry_price,
        )
        for row in wallet_rows
    ]

    elite_disagreement = [
        w for w in wallets
        if w.get("tier") in {"elite", "trusted"} and w.get("side") and w["side"] != signal.side
    ]

    raw_score = float(snapshot.raw_score) if (snapshot and snapshot.raw_score is not None) else (
        float(signal.score) if signal.score is not None else None
    )
    calibrated = (
        snapshot.calibrated_probability if snapshot else None
    )
    if calibrated is None and raw_score is not None:
        calibrated = lookup_calibrated_probability(
            db,
            raw_score=raw_score,
            sport=sport,
            market_type=market_type,
        )

    return {
        "signal_id": signal_id,
        "market_id": signal.market_id,
        "side": signal.side,
        "outcome": signal.outcome,
        "raw_score": raw_score,
        "calibrated_probability": calibrated,
        "factors": {
            "available": bool(factors),
            "rows": factors,
        },
        "wallets": {
            "available": bool(wallets),
            "rows": wallets,
            "elite_disagreement_count": len(elite_disagreement),
        },
        "regime": {
            "available": regime_row is not None,
            "rows": regime_row.raw_payload if regime_row else None,
            "summary": _summarise_regime(regime_row) if regime_row else None,
        },
        "regime_snapshot": _snapshot_block(db, snapshot_row, sport, market_type),
        "conflict": (snapshot.conflict_payload if snapshot else None) or {},
        "generated_at": datetime.utcnow().isoformat(),
    }


def _snapshot_block(
    db: Session,
    snapshot: SignalRegimeSnapshot | None,
    sport: str | None,
    market_type: str | None,
) -> dict[str, Any]:
    """Immutable per-signal regime snapshot + historical ROI of similar regimes."""
    if snapshot is None:
        return {"available": False, "reason": "no snapshot persisted yet"}
    historical = lookup_regime_stats(
        db,
        classification=snapshot.regime_classification or "",
        sport=sport,
        market_type=market_type,
    ) if snapshot.regime_classification else None
    return {
        "available": True,
        "captured_at": snapshot.captured_at.isoformat() if snapshot.captured_at else None,
        "enrichment_status": snapshot.enrichment_status,
        "components": dict(snapshot.components or {}),
        "regime_classification": snapshot.regime_classification,
        "market_price": snapshot.market_price,
        "line_velocity": snapshot.line_velocity,
        "line_acceleration": snapshot.line_acceleration,
        "volatility_score": snapshot.volatility_score,
        "liquidity_score": snapshot.liquidity_score,
        "orderflow_state": snapshot.orderflow_state,
        "steam_state": snapshot.steam_state,
        "sentiment_state": snapshot.sentiment_state,
        "orderbook_imbalance": snapshot.orderbook_imbalance,
        "consensus_concentration": snapshot.consensus_concentration,
        "elite_disagreement_count": snapshot.elite_disagreement_count,
        "whale_activity_score": snapshot.whale_activity_score,
        "candlestick_state": snapshot.candlestick_state,
        "conflict_flags": snapshot.conflict_flags or {},
        "errors": list(snapshot.errors or [])[:5],
        "historical_regime_stats": historical,
    }


def _summarise_regime(row: SignalRegimeFeatures) -> dict[str, Any]:
    return {
        "line_movement_velocity": row.line_movement_velocity,
        "market_volatility": row.market_volatility,
        "consensus_concentration": row.consensus_concentration,
        "orderbook_imbalance": row.public_sharp_divergence,
        "sentiment_score": row.sentiment_score,
    }
