"""Tests for the Falcon adaptive learning subsystem."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import pytest

from app.models import (
    AdaptiveFactorWeight,
    ConfidenceBandLearning,
    Signal,
    SignalFactorAttribution,
    SignalLearningSnapshot,
    SignalWalletContribution,
    WalletBehaviorProfile,
    WalletLearningStats,
    WalletMarketSpecialization,
    WalletTierHistory,
)
from app.providers.falcon import FalconResult
from app.services.falcon_intelligence import (
    derive_behavior_profile,
    detect_conflict,
    upsert_behavior_profile,
)
from app.services.falcon_learning import (
    backfill_tracked_wallets,
    capture_signal_attribution,
    record_signal_outcome,
)
from app.services.falcon_retraining import (
    adaptive_weight_for,
    lookup_calibrated_probability,
    recompute_adaptive_factor_weights,
    recompute_confidence_bands,
    recompute_wallet_tiers,
    run_full_retraining,
)
from app.services.falcon_signal_explainer import explain_signal


# ---- shared helpers ------------------------------------------------------

WALLET_A = "0x" + "a" * 40
WALLET_B = "0x" + "b" * 40


def _signal(db, **kwargs):
    defaults = {
        "market_id": 1,
        "trader_id": None,
        "signal_type": "trusted_wallet_entry",
        "side": "BUY",
        "outcome": "YES",
        "entry_price": 0.5,
        "size_usd": 1000.0,
        "score": 70.0,
        "confidence": 0.7,
        "reason": "test",
        "source": "Falcon",
        "score_breakdown": {},
    }
    defaults.update(kwargs)
    sig = Signal(**defaults)
    db.add(sig)
    db.flush()
    return sig


# ---- backfill ------------------------------------------------------------


class _FakeFalcon:
    """Provider stub that returns canned FalconResult payloads per wallet."""

    def __init__(self, payloads: dict[str, dict[str, Any] | None]):
        self._payloads = payloads

    async def fetch_wallet_360(self, *, wallet: str, **_: Any) -> FalconResult:
        body = self._payloads.get(wallet)
        if body is None:
            return FalconResult(agent_id=581, available=False, reason="not found")
        return FalconResult(
            agent_id=581, available=True, rows=[body], summary=body, raw=body,
        )


def test_backfill_upserts_wallet_stats_and_specialisations(db_session):
    payloads = {
        WALLET_A: {
            "win_rate": 0.62,
            "roi": 18.0,
            "pnl_last_30day": 12500.0,
            "performance_by_category": [
                {"category": "Sports", "total_trades": 120, "win_rate": 0.6, "roi": 12.0},
                {"category": "Politics", "total_trades": 30, "win_rate": 0.4, "roi": -5.0},
            ],
        },
        WALLET_B: None,  # unavailable
    }
    falcon = _FakeFalcon(payloads)

    summary = asyncio.run(
        backfill_tracked_wallets(
            db_session, falcon, [(WALLET_A, "alpha"), (WALLET_B, "beta")],
        )
    )

    assert summary.wallets_seen == 2
    assert summary.wallets_backfilled == 1
    assert summary.wallets_unavailable == 1
    assert summary.specialisations_written == 2

    stats = db_session.get(WalletLearningStats, WALLET_A)
    assert stats is not None
    assert stats.wallet_name == "alpha"
    assert stats.roi == 18.0
    assert stats.total_signals == 150  # 120 + 30
    # 120*0.6 + 30*0.4 = 84 wins
    assert stats.wins == 84

    specs = list(db_session.scalars(
        WalletMarketSpecialization.__table__.select().where(
            WalletMarketSpecialization.wallet_address == WALLET_A,
        )
    ))
    # raw `select` returns Row objects; just use the count from summary above.
    by_cat = list(
        db_session.query(WalletMarketSpecialization).filter(
            WalletMarketSpecialization.wallet_address == WALLET_A,
        ).all()
    )
    assert {s.market_type for s in by_cat} == {"Sports", "Politics"}


def test_backfill_is_idempotent(db_session):
    payloads = {
        WALLET_A: {
            "win_rate": 0.55,
            "roi": 10.0,
            "performance_by_category": [
                {"category": "Sports", "total_trades": 50, "win_rate": 0.55, "roi": 10.0},
            ],
        }
    }
    falcon = _FakeFalcon(payloads)
    asyncio.run(backfill_tracked_wallets(db_session, falcon, [(WALLET_A, "alpha")]))
    asyncio.run(backfill_tracked_wallets(db_session, falcon, [(WALLET_A, "alpha")]))

    rows = list(
        db_session.query(WalletMarketSpecialization).filter(
            WalletMarketSpecialization.wallet_address == WALLET_A,
        )
    )
    assert len(rows) == 1  # second call updated in place


# ---- emit-time capture --------------------------------------------------


def test_capture_signal_attribution_persists_rows(db_session):
    sig = _signal(db_session)
    capture_signal_attribution(
        db_session,
        signal_id=sig.id,
        factors={"wallet_quality": 0.8, "multi_wallet_consensus": 0.4},
        weights={"wallet_quality": 0.35, "multi_wallet_consensus": 0.25},
        sport="basketball",
        market_type="trusted_wallet_entry",
        contributing_wallets=[
            {"wallet_address": WALLET_A, "contribution_weight": 1.0, "side": "BUY", "size_usd": 5000},
        ],
        raw_score=72.0,
    )
    db_session.commit()

    factor_rows = list(
        db_session.query(SignalFactorAttribution).filter(
            SignalFactorAttribution.signal_id == sig.id,
        )
    )
    assert {r.factor_name for r in factor_rows} == {"wallet_quality", "multi_wallet_consensus"}
    contributions = list(
        db_session.query(SignalWalletContribution).filter(
            SignalWalletContribution.signal_id == sig.id,
        )
    )
    assert len(contributions) == 1
    snapshot = db_session.get(SignalLearningSnapshot, sig.id)
    assert snapshot is not None
    assert snapshot.raw_score == 72.0
    assert snapshot.sport == "basketball"


def test_capture_is_idempotent(db_session):
    sig = _signal(db_session)
    capture_signal_attribution(
        db_session, signal_id=sig.id, factors={"liquidity": 0.5},
        weights={"liquidity": 0.15}, sport="*", market_type="*",
    )
    capture_signal_attribution(
        db_session, signal_id=sig.id, factors={"liquidity": 0.9},
        weights={"liquidity": 0.20}, sport="*", market_type="*",
    )
    db_session.commit()

    rows = list(
        db_session.query(SignalFactorAttribution).filter(
            SignalFactorAttribution.signal_id == sig.id,
        )
    )
    # Replaced, not duplicated.
    assert len(rows) == 1
    assert rows[0].factor_value == 0.9
    assert rows[0].factor_weight == 0.20


# ---- grading-time update ------------------------------------------------


def test_record_signal_outcome_updates_factor_rows_and_wallet_stats(db_session):
    sig = _signal(db_session)
    capture_signal_attribution(
        db_session, signal_id=sig.id,
        factors={"wallet_quality": 0.7, "liquidity": 0.6},
        weights={"wallet_quality": 0.35, "liquidity": 0.15},
        sport="basketball", market_type="trusted_wallet_entry",
        contributing_wallets=[
            {"wallet_address": WALLET_A, "contribution_weight": 1.0, "side": "BUY"},
        ],
    )
    db_session.commit()

    updated = record_signal_outcome(
        db_session,
        signal_id=sig.id,
        win_loss_push="win",
        realized_pnl=125.0,
        clv_points=0.03,
    )
    db_session.commit()

    assert updated == 2
    factor_rows = list(
        db_session.query(SignalFactorAttribution).filter(
            SignalFactorAttribution.signal_id == sig.id,
        )
    )
    assert all(r.win_loss_push == "win" for r in factor_rows)
    assert all(r.realized_pnl == 125.0 for r in factor_rows)

    stats = db_session.get(WalletLearningStats, WALLET_A)
    assert stats.wins == 1
    assert stats.sample_size == 1
    assert stats.confidence_weight > 0.5  # one win nudges above neutral

    # Re-running should not double-count (graded_at IS NULL filter).
    record_signal_outcome(
        db_session, signal_id=sig.id, win_loss_push="win",
        realized_pnl=125.0, clv_points=0.03,
    )
    db_session.commit()
    stats = db_session.get(WalletLearningStats, WALLET_A)
    assert stats.wins == 1


# ---- adaptive factor weight Bayesian behaviour --------------------------


def _build_graded_factor_dataset(db_session, *, signals: int, win_rate: float):
    """Helper: create N graded signals with one factor, given win rate."""
    wins_needed = int(round(signals * win_rate))
    for i in range(signals):
        sig = _signal(db_session)
        won = i < wins_needed
        capture_signal_attribution(
            db_session, signal_id=sig.id,
            factors={"wallet_quality": 0.9 if won else 0.4},
            weights={"wallet_quality": 0.35},
            sport="basketball", market_type="trusted_wallet_entry",
            raw_score=72.0,
        )
        db_session.commit()
        record_signal_outcome(
            db_session, signal_id=sig.id,
            win_loss_push="win" if won else "loss",
            realized_pnl=1.0 if won else -1.0,
            clv_points=0.02 if won else -0.02,
        )
    db_session.commit()


def test_adaptive_weights_below_min_sample_keep_static_prior(db_session):
    _build_graded_factor_dataset(db_session, signals=5, win_rate=1.0)
    summary = recompute_adaptive_factor_weights(db_session, min_sample=30)

    assert summary.rows_examined == 5
    assert summary.weights_updated == 0
    assert summary.weights_below_min_sample == 1
    weight = adaptive_weight_for(
        db_session, "wallet_quality", sport="basketball", market_type="trusted_wallet_entry",
    )
    # Below the floor → static prior 0.35.
    assert abs(weight - 0.35) < 1e-9


def test_adaptive_weights_react_with_sufficient_sample(db_session):
    _build_graded_factor_dataset(db_session, signals=60, win_rate=0.9)
    recompute_adaptive_factor_weights(db_session, min_sample=30)

    row = db_session.query(AdaptiveFactorWeight).filter(
        AdaptiveFactorWeight.factor_name == "wallet_quality",
        AdaptiveFactorWeight.sport == "basketball",
        AdaptiveFactorWeight.market_type == "trusted_wallet_entry",
    ).one()
    # Higher value paired with wins → predictive power positive → weight > static.
    assert row.sample_size == 60
    assert row.current_weight > 0.35
    # Bayesian shrinkage caps multiplicative adjustment at 0.5x..1.5x prior.
    assert row.current_weight <= 0.35 * 1.5 + 1e-9


def test_adaptive_weights_do_not_overreact_to_one_game(db_session):
    """Single graded win must not flip a factor weight wildly."""
    _build_graded_factor_dataset(db_session, signals=1, win_rate=1.0)
    recompute_adaptive_factor_weights(db_session, min_sample=30)
    weight = adaptive_weight_for(
        db_session, "wallet_quality", sport="basketball", market_type="trusted_wallet_entry",
    )
    assert abs(weight - 0.35) < 1e-9


# ---- calibration --------------------------------------------------------


def test_confidence_bands_recompute_and_lookup(db_session):
    # 10 signals in [70, 75) band — 7 wins, 3 losses.
    wins = 7
    losses = 3
    for i in range(wins + losses):
        sig = _signal(db_session)
        capture_signal_attribution(
            db_session, signal_id=sig.id,
            factors={"wallet_quality": 0.7},
            weights={"wallet_quality": 0.35},
            sport="basketball", market_type="trusted_wallet_entry",
            raw_score=72.0,
        )
        db_session.commit()
        record_signal_outcome(
            db_session, signal_id=sig.id,
            win_loss_push="win" if i < wins else "loss",
            realized_pnl=1.0 if i < wins else -1.0,
            clv_points=0.01,
        )
    db_session.commit()
    recompute_confidence_bands(db_session, sport="basketball", market_type="trusted_wallet_entry")

    cal = lookup_calibrated_probability(
        db_session, raw_score=72.0,
        sport="basketball", market_type="trusted_wallet_entry",
        min_signals=5,
    )
    assert cal is not None
    # Laplace-smoothed (7+1)/(10+2) = 0.6667
    assert abs(cal - 0.6667) < 1e-3

    # No calibration for an empty scope.
    empty = lookup_calibrated_probability(
        db_session, raw_score=72.0,
        sport="hockey", market_type="trusted_wallet_entry", min_signals=5,
    )
    assert empty is None


# ---- tiers ---------------------------------------------------------------


def test_wallet_tier_recompute_writes_history_per_recompute(db_session):
    stats = WalletLearningStats(
        wallet_address=WALLET_A,
        total_signals=40,
        wins=28,
        losses=12,
        roi=18.0,
        avg_clv=0.04,
        sample_size=40,
        confidence_weight=0.7,
    )
    db_session.add(stats)
    db_session.commit()

    recompute_wallet_tiers(db_session)
    rows = list(
        db_session.query(WalletTierHistory).filter(
            WalletTierHistory.wallet_address == WALLET_A,
            WalletTierHistory.sport.is_(None),
        )
    )
    assert len(rows) == 1
    assert rows[0].tier in {"elite", "trusted"}

    # Re-running appends a new row so movement is auditable.
    recompute_wallet_tiers(db_session)
    rows = list(
        db_session.query(WalletTierHistory).filter(
            WalletTierHistory.wallet_address == WALLET_A,
            WalletTierHistory.sport.is_(None),
        )
    )
    assert len(rows) == 2


def test_wallet_tier_respects_min_sample(db_session):
    stats = WalletLearningStats(
        wallet_address=WALLET_A,
        total_signals=2,
        wins=2,
        losses=0,
        roi=99.0,
        sample_size=2,
        confidence_weight=0.9,
    )
    db_session.add(stats)
    db_session.commit()

    summary = recompute_wallet_tiers(db_session, min_sample=10)
    assert summary.tiers_written == 0
    assert summary.skipped_insufficient_sample == 1


# ---- conflict detector --------------------------------------------------


def test_detect_conflict_flags_crowded_with_elite_disagreement():
    flags = detect_conflict(
        same_side_wallets=6,
        total_watched=8,
        elite_disagreement_count=2,
        consensus_concentration=0.9,
        orderbook_imbalance=-0.4,
    )
    assert flags["crowded_side"] is True
    assert flags["elite_disagreement"] is True
    assert flags["trap_signal"] is True
    assert flags["conviction_penalty"] > 0.25


def test_detect_conflict_quiet_when_signal_clean():
    flags = detect_conflict(
        same_side_wallets=1,
        total_watched=10,
        elite_disagreement_count=0,
    )
    assert flags["conviction_penalty"] == 0.0
    assert not flags["crowded_side"]


# ---- behaviour profile --------------------------------------------------


def test_behavior_profile_clusterer_scores_all_archetypes():
    profiles = derive_behavior_profile(
        wallet_360_summary={
            "win_rate_last_30day": 0.7,
            "profit_factor": 2.0,
            "entry_timing_score": 0.9,
            "total_trades": 200,
            "market_concentration_ratio": 0.2,
            "roi": 25.0,
        },
    )
    assert len(profiles) == 5
    by_arch = {p["archetype"]: p["score"] for p in profiles}
    # Sharp + early entry → sharp_steam should dominate.
    assert by_arch["sharp_steam"] > by_arch["late_momentum_chaser"]
    assert all(0.0 <= v <= 1.0 for v in by_arch.values())


def test_upsert_behavior_profile_writes_rows(db_session):
    profiles = derive_behavior_profile(
        wallet_360_summary={"win_rate": 0.6, "profit_factor": 1.5},
    )
    written = upsert_behavior_profile(db_session, wallet_address=WALLET_A, profiles=profiles)
    db_session.commit()
    assert written == 5
    rows = list(
        db_session.query(WalletBehaviorProfile).filter(
            WalletBehaviorProfile.wallet_address == WALLET_A,
        )
    )
    assert {r.archetype for r in rows} == {
        "sharp_steam", "late_momentum_chaser", "high_volume_low_edge",
        "contrarian_sniper", "market_maker_follower",
    }


# ---- explainer ----------------------------------------------------------


def test_explain_signal_aggregates_factors_wallets_calibration(db_session):
    sig = _signal(db_session, side="BUY")
    capture_signal_attribution(
        db_session, signal_id=sig.id,
        factors={"wallet_quality": 0.7, "liquidity": 0.5},
        weights={"wallet_quality": 0.35, "liquidity": 0.15},
        sport="basketball", market_type="trusted_wallet_entry",
        contributing_wallets=[
            {"wallet_address": WALLET_A, "contribution_weight": 1.0, "side": "SELL"},
        ],
        raw_score=72.0,
    )
    db_session.commit()
    record_signal_outcome(
        db_session, signal_id=sig.id, win_loss_push="win",
        realized_pnl=1.0, clv_points=0.02,
    )
    db_session.commit()

    payload = explain_signal(db_session, sig.id)
    assert payload is not None
    assert payload["raw_score"] == 72.0
    # Wallet contributed SELL while signal side is BUY → elite_disagreement is
    # only counted if the wallet is elite/trusted; with one win it shouldn't be
    # yet. Just assert the field exists.
    assert "elite_disagreement_count" in payload["wallets"]
    assert payload["factors"]["available"] is True
    assert len(payload["factors"]["rows"]) == 2


def test_explain_signal_returns_none_for_missing_signal(db_session):
    assert explain_signal(db_session, 9999) is None


# ---- full retraining ----------------------------------------------------


def test_full_retraining_runs_all_three_passes(db_session):
    _build_graded_factor_dataset(db_session, signals=40, win_rate=0.6)
    summary = run_full_retraining(db_session)
    assert summary.factor_weights.rows_examined == 40
    assert summary.calibration.bands_updated > 0
    # No wallets registered → tiers pass examines 0.
    assert summary.tiers.wallets_examined == 0
