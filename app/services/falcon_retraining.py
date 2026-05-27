"""Falcon retraining / reweighting pipeline.

Recomputes:

* **Adaptive factor weights** — rolling ROI and predictive power per
  ``(factor_name, sport, market_type)`` triple, Bayesian-smoothed against
  the static prior so a small sample can't dominate.
* **Confidence calibration bands** — score-band to realised win-rate.
* **Wallet tiers** — elite / trusted / neutral / weak / fade, with one
  append-only ``wallet_tier_history`` row per recompute so movement is
  auditable.

Every recompute is idempotent. Re-running on the same data produces the
same outputs. Sample-size guards (``min_sample``) prevent tier flips on a
single graded signal.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import (
    AdaptiveFactorWeight,
    ConfidenceBandLearning,
    RegimeLearningStats,
    SignalFactorAttribution,
    SignalRegimeSnapshot,
    WalletLearningStats,
    WalletMarketSpecialization,
    WalletTierHistory,
)

logger = logging.getLogger(__name__)


# --- helpers --------------------------------------------------------------


def _bayes_mean(values: list[float], *, prior_mean: float, prior_weight: float) -> float:
    """Shrinkage to a prior. With ``prior_weight=0`` this is the raw mean;
    with ``prior_weight≫len(values)`` it stays near ``prior_mean``."""
    if not values and prior_weight <= 0:
        return prior_mean
    s = sum(values) + prior_mean * prior_weight
    n = len(values) + prior_weight
    return s / n if n else prior_mean


def _predictive_power(values: list[float], outcomes_pnl: list[float]) -> float | None:
    """Point-biserial-style correlation between a factor value (0..1) and a
    realised-PnL sign. Returns a number in [-1, 1] or None when sample is
    too small / no variance."""
    n = len(values)
    if n < 5 or n != len(outcomes_pnl):
        return None
    mean_v = sum(values) / n
    mean_o = sum(outcomes_pnl) / n
    num = sum((v - mean_v) * (o - mean_o) for v, o in zip(values, outcomes_pnl))
    den_v = math.sqrt(sum((v - mean_v) ** 2 for v in values))
    den_o = math.sqrt(sum((o - mean_o) ** 2 for o in outcomes_pnl))
    if den_v <= 0 or den_o <= 0:
        return None
    return round(max(-1.0, min(1.0, num / (den_v * den_o))), 4)


# --- adaptive factor weights ---------------------------------------------


@dataclass
class WeightRetrainSummary:
    rows_examined: int = 0
    weights_updated: int = 0
    weights_below_min_sample: int = 0
    triples_seen: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows_examined": self.rows_examined,
            "weights_updated": self.weights_updated,
            "weights_below_min_sample": self.weights_below_min_sample,
            "triples_seen": self.triples_seen,
        }


def recompute_adaptive_factor_weights(
    db: Session,
    *,
    min_sample: int = 30,
    smoothing_prior_weight: float = 20.0,
) -> WeightRetrainSummary:
    """Recompute one ``AdaptiveFactorWeight`` row per (factor, sport, market_type).

    ``min_sample`` is the absolute floor — below it ``current_weight`` falls
    back to the static prior (1.0). ``smoothing_prior_weight`` controls how
    fast the weight diverges from the prior: larger = slower adaptation.
    """
    summary = WeightRetrainSummary()
    rows = list(
        db.scalars(
            select(SignalFactorAttribution).where(
                SignalFactorAttribution.graded_at.is_not(None)
            )
        )
    )
    summary.rows_examined = len(rows)
    static_weights = _static_weights_lookup()

    buckets: dict[tuple[str, str, str], list[SignalFactorAttribution]] = {}
    for row in rows:
        key = (
            row.factor_name,
            row.sport or "*",
            row.market_type or "*",
        )
        buckets.setdefault(key, []).append(row)
    summary.triples_seen = len(buckets)

    for (factor_name, sport, market_type), entries in buckets.items():
        pnl_list = [
            float(e.realized_pnl) for e in entries if e.realized_pnl is not None
        ]
        outcome_sign_list = [
            1.0 if (e.win_loss_push or "").lower() == "win"
            else (-1.0 if (e.win_loss_push or "").lower() == "loss" else 0.0)
            for e in entries
        ]
        values = [float(e.factor_value) for e in entries]
        rolling_roi = _bayes_mean(
            pnl_list, prior_mean=0.0, prior_weight=smoothing_prior_weight,
        ) if pnl_list else None
        rolling_clv = _bayes_mean(
            [float(e.clv_points) for e in entries if e.clv_points is not None],
            prior_mean=0.0, prior_weight=smoothing_prior_weight,
        ) if any(e.clv_points is not None for e in entries) else None
        predictive = _predictive_power(values, outcome_sign_list)

        static = static_weights.get(factor_name, 1.0)
        sample = len(entries)
        if sample < min_sample:
            # Below the floor: keep the static prior; record what we have for
            # diagnostics but do not adjust the live weight.
            summary.weights_below_min_sample += 1
            current_weight = static
            confidence = 0.0
        else:
            # ``adjustment`` is a multiplicative factor in [0.5, 1.5] driven
            # by predictive power. Bayesian smoothing toward 1.0 keeps the
            # learned weight gentle; large samples earn larger swings.
            adj_signal = predictive or 0.0
            shrink = sample / (sample + smoothing_prior_weight)
            adjustment = 1.0 + shrink * adj_signal * 0.5
            current_weight = round(static * max(0.5, min(1.5, adjustment)), 4)
            confidence = round(min(1.0, sample / (sample + smoothing_prior_weight)), 4)
            summary.weights_updated += 1

        existing = db.scalar(
            select(AdaptiveFactorWeight).where(
                AdaptiveFactorWeight.factor_name == factor_name,
                AdaptiveFactorWeight.sport == sport,
                AdaptiveFactorWeight.market_type == market_type,
            )
        )
        if existing is None:
            db.add(
                AdaptiveFactorWeight(
                    factor_name=factor_name,
                    sport=sport,
                    market_type=market_type,
                    rolling_roi=rolling_roi,
                    rolling_clv=rolling_clv,
                    predictive_power=predictive,
                    confidence=confidence,
                    sample_size=sample,
                    current_weight=current_weight,
                )
            )
        else:
            existing.rolling_roi = rolling_roi
            existing.rolling_clv = rolling_clv
            existing.predictive_power = predictive
            existing.confidence = confidence
            existing.sample_size = sample
            existing.current_weight = current_weight
    db.commit()
    logger.info("Adaptive factor weight recompute: %s", summary.as_dict())
    return summary


def _static_weights_lookup() -> dict[str, float]:
    """The static ScoringWeights prior, exposed by factor name.

    The adaptive layer modulates these multiplicatively — never overwrites
    them — so the static config is still the floor / ceiling reference.
    """
    sw = get_settings().scoring
    return {
        "wallet_quality": sw.wallet_quality,
        "multi_wallet_consensus": sw.multi_wallet_consensus,
        "liquidity": sw.liquidity,
        "entry_timing": sw.entry_timing,
        "price_inefficiency": sw.price_inefficiency,
    }


def adaptive_weight_for(
    db: Session,
    factor_name: str,
    *,
    sport: str | None = None,
    market_type: str | None = None,
    min_sample: int = 30,
) -> float:
    """Return the adaptive weight for a factor, falling back to the static
    prior when the sample for the specific scope is below ``min_sample``.

    Lookup order: ``(factor, sport, market_type)`` → ``(factor, sport, *)``
    → ``(factor, *, *)`` → static prior.
    """
    static = _static_weights_lookup().get(factor_name, 1.0)
    scopes: list[tuple[str, str]] = []
    if sport and market_type:
        scopes.append((sport, market_type))
    if sport:
        scopes.append((sport, "*"))
    scopes.append(("*", "*"))
    for s, m in scopes:
        row = db.scalar(
            select(AdaptiveFactorWeight).where(
                AdaptiveFactorWeight.factor_name == factor_name,
                AdaptiveFactorWeight.sport == s,
                AdaptiveFactorWeight.market_type == m,
            )
        )
        if row and row.sample_size >= min_sample:
            return float(row.current_weight)
    return static


# --- calibration bands ---------------------------------------------------


_DEFAULT_BANDS: tuple[tuple[float, float], ...] = (
    (0, 50),
    (50, 60),
    (60, 65),
    (65, 70),
    (70, 75),
    (75, 80),
    (80, 85),
    (85, 90),
    (90, 100.0001),
)


@dataclass
class CalibrationSummary:
    bands_updated: int = 0
    rows_examined: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"bands_updated": self.bands_updated, "rows_examined": self.rows_examined}


def recompute_confidence_bands(
    db: Session,
    *,
    sport: str = "*",
    market_type: str = "*",
    bands: Iterable[tuple[float, float]] = _DEFAULT_BANDS,
) -> CalibrationSummary:
    """Recompute one calibration row per band.

    Pulls every graded ``SignalFactorAttribution`` row tagged with the
    requested ``sport``/``market_type`` scope, aggregates by the originating
    signal (so each signal contributes once), and persists realised win-rate
    per band.
    """
    summary = CalibrationSummary()
    # We need (signal_id, raw_score, win_loss_push). The factor table has
    # signal_id + win_loss_push; raw score lives in SignalLearningSnapshot.
    from app.models import SignalLearningSnapshot  # local import avoids cycle

    snap_q = select(
        SignalLearningSnapshot.signal_id,
        SignalLearningSnapshot.raw_score,
    ).where(SignalLearningSnapshot.raw_score.is_not(None))
    if sport != "*":
        snap_q = snap_q.where(SignalLearningSnapshot.sport == sport)
    if market_type != "*":
        snap_q = snap_q.where(SignalLearningSnapshot.market_type == market_type)
    score_by_signal = {sig_id: float(score) for sig_id, score in db.execute(snap_q).all()}

    outcomes_q = select(
        SignalFactorAttribution.signal_id,
        SignalFactorAttribution.win_loss_push,
        SignalFactorAttribution.realized_pnl,
        SignalFactorAttribution.clv_points,
    ).where(SignalFactorAttribution.graded_at.is_not(None))
    outcomes_by_signal: dict[int, dict[str, Any]] = {}
    for sig_id, wlp, pnl, clv in db.execute(outcomes_q).all():
        if sig_id not in outcomes_by_signal:
            outcomes_by_signal[sig_id] = {"wlp": wlp, "pnl": pnl, "clv": clv}
    summary.rows_examined = len(outcomes_by_signal)

    for lo, hi in bands:
        wins = losses = pushes = 0
        roi_values: list[float] = []
        clv_values: list[float] = []
        for sig_id, score in score_by_signal.items():
            if not (lo <= score < hi):
                continue
            outcome = outcomes_by_signal.get(sig_id)
            if not outcome:
                continue
            wlp = (outcome["wlp"] or "").lower()
            if wlp == "win":
                wins += 1
            elif wlp == "loss":
                losses += 1
            elif wlp == "push":
                pushes += 1
            if outcome["pnl"] is not None:
                roi_values.append(float(outcome["pnl"]))
            if outcome["clv"] is not None:
                clv_values.append(float(outcome["clv"]))
        decided = wins + losses
        win_rate = round(wins / decided, 4) if decided else None
        # Laplace-smoothed calibrated probability — keeps narrow-band cells
        # honest when the sample is small.
        calibrated = round((wins + 1) / max(decided + 2, 1), 4) if decided else None
        roi = round(sum(roi_values) / len(roi_values), 4) if roi_values else None
        avg_clv = round(sum(clv_values) / len(clv_values), 4) if clv_values else None

        existing = db.scalar(
            select(ConfidenceBandLearning).where(
                ConfidenceBandLearning.sport == sport,
                ConfidenceBandLearning.market_type == market_type,
                ConfidenceBandLearning.score_min == lo,
                ConfidenceBandLearning.score_max == hi,
            )
        )
        if existing is None:
            db.add(
                ConfidenceBandLearning(
                    sport=sport,
                    market_type=market_type,
                    score_min=lo,
                    score_max=hi,
                    signals=decided + pushes,
                    win_rate=win_rate,
                    roi=roi,
                    avg_clv=avg_clv,
                    calibrated_probability=calibrated,
                )
            )
        else:
            existing.signals = decided + pushes
            existing.win_rate = win_rate
            existing.roi = roi
            existing.avg_clv = avg_clv
            existing.calibrated_probability = calibrated
        summary.bands_updated += 1
    db.commit()
    logger.info("Confidence band recompute: %s", summary.as_dict())
    return summary


def lookup_calibrated_probability(
    db: Session,
    *,
    raw_score: float,
    sport: str | None = None,
    market_type: str | None = None,
    min_signals: int = 10,
) -> float | None:
    """Best-match calibrated probability for a score.

    Lookup order: specific scope → sport-only → global. Returns ``None``
    when no scope has enough data (so the dashboard can show the raw score
    rather than a fabricated calibration).
    """
    scopes = []
    if sport and market_type:
        scopes.append((sport, market_type))
    if sport:
        scopes.append((sport, "*"))
    scopes.append(("*", "*"))
    for s, m in scopes:
        row = db.scalar(
            select(ConfidenceBandLearning).where(
                ConfidenceBandLearning.sport == s,
                ConfidenceBandLearning.market_type == m,
                ConfidenceBandLearning.score_min <= raw_score,
                ConfidenceBandLearning.score_max > raw_score,
            )
        )
        if row and (row.signals or 0) >= min_signals and row.calibrated_probability is not None:
            return float(row.calibrated_probability)
    return None


# --- wallet tiers --------------------------------------------------------


@dataclass
class TierRecomputeSummary:
    wallets_examined: int = 0
    tiers_written: int = 0
    skipped_insufficient_sample: int = 0
    distribution: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "wallets_examined": self.wallets_examined,
            "tiers_written": self.tiers_written,
            "skipped_insufficient_sample": self.skipped_insufficient_sample,
            "distribution": self.distribution,
        }


def _tier_from_stats(
    *,
    roi: float | None,
    confidence_weight: float,
    sample_size: int,
    avg_clv: float | None,
    min_sample: int = 10,
) -> tuple[str, str]:
    """Map smoothed stats → tier label. Returns ``(tier, reason)``."""
    if sample_size < min_sample:
        return "neutral", f"sample below min ({sample_size} < {min_sample})"
    score = (confidence_weight - 0.5) * 100
    if roi is not None:
        score += max(min(roi * 10, 25), -25)
    if avg_clv is not None:
        score += max(min(avg_clv * 100, 10), -10)
    if score >= 25:
        return "elite", f"score={score:.1f}"
    if score >= 10:
        return "trusted", f"score={score:.1f}"
    if score >= -5:
        return "neutral", f"score={score:.1f}"
    if score >= -15:
        return "weak", f"score={score:.1f}"
    return "fade", f"score={score:.1f}"


def recompute_wallet_tiers(
    db: Session,
    *,
    min_sample: int = 10,
) -> TierRecomputeSummary:
    """Recompute global wallet tiers and append one history row per wallet.

    Per-(sport, market_type) tiers come from the specialisation table and
    are appended in the same pass — that's what lets a wallet be "elite in
    MLB totals, weak in NBA spreads".
    """
    summary = TierRecomputeSummary()
    wallets = list(db.scalars(select(WalletLearningStats)))
    summary.wallets_examined = len(wallets)
    now = datetime.utcnow()

    for stats in wallets:
        tier, reason = _tier_from_stats(
            roi=stats.roi,
            confidence_weight=float(stats.confidence_weight or 0.5),
            sample_size=int(stats.sample_size or 0),
            avg_clv=stats.avg_clv,
            min_sample=min_sample,
        )
        if (stats.sample_size or 0) < min_sample:
            summary.skipped_insufficient_sample += 1
        else:
            db.add(
                WalletTierHistory(
                    wallet_address=stats.wallet_address,
                    tier=tier,
                    sport=None,
                    market_type=None,
                    rolling_roi=stats.roi,
                    rolling_clv=stats.avg_clv,
                    sample_size=stats.sample_size,
                    reason=reason,
                    captured_at=now,
                )
            )
            summary.tiers_written += 1
            summary.distribution[tier] = summary.distribution.get(tier, 0) + 1

        # Per-market specialisation tiers.
        specs = list(
            db.scalars(
                select(WalletMarketSpecialization).where(
                    WalletMarketSpecialization.wallet_address == stats.wallet_address
                )
            )
        )
        for spec in specs:
            if (spec.signals or 0) < min_sample:
                continue
            sport_tier, sport_reason = _tier_from_stats(
                roi=spec.roi,
                confidence_weight=float(spec.win_rate or 0.5),
                sample_size=int(spec.signals or 0),
                avg_clv=spec.avg_clv,
                min_sample=min_sample,
            )
            db.add(
                WalletTierHistory(
                    wallet_address=spec.wallet_address,
                    tier=sport_tier,
                    sport=spec.sport,
                    market_type=spec.market_type,
                    rolling_roi=spec.roi,
                    rolling_clv=spec.avg_clv,
                    sample_size=spec.signals,
                    reason=sport_reason,
                    captured_at=now,
                )
            )
            summary.tiers_written += 1
    db.commit()
    logger.info("Wallet tier recompute: %s", summary.as_dict())
    return summary


# --- regime learning -----------------------------------------------------


@dataclass
class RegimeRetrainSummary:
    classifications_seen: int = 0
    rows_written: int = 0
    rows_examined: int = 0
    distribution: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "classifications_seen": self.classifications_seen,
            "rows_written": self.rows_written,
            "rows_examined": self.rows_examined,
            "distribution": self.distribution,
        }


def recompute_regime_learning_stats(db: Session) -> RegimeRetrainSummary:
    """Aggregate per-regime realised performance from graded factor rows.

    A snapshot contributes once even if its signal has many factor rows —
    we group factor rows by ``signal_id`` and pick the first (outcomes are
    consistent within a signal). Only graded signals count; snapshots
    without a graded outcome are ignored.
    """
    summary = RegimeRetrainSummary()
    # snapshot.signal_id → classification (+ sport/market_type from the
    # learning-snapshot table when available).
    from app.models import SignalLearningSnapshot

    snap_rows = list(
        db.execute(
            select(
                SignalRegimeSnapshot.signal_id,
                SignalRegimeSnapshot.regime_classification,
            ).where(SignalRegimeSnapshot.regime_classification.is_not(None))
        ).all()
    )
    classification_by_signal: dict[int, str] = {
        sig_id: cls for sig_id, cls in snap_rows if cls
    }
    scope_q = select(
        SignalLearningSnapshot.signal_id,
        SignalLearningSnapshot.sport,
        SignalLearningSnapshot.market_type,
    ).where(SignalLearningSnapshot.signal_id.in_(classification_by_signal.keys()))
    scopes_by_signal: dict[int, tuple[str, str]] = {
        sig_id: (sport or "*", market_type or "*")
        for sig_id, sport, market_type in db.execute(scope_q).all()
    }

    # One factor row per signal is enough for the outcome — pick any.
    outcomes_q = select(
        SignalFactorAttribution.signal_id,
        SignalFactorAttribution.win_loss_push,
        SignalFactorAttribution.realized_pnl,
        SignalFactorAttribution.clv_points,
    ).where(
        SignalFactorAttribution.signal_id.in_(classification_by_signal.keys()),
        SignalFactorAttribution.graded_at.is_not(None),
    )
    outcomes_by_signal: dict[int, dict[str, Any]] = {}
    for sig_id, wlp, pnl, clv in db.execute(outcomes_q).all():
        outcomes_by_signal.setdefault(
            sig_id, {"wlp": wlp, "pnl": pnl, "clv": clv}
        )
    summary.rows_examined = len(outcomes_by_signal)

    # Bucket by (classification, sport, market_type).
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for sig_id, classification in classification_by_signal.items():
        outcome = outcomes_by_signal.get(sig_id)
        if not outcome:
            continue
        sport, market_type = scopes_by_signal.get(sig_id, ("*", "*"))
        buckets.setdefault(
            (classification, sport, market_type), []
        ).append(outcome)
    summary.classifications_seen = len({k[0] for k in buckets})

    for (classification, sport, market_type), entries in buckets.items():
        wins = sum(1 for e in entries if (e["wlp"] or "").lower() == "win")
        losses = sum(1 for e in entries if (e["wlp"] or "").lower() == "loss")
        pushes = sum(1 for e in entries if (e["wlp"] or "").lower() == "push")
        decided = wins + losses
        pnls = [float(e["pnl"]) for e in entries if e["pnl"] is not None]
        clvs = [float(e["clv"]) for e in entries if e["clv"] is not None]
        avg_roi = round(sum(pnls) / len(pnls), 4) if pnls else None
        avg_clv = round(sum(clvs) / len(clvs), 4) if clvs else None
        positive_clv = sum(1 for c in clvs if c > 0)
        positive_clv_rate = round(positive_clv / len(clvs), 4) if clvs else None
        win_rate = round(wins / decided, 4) if decided else None

        existing = db.scalar(
            select(RegimeLearningStats).where(
                RegimeLearningStats.regime_classification == classification,
                RegimeLearningStats.sport == sport,
                RegimeLearningStats.market_type == market_type,
            )
        )
        if existing is None:
            existing = RegimeLearningStats(
                regime_classification=classification,
                sport=sport,
                market_type=market_type,
            )
            db.add(existing)
        existing.signals = wins + losses + pushes
        existing.wins = wins
        existing.losses = losses
        existing.pushes = pushes
        existing.avg_roi = avg_roi
        existing.avg_clv = avg_clv
        existing.positive_clv_rate = positive_clv_rate
        existing.win_rate = win_rate
        summary.rows_written += 1
        summary.distribution[classification] = summary.distribution.get(classification, 0) + (wins + losses + pushes)
    db.commit()
    logger.info("Regime learning recompute: %s", summary.as_dict())
    return summary


def lookup_regime_stats(
    db: Session,
    *,
    classification: str,
    sport: str | None = None,
    market_type: str | None = None,
    min_signals: int = 5,
) -> dict[str, Any] | None:
    """Best-match regime stats row. Falls back from specific to global scope."""
    scopes = []
    if sport and market_type:
        scopes.append((sport, market_type))
    if sport:
        scopes.append((sport, "*"))
    scopes.append(("*", "*"))
    for s, m in scopes:
        row = db.scalar(
            select(RegimeLearningStats).where(
                RegimeLearningStats.regime_classification == classification,
                RegimeLearningStats.sport == s,
                RegimeLearningStats.market_type == m,
            )
        )
        if row and (row.signals or 0) >= min_signals:
            return {
                "regime_classification": row.regime_classification,
                "sport": row.sport,
                "market_type": row.market_type,
                "signals": row.signals,
                "win_rate": row.win_rate,
                "avg_roi": row.avg_roi,
                "avg_clv": row.avg_clv,
                "positive_clv_rate": row.positive_clv_rate,
            }
    return None


# --- top-level orchestrator ----------------------------------------------


@dataclass
class FullRetrainingSummary:
    factor_weights: WeightRetrainSummary = field(default_factory=WeightRetrainSummary)
    calibration: CalibrationSummary = field(default_factory=CalibrationSummary)
    tiers: TierRecomputeSummary = field(default_factory=TierRecomputeSummary)
    regimes: RegimeRetrainSummary = field(default_factory=RegimeRetrainSummary)
    completed_at: datetime = field(default_factory=datetime.utcnow)

    def as_dict(self) -> dict[str, Any]:
        return {
            "factor_weights": self.factor_weights.as_dict(),
            "calibration": self.calibration.as_dict(),
            "tiers": self.tiers.as_dict(),
            "regimes": self.regimes.as_dict(),
            "completed_at": self.completed_at.isoformat(),
        }


def run_full_retraining(db: Session) -> FullRetrainingSummary:
    """One-shot retraining pass. Safe to call any time; idempotent."""
    return FullRetrainingSummary(
        factor_weights=recompute_adaptive_factor_weights(db),
        calibration=recompute_confidence_bands(db),
        tiers=recompute_wallet_tiers(db),
        regimes=recompute_regime_learning_stats(db),
    )
