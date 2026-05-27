"""Falcon learning ingestion.

Owns three responsibilities:

1. **Backfill** — pull every tracked wallet through Wallet 360 (581) and
   Polymarket PnL (569). Persists/upserts ``WalletLearningStats`` and one
   ``WalletMarketSpecialization`` row per category. Uses Falcon historical
   data so the learning system has a substrate from day one rather than
   needing to wait for future graded signals.

2. **Signal emit-time capture** — at the moment a signal is emitted, persist
   one ``SignalFactorAttribution`` row per factor and one
   ``SignalWalletContribution`` row per contributing wallet, plus a single
   ``SignalLearningSnapshot`` recording the full context.

3. **Grading-time update** — after a signal is graded (win / loss / push
   plus realised PnL / CLV), backfill the corresponding factor attribution
   rows and update the per-wallet rolling stats.

The retraining pass (``falcon_retraining``) consumes the rows this module
writes — keeping ingestion and retraining as separate services makes both
testable in isolation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    SignalFactorAttribution,
    SignalLearningSnapshot,
    SignalWalletContribution,
    WalletLearningStats,
    WalletMarketSpecialization,
)
from app.providers.falcon import FalconProvider, FalconResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------


@dataclass
class WalletBackfillSummary:
    """What ``backfill_tracked_wallets`` produced for telemetry."""

    wallets_seen: int = 0
    wallets_backfilled: int = 0
    wallets_unavailable: int = 0
    specialisations_written: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "wallets_seen": self.wallets_seen,
            "wallets_backfilled": self.wallets_backfilled,
            "wallets_unavailable": self.wallets_unavailable,
            "specialisations_written": self.specialisations_written,
            "errors": self.errors[:20],
        }


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _decode_perf_by_category(value: Any) -> list[dict[str, Any]]:
    """Wallet 360 returns ``performance_by_category`` as a JSON-encoded string
    in many builds. Be permissive: accept str → list, list → list, else []."""
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    if isinstance(value, str) and value:
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            return []
        if isinstance(decoded, list):
            return [v for v in decoded if isinstance(v, dict)]
    return []


def _upsert_wallet_stats(
    db: Session,
    *,
    wallet: str,
    wallet_name: str | None,
    summary: dict[str, Any],
) -> WalletLearningStats:
    """Idempotent insert/update of ``wallet_learning_stats`` from Wallet-360 data."""
    row = db.get(WalletLearningStats, wallet)
    perf_by_category = _decode_perf_by_category(summary.get("performance_by_category"))

    aggregated_signals = sum(_to_int(c.get("total_trades")) for c in perf_by_category)
    aggregated_wins = 0
    aggregated_clv: list[float] = []
    for cat in perf_by_category:
        trades = _to_int(cat.get("total_trades"))
        wr = _to_float(cat.get("win_rate"))
        if trades and wr is not None:
            aggregated_wins += int(round(trades * wr))
        clv = _to_float(cat.get("avg_clv"))
        if clv is not None:
            aggregated_clv.append(clv)

    win_rate_overall = _to_float(summary.get("win_rate")) or _to_float(
        summary.get("win_rate_last_30day")
    )
    roi = _to_float(summary.get("roi"))
    pnl = _to_float(summary.get("pnl_last_30day")) or _to_float(summary.get("total_pnl"))
    avg_clv = (
        round(sum(aggregated_clv) / len(aggregated_clv), 4) if aggregated_clv else None
    )

    if row is None:
        row = WalletLearningStats(
            wallet_address=wallet,
            wallet_name=wallet_name,
            total_signals=aggregated_signals,
            wins=aggregated_wins,
            losses=max(aggregated_signals - aggregated_wins, 0) if aggregated_signals else 0,
            pushes=0,
            realized_pnl=pnl or 0.0,
            roi=roi,
            avg_clv=avg_clv,
            sample_size=aggregated_signals,
            confidence_weight=_bayes_confidence_weight(aggregated_signals, win_rate_overall),
            last_seen=datetime.utcnow(),
        )
        db.add(row)
        return row
    # Update in place — keep wins/losses additive so future grading still
    # contributes; backfill only refreshes the Falcon-derived view.
    row.wallet_name = wallet_name or row.wallet_name
    row.realized_pnl = pnl if pnl is not None else row.realized_pnl
    row.roi = roi if roi is not None else row.roi
    if avg_clv is not None:
        row.avg_clv = avg_clv
    if aggregated_signals > row.sample_size:
        # Trust the larger Falcon sample.
        row.total_signals = aggregated_signals
        row.wins = aggregated_wins
        row.losses = max(aggregated_signals - aggregated_wins, 0)
        row.sample_size = aggregated_signals
        row.confidence_weight = _bayes_confidence_weight(aggregated_signals, win_rate_overall)
    row.last_seen = datetime.utcnow()
    return row


def _upsert_specialization(
    db: Session,
    *,
    wallet: str,
    sport: str,
    market_type: str,
    cat_payload: dict[str, Any],
) -> WalletMarketSpecialization:
    row = db.scalar(
        select(WalletMarketSpecialization).where(
            WalletMarketSpecialization.wallet_address == wallet,
            WalletMarketSpecialization.sport == sport,
            WalletMarketSpecialization.market_type == market_type,
        )
    )
    signals = _to_int(cat_payload.get("total_trades"))
    win_rate = _to_float(cat_payload.get("win_rate"))
    roi = _to_float(cat_payload.get("roi"))
    avg_clv = _to_float(cat_payload.get("avg_clv"))
    side_bias = _to_float(cat_payload.get("favorite_pct") or cat_payload.get("side_bias"))
    volatility = _to_float(cat_payload.get("volatility") or cat_payload.get("variance"))

    if row is None:
        row = WalletMarketSpecialization(
            wallet_address=wallet,
            sport=sport,
            market_type=market_type,
            signals=signals,
            roi=roi,
            win_rate=win_rate,
            avg_clv=avg_clv,
            side_bias=side_bias,
            volatility_score=volatility,
        )
        db.add(row)
    else:
        # Trust the larger sample.
        if signals >= row.signals:
            row.signals = signals
            row.roi = roi if roi is not None else row.roi
            row.win_rate = win_rate if win_rate is not None else row.win_rate
            row.avg_clv = avg_clv if avg_clv is not None else row.avg_clv
            row.side_bias = side_bias if side_bias is not None else row.side_bias
            row.volatility_score = volatility if volatility is not None else row.volatility_score
    return row


def _bayes_confidence_weight(sample_size: int, win_rate: float | None) -> float:
    """Bayesian-smoothed confidence in [0, 1].

    Beta(2,2) prior keeps tiny samples near 0.5 and avoids overreaction to
    one or two graded signals. As the sample grows, the smoothed value
    converges to the observed win rate.
    """
    if win_rate is None:
        return 0.5
    alpha = 2.0
    beta = 2.0
    wins = win_rate * sample_size
    losses = sample_size - wins
    smoothed = (wins + alpha) / max(wins + losses + alpha + beta, 1e-9)
    return round(max(0.0, min(1.0, smoothed)), 4)


async def backfill_tracked_wallets(
    db: Session,
    falcon: FalconProvider,
    wallets: Iterable[tuple[str, str | None]],
) -> WalletBackfillSummary:
    """Pull Wallet-360 + PnL for each wallet and persist learning rows.

    Each ``wallets`` entry is ``(wallet_address, wallet_name_or_None)``. The
    caller is responsible for filtering to 0x-addresses — this function skips
    non-addresses defensively but won't complain about them.
    """
    summary = WalletBackfillSummary()
    for wallet, name in wallets:
        if not (wallet and wallet.startswith("0x") and len(wallet) == 42):
            continue
        summary.wallets_seen += 1
        result: FalconResult = await falcon.fetch_wallet_360(wallet=wallet)
        if not result.available or result.summary is None:
            summary.wallets_unavailable += 1
            if result.reason:
                summary.errors.append(f"{wallet[:10]}…: {result.reason}")
            continue
        stats_row = _upsert_wallet_stats(
            db,
            wallet=wallet,
            wallet_name=name or result.summary.get("wallet_name"),
            summary=result.summary,
        )
        summary.wallets_backfilled += 1

        # Per-category specialisation. Falcon doesn't separate sport/market_type
        # cleanly, so we use the category string as ``market_type`` and the
        # provider's coarse ``sport`` field when present.
        for cat in _decode_perf_by_category(result.summary.get("performance_by_category")):
            category = str(cat.get("category") or "unknown").strip() or "unknown"
            sport = str(cat.get("sport") or "*")
            _upsert_specialization(
                db,
                wallet=wallet,
                sport=sport,
                market_type=category,
                cat_payload=cat,
            )
            summary.specialisations_written += 1
    db.commit()
    logger.info("Wallet backfill: %s", summary.as_dict())
    return summary


# ---------------------------------------------------------------------------
# Signal emit-time capture
# ---------------------------------------------------------------------------


def capture_signal_attribution(
    db: Session,
    *,
    signal_id: int,
    factors: dict[str, float],
    weights: dict[str, float],
    sport: str | None = None,
    market_type: str | None = None,
    contributing_wallets: list[dict[str, Any]] | None = None,
    raw_score: float | None = None,
    calibrated_probability: float | None = None,
    regime_payload: dict[str, Any] | None = None,
    conflict_payload: dict[str, Any] | None = None,
) -> None:
    """Persist per-factor, per-wallet, and snapshot rows for a fresh signal.

    Idempotent: re-calling for the same ``signal_id`` replaces existing rows
    so a re-emit (e.g. from a retry) doesn't double-count contributions.
    """
    # Force the session to materialise any pending inserts so the followup
    # bulk deletes see a consistent view. Then wipe prior emit-time rows
    # (graded rows are preserved) so a re-emit doesn't double-count.
    db.flush()
    db.query(SignalFactorAttribution).filter(
        SignalFactorAttribution.signal_id == signal_id,
        SignalFactorAttribution.graded_at.is_(None),
    ).delete(synchronize_session="fetch")
    db.query(SignalWalletContribution).filter(
        SignalWalletContribution.signal_id == signal_id,
    ).delete(synchronize_session="fetch")
    db.flush()

    for name, value in factors.items():
        db.add(
            SignalFactorAttribution(
                signal_id=signal_id,
                factor_name=name,
                factor_value=float(value),
                factor_weight=float(weights.get(name, 0.0)),
                sport=sport,
                market_type=market_type,
            )
        )
    for wallet in contributing_wallets or []:
        addr = wallet.get("wallet_address") or wallet.get("wallet")
        if not addr:
            continue
        db.add(
            SignalWalletContribution(
                signal_id=signal_id,
                wallet_address=str(addr),
                contribution_weight=float(wallet.get("contribution_weight") or 0.0),
                side=wallet.get("side"),
                size_usd=_to_float(wallet.get("size_usd")),
                entry_price=_to_float(wallet.get("entry_price")),
            )
        )

    # Upsert the snapshot row in place so unique-PK conflicts can't fire.
    existing_snapshot = db.get(SignalLearningSnapshot, signal_id)
    snapshot_kwargs = dict(
        sport=sport,
        market_type=market_type,
        raw_score=raw_score,
        calibrated_probability=calibrated_probability,
        factor_payload={"factors": factors, "weights": weights},
        regime_payload=regime_payload or {},
        conflict_payload=conflict_payload or {},
    )
    if existing_snapshot is None:
        db.add(SignalLearningSnapshot(signal_id=signal_id, **snapshot_kwargs))
    else:
        for key, value in snapshot_kwargs.items():
            setattr(existing_snapshot, key, value)


# ---------------------------------------------------------------------------
# Grading-time update
# ---------------------------------------------------------------------------


def record_signal_outcome(
    db: Session,
    *,
    signal_id: int,
    win_loss_push: str,
    realized_pnl: float | None,
    clv_points: float | None,
    contributing_wallets: list[str] | None = None,
) -> int:
    """Backfill outcome on every factor-attribution row for the signal and
    bump per-wallet rolling stats. Returns the number of factor rows updated.

    Idempotent — re-running with the same outcome leaves stats unchanged
    because we operate on ``graded_at IS NULL`` rows only.
    """
    rows = list(
        db.scalars(
            select(SignalFactorAttribution).where(
                SignalFactorAttribution.signal_id == signal_id,
                SignalFactorAttribution.graded_at.is_(None),
            )
        )
    )
    if not rows:
        # Already graded (or never captured) — nothing to do. Idempotent
        # so the dashboard / scheduler can retry safely.
        return 0

    now = datetime.utcnow()
    outcome = win_loss_push.lower()
    for row in rows:
        row.win_loss_push = outcome
        row.realized_pnl = realized_pnl
        row.clv_points = clv_points
        row.graded_at = now

    if not contributing_wallets:
        contributing_wallets = [
            w
            for (w,) in db.execute(
                select(SignalWalletContribution.wallet_address).where(
                    SignalWalletContribution.signal_id == signal_id
                )
            ).all()
        ]

    for addr in contributing_wallets:
        stats = db.get(WalletLearningStats, addr)
        if stats is None:
            stats = WalletLearningStats(wallet_address=addr, sample_size=0)
            db.add(stats)
            db.flush()
        stats.total_signals = (stats.total_signals or 0) + 1
        stats.sample_size = (stats.sample_size or 0) + 1
        if outcome == "win":
            stats.wins = (stats.wins or 0) + 1
        elif outcome == "loss":
            stats.losses = (stats.losses or 0) + 1
        elif outcome == "push":
            stats.pushes = (stats.pushes or 0) + 1
        if realized_pnl is not None:
            stats.realized_pnl = (stats.realized_pnl or 0.0) + float(realized_pnl)
        decided = (stats.wins or 0) + (stats.losses or 0)
        win_rate = (stats.wins or 0) / decided if decided else None
        stats.confidence_weight = _bayes_confidence_weight(decided, win_rate)
        if clv_points is not None:
            prior = stats.avg_clv or 0.0
            n = max(stats.sample_size, 1)
            stats.avg_clv = round(prior + (float(clv_points) - prior) / n, 4)
        stats.last_seen = now

    return len(rows)
