"""Signal generation rules.

The engine inspects recent trades + market state and emits Signal rows for any
of the following conditions:

  * trusted_wallet_entry       — a watched wallet entered a position
  * multi_wallet_consensus     — 2+ watched wallets bought the same side
  * size_threshold             — a single position crossed a USD size threshold
  * post_entry_price_move      — price moved meaningfully after a smart entry
  * cross_market_price_gap     — equivalent markets disagree across venues

Each emitted Signal carries a scored breakdown + a human-readable reason.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Market, MarketSnapshot, Signal, Trade, Trader
from app.providers.base import BaseProvider, ProviderSource
from app.services import scoring

logger = logging.getLogger(__name__)

# Recent = last 24h for the multi-wallet / size rules.
_RECENT_WINDOW = timedelta(hours=24)
# A single position counts as "large" above this USD value.
_LARGE_POSITION_USD = 5_000.0
# Post-entry price move that qualifies as "validated".
_PRICE_MOVE_THRESHOLD = 0.03


@dataclass
class SignalCandidate:
    market_id: int
    trader_id: int | None
    signal_type: str
    side: str | None
    outcome: str | None
    entry_price: float | None
    size_usd: float | None
    reason: str
    source: str = ProviderSource.MOCK.value
    score: float = 0.0
    confidence: float = 0.0
    score_breakdown: dict[str, Any] = field(default_factory=dict)

    def to_model(self) -> Signal:
        return Signal(
            market_id=self.market_id,
            trader_id=self.trader_id,
            signal_type=self.signal_type,
            side=self.side,
            outcome=self.outcome,
            entry_price=self.entry_price,
            size_usd=self.size_usd,
            score=self.score,
            confidence=self.confidence,
            reason=self.reason,
            source=self.source,
            score_breakdown=self.score_breakdown,
            created_at=datetime.utcnow(),
        )


def _recent_trades(db: Session) -> list[Trade]:
    since = datetime.utcnow() - _RECENT_WINDOW
    return list(
        db.scalars(
            select(Trade).where(Trade.timestamp >= since).order_by(Trade.timestamp.desc())
        )
    )


def _has_ambiguous_falcon_outcome(trade: Trade) -> bool:
    return (
        trade.source == ProviderSource.FALCON.value
        and trade.side in {"BUY", "SELL"}
        and not trade.outcome
    )


def _latest_snapshot(db: Session, market_id: int) -> MarketSnapshot | None:
    return db.scalar(
        select(MarketSnapshot)
        .where(MarketSnapshot.market_id == market_id)
        .order_by(MarketSnapshot.captured_at.desc())
    )


def _signal_key(
    market_id: int,
    trader_id: int | None,
    signal_type: str,
    side: str | None,
    outcome: str | None,
    entry_price: float | None,
    size_usd: float | None,
    source: str,
    reason: str,
) -> tuple[Any, ...]:
    return (
        market_id,
        trader_id,
        signal_type,
        side,
        outcome,
        round(entry_price or 0.0, 8),
        round(size_usd or 0.0, 2),
        source,
        reason,
    )


def _candidate_key(candidate: SignalCandidate) -> tuple[Any, ...]:
    return _signal_key(
        candidate.market_id,
        candidate.trader_id,
        candidate.signal_type,
        candidate.side,
        candidate.outcome,
        candidate.entry_price,
        candidate.size_usd,
        candidate.source,
        candidate.reason,
    )


def _existing_signal_keys(db: Session) -> set[tuple[Any, ...]]:
    since = datetime.utcnow() - _RECENT_WINDOW
    signals = db.scalars(select(Signal).where(Signal.created_at >= since))
    return {
        _signal_key(
            s.market_id,
            s.trader_id,
            s.signal_type,
            s.side,
            s.outcome,
            s.entry_price,
            s.size_usd,
            s.source,
            s.reason,
        )
        for s in signals
    }


async def _score_candidate(
    db: Session,
    candidate: SignalCandidate,
    trader: Trader | None,
    market: Market | None,
    same_side_wallets: int,
    total_watched: int,
    cross_market_gap: float = 0.0,
    conviction_penalty: float = 0.0,
) -> None:
    snap = _latest_snapshot(db, candidate.market_id) if market else None
    current_price = snap.yes_price if snap else (market.yes_price if market else None)

    inputs = scoring.ScoreInputs(
        trader=trader,
        same_side_wallets=same_side_wallets,
        total_watched_wallets=max(total_watched, 1),
        market=market,
        entry_price=candidate.entry_price,
        current_price=current_price,
        price_gap=cross_market_gap,
    )
    sport = (market.category if market else None) or None
    market_type = candidate.signal_type
    breakdown = scoring.score_signal(
        inputs,
        db=db,
        sport=sport,
        market_type=market_type,
        conviction_penalty=conviction_penalty,
    )
    candidate.score = breakdown.total
    candidate.score_breakdown = breakdown.as_dict()
    candidate.score_breakdown["effective_weights"] = breakdown.__dict__.get(
        "effective_weights", {}
    )
    if conviction_penalty:
        candidate.score_breakdown["conviction_penalty"] = round(conviction_penalty, 4)
    wallet360_confidence = (
        max(0.0, min(1.0, (trader.trust_score or 50.0) / 100.0))
        if trader
        else 0.5
    )
    score_confidence = breakdown.total / 100.0
    candidate.confidence = round(0.65 * score_confidence + 0.35 * wallet360_confidence, 4)
    candidate.score_breakdown["wallet360_confidence"] = round(wallet360_confidence, 4)
    candidate.score_breakdown["confidence_blend"] = round(candidate.confidence, 4)


async def generate_signals(
    db: Session,
    providers: dict[str, BaseProvider] | None = None,
) -> list[Signal]:
    """Run every rule against current DB state. Returns persisted Signal rows."""
    settings = get_settings()
    candidates: list[SignalCandidate] = []

    trades = _recent_trades(db)
    if not trades:
        logger.info("signal_engine: no recent trades to evaluate")
        return []

    traders_by_id: dict[int, Trader] = {
        t.id: t for t in db.scalars(select(Trader))
    }
    markets_by_id: dict[int, Market] = {
        m.id: m for m in db.scalars(select(Market))
    }
    total_watched = len(traders_by_id)

    # --- Rule 1+2: trusted_wallet_entry / multi_wallet_consensus -------------
    by_market_side: dict[tuple[int, str, str | None], list[Trade]] = defaultdict(list)
    for t in trades:
        if _has_ambiguous_falcon_outcome(t):
            continue
        by_market_side[(t.market_id, t.side, t.outcome)].append(t)

    for (market_id, side, outcome), grouped in by_market_side.items():
        market = markets_by_id.get(market_id)
        unique_traders = {t.trader_id for t in grouped}
        same_side_wallets = len(unique_traders)

        # Trusted wallet entry — emitted for the most recent trade per market/side.
        most_recent = max(grouped, key=lambda t: t.timestamp)
        trader = traders_by_id.get(most_recent.trader_id)
        if trader is None:
            continue
        cand = SignalCandidate(
            market_id=market_id,
            trader_id=trader.id,
            signal_type="trusted_wallet_entry",
            side=side,
            outcome=outcome,
            entry_price=most_recent.price,
            size_usd=most_recent.size_usd,
            reason=(
                f"Watched wallet {trader.nickname} {side} {outcome or 'the outcome'} on "
                f"'{market.title if market else market_id}' at {most_recent.price:.3f} "
                f"(${most_recent.size_usd:,.0f})"
            ),
            source=most_recent.source or ProviderSource.MOCK.value,
        )
        await _score_candidate(
            db, cand, trader, market, same_side_wallets, total_watched
        )
        if cand.score >= settings.signal_score_threshold:
            candidates.append(cand)

        if same_side_wallets >= 2:
            consensus = SignalCandidate(
                market_id=market_id,
                trader_id=trader.id,
                signal_type="multi_wallet_consensus",
                side=side,
                outcome=outcome,
                entry_price=most_recent.price,
                size_usd=sum(t.size_usd for t in grouped),
                reason=(
                    f"{same_side_wallets} watched wallets {side} {outcome or 'the outcome'} on "
                    f"'{market.title if market else market_id}' in the last 24h "
                    f"(total ${sum(t.size_usd for t in grouped):,.0f})"
                ),
                source=most_recent.source or ProviderSource.MOCK.value,
            )
            await _score_candidate(
                db, consensus, trader, market, same_side_wallets, total_watched
            )
            if consensus.score >= settings.signal_score_threshold:
                candidates.append(consensus)

    # --- Rule 3: size_threshold ---------------------------------------------
    for trade in trades:
        if _has_ambiguous_falcon_outcome(trade):
            continue
        if trade.size_usd < _LARGE_POSITION_USD:
            continue
        trader = traders_by_id.get(trade.trader_id)
        market = markets_by_id.get(trade.market_id)
        cand = SignalCandidate(
            market_id=trade.market_id,
            trader_id=trade.trader_id,
            signal_type="size_threshold",
            side=trade.side,
            outcome=trade.outcome,
            entry_price=trade.price,
            size_usd=trade.size_usd,
            reason=(
                f"Large position: ${trade.size_usd:,.0f} {trade.side} "
                f"{trade.outcome or 'the outcome'} on "
                f"'{market.title if market else trade.market_id}' "
                f"(threshold ${_LARGE_POSITION_USD:,.0f})"
            ),
            source=trade.source or ProviderSource.MOCK.value,
        )
        await _score_candidate(db, cand, trader, market, 1, total_watched)
        if cand.score >= settings.signal_score_threshold:
            candidates.append(cand)

    # --- Rule 4: post_entry_price_move --------------------------------------
    for trade in trades:
        if _has_ambiguous_falcon_outcome(trade):
            continue
        market = markets_by_id.get(trade.market_id)
        snap = _latest_snapshot(db, trade.market_id)
        if not (market and snap and snap.yes_price is not None):
            continue
        ref_price = trade.price
        move = snap.yes_price - ref_price
        if abs(move) < _PRICE_MOVE_THRESHOLD:
            continue
        # only emit when the move confirms the trader's side
        confirms = (trade.side == "YES" and move > 0) or (trade.side == "NO" and move < 0)
        if not confirms:
            continue
        trader = traders_by_id.get(trade.trader_id)
        cand = SignalCandidate(
            market_id=trade.market_id,
            trader_id=trade.trader_id,
            signal_type="post_entry_price_move",
            side=trade.side,
            outcome=trade.outcome,
            entry_price=trade.price,
            size_usd=trade.size_usd,
            reason=(
                f"Price moved {move:+.3f} in favor of {trader.nickname if trader else 'wallet'} "
                f"after their {trade.side} entry on '{market.title}'"
            ),
            source=trade.source or ProviderSource.MOCK.value,
        )
        await _score_candidate(db, cand, trader, market, 1, total_watched)
        if cand.score >= settings.signal_score_threshold:
            candidates.append(cand)

    # --- Rule 5: cross_market_price_gap (disabled) ---------------------------
    # Cross-market comparison has no proven Falcon agent yet. Running this
    # against mock-only data would emit fake "Falcon" signals, so the rule is
    # disabled until we wire a real cross-market agent_id.

    # --- persist -------------------------------------------------------------
    from app.services.falcon_learning import capture_signal_attribution

    persisted: list[Signal] = []
    persisted_pairs: list[tuple[Signal, SignalCandidate]] = []
    existing_keys = _existing_signal_keys(db)
    seen_keys: set[tuple[Any, ...]] = set()
    for cand in candidates:
        key = _candidate_key(cand)
        if key in existing_keys or key in seen_keys:
            continue
        seen_keys.add(key)
        sig = cand.to_model()
        db.add(sig)
        persisted.append(sig)
        persisted_pairs.append((sig, cand))
    db.flush()

    # Capture learning attribution for every freshly persisted signal. We do
    # this in a separate pass so each Signal already has its primary key.
    for sig, cand in persisted_pairs:
        market = markets_by_id.get(sig.market_id)
        sport = (market.category if market else None) or None
        score_breakdown = cand.score_breakdown or {}
        factor_values = {
            k: float(v) for k, v in score_breakdown.items()
            if k in {
                "wallet_quality",
                "multi_wallet_consensus",
                "liquidity",
                "entry_timing",
                "price_inefficiency",
            } and isinstance(v, (int, float))
        }
        effective = score_breakdown.get("effective_weights") or {}
        contributing = []
        if cand.trader_id is not None:
            trader = traders_by_id.get(cand.trader_id)
            if trader and trader.wallet_address:
                contributing.append(
                    {
                        "wallet_address": trader.wallet_address,
                        "contribution_weight": 1.0,
                        "side": cand.side,
                        "size_usd": cand.size_usd,
                        "entry_price": cand.entry_price,
                    }
                )
        try:
            capture_signal_attribution(
                db,
                signal_id=sig.id,
                factors=factor_values,
                weights=effective,
                sport=sport,
                market_type=cand.signal_type,
                contributing_wallets=contributing,
                raw_score=cand.score,
                regime_payload=score_breakdown.get("regime_payload"),
                conflict_payload={
                    "conviction_penalty": score_breakdown.get("conviction_penalty", 0.0),
                },
            )
        except Exception:  # noqa: BLE001
            # Learning-layer hiccups must never break signal persistence.
            logger.exception("falcon_learning capture failed for signal=%s", sig.id)

    # Fire-and-forget regime capture for every freshly persisted signal.
    # The capture task opens its own DB session, retries each Falcon agent
    # with bounded backoff, and persists an immutable SignalRegimeSnapshot.
    # Signal persistence must not wait on Falcon latency — by the time this
    # function returns, signals are already committed and consumable.
    if persisted_pairs:
        try:
            from app.services.falcon_regime_capture import schedule_capture

            capture_requests = []
            for sig, cand in persisted_pairs:
                market = markets_by_id.get(sig.market_id)
                if market is None or not market.slug:
                    continue
                capture_requests.append({
                    "signal_id": sig.id,
                    "market_slug": market.slug,
                    "sport": (market.category if market else None) or None,
                    "market_type": cand.signal_type,
                    "same_side_wallets": (cand.score_breakdown or {}).get(
                        "same_side_wallets"
                    ),
                    "total_watched": total_watched,
                    "elite_disagreement_count": (cand.score_breakdown or {}).get(
                        "elite_disagreement_count", 0,
                    ),
                })
            if capture_requests:
                schedule_capture(capture_requests)
        except Exception:  # noqa: BLE001
            logger.exception("failed to schedule regime capture")

    logger.info("signal_engine: produced %d signals", len(persisted))
    return persisted
