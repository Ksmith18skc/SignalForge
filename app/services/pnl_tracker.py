"""Personal P&L math.

Everything here is a pure function operating on plain data shapes (or ORM
rows treated as data). No HTTP, no Discord, no side effects beyond what
`rebuild_positions_for_wallet` writes to the DB. That makes the math easy
to unit test and easy to re-use from the Streamlit page and FastAPI
routes alike.

Key conventions used throughout:
  - Prediction-market prices are 0.0-1.0 probabilities. We never convert
    to American odds here — the dashboard owns presentation.
  - All P&L numbers are in USD (Polymarket USDC and Kalshi USD both map
    cleanly to dollars).
  - `cost_basis` of an open position is the *remaining* cost basis after
    any partial sells, not the lifetime gross. This matches how a
    brokerage statement reads.
  - `realized_pnl` includes settlement payouts (1.0 for a winning binary
    contract, 0.0 for a loser) as recorded in `ClosedTradeOutcome`.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, Mapping

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.storage.pnl_store import (
    ClosedTradeOutcome,
    MyPosition,
    MyTrade,
    MyWallet,
    WalletSnapshot,
)

logger = logging.getLogger(__name__)

# Tracks how recently the dashboard considers a `current_price` reliable.
# Past this age the position is rendered with a "stale" badge and any P&L
# derived from it is labelled "estimated".
STALE_PRICE_AFTER = timedelta(minutes=20)


# ---------------------------------------------------------------------------
# Data shapes returned to UI/API callers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioSummary:
    total_value_usd: float
    cash_usd: float | None
    open_position_value_usd: float
    realized_pnl_usd: float
    unrealized_pnl_usd: float
    total_pnl_usd: float
    roi_percent: float | None
    daily_pnl_usd: float | None
    daily_pnl_percent: float | None
    open_position_count: int
    closed_position_count: int
    is_estimated: bool
    stale_positions: int
    largest_winner_usd: float
    largest_loser_usd: float


@dataclass(frozen=True)
class ExposureSlice:
    key: str
    label: str
    notional_usd: float
    portfolio_pct: float
    position_count: int


@dataclass
class PnlInputs:
    """Everything `compute_portfolio_summary` needs in one bag."""

    positions: list[MyPosition]
    cash_by_wallet: Mapping[int, float] = field(default_factory=dict)
    previous_total_value_usd: float | None = None
    realized_overrides_by_position_id: Mapping[int, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tiny math helpers — kept tiny so they're trivially testable
# ---------------------------------------------------------------------------


def compute_position_value(shares: float, price: float | None) -> float:
    """Value of a one-sided binary position. None price -> 0.0 estimate
    (caller should label the row stale, see `is_stale_price`)."""
    if shares <= 0:
        return 0.0
    if price is None or math.isnan(price):
        return 0.0
    return float(shares) * float(price)


def compute_unrealized_pnl(
    shares: float, avg_entry_price: float, current_price: float | None
) -> float:
    """(current_price - avg_entry) * shares. Returns 0 when we can't
    price the position; the caller flags it as estimated."""
    if shares <= 0 or current_price is None:
        return 0.0
    return float(shares) * (float(current_price) - float(avg_entry_price))


def compute_clv(entry_price: float | None, closing_price: float | None) -> tuple[float | None, float | None]:
    """Closing-Line Value in raw price points and as a percent of entry.

    CLV is a price-move measurement, so it has to be sign-aware. For a
    BUY (you own YES), CLV is positive when the price moves *up* (the
    market agreed with you). The caller is responsible for flipping the
    sign when the position is a SELL — we don't try to guess here
    because `position_matcher` already knows the side.

    Returns (None, None) if either price is missing or entry is 0.
    """
    if entry_price is None or closing_price is None:
        return None, None
    try:
        ep = float(entry_price)
        cp = float(closing_price)
    except (TypeError, ValueError):
        return None, None
    if ep <= 0 or math.isnan(ep) or math.isnan(cp):
        return None, None
    points = cp - ep
    pct = (points / ep) * 100.0
    return points, pct


def compute_edge(fair_probability: float | None, market_price: float | None) -> float | None:
    """Edge = fair_probability - market_price, both on [0, 1]. Returns
    None when either side is missing so the dashboard can show '—'."""
    if fair_probability is None or market_price is None:
        return None
    try:
        return float(fair_probability) - float(market_price)
    except (TypeError, ValueError):
        return None


def is_stale_price(updated_at: datetime | None, *, now: datetime | None = None) -> bool:
    if updated_at is None:
        return True
    now = now or datetime.utcnow()
    return (now - updated_at) > STALE_PRICE_AFTER


def grade_trade(
    *,
    pnl_usd: float,
    edge_at_entry: float | None,
    clv_points: float | None,
    is_win: bool | None,
) -> str:
    """Letter grade for a closed trade. The bar is intentionally about
    *process*, not outcome — a coinflip win on a bad price still grades
    worse than a marginal loss on a strong CLV move."""
    edge = edge_at_entry or 0.0
    clv = clv_points or 0.0
    score = 0
    score += 30 if edge >= 0.05 else 20 if edge >= 0.02 else 10 if edge > 0 else 0
    score += 30 if clv >= 0.04 else 20 if clv >= 0.02 else 10 if clv > 0 else 0
    if is_win is True:
        score += 25
    elif is_win is False and pnl_usd < 0:
        score += 0
    else:
        score += 10  # open / unsettled
    if pnl_usd > 0:
        score += 15
    elif pnl_usd < 0:
        score -= 5
    if score >= 80:
        return "A"
    if score >= 65:
        return "B"
    if score >= 50:
        return "C"
    if score >= 35:
        return "D"
    return "F"


# ---------------------------------------------------------------------------
# Position rebuild — convert raw fills into one MyPosition row per slot
# ---------------------------------------------------------------------------


def _slot_key(trade: MyTrade) -> tuple[str, str | None]:
    return trade.market_slug, (trade.outcome or None)


def rebuild_positions_for_wallet(db: Session, wallet: MyWallet) -> list[MyPosition]:
    """Recompute `MyPosition` rows from the wallet's fill history.

    Walks all `MyTrade` rows for the wallet in chronological order and
    folds them into one position per (market, outcome) slot. BUY adds
    shares at the trade price, SELL subtracts shares and crystallises
    realised P&L for the size that crossed back. Anything we can't
    classify (unknown side) is logged and skipped — never silently
    corrupts the running average.
    """
    trades = list(
        db.scalars(
            select(MyTrade)
            .where(MyTrade.wallet_id == wallet.id)
            .order_by(MyTrade.timestamp.asc())
        )
    )

    # In-memory accumulator: slot -> running state.
    state: dict[tuple[str, str | None], dict] = {}
    for trade in trades:
        key = _slot_key(trade)
        s = state.setdefault(
            key,
            {
                "platform": trade.platform,
                "market_slug": trade.market_slug,
                "market_title": trade.market_title,
                "outcome": trade.outcome,
                "shares": 0.0,
                "cost_basis_usd": 0.0,
                "realized_pnl_usd": 0.0,
                "opened_at": trade.timestamp,
                "closed_at": None,
                "sport": trade.sport,
                "event_date": trade.event_date,
                "side": trade.side,
            },
        )
        # `notional` is the venue-reported USD where available; fall back to
        # shares*price so a missing/zero notional doesn't zero out cost basis.
        notional = trade.size_usd or (trade.size_shares * trade.price)
        side = (trade.side or "").upper()
        if side == "BUY":
            s["shares"] += trade.size_shares
            s["cost_basis_usd"] += notional
        elif side == "SELL":
            # Realised P&L = sale proceeds - proportional cost basis.
            if s["shares"] > 0:
                avg_cost = s["cost_basis_usd"] / s["shares"]
                closing_shares = min(trade.size_shares, s["shares"])
                proportional_cost = avg_cost * closing_shares
                s["realized_pnl_usd"] += notional - proportional_cost
                s["cost_basis_usd"] -= proportional_cost
                s["shares"] -= closing_shares
                if s["shares"] <= 1e-6:
                    s["shares"] = 0.0
                    s["cost_basis_usd"] = 0.0
                    s["closed_at"] = trade.timestamp
            else:
                # Pre-existing short or no-position sell. We do not model
                # short interest on prediction markets yet; record as
                # realised income.
                s["realized_pnl_usd"] += notional
        else:
            logger.debug(
                "Unknown trade side %r for wallet=%s trade=%s; skipping",
                trade.side, wallet.id, trade.external_id,
            )
            continue

    # Crystallise any settlement payouts that exist for this wallet's slots.
    settlements_by_slot: dict[tuple[str, str | None], float] = defaultdict(float)
    pos_id_to_slot: dict[int, tuple[str, str | None]] = {}
    existing_positions = list(
        db.scalars(select(MyPosition).where(MyPosition.wallet_id == wallet.id))
    )
    for pos in existing_positions:
        pos_id_to_slot[pos.id] = (pos.market_slug, pos.outcome or None)
    if pos_id_to_slot:
        outcomes = list(
            db.scalars(
                select(ClosedTradeOutcome).where(
                    ClosedTradeOutcome.my_position_id.in_(pos_id_to_slot.keys())
                )
            )
        )
        for outcome in outcomes:
            slot = pos_id_to_slot.get(outcome.my_position_id)
            if slot is not None:
                settlements_by_slot[slot] += outcome.realized_pnl_usd

    # Upsert one row per slot.
    existing_by_slot: dict[tuple[str, str | None], MyPosition] = {
        (p.market_slug, p.outcome or None): p for p in existing_positions
    }
    written: list[MyPosition] = []
    for key, s in state.items():
        pos = existing_by_slot.get(key)
        if pos is None:
            pos = MyPosition(
                wallet_id=wallet.id,
                platform=s["platform"],
                market_slug=s["market_slug"],
                market_title=s["market_title"],
                outcome=s["outcome"],
                opened_at=s["opened_at"],
            )
            db.add(pos)
        pos.platform = s["platform"]
        pos.market_title = s["market_title"] or pos.market_title
        pos.side = s["side"]
        pos.shares = round(s["shares"], 6)
        pos.cost_basis_usd = round(s["cost_basis_usd"], 4)
        pos.avg_entry_price = (
            round(s["cost_basis_usd"] / s["shares"], 6) if s["shares"] > 0 else pos.avg_entry_price
        )
        pos.realized_pnl_usd = round(s["realized_pnl_usd"] + settlements_by_slot.get(key, 0.0), 4)
        pos.sport = s["sport"] or pos.sport
        pos.event_date = s["event_date"] or pos.event_date
        if s["shares"] <= 0:
            pos.status = "closed"
            pos.closed_at = s.get("closed_at") or pos.closed_at
        else:
            pos.status = "open"
            pos.closed_at = None
        pos.unrealized_pnl_usd = compute_unrealized_pnl(
            pos.shares, pos.avg_entry_price, pos.current_price
        )
        pos.current_value_usd = compute_position_value(pos.shares, pos.current_price)
        pos.is_stale_price = is_stale_price(pos.last_updated)
        written.append(pos)

    db.flush()
    return written


# ---------------------------------------------------------------------------
# Portfolio aggregation
# ---------------------------------------------------------------------------


def compute_portfolio_summary(inputs: PnlInputs) -> PortfolioSummary:
    """Roll positions + cash into a single summary card row."""
    positions = inputs.positions
    open_positions = [p for p in positions if p.status == "open" and p.shares > 0]
    closed_positions = [p for p in positions if p.status == "closed" or p.shares <= 0]

    open_value = sum(p.current_value_usd or 0.0 for p in open_positions)
    realized = sum(
        inputs.realized_overrides_by_position_id.get(p.id, p.realized_pnl_usd or 0.0)
        for p in positions
    )
    unrealized = sum(p.unrealized_pnl_usd or 0.0 for p in open_positions)
    cash_total = sum(inputs.cash_by_wallet.values()) if inputs.cash_by_wallet else 0.0
    total_value = open_value + cash_total
    total_pnl = realized + unrealized

    cost_basis_total = sum(p.cost_basis_usd or 0.0 for p in open_positions)
    invested_total = cost_basis_total + sum(
        max(0.0, (p.cost_basis_usd or 0.0)) for p in closed_positions if (p.realized_pnl_usd or 0.0)
    )
    roi = (total_pnl / invested_total * 100.0) if invested_total > 0 else None

    daily_pnl: float | None = None
    daily_pnl_pct: float | None = None
    prev = inputs.previous_total_value_usd
    if prev is not None and prev > 0:
        daily_pnl = total_value - prev
        daily_pnl_pct = (daily_pnl / prev) * 100.0

    pnl_per_position = [
        ((p.realized_pnl_usd or 0.0) + (p.unrealized_pnl_usd or 0.0)) for p in positions
    ]
    largest_winner = max(pnl_per_position, default=0.0)
    largest_loser = min(pnl_per_position, default=0.0)
    stale = sum(1 for p in open_positions if p.is_stale_price)

    is_estimated = (
        stale > 0
        or any(p.current_price is None for p in open_positions)
        or not inputs.cash_by_wallet
    )

    return PortfolioSummary(
        total_value_usd=round(total_value, 2),
        cash_usd=round(cash_total, 2) if inputs.cash_by_wallet else None,
        open_position_value_usd=round(open_value, 2),
        realized_pnl_usd=round(realized, 2),
        unrealized_pnl_usd=round(unrealized, 2),
        total_pnl_usd=round(total_pnl, 2),
        roi_percent=round(roi, 2) if roi is not None else None,
        daily_pnl_usd=round(daily_pnl, 2) if daily_pnl is not None else None,
        daily_pnl_percent=round(daily_pnl_pct, 2) if daily_pnl_pct is not None else None,
        open_position_count=len(open_positions),
        closed_position_count=len(closed_positions),
        is_estimated=is_estimated,
        stale_positions=stale,
        largest_winner_usd=round(largest_winner, 2),
        largest_loser_usd=round(largest_loser, 2),
    )


def exposure_breakdown(
    positions: Iterable[MyPosition],
    *,
    dimension: str = "sport",
    portfolio_value: float | None = None,
) -> list[ExposureSlice]:
    """Group open notional by sport / market / platform / event.

    `portfolio_value` is used as the denominator for `portfolio_pct`; if
    None, percentages are reported against the slice total. That avoids
    showing 9999% when the dashboard hasn't priced any positions yet.
    """
    if dimension not in {"sport", "market", "platform", "event_date"}:
        raise ValueError(f"unsupported exposure dimension: {dimension}")
    buckets: dict[str, dict] = defaultdict(lambda: {"notional": 0.0, "count": 0})
    for p in positions:
        if p.status != "open" or p.shares <= 0:
            continue
        notional = p.current_value_usd or p.cost_basis_usd or 0.0
        if dimension == "sport":
            key = (p.sport or "unknown").lower()
        elif dimension == "market":
            key = p.market_slug or "unknown"
        elif dimension == "platform":
            key = p.platform or "unknown"
        else:
            key = p.event_date or "unknown"
        bucket = buckets[key]
        bucket["notional"] += notional
        bucket["count"] += 1

    total_for_pct = portfolio_value or sum(b["notional"] for b in buckets.values()) or 1.0
    slices = [
        ExposureSlice(
            key=key,
            label=_humanize_bucket(dimension, key),
            notional_usd=round(b["notional"], 2),
            portfolio_pct=round(b["notional"] / total_for_pct * 100.0, 2),
            position_count=b["count"],
        )
        for key, b in buckets.items()
    ]
    slices.sort(key=lambda s: s.notional_usd, reverse=True)
    return slices


def _humanize_bucket(dimension: str, raw: str) -> str:
    if dimension == "sport":
        return raw.upper() if raw and raw != "unknown" else "Unknown sport"
    if dimension == "platform":
        return raw.capitalize() if raw else "Unknown"
    if dimension == "event_date":
        return raw or "No date"
    return raw or "Unknown"


# ---------------------------------------------------------------------------
# Daily-P&L helpers (built on `WalletSnapshot`)
# ---------------------------------------------------------------------------


def previous_total_value(snapshots: Iterable[WalletSnapshot]) -> float | None:
    """Pick the latest snapshot strictly older than 12h ago — the
    "previous" reference point for daily P&L. Returns None if we don't
    have history yet."""
    snapshots = sorted(snapshots, key=lambda s: s.captured_at, reverse=True)
    cutoff = datetime.utcnow() - timedelta(hours=12)
    for snap in snapshots:
        if snap.captured_at <= cutoff:
            return snap.total_value_usd
    return None
