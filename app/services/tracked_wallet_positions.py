"""Tracked-wallet *live position* reader.

This service exists to answer the question the dashboard's signal-based
pipeline cannot:

    "Do my tracked wallets have any open positions today, regardless of
    whether a Signal row was generated or an MLB edge matches?"

The Signal pipeline drops trades that don't clear the score threshold
(low-conviction trusted_wallet_entry rows) and trades whose markets
don't normalize to today's card date. Both of those are *correct* for
the curated "high-conviction wallet flow" view, but they were also the
cause of the "No current-card wallet flow found" empty state showing up
even while ``positions_rejected_date_mismatch=5383`` in the same scan.

The output here intentionally has NO score threshold. The only filter
is: "is this trade by a tracked wallet, and is its market plausibly
today's card?" Everything that falls off the bus comes back via
:func:`live_position_debug` with a human-readable rejection reason so
the operator can spot the bug rather than guessing.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Market, Trade, Trader
from app.services.card_date import market_card_date, parse_iso_date, parse_slug_date
from app.services.wallet_normalize import (
    NormalizedMarketKey,
    looks_like_same_card_date,
    normalize_market_key,
)

# How far back we look for trades. The Polymarket trade feed is push-y
# enough that a 36-hour window safely covers anything that opened in the
# last evening through the current Arizona afternoon without dragging in
# truly stale rows.
RECENT_TRADE_WINDOW_HOURS = 36
# Cap on the debug response so a 5000-row rejection log can't blow up
# the dashboard render budget.
DEFAULT_DEBUG_LIMIT = 50
# Per-team known abbreviations used to spot a same-matchup partial join
# when MLB edges store ``DET vs CWS`` while wallets use ``det-cws``.
# Kept tiny on purpose â€” the full canonical-team table lives elsewhere
# in the codebase; this is only the loose-join fallback.
_MLB_MATCHUP_DELIMITER_RE = "[-_ ]"


def live_positions(
    db: Session, *, card_date: str,
) -> list[dict[str, Any]]:
    """All tracked-wallet trades whose market plausibly belongs to
    ``card_date``. NO score threshold, NO sport filter.

    "Plausibly belongs" means either the market's parsed date equals
    ``card_date``, OR the trade timestamp is within the last 36 hours
    AND the market has no parseable date (handles ATP / WNBA / generic
    markets that the date regex can't crack).
    """
    rows = _candidate_trades(db)
    out: list[dict[str, Any]] = []
    for trade, trader, market in rows:
        key = normalize_market_key(getattr(market, "slug", None))
        if not _row_belongs_to_card_date(trade, market, key, card_date):
            continue
        out.append(_serialize(trade, trader, market, key))
    # Largest position first â€” operators want to see the size-y rows.
    out.sort(key=lambda row: row.get("size_usd") or 0.0, reverse=True)
    return out


def live_position_debug(
    db: Session, *, card_date: str, limit: int = DEFAULT_DEBUG_LIMIT,
) -> dict[str, Any]:
    """Return per-rejection diagnostics for the Wallet Flow debug panel.

    Aggregates the rejection reasons and surfaces up to ``limit`` worked
    examples â€” never just a count. The operator should be able to read
    this and answer "WHY didn't VeryLucky888's position appear?" in one
    glance.
    """
    rows = _candidate_trades(db)
    raw_total = 0
    accepted_total = 0
    rejection_reasons: dict[str, int] = {}
    examples: list[dict[str, Any]] = []

    for trade, trader, market in rows:
        raw_total += 1
        key = normalize_market_key(getattr(market, "slug", None))
        accepted, reason = _classify_row(trade, market, key, card_date)
        if accepted:
            accepted_total += 1
            continue
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        if len(examples) < limit:
            examples.append({
                "wallet_nickname": getattr(trader, "nickname", None),
                "wallet_address": getattr(trader, "wallet_address", None),
                "market_slug": getattr(market, "slug", None),
                "market_title": getattr(market, "title", None),
                "parsed_event_date": key.event_date if key else None,
                "dashboard_card_date": card_date,
                "normalized_market_key": key.canonical if key else None,
                "rejection_reason": reason,
                "trade_timestamp": (
                    trade.timestamp.isoformat() if trade.timestamp else None
                ),
            })

    return {
        "card_date": card_date,
        "raw_recent_trades": raw_total,
        "accepted_for_card_date": accepted_total,
        "rejected": raw_total - accepted_total,
        "rejection_reasons": rejection_reasons,
        "top_rejected_examples": examples,
    }


def _candidate_trades(
    db: Session,
) -> list[tuple[Trade, Trader, Market]]:
    """All recent trades joined with their trader + market rows.

    The window is intentionally generous â€” the goal is to expose any
    Polymarket position that might still be open on today's card; downstream
    filters do the actual card_date decision.
    """
    cutoff = datetime.utcnow() - timedelta(hours=RECENT_TRADE_WINDOW_HOURS)
    query = (
        select(Trade, Trader, Market)
        .join(Trader, Trade.trader_id == Trader.id)
        .join(Market, Trade.market_id == Market.id)
        .where(Trade.timestamp >= cutoff)
        .order_by(Trade.timestamp.desc())
    )
    return list(db.execute(query).all())


def _row_belongs_to_card_date(
    trade: Trade,
    market: Market,
    key: NormalizedMarketKey | None,
    card_date: str,
) -> bool:
    """Accept the trade for ``card_date`` only when the evidence says so.

    Order of evidence â€” first hit wins:
      1. The market's persisted ``end_date`` / slug-derived card date.
      2. The slug-parsed event date (from ``normalize_market_key``).
      3. Recency fallback: trade timestamp within today's Arizona window
         AND the market has no parseable date (so we can't reject it
         purely because the slug shape is unusual).
    """
    market_date = market_card_date(market)
    if market_date:
        return market_date == card_date
    if key and key.event_date:
        return key.event_date == card_date
    # No date anywhere â€” fall back to "is this a recent trade?". This is
    # what keeps an ATP / unknown-sport position from disappearing.
    return _trade_is_recent_for_card_date(trade, card_date)


def _classify_row(
    trade: Trade,
    market: Market,
    key: NormalizedMarketKey | None,
    card_date: str,
) -> tuple[bool, str]:
    """Like ``_row_belongs_to_card_date`` but returns the rejection
    reason instead of a bool so the debug view can show *why*.
    """
    market_date = market_card_date(market)
    if market_date and market_date == card_date:
        return True, "accepted:market_date"
    if market_date and market_date != card_date:
        return False, f"market_date_mismatch:{market_date}"
    # No persisted market date â€” try the slug parser.
    if key and key.event_date:
        if key.event_date == card_date:
            return True, "accepted:slug_date"
        return False, f"slug_date_mismatch:{key.event_date}"
    # Last chance: recency-based fallback.
    if _trade_is_recent_for_card_date(trade, card_date):
        return True, "accepted:recent_fallback"
    return False, "stale_trade_no_date"


def _trade_is_recent_for_card_date(trade: Trade, card_date: str) -> bool:
    """True when the trade is fresh enough to plausibly belong to the
    card_date by recency alone. Used only when no slug/market date is
    available â€” we don't want to silently rebrand a 2-day-old trade as
    today's position.
    """
    if trade.timestamp is None:
        return False
    target = parse_iso_date(card_date)
    if target is None:
        return False
    # Allow the day before/after to absorb the Arizona/UTC boundary.
    delta_days = abs((trade.timestamp.date() - target).days)
    return delta_days <= 1


def _serialize(
    trade: Trade,
    trader: Trader,
    market: Market,
    key: NormalizedMarketKey | None,
) -> dict[str, Any]:
    """Shape the row for the API + dashboard. Keeps every field a UI
    might need on the same payload so the front-end doesn't have to
    second-fetch."""
    return {
        "trader_id": trader.id,
        "wallet_nickname": trader.nickname,
        "wallet_address": trader.wallet_address,
        "market_id": market.id,
        "market_slug": market.slug,
        "market_title": market.title,
        "market_platform": getattr(market, "platform", None),
        "market_category": getattr(market, "category", None),
        "sport": key.sport if key else None,
        "normalized_market_key": key.canonical if key else None,
        "matchup_date_key": key.matchup_date_key if key else None,
        "sport_date_key": key.sport_date_key if key else None,
        "parsed_event_date": key.event_date if key else None,
        "line": key.line if key else None,
        "market_subtype": key.market_subtype if key else None,
        "side": trade.side,
        "outcome": trade.outcome,
        "entry_price": trade.price,
        "size_usd": trade.size_usd,
        "source": trade.source,
        "opened_at": trade.timestamp.isoformat() if trade.timestamp else None,
        # No closing/current_price on Trade â€” left for the dashboard to
        # join from the market snapshot if it wants. Returning yes_price
        # here so the simple "current price" column has something to
        # show without an extra fetch.
        "current_yes_price": getattr(market, "yes_price", None),
        "current_no_price": getattr(market, "no_price", None),
    }
