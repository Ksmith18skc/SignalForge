"""End-to-end pipeline funnel diagnostics.

Answers ONE operator question: *"My scans succeed and tracked wallets are
trading -- so why is the Aligned Consensus view empty?"*

It walks the same path the live pipeline walks and reports a per-stage count so
the exact stage where records disappear is obvious:

    wallet trades -> trades on card date -> in the signal window
        -> candidate market-sides -> 2+ wallet consensus groups
        -> persisted signals

Market-agnostic: every tracked-wallet trade counts, regardless of sport. Every
stage is read-only and cheap (no scoring, no provider calls), so this is safe to
call on every dashboard render. The shape is intentionally flat JSON so the
dashboard debug panel and an LLM reviewer can both consume it.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Market, Signal, Trade, Trader
from app.services.card_date import arizona_today, market_card_date


def _market_index(db: Session) -> dict[int, Market]:
    return {m.id: m for m in db.scalars(select(Market))}


# ---------------------------------------------------------------------------
# Section 1 -- wallet ingestion audit
# ---------------------------------------------------------------------------


def ingestion_audit(db: Session, *, card_date: str, markets: dict[int, Market]) -> dict[str, Any]:
    trades = list(db.scalars(select(Trade)))
    card_trades = [
        t for t in trades
        if market_card_date(markets.get(t.market_id)) == card_date
    ]
    return {
        "wallet_trades_total": len(trades),
        "wallet_trades_for_card_date": len(card_trades),
        "unique_wallets": len({t.trader_id for t in trades}),
        "unique_markets": len({t.market_id for t in trades}),
        "unique_markets_for_card_date": len({t.market_id for t in card_trades}),
    }


# ---------------------------------------------------------------------------
# Section 2 -- signal / consensus funnel (read-only replica of signal_engine)
# ---------------------------------------------------------------------------


def signal_funnel(db: Session, *, card_date: str, markets: dict[int, Market]) -> dict[str, Any]:
    settings = get_settings()
    window_h = int(getattr(settings, "signal_recent_trade_window_hours", 48))
    since = datetime.utcnow() - timedelta(hours=max(window_h, 1))

    recent = list(db.scalars(select(Trade).where(Trade.timestamp >= since)))
    on_card = [t for t in recent if market_card_date(markets.get(t.market_id)) == card_date]

    # The signal engine drops Falcon BUY/SELL rows that carry no outcome.
    def _ambiguous(t: Trade) -> bool:
        return t.source == "Falcon" and t.side in {"BUY", "SELL"} and not t.outcome

    usable = [t for t in on_card if not _ambiguous(t)]
    ambiguous_dropped = len(on_card) - len(usable)

    groups: dict[tuple[int, str, str | None], list[Trade]] = defaultdict(list)
    for t in usable:
        groups[(t.market_id, t.side, t.outcome)].append(t)
    multi_wallet = sum(1 for g in groups.values() if len({t.trader_id for t in g}) >= 2)

    # What actually landed in the DB for this card date.
    persisted = list(
        db.scalars(select(Signal).where(Signal.generated_for_date == card_date))
    )
    threshold = float(settings.signal_score_threshold)
    above = [s for s in persisted if (s.score or 0.0) >= threshold]

    return {
        "trade_window_hours": window_h,
        "recent_trades_in_window": len(recent),
        "recent_trades_on_card_date": len(on_card),
        "dropped_ambiguous_falcon_outcome": ambiguous_dropped,
        "usable_trades": len(usable),
        "candidate_market_sides": len(groups),
        "consensus_groups_2plus_wallets": multi_wallet,
        "score_threshold": threshold,
        "persisted_signals_for_card_date": len(persisted),
        "persisted_signals_above_threshold": len(above),
    }


# ---------------------------------------------------------------------------
# Section 3 -- alignment / consensus cards
# ---------------------------------------------------------------------------


def alignment_audit(
    db: Session, *, card_date: str, markets: dict[int, Market], min_wallets: int = 2,
) -> dict[str, Any]:
    """Group this card's tracked-wallet trades by (market, side, outcome) and
    count markets where >= ``min_wallets`` distinct wallets align -- i.e. the
    Aligned Consensus cards the dashboard renders."""
    trades = [
        t for t in db.scalars(select(Trade))
        if market_card_date(markets.get(t.market_id)) == card_date
    ]
    by_market_side: dict[tuple[int, str | None, str | None], set[int]] = defaultdict(set)
    for t in trades:
        by_market_side[(t.market_id, t.side, t.outcome)].add(t.trader_id)

    cards: list[dict[str, Any]] = []
    for (market_id, side, outcome), traders in by_market_side.items():
        if len(traders) < min_wallets:
            continue
        market = markets.get(market_id)
        cards.append({
            "market": market.slug if market else market_id,
            "title": market.title if market else None,
            "side": side,
            "outcome": outcome,
            "wallet_count": len(traders),
        })
    cards.sort(key=lambda c: c["wallet_count"], reverse=True)
    return {
        "card_date_trades": len(trades),
        "distinct_market_sides": len(by_market_side),
        "aligned_cards": len(cards),
        "examples": cards[:15],
    }


# ---------------------------------------------------------------------------
# Section 4 -- duplicate wallet check
# ---------------------------------------------------------------------------


def duplicate_wallet_audit(db: Session) -> dict[str, Any]:
    traders = list(db.scalars(select(Trader)))
    trade_counts = dict(
        db.execute(
            select(Trade.trader_id, func.count()).group_by(Trade.trader_id)
        ).all()
    )
    addr_seen: dict[str, list[str]] = defaultdict(list)
    nick_seen: dict[str, list[str]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for t in traders:
        rows.append({
            "wallet_address": t.wallet_address,
            "wallet_label": t.nickname,
            "trade_count": int(trade_counts.get(t.id, 0)),
        })
        if t.wallet_address:
            addr_seen[t.wallet_address.lower()].append(t.nickname)
        if t.nickname:
            nick_seen[t.nickname.lower()].append(t.wallet_address or "?")
    dup_addr = {a: names for a, names in addr_seen.items() if len(names) > 1}
    dup_nick = {n: addrs for n, addrs in nick_seen.items() if len(addrs) > 1}
    rows.sort(key=lambda r: r["trade_count"], reverse=True)
    return {
        "trader_rows": len(traders),
        "duplicate_addresses": dup_addr,
        "duplicate_nicknames": dup_nick,
        "wallets": rows,
    }


# ---------------------------------------------------------------------------
# Top-level funnel
# ---------------------------------------------------------------------------


def _safe(label: str, fn: Any) -> dict[str, Any]:
    """Run one audit section; never let a single failure blank the panel."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "section": label}


def pipeline_funnel(db: Session, *, card_date: str | None = None) -> dict[str, Any]:
    """Full per-stage funnel for ``card_date`` (defaults to Arizona today)."""
    card_date = card_date or arizona_today()
    markets = _market_index(db)

    ingestion = _safe("ingestion", lambda: ingestion_audit(db, card_date=card_date, markets=markets))
    signals = _safe("signals", lambda: signal_funnel(db, card_date=card_date, markets=markets))
    alignment = _safe("alignment", lambda: alignment_audit(db, card_date=card_date, markets=markets))
    duplicates = _safe("duplicates", lambda: duplicate_wallet_audit(db))

    drop_stage = _first_empty_stage(ingestion, signals, alignment)

    return {
        "card_date": card_date,
        "arizona_today": arizona_today(),
        "drop_stage": drop_stage,
        "funnel": {
            "wallet_trades_today": ingestion.get("wallet_trades_for_card_date"),
            "recent_trades_on_card_date": signals.get("recent_trades_on_card_date"),
            "candidate_market_sides": signals.get("candidate_market_sides"),
            "consensus_groups": signals.get("consensus_groups_2plus_wallets"),
            "aligned_cards": alignment.get("aligned_cards"),
            "signals_generated": signals.get("persisted_signals_for_card_date"),
        },
        "ingestion": ingestion,
        "signals": signals,
        "alignment": alignment,
        "duplicates": duplicates,
    }


def _first_empty_stage(
    ingestion: dict[str, Any],
    signals: dict[str, Any],
    alignment: dict[str, Any],
) -> str:
    """Name the earliest funnel stage that hit zero -- the prime suspect."""
    if ingestion.get("wallet_trades_total") == 0:
        return "ingestion: no tracked-wallet trades in DB at all (wallet scan produced nothing)"
    if ingestion.get("wallet_trades_for_card_date") == 0:
        return (
            "card_date_mismatch: trades exist but NONE map to this card date "
            "-- market slug/end dates differ from the Arizona card date (timezone/date drift)"
        )
    if signals.get("recent_trades_on_card_date") == 0:
        return (
            "trade_window: card-date trades exist but none fall inside the signal "
            "engine's recency window -- widen signal_recent_trade_window_hours"
        )
    if alignment.get("aligned_cards") == 0:
        return "alignment: trades exist but no market has >=2 distinct wallets on the same side"
    if signals.get("persisted_signals_for_card_date") == 0:
        return "signal_persistence: aligned trades exist but no signals were persisted (score threshold or generated_for_date)"
    return "none: pipeline is producing records at every stage"
