"""End-to-end pipeline funnel diagnostics.

Answers ONE operator question: *"My scans succeed and there are 1000+ MLB
trades for today -- so why are Signals / High Conviction / Wallet Flow empty?"*

It walks the same path the live pipeline walks and reports a per-stage count
so the exact stage where records disappear is obvious:

    wallet trades -> MLB trades -> card-date trades -> signal candidates
        -> persisted signals -> aligned (consensus) cards -> high conviction

Every stage is read-only and cheap (no scoring, no provider calls), so this is
safe to call on every dashboard render. The shape is intentionally flat JSON so
the dashboard debug panel and an LLM reviewer can both consume it.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import MlbEdge, Signal, Trade, Trader, Market
from app.services.card_date import arizona_today, market_card_date
from app.services.wallet_market_resolver import parse_market_slug


def _market_index(db: Session) -> dict[int, Market]:
    return {m.id: m for m in db.scalars(select(Market))}


def _is_mlb(market: Market | None) -> bool:
    return bool(market and (market.slug or "").lower().startswith("mlb-"))


def _game_key(market: Market | None) -> str | None:
    """``mlb-sea-oak-2026-05-26`` matchup+date key shared by all of a game's
    total/spread/moneyline markets, so we can count distinct *games*."""
    if market is None:
        return None
    parsed = parse_market_slug(market.slug)
    if parsed is None or not parsed.event_date:
        return None
    return f"{parsed.league}:{'-'.join(sorted(parsed.team_pair()))}:{parsed.event_date}"


# ---------------------------------------------------------------------------
# Section 1 -- wallet ingestion audit
# ---------------------------------------------------------------------------


def ingestion_audit(db: Session, *, card_date: str, markets: dict[int, Market]) -> dict[str, Any]:
    trades = list(db.scalars(select(Trade)))
    mlb_trades = [t for t in trades if _is_mlb(markets.get(t.market_id))]
    card_trades = [
        t for t in mlb_trades
        if market_card_date(markets.get(t.market_id)) == card_date
    ]
    games = {g for t in card_trades if (g := _game_key(markets.get(t.market_id)))}
    return {
        "wallet_trades_total": len(trades),
        "mlb_trades_total": len(mlb_trades),
        "mlb_trades_for_card_date": len(card_trades),
        "unique_wallets": len({t.trader_id for t in mlb_trades}),
        "unique_markets": len({t.market_id for t in mlb_trades}),
        "unique_games_for_card_date": len(games),
    }


# ---------------------------------------------------------------------------
# Section 3 -- signal generation funnel (read-only replica of signal_engine)
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
        "candidate_multi_wallet_groups": multi_wallet,
        "score_threshold": threshold,
        "persisted_signals_for_card_date": len(persisted),
        "persisted_signals_above_threshold": len(above),
    }


# ---------------------------------------------------------------------------
# Section 2 + 5 -- alignment / wallet-flow aligned game cards
# ---------------------------------------------------------------------------


def alignment_audit(
    db: Session, *, card_date: str, markets: dict[int, Market], min_wallets: int = 2,
) -> dict[str, Any]:
    """Group this card's signals by (market, side, outcome) and count games
    where >= ``min_wallets`` distinct wallets align -- i.e. Wallet Flow cards."""
    signals = list(
        db.scalars(select(Signal).where(Signal.generated_for_date == card_date))
    )
    by_market_side: dict[tuple[int, str | None, str | None], set[int | None]] = defaultdict(set)
    for s in signals:
        by_market_side[(s.market_id, s.side, s.outcome)].add(s.trader_id)

    cards: list[dict[str, Any]] = []
    for (market_id, side, outcome), traders in by_market_side.items():
        if len(traders) < min_wallets:
            continue
        market = markets.get(market_id)
        cards.append({
            "game": _game_key(market) or (market.slug if market else market_id),
            "market": market.slug if market else market_id,
            "side": side,
            "outcome": outcome,
            "wallet_count": len(traders),
        })
    cards.sort(key=lambda c: c["wallet_count"], reverse=True)
    return {
        "signals_for_card_date": len(signals),
        "distinct_market_sides": len(by_market_side),
        "aligned_cards": len(cards),
        "examples": cards[:15],
    }


# ---------------------------------------------------------------------------
# Section 4 -- MLB game mapping audit
# ---------------------------------------------------------------------------


def mapping_audit(db: Session, *, card_date: str, markets: dict[int, Market]) -> dict[str, Any]:
    """For markets that carry card-date trades, can we parse the slug into a
    game key? Surface the slugs that fail so a team/date/format drift shows."""
    traded_market_ids = {
        t.market_id for t in db.scalars(select(Trade)) if _is_mlb(markets.get(t.market_id))
    }
    parsed_ok = 0
    failures: list[dict[str, Any]] = []
    wrong_date: list[dict[str, Any]] = []
    for mid in traded_market_ids:
        market = markets.get(mid)
        parsed = parse_market_slug(market.slug if market else None)
        if parsed is None:
            if len(failures) < 15:
                failures.append({"market_slug": market.slug if market else None,
                                 "reason": "slug did not parse to a game"})
            continue
        parsed_ok += 1
        if parsed.event_date and parsed.event_date != card_date and len(wrong_date) < 15:
            wrong_date.append({
                "market_slug": market.slug,
                "parsed_event_date": parsed.event_date,
                "card_date": card_date,
            })
    return {
        "mlb_markets_with_trades": len(traded_market_ids),
        "parsed_to_game": parsed_ok,
        "parse_failures": len(failures),
        "parse_failure_examples": failures,
        "date_mismatch_vs_card": wrong_date,
    }


# ---------------------------------------------------------------------------
# Section 6 -- duplicate wallet check
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
# High conviction (edges)
# ---------------------------------------------------------------------------


def high_conviction_audit(db: Session, *, card_date: str, min_prediction: float = 85.0) -> dict[str, Any]:
    edges = list(db.scalars(select(MlbEdge).where(MlbEdge.generated_for_date == card_date)))

    def _pred(e: MlbEdge) -> float:
        return float(e.prediction_score if e.prediction_score is not None else (e.score or 0.0))

    hc = [e for e in edges if _pred(e) >= min_prediction]
    return {
        "edges_for_card_date": len(edges),
        "high_conviction_threshold": min_prediction,
        "high_conviction_cards": len(hc),
        "max_prediction_score": round(max((_pred(e) for e in edges), default=0.0), 1),
    }


# ---------------------------------------------------------------------------
# Top-level funnel
# ---------------------------------------------------------------------------


def _safe(label: str, fn: Any) -> dict[str, Any]:
    """Run one audit section; never let a single failure blank the panel.

    A schema drift on one table (e.g. a not-yet-migrated edge column) must
    not hide the ingestion/signal funnel that actually pinpoints the bug.
    """
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
    mapping = _safe("mapping", lambda: mapping_audit(db, card_date=card_date, markets=markets))
    duplicates = _safe("duplicates", lambda: duplicate_wallet_audit(db))
    high_conv = _safe("high_conviction", lambda: high_conviction_audit(db, card_date=card_date))

    # The single most useful line: where did the records vanish?
    drop_stage = _first_empty_stage(ingestion, signals, alignment, high_conv)

    return {
        "card_date": card_date,
        "arizona_today": arizona_today(),
        "drop_stage": drop_stage,
        "funnel": {
            "wallet_trades_today": ingestion.get("mlb_trades_for_card_date"),
            "recent_trades_on_card_date": signals.get("recent_trades_on_card_date"),
            "signal_candidates": signals.get("candidate_market_sides"),
            "signals_generated": signals.get("persisted_signals_for_card_date"),
            "aligned_wallet_cards": alignment.get("aligned_cards"),
            "high_conviction_cards": high_conv.get("high_conviction_cards"),
        },
        "ingestion": ingestion,
        "signals": signals,
        "alignment": alignment,
        "mapping": mapping,
        "duplicates": duplicates,
        "high_conviction": high_conv,
    }


def _first_empty_stage(
    ingestion: dict[str, Any],
    signals: dict[str, Any],
    alignment: dict[str, Any],
    high_conv: dict[str, Any],
) -> str:
    """Name the earliest funnel stage that hit zero -- the prime suspect."""
    if ingestion.get("mlb_trades_total") == 0:
        return "ingestion: no MLB trades in DB at all (wallet scan produced nothing)"
    if ingestion.get("mlb_trades_for_card_date") == 0:
        return (
            "card_date_mismatch: MLB trades exist but NONE map to this card date "
            "-- market slug dates differ from the Arizona card date (timezone/date drift)"
        )
    if signals.get("recent_trades_on_card_date") == 0:
        return (
            "trade_window: card-date trades exist but none fall inside the signal "
            "engine's recency window -- widen signal_recent_trade_window_hours"
        )
    if signals.get("persisted_signals_for_card_date") == 0:
        return "signal_persistence: usable trades exist but no signals were persisted for this card date (score threshold or generated_for_date)"
    if alignment.get("aligned_cards") == 0:
        return "alignment: signals exist but no game has >=2 distinct wallets on the same side"
    if high_conv.get("high_conviction_cards") == 0:
        return "high_conviction: edges exist but none clear the prediction-score bar (data ok; just no 85+ plays)"
    return "none: pipeline is producing records at every stage"
