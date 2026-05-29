"""Alert dispatch."""

from __future__ import annotations

import logging
import re
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Literal

import httpx
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models import Alert, Signal, Trade, Trader
from app.services.card_date import TZ_ARIZONA, parse_slug_date
from app.utils.dashboard_format import (
    american_to_implied_probability,
    confidence_label,
    factor_label,
)

logger = logging.getLogger(__name__)

AlertTier = Literal["ignore", "log_only", "discord_alert", "high_conviction", "possible_entry"]
DISCORD_TIERS: set[AlertTier] = {"discord_alert", "high_conviction", "possible_entry"}
TIER_RANK: dict[AlertTier, int] = {
    "ignore": 0,
    "log_only": 1,
    "discord_alert": 2,
    "high_conviction": 3,
    "possible_entry": 4,
}
HARD_MIN_SCORE = 65.0
ELITE_TRUST_SCORE = 75.0
MIN_LIQUIDITY_USD = 5_000.0
MARKET_CONTEXT_HOURS = 24
MARKET_SIDE_OUTCOME_WINDOW_MINUTES = 60
MARKET_SUMMARY_WINDOW_HOURS = 3


@dataclass(frozen=True)
class AlertContext:
    trader_count: int = 0
    total_tracked_size: float = 0.0
    entry_price_min: float | None = None
    entry_price_max: float | None = None
    current_price: float | None = None
    first_entry_price: float | None = None
    first_entry_at: datetime | None = None
    market_volume: float | None = None
    liquidity: float | None = None
    price_movement: float | None = None
    aligned_traders: tuple[Trader, ...] = ()


@dataclass(frozen=True)
class AlertDecision:
    tier: AlertTier
    context: AlertContext
    action: str
    chase_risk: str
    reason: str


# A "human" market title contains a space or capital letter — anything else
# (mlb-mia-tor-2026-05-25-spread-home-1pt5) is treated as the raw slug.
_SLUG_LIKE_RE = re.compile(r"^[a-z0-9.\-]+$")


def event_date_from_slug(slug: str | None) -> date | None:
    """Extract the calendar date encoded in slugs like
    `mlb-mia-tor-2026-05-25-spread-home-1pt5`. Returns None when the slug
    has no parseable date — never throws."""
    return parse_slug_date(slug)


def market_expiration_reason(
    signal: Signal,
    *,
    now: datetime | None = None,
    grace_hours: int = 3,
) -> str | None:
    """Return a reason string if the market's event is already over.

    Checks (cheapest first):
      1. `Market.is_active is False`
      2. `Market.end_date` is in the past (with `grace_hours` slack for late
         settlement of live markets — without it, a market mid-game could be
         marked done if its `end_date` is the scheduled first-pitch time).
      3. The slug's game date is before today's Arizona card date.

    The slug encodes the *Arizona* calendar date of the game (the same card
    date the dashboard and scanner use), so it must be compared against the
    Arizona date — not UTC. Comparing against `datetime.utcnow().date()` marked
    every evening game "expired" the instant UTC rolled past midnight (5pm
    Arizona), killing alerts for games that were just tipping off.

    `now` is a naive UTC datetime (as produced by `datetime.utcnow()`).

    Returns None when the market is still in-play or its lifecycle can't be
    determined (we'd rather alert on an unknown-state market than silently
    drop signals from one without an `end_date`).
    """
    market = signal.market
    if market is None:
        return None
    now = now or datetime.utcnow()
    if market.is_active is False:
        return "market is no longer active"
    # Defensive: ingestion now normalizes end_date to naive UTC, but a
    # Postgres ``TIMESTAMP WITH TIME ZONE`` column can still hand back
    # an aware datetime. Strip the tz here so the comparison can't blow
    # up the scan with "can't compare offset-naive and offset-aware".
    end_date_naive = market.end_date
    if end_date_naive is not None and end_date_naive.tzinfo is not None:
        end_date_naive = end_date_naive.astimezone(timezone.utc).replace(tzinfo=None)
    if end_date_naive is not None and end_date_naive + timedelta(hours=grace_hours) < now:
        return f"market end_date {market.end_date.isoformat()} already passed"
    event_date = event_date_from_slug(market.slug)
    arizona_today = now.replace(tzinfo=timezone.utc).astimezone(TZ_ARIZONA).date()
    if event_date is not None and event_date < arizona_today:
        return f"event date {event_date.isoformat()} in the past"
    return None


def _humanize_market(signal: Signal) -> dict[str, str]:
    """Best-effort: return {matchup, contract, league} from slug + title.

    Keeps a human-readable Market.title untouched. For slug-style titles like
    `mlb-mia-tor-2026-05-25-spread-home-1pt5` we extract league, teams, and
    the contract description so alerts read like a betting card.
    """
    market = signal.market
    title = (market.title or "") if market else ""
    slug = (market.slug or "") if market else ""

    if title and not _SLUG_LIKE_RE.match(title):
        return {"matchup": title, "contract": "", "league": ""}

    if not slug:
        return {"matchup": title or "Unknown market", "contract": "", "league": ""}

    parts = slug.split("-")
    league = parts[0].upper() if parts else ""

    date_idx = None
    for i in range(max(len(parts) - 2, 0)):
        if (
            re.fullmatch(r"20\d{2}", parts[i] or "")
            and re.fullmatch(r"\d{2}", parts[i + 1] or "")
            and re.fullmatch(r"\d{2}", parts[i + 2] or "")
        ):
            date_idx = i
            break

    if date_idx is None or date_idx < 2:
        return {"matchup": title or slug, "contract": "", "league": league}

    teams = [p.upper() for p in parts[1:date_idx]]
    matchup = f"{teams[0]} @ {teams[1]}" if len(teams) >= 2 else " ".join(teams)
    rest = parts[date_idx + 3:]
    contract = ""
    if rest[:1] == ["spread"] and len(rest) >= 3:
        contract = f"Spread {rest[1].title()} {rest[2].replace('pt', '.')}"
    elif rest[:1] == ["total"] and len(rest) >= 2:
        contract = f"Total {rest[1].replace('pt', '.')}"
    elif rest[:1] == ["moneyline"] and len(rest) >= 2:
        contract = f"Moneyline {' '.join(p.title() for p in rest[1:])}"
    elif rest:
        contract = " ".join(p.title() for p in rest)

    return {"matchup": f"{league} {matchup}".strip(), "contract": contract, "league": league}


def _market_url(signal: Signal) -> str | None:
    from app.services.wallet_market_resolver import market_url_for as _impl
    market = signal.market
    if not market or not market.slug:
        return None
    return _impl(market.slug, market.platform)


def _trader_url(signal: Signal) -> str | None:
    trader = signal.trader
    if not (trader and trader.wallet_address):
        return None
    return f"https://polymarketanalytics.com/traders/{trader.wallet_address}"


def _format_signal(signal: Signal, decision: AlertDecision | None = None) -> str:
    market_title = signal.market.title if signal.market else f"market#{signal.market_id}"
    trader = signal.trader
    nickname = trader.nickname if trader else "-"
    wallet = (trader.wallet_address if trader else None) or "-"
    market_url = _market_url(signal)
    trader_url = _trader_url(signal)
    outcome = f" outcome={signal.outcome}" if signal.outcome else ""
    tier = f"[{decision.tier}] " if decision else ""
    return (
        f"{tier}[{signal.source}] {signal.signal_type} | score={signal.score:.1f} "
        f"| market='{market_title}' | trader={nickname} ({wallet}) "
        f"| side={signal.side}{outcome} entry={signal.entry_price} size=${signal.size_usd or 0:,.0f} "
        f"| market_url={market_url or 'n/a'} "
        f"| trader_url={trader_url or 'n/a'} "
        f"| reason: {signal.reason}"
    )


def _money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.0f}"


def _price(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _entry_range(ctx: AlertContext) -> str:
    if ctx.entry_price_min is None or ctx.entry_price_max is None:
        return "n/a"
    if abs(ctx.entry_price_min - ctx.entry_price_max) < 0.0001:
        return _price(ctx.entry_price_min)
    return f"{_price(ctx.entry_price_min)}-{_price(ctx.entry_price_max)}"


def _current_price(signal: Signal) -> float | None:
    market = signal.market
    if not market:
        return None
    outcome = (signal.outcome or "").strip().lower()
    if outcome in {"no", "under"}:
        if market.no_price is not None:
            return market.no_price
        if market.yes_price is not None:
            return 1 - market.yes_price
    return market.yes_price


def _same_outcome_filter(signal: Signal):
    if signal.outcome:
        return Trade.outcome == signal.outcome
    return or_(Trade.outcome.is_(None), Trade.outcome == "")


def _load_alert_context(db: Session, signal: Signal) -> AlertContext:
    created_at = signal.created_at or datetime.utcnow()
    since = created_at - timedelta(hours=MARKET_CONTEXT_HOURS)
    filters = [
        Trade.market_id == signal.market_id,
        Trade.timestamp >= since,
    ]
    if signal.side:
        filters.append(Trade.side == signal.side)
    filters.append(_same_outcome_filter(signal))

    trades = list(
        db.scalars(
            select(Trade)
            .where(*filters)
            .order_by(Trade.timestamp.asc())
        )
    )
    prices = [t.price for t in trades if t.price is not None and t.price > 0]
    total_size = sum(t.size_usd or 0 for t in trades)
    trader_ids = {t.trader_id for t in trades if t.trader_id is not None}
    current = _current_price(signal)
    first_price = prices[0] if prices else signal.entry_price
    movement = None
    if current is not None and first_price is not None:
        movement = abs(current - first_price)

    aligned_traders: tuple[Trader, ...] = ()
    if trader_ids:
        aligned_traders = tuple(
            db.scalars(
                select(Trader)
                .where(Trader.id.in_(trader_ids))
                .order_by(Trader.trust_score.desc(), Trader.nickname.asc())
            )
        )

    market = signal.market
    return AlertContext(
        trader_count=len(trader_ids),
        total_tracked_size=total_size,
        entry_price_min=min(prices) if prices else signal.entry_price,
        entry_price_max=max(prices) if prices else signal.entry_price,
        current_price=current,
        first_entry_price=first_price,
        first_entry_at=trades[0].timestamp if trades else signal.created_at,
        market_volume=market.volume_24h_usd if market else None,
        liquidity=market.liquidity_usd if market else None,
        price_movement=movement,
        aligned_traders=aligned_traders,
    )


def _is_duplicate_discord_alert(db: Session, signal: Signal, window_minutes: int) -> bool:
    if not signal.trader_id or not signal.side:
        return False
    cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
    filters = [
        Alert.channel == "discord",
        Alert.status == "sent",
        Alert.created_at >= cutoff,
        Signal.id != signal.id,
        Signal.trader_id == signal.trader_id,
        Signal.market_id == signal.market_id,
        Signal.side == signal.side,
    ]
    if signal.outcome:
        filters.append(Signal.outcome == signal.outcome)

    existing = db.scalar(select(Alert.id).join(Signal).where(*filters).limit(1))
    return existing is not None


def _same_outcome_signal_filter(signal: Signal):
    if signal.outcome:
        return Signal.outcome == signal.outcome
    return or_(Signal.outcome.is_(None), Signal.outcome == "")


def _recent_sent_discord_signals(db: Session, signal: Signal, since: datetime) -> list[Signal]:
    return list(
        db.scalars(
            select(Signal)
            .join(Alert)
            .where(
                Alert.channel == "discord",
                Alert.status == "sent",
                Alert.created_at >= since,
                Signal.id != signal.id,
                Signal.market_id == signal.market_id,
            )
            .order_by(Alert.created_at.desc())
        )
    )


def _has_recent_market_side_outcome_alert(db: Session, signal: Signal) -> bool:
    if not signal.side:
        return False
    since = datetime.utcnow() - timedelta(minutes=MARKET_SIDE_OUTCOME_WINDOW_MINUTES)
    existing = db.scalar(
        select(Alert.id)
        .join(Signal)
        .where(
            Alert.channel == "discord",
            Alert.status == "sent",
            Alert.created_at >= since,
            Signal.id != signal.id,
            Signal.market_id == signal.market_id,
            Signal.side == signal.side,
            _same_outcome_signal_filter(signal),
        )
        .limit(1)
    )
    return existing is not None


def _binary_outcome_key(outcome: str | None) -> str | None:
    normalized = (outcome or "").strip().lower()
    if normalized in {"yes", "y", "over"}:
        return "yes"
    if normalized in {"no", "n", "under"}:
        return "no"
    return None


def _is_opposite_binary_side(current: Signal, previous: Signal) -> bool:
    current_key = _binary_outcome_key(current.outcome)
    previous_key = _binary_outcome_key(previous.outcome)
    if not current_key or not previous_key:
        return False
    return current_key != previous_key


def _tier_from_alert_message(message: str | None) -> AlertTier:
    message = message or ""
    if "[possible_entry]" in message:
        return "possible_entry"
    if "[high_conviction]" in message:
        return "high_conviction"
    if "[discord_alert]" in message:
        return "discord_alert"
    if "[log_only]" in message:
        return "log_only"
    return "ignore"


def _should_suppress_discord(db: Session, signal: Signal, decision: AlertDecision) -> str | None:
    if decision.action == "Avoid chasing":
        return "avoid chasing action"

    if _has_recent_market_side_outcome_alert(db, signal):
        return "same market + side + outcome alerted in last hour"

    recent_market_signals = _recent_sent_discord_signals(
        db,
        signal,
        datetime.utcnow() - timedelta(hours=MARKET_SUMMARY_WINDOW_HOURS),
    )
    for previous in recent_market_signals:
        if _is_opposite_binary_side(signal, previous):
            return "opposite side of same binary market already alerted"

    if recent_market_signals:
        highest_previous_rank = 0
        for previous in recent_market_signals:
            latest_alert = max(previous.alerts, key=lambda alert: alert.created_at, default=None)
            highest_previous_rank = max(
                highest_previous_rank,
                TIER_RANK[_tier_from_alert_message(latest_alert.message if latest_alert else None)],
            )
        if TIER_RANK[decision.tier] <= highest_previous_rank:
            return "market summary already sent in last 3 hours without tier upgrade"

    return None


def _has_category_match(signal: Signal) -> bool:
    trader = signal.trader
    market = signal.market
    strengths = trader.category_strengths if trader else None
    category = (market.category if market else None) or ""
    if not strengths or not category:
        return False
    normalized = category.lower().split()[0]
    for key, value in strengths.items():
        if key.lower() in normalized or normalized in key.lower():
            return float(value) >= 0.6
    return False


def _has_elite_wallet(signal: Signal) -> bool:
    trader = signal.trader
    return bool(trader and trader.trust_score >= ELITE_TRUST_SCORE)


def _market_volume_ok(ctx: AlertContext) -> bool:
    return ctx.market_volume is None or ctx.market_volume >= 25_000


def _liquidity_ok(ctx: AlertContext) -> bool:
    return ctx.liquidity is None or ctx.liquidity >= MIN_LIQUIDITY_USD


def _movement_ok(ctx: AlertContext, max_move: float) -> bool:
    return ctx.price_movement is None or ctx.price_movement <= max_move


def _chase_risk(ctx: AlertContext) -> str:
    if ctx.price_movement is None:
        return "Unknown"
    if ctx.price_movement <= 0.03:
        return "Low"
    if ctx.price_movement <= 0.07:
        return "Medium"
    return "High"


def _action_for(tier: AlertTier, ctx: AlertContext) -> str:
    if ctx.price_movement is not None and ctx.price_movement > 0.07:
        return "Avoid chasing"
    if tier == "possible_entry":
        return "Possible entry"
    if tier == "high_conviction":
        limit = ctx.current_price + 0.01 if ctx.current_price is not None else None
        return f"Watch / small entry only below {_price(limit)}" if limit else "Watch / small entry only"
    return "Watch only"


def evaluate_alert_decision(db: Session, signal: Signal, settings: Settings | None = None) -> AlertDecision:
    settings = settings or get_settings()
    ctx = _load_alert_context(db, signal)
    min_trade_size = max(100.0, settings.min_trade_size_usd)

    # Lifecycle gate first — alerts for already-finished events are pure
    # noise. Caught here even before the trade-size floor so we never page
    # an operator about a yesterday game.
    expiration_reason = market_expiration_reason(signal)
    if expiration_reason:
        return AlertDecision(
            "ignore", ctx, "Watch only", _chase_risk(ctx),
            f"event expired ({expiration_reason})",
        )
    if signal.size_usd is None or signal.size_usd <= min_trade_size:
        return AlertDecision("ignore", ctx, "Watch only", _chase_risk(ctx), "trade size below alert floor")
    if signal.entry_price is None or signal.entry_price <= 0:
        return AlertDecision("ignore", ctx, "Watch only", _chase_risk(ctx), "missing or invalid entry price")
    if signal.score < HARD_MIN_SCORE:
        return AlertDecision("ignore", ctx, "Watch only", _chase_risk(ctx), "score below hard alert floor")
    if _is_duplicate_discord_alert(db, signal, settings.duplicate_window_minutes):
        return AlertDecision("ignore", ctx, "Watch only", _chase_risk(ctx), "duplicate wallet/market/side alert window")
    if signal.score < settings.min_discord_score:
        return AlertDecision("log_only", ctx, "Watch only", _chase_risk(ctx), "score below Discord alert floor")

    elite_wallet = _has_elite_wallet(signal)
    category_match = _has_category_match(signal)
    high_conviction = (
        signal.score >= settings.high_conviction_score
        and ctx.total_tracked_size >= 1_000
        and (ctx.trader_count >= 2 or (elite_wallet and (signal.size_usd or 0) >= 2_500))
        and _market_volume_ok(ctx)
        and _movement_ok(ctx, 0.05)
    )
    possible_entry = (
        signal.score >= settings.possible_entry_score
        and ctx.total_tracked_size >= 2_500
        and (ctx.trader_count >= 3 or (elite_wallet and category_match))
        and _liquidity_ok(ctx)
        and _movement_ok(ctx, 0.07)
    )

    if possible_entry:
        tier: AlertTier = "possible_entry"
        reason = "possible entry rules matched"
    elif high_conviction:
        tier = "high_conviction"
        reason = "high conviction rules matched"
    else:
        tier = "discord_alert"
        reason = "base Discord alert rules matched"

    return AlertDecision(tier, ctx, _action_for(tier, ctx), _chase_risk(ctx), reason)


def _tier_title(tier: AlertTier) -> str:
    return {
        "ignore": "IGNORE",
        "log_only": "LOG ONLY",
        "discord_alert": "DISCORD ALERT",
        "high_conviction": "HIGH CONVICTION SIGNAL",
        "possible_entry": "POSSIBLE ENTRY",
    }[tier]


def _tier_icon(tier: AlertTier) -> str:
    return {
        "discord_alert": "🚨",
        "high_conviction": "🚨",
        "possible_entry": "🎯",
    }.get(tier, "")


def _discord_color(tier: AlertTier) -> int:
    return {
        "discord_alert": 0xF59E0B,
        "high_conviction": 0xDC2626,
        "possible_entry": 0x16A34A,
    }.get(tier, 0x64748B)


def _trader_links(ctx: AlertContext, signal: Signal) -> str:
    traders = list(ctx.aligned_traders)
    if signal.trader and all(t.id != signal.trader.id for t in traders):
        traders.insert(0, signal.trader)
    links: list[str] = []
    for trader in traders[:6]:
        label = trader.nickname or trader.wallet_address or f"trader#{trader.id}"
        if trader.wallet_address:
            links.append(f"[{label}](https://polymarketanalytics.com/traders/{trader.wallet_address})")
        else:
            links.append(label)
    if len(traders) > 6:
        links.append(f"+{len(traders) - 6} more")
    return " | ".join(links) or "n/a"


def _prob_pct(price: float | None) -> str | None:
    """Wallet trades quote `price` as a 0–1 probability already (the
    Polymarket convention). Skip the American-odds conversion path."""
    if price is None:
        return None
    try:
        p = float(price)
    except (TypeError, ValueError):
        return None
    if p <= 0 or p >= 1:
        return None
    return f"{p * 100:.1f}%"


def _entry_range_with_prob(ctx: AlertContext) -> str:
    lo = ctx.entry_price_min
    hi = ctx.entry_price_max
    if lo is None or hi is None:
        return "n/a"
    if abs(lo - hi) < 0.0001:
        prob = _prob_pct(lo)
        suffix = f" ({prob})" if prob else ""
        return f"{_price(lo)}{suffix}"
    prob_lo = _prob_pct(lo)
    prob_hi = _prob_pct(hi)
    if prob_lo and prob_hi:
        return f"{_price(lo)} ({prob_lo}) – {_price(hi)} ({prob_hi})"
    return f"{_price(lo)} – {_price(hi)}"


def _current_with_prob(ctx: AlertContext) -> str:
    price = ctx.current_price
    if price is None:
        return "n/a"
    prob = _prob_pct(price)
    return f"{_price(price)} ({prob})" if prob else _price(price)


def _price_movement_summary(ctx: AlertContext) -> str | None:
    """Entry → current, with delta and direction. None if no movement data."""
    first = ctx.first_entry_price
    current = ctx.current_price
    if first is None or current is None:
        return None
    delta = current - first
    arrow = "→" if abs(delta) < 0.0001 else ("↑" if delta > 0 else "↓")
    pct_delta = (delta / first * 100.0) if first else 0.0
    return f"{_price(first)} {arrow} {_price(current)} ({pct_delta:+.1f}%)"


def _time_to_event(market) -> tuple[str, str] | None:
    """Returns ('Starts in 4h 12m', '2026-05-26 23:00 UTC') or None."""
    if market is None:
        return None
    end = market.end_date
    if end is None:
        return None
    now = datetime.utcnow()
    delta = (end - now).total_seconds()
    if delta < 0:
        absdelta = -delta
        if absdelta < 3600:
            label = f"ended {int(absdelta // 60)}m ago"
        elif absdelta < 86400:
            label = f"ended {int(absdelta // 3600)}h ago"
        else:
            label = f"ended {int(absdelta // 86400)}d ago"
    else:
        if delta < 3600:
            label = f"in {int(delta // 60)}m"
        elif delta < 86400:
            hours = int(delta // 3600)
            mins = int((delta % 3600) // 60)
            label = f"in {hours}h {mins:02d}m"
        else:
            days = int(delta // 86400)
            hours = int((delta % 86400) // 3600)
            label = f"in {days}d {hours}h"
    iso = end.strftime("%Y-%m-%d %H:%M UTC")
    return label, iso


def _chase_emoji(risk: str) -> str:
    return {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk, "·")


def _tier_emoji(tier: AlertTier) -> str:
    """Public-facing tier glyph that matches the dashboard's vocabulary
    (HIGH CONV gold / ACTIONABLE green / WATCH gray)."""
    return {
        "high_conviction": "🟡",
        "possible_entry": "🟢",
        "discord_alert": "⚪",
    }.get(tier, "·")


def _public_tier_label(decision: AlertDecision, signal: Signal) -> str:
    """Combine the score-based dashboard label with the engine's tier so
    operators see the same vocabulary across Discord and the terminal."""
    label, _ = confidence_label(signal.score, decision.action)
    tier_label = _tier_title(decision.tier)
    if label and label != tier_label:
        return f"{label} · {tier_label}"
    return tier_label


def _build_discord_embed(signal: Signal, decision: AlertDecision) -> dict:
    ctx = decision.context
    market = signal.market
    parts = _humanize_market(signal)
    matchup = parts["matchup"]
    contract = parts["contract"]
    market_url = _market_url(signal)
    side = signal.side or "?"
    outcome = signal.outcome or ""
    side_block = f"{side} {outcome}".strip()

    first_at = ctx.first_entry_at or signal.created_at
    minutes = None
    if first_at and signal.created_at:
        minutes = max(0, int((signal.created_at - first_at).total_seconds() // 60))
    reason = signal.reason or decision.reason
    if minutes is not None and ctx.trader_count > 1:
        reason = f"{reason} ({ctx.trader_count} watched wallets within {minutes} min.)"

    tier_label = _public_tier_label(decision, signal)
    chase = _chase_emoji(decision.chase_risk)

    header_bits = [
        f"**Score:** {signal.score:.0f} · **{tier_label}**",
        f"**Smart money:** {ctx.trader_count} wallet{'s' if ctx.trader_count != 1 else ''} · {_money(ctx.total_tracked_size)}",
        f"**Side:** {side_block or 'n/a'}",
        f"**Entry:** {_entry_range_with_prob(ctx)}",
        f"**Current:** {_current_with_prob(ctx)}",
    ]
    movement = _price_movement_summary(ctx)
    if movement:
        header_bits.append(f"**Move:** {movement}")
    header_bits.append(f"**Chase risk:** {chase} {decision.chase_risk}")

    event_info = _time_to_event(market)
    if event_info:
        when_label, when_iso = event_info
        header_bits.append(f"**Event:** {when_label} · {when_iso}")
    elif market and getattr(market, "slug", None):
        slug_date = event_date_from_slug(market.slug)
        if slug_date:
            header_bits.append(f"**Event date:** {slug_date.isoformat()}")

    book_factors: list[str] = []
    breakdown = signal.score_breakdown if isinstance(signal.score_breakdown, dict) else None
    if breakdown:
        for key in (
            "trust_score",
            "consensus_score",
            "size_score",
            "category_match",
            "price_movement",
        ):
            value = breakdown.get(key)
            if value is None:
                continue
            try:
                book_factors.append(f"{factor_label(key)}: {float(value):.0f}/100")
            except (TypeError, ValueError):
                continue

    description = "\n".join(header_bits)

    fields = [
        {"name": "Action", "value": decision.action, "inline": False},
    ]
    if book_factors:
        fields.append({"name": "Edge composition", "value": "\n".join(book_factors[:5])[:1024], "inline": False})
    fields.append({"name": "Reason", "value": (reason or "n/a")[:1024], "inline": False})
    fields.append({"name": "Traders", "value": _trader_links(ctx, signal)[:1024], "inline": False})

    links = []
    if market_url:
        links.append(f"[Market]({market_url})")
    trader_url = _trader_url(signal)
    if trader_url:
        links.append(f"[Lead trader]({trader_url})")
    source = (signal.source or "").strip()
    fields.append({
        "name": "Links",
        "value": (" · ".join(links) or "n/a") + (f"  · source: {source}" if source else ""),
        "inline": False,
    })

    title_main = matchup or "Unknown market"
    if contract:
        title_main = f"{title_main} · {contract}"
    title = f"{_tier_emoji(decision.tier)} {tier_label} — {title_main}"

    embed = {
        "title": title[:256],
        "description": description[:3900],
        "color": _discord_color(decision.tier),
        "fields": fields,
        "timestamp": (signal.created_at or datetime.utcnow()).isoformat(),
        "footer": {"text": (market.slug or "")[:2048]} if market and market.slug else {"text": ""},
    }
    if market_url:
        embed["url"] = market_url
    if not embed["footer"]["text"]:
        embed.pop("footer", None)
    return embed


def _discord_payload(signal: Signal, decision: AlertDecision) -> dict:
    return {"embeds": [_build_discord_embed(signal, decision)]}


def _should_send_remote(ch: "AlertChannel", decision: AlertDecision) -> bool:
    if ch.name == "console":
        return True
    return decision.tier in DISCORD_TIERS


def _skipped_alert(
    signal: Signal,
    channel: str,
    message: str,
    decision: AlertDecision,
    reason: str | None = None,
) -> Alert:
    return Alert(
        signal_id=signal.id,
        channel=channel,
        status="skipped",
        message=message,
        error=f"{decision.tier}: {reason or decision.reason}",
        generated_for_date=signal.generated_for_date,
        created_at=datetime.utcnow(),
    )


class AlertChannel(ABC):
    name: str

    @abstractmethod
    def send(self, message: str) -> tuple[bool, str | None]:
        """Return (sent_ok, error_string_or_None)."""


class ConsoleChannel(AlertChannel):
    name = "console"

    def send(self, message: str) -> tuple[bool, str | None]:
        logger.info("[ALERT] %s", message)
        return True, None


class DiscordChannel(AlertChannel):
    name = "discord"

    def __init__(self, webhook_url: str | None) -> None:
        self._webhook_url = webhook_url

    def send(self, message: str) -> tuple[bool, str | None]:
        return self.send_payload({"content": message[:1900]})

    def send_payload(self, payload: dict) -> tuple[bool, str | None]:
        if not self._webhook_url:
            return False, "Discord webhook URL not configured"
        try:
            resp = httpx.post(self._webhook_url, json=payload, timeout=10)
            if resp.status_code >= 400:
                return False, f"Discord HTTP {resp.status_code}: {resp.text[:200]}"
            return True, None
        except httpx.HTTPError as exc:
            return False, f"{type(exc).__name__}: {exc}"


class TelegramChannel(AlertChannel):
    name = "telegram"

    def __init__(self, bot_token: str | None, chat_id: str | None) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id

    def send(self, message: str) -> tuple[bool, str | None]:
        if not (self._bot_token and self._chat_id):
            return False, "Telegram bot token or chat id not configured"
        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        try:
            resp = httpx.post(
                url,
                json={"chat_id": self._chat_id, "text": message[:4000]},
                timeout=10,
            )
            if resp.status_code >= 400:
                return False, f"Telegram HTTP {resp.status_code}: {resp.text[:200]}"
            return True, None
        except httpx.HTTPError as exc:
            return False, f"{type(exc).__name__}: {exc}"


class EmailChannel(AlertChannel):
    name = "email"

    def __init__(self, settings: Settings) -> None:
        self._to_addr = settings.alert_email_to
        self._from_addr = settings.alert_email_from or settings.smtp_username
        self._smtp_host = settings.smtp_host
        self._smtp_port = settings.smtp_port
        self._smtp_username = settings.smtp_username
        self._smtp_password = settings.smtp_password
        self._smtp_use_tls = settings.smtp_use_tls

    def send(self, message: str) -> tuple[bool, str | None]:
        if not self._to_addr:
            return False, "Email recipient not configured"
        if not (self._smtp_host and self._from_addr):
            return False, "SMTP host/from address not configured"

        email = EmailMessage()
        email["Subject"] = "SignalForge alert"
        email["From"] = self._from_addr
        email["To"] = self._to_addr
        email.set_content(message)

        try:
            with smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=10) as smtp:
                if self._smtp_use_tls:
                    smtp.starttls()
                if self._smtp_username and self._smtp_password:
                    smtp.login(self._smtp_username, self._smtp_password)
                smtp.send_message(email)
            return True, None
        except (OSError, smtplib.SMTPException) as exc:
            return False, f"{type(exc).__name__}: {exc}"


class AlertDispatcher:
    """Apply alert tiers, send eligible notifications, and persist Alert rows."""

    def __init__(self) -> None:
        s = get_settings()
        self.channels: list[AlertChannel] = [
            ConsoleChannel(),
            DiscordChannel(s.discord_webhook_url),
            TelegramChannel(s.telegram_bot_token, s.telegram_chat_id),
            EmailChannel(s),
        ]
        self.settings = s

    def dispatch(self, db: Session, signal: Signal) -> list[Alert]:
        decision = evaluate_alert_decision(db, signal, self.settings)
        message = _format_signal(signal, decision)
        alerts: list[Alert] = []
        discord_suppression = _should_suppress_discord(db, signal, decision)

        for ch in self.channels:
            if not _should_send_remote(ch, decision):
                alert = _skipped_alert(signal, ch.name, message, decision)
                db.add(alert)
                alerts.append(alert)
                continue

            if ch.name == "discord":
                if discord_suppression:
                    alert = _skipped_alert(signal, ch.name, message, decision, discord_suppression)
                    db.add(alert)
                    alerts.append(alert)
                    continue
            if isinstance(ch, DiscordChannel):
                ok, err = ch.send_payload(_discord_payload(signal, decision))
            else:
                ok, err = ch.send(message)

            alert = Alert(
                signal_id=signal.id,
                channel=ch.name,
                status="sent" if ok else "failed",
                message=message,
                error=err,
                generated_for_date=signal.generated_for_date,
                created_at=datetime.utcnow(),
            )
            db.add(alert)
            alerts.append(alert)

        db.flush()
        return alerts
