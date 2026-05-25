"""Alert dispatch."""

from __future__ import annotations

import logging
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.message import EmailMessage
from typing import Literal

import httpx
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models import Alert, Signal, Trade, Trader

logger = logging.getLogger(__name__)

AlertTier = Literal["ignore", "log_only", "discord_alert", "high_conviction", "possible_entry"]
DISCORD_TIERS: set[AlertTier] = {"discord_alert", "high_conviction", "possible_entry"}
HARD_MIN_SCORE = 65.0
ELITE_TRUST_SCORE = 75.0
MIN_LIQUIDITY_USD = 5_000.0
MARKET_CONTEXT_HOURS = 24


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


def _market_url(signal: Signal) -> str | None:
    market = signal.market
    if not market or not market.slug:
        return None
    if market.platform and market.platform.lower() == "kalshi":
        return f"https://kalshi.com/markets/{market.slug.upper()}"
    return f"https://polymarket.com/event/{market.slug}"


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


def _build_discord_embed(signal: Signal, decision: AlertDecision) -> dict:
    ctx = decision.context
    market_title = signal.market.title if signal.market else f"market#{signal.market_id}"
    market_url = _market_url(signal)
    side = signal.side or "n/a"
    if signal.outcome:
        side = f"{side} / {signal.outcome}"
    smart_money = f"{ctx.trader_count} wallet{'s' if ctx.trader_count != 1 else ''} aligned"
    first_at = ctx.first_entry_at or signal.created_at
    minutes = None
    if first_at and signal.created_at:
        minutes = max(0, int((signal.created_at - first_at).total_seconds() // 60))
    reason = signal.reason or decision.reason
    if minutes is not None and ctx.trader_count > 1:
        reason = f"{reason} ({ctx.trader_count} watched wallets within {minutes} min.)"

    description = "\n".join(
        [
            f"Market: {market_title}",
            f"Side: {side}",
            f"Score: {signal.score:.0f}",
            f"Smart money: {smart_money}",
            f"Total tracked size: {_money(ctx.total_tracked_size)}",
            f"Entry range: {_entry_range(ctx)}",
            f"Current price: {_price(ctx.current_price)}",
            f"Chase risk: {decision.chase_risk}",
            f"Reason: {reason}",
        ]
    )
    links = []
    if market_url:
        links.append(f"[Market]({market_url})")
    trader_url = _trader_url(signal)
    if trader_url:
        links.append(f"[Lead trader]({trader_url})")

    embed = {
        "title": f"{_tier_icon(decision.tier)} {_tier_title(decision.tier)} - {market_title}",
        "description": description[:3900],
        "color": _discord_color(decision.tier),
        "fields": [
            {"name": "Action", "value": decision.action, "inline": False},
            {"name": "Links", "value": " | ".join(links) or "n/a", "inline": False},
            {"name": "Traders", "value": _trader_links(ctx, signal)[:1024], "inline": False},
        ],
        "timestamp": (signal.created_at or datetime.utcnow()).isoformat(),
    }
    if market_url:
        embed["url"] = market_url
    return embed


def _discord_payload(signal: Signal, decision: AlertDecision) -> dict:
    return {"embeds": [_build_discord_embed(signal, decision)]}


def _should_send_remote(ch: "AlertChannel", decision: AlertDecision) -> bool:
    if ch.name == "console":
        return True
    return decision.tier in DISCORD_TIERS


def _skipped_alert(signal: Signal, channel: str, message: str, decision: AlertDecision) -> Alert:
    return Alert(
        signal_id=signal.id,
        channel=channel,
        status="skipped",
        message=message,
        error=f"{decision.tier}: {decision.reason}",
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

        for ch in self.channels:
            if not _should_send_remote(ch, decision):
                alert = _skipped_alert(signal, ch.name, message, decision)
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
                created_at=datetime.utcnow(),
            )
            db.add(alert)
            alerts.append(alert)

        db.flush()
        return alerts
