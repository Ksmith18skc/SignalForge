from __future__ import annotations

from datetime import datetime, timedelta

import httpx

from app.config import Settings
from app.models import Alert, Market, Signal, Trade, Trader
from app.services.alerts import (
    AlertDispatcher,
    DiscordChannel,
    TelegramChannel,
    _discord_payload,
    _format_signal,
    evaluate_alert_decision,
)


class _Response:
    status_code = 204
    text = ""


def test_discord_channel_posts_webhook(monkeypatch):
    calls = []

    def fake_post(url, json, timeout):  # noqa: ANN001
        calls.append((url, json, timeout))
        return _Response()

    monkeypatch.setattr(httpx, "post", fake_post)

    ok, err = DiscordChannel("https://discord.example/webhook").send("hello")

    assert ok is True
    assert err is None
    assert calls == [("https://discord.example/webhook", {"content": "hello"}, 10)]


def test_discord_channel_posts_embed_payload(monkeypatch):
    calls = []

    def fake_post(url, json, timeout):  # noqa: ANN001
        calls.append((url, json, timeout))
        return _Response()

    monkeypatch.setattr(httpx, "post", fake_post)

    payload = {"embeds": [{"title": "HIGH CONVICTION SIGNAL - Test"}]}
    ok, err = DiscordChannel("https://discord.example/webhook").send_payload(payload)

    assert ok is True
    assert err is None
    assert calls == [("https://discord.example/webhook", payload, 10)]


def test_telegram_channel_requires_chat_id():
    ok, err = TelegramChannel("token", None).send("hello")

    assert ok is False
    assert err == "Telegram bot token or chat id not configured"


def test_telegram_channel_posts_send_message(monkeypatch):
    calls = []

    def fake_post(url, json, timeout):  # noqa: ANN001
        calls.append((url, json, timeout))
        return _Response()

    monkeypatch.setattr(httpx, "post", fake_post)

    ok, err = TelegramChannel("token", "123").send("hello")

    assert ok is True
    assert err is None
    assert calls == [
        (
            "https://api.telegram.org/bottoken/sendMessage",
            {"chat_id": "123", "text": "hello"},
            10,
        )
    ]


def test_format_signal_includes_market_and_trader_links():
    trader = Trader(
        nickname="sharp",
        wallet_address="0x0720803c7cb0d0c5a928787b3b7ea148c6831cdb",
    )
    market = Market(
        slug="wnba-por-nyl-2026-05-25-total-176pt5",
        title="wnba-por-nyl-2026-05-25-total-176pt5",
        platform="polymarket",
    )
    signal = Signal(
        market_id=1,
        trader_id=1,
        signal_type="trusted_wallet_entry",
        side="BUY",
        outcome="Over",
        entry_price=0.49,
        size_usd=100,
        score=50.5,
        source="Falcon",
        reason="test",
    )
    signal.trader = trader
    signal.market = market

    message = _format_signal(signal)

    assert "outcome=Over" in message
    assert "https://polymarket.com/event/wnba-por-nyl-2026-05-25-total-176pt5" in message
    assert (
        "https://polymarketanalytics.com/traders/"
        "0x0720803c7cb0d0c5a928787b3b7ea148c6831cdb"
    ) in message


def test_alert_decision_ignores_small_trade(db_session):
    market = Market(slug="small-market", title="Small Market", yes_price=0.5)
    trader = Trader(nickname="sharp", wallet_address="0xabc", trust_score=80)
    db_session.add_all([market, trader])
    db_session.flush()
    signal = Signal(
        market_id=market.id,
        trader_id=trader.id,
        signal_type="trusted_wallet_entry",
        side="BUY",
        outcome="Yes",
        entry_price=0.5,
        size_usd=100,
        score=90,
        source="Falcon",
    )
    signal.market = market
    signal.trader = trader
    db_session.add(signal)
    db_session.flush()

    decision = evaluate_alert_decision(db_session, signal, Settings())

    assert decision.tier == "ignore"
    assert "trade size" in decision.reason


def test_alert_decision_high_conviction_from_aligned_wallets(db_session):
    now = datetime.utcnow()
    market = Market(
        slug="knicks-cavs-spread",
        title="Knicks +2.5 vs Cavs",
        yes_price=0.51,
        volume_24h_usd=30_000,
        liquidity_usd=10_000,
    )
    traders = [
        Trader(nickname="HomeRunHazard", wallet_address="0xaaa", trust_score=82),
        Trader(nickname="sharp2", wallet_address="0xbbb", trust_score=70),
    ]
    db_session.add(market)
    db_session.add_all(traders)
    db_session.flush()
    db_session.add_all(
        [
            Trade(
                trader_id=traders[0].id,
                market_id=market.id,
                side="BUY",
                outcome="Knicks",
                price=0.47,
                size_usd=2_000,
                source="Falcon",
                timestamp=now - timedelta(minutes=10),
            ),
            Trade(
                trader_id=traders[1].id,
                market_id=market.id,
                side="BUY",
                outcome="Knicks",
                price=0.50,
                size_usd=1_420,
                source="Falcon",
                timestamp=now,
            ),
        ]
    )
    signal = Signal(
        market_id=market.id,
        trader_id=traders[0].id,
        signal_type="trusted_wallet_entry",
        side="BUY",
        outcome="Knicks",
        entry_price=0.50,
        size_usd=2_000,
        score=82,
        source="Falcon",
        reason="HomeRunHazard bought Knicks",
        created_at=now,
    )
    signal.market = market
    signal.trader = traders[0]
    db_session.add(signal)
    db_session.flush()

    decision = evaluate_alert_decision(db_session, signal, Settings())
    payload = _discord_payload(signal, decision)

    assert decision.tier == "high_conviction"
    assert decision.context.trader_count == 2
    assert decision.context.total_tracked_size == 3_420
    embed = payload["embeds"][0]
    assert "HIGH CONVICTION SIGNAL" in embed["title"]
    assert "Market: Knicks +2.5 vs Cavs" in embed["description"]
    assert "Side: BUY / Knicks" in embed["description"]
    assert "Total tracked size: $3,420" in embed["description"]
    assert "Entry range: 0.47-0.50" in embed["description"]
    assert "0xaaa" not in embed["title"]


def test_alert_decision_suppresses_duplicate_discord_alert(db_session):
    now = datetime.utcnow()
    market = Market(slug="dupe-market", title="Dupe Market", yes_price=0.5)
    trader = Trader(nickname="sharp", wallet_address="0xabc", trust_score=80)
    db_session.add_all([market, trader])
    db_session.flush()
    previous_signal = Signal(
        market_id=market.id,
        trader_id=trader.id,
        signal_type="trusted_wallet_entry",
        side="BUY",
        outcome="Yes",
        entry_price=0.5,
        size_usd=500,
        score=75,
        source="Falcon",
        created_at=now - timedelta(minutes=5),
    )
    db_session.add(previous_signal)
    db_session.flush()
    db_session.add(
        Alert(
            signal_id=previous_signal.id,
            channel="discord",
            status="sent",
            message="sent",
            created_at=now - timedelta(minutes=5),
        )
    )
    signal = Signal(
        market_id=market.id,
        trader_id=trader.id,
        signal_type="trusted_wallet_entry",
        side="BUY",
        outcome="Yes",
        entry_price=0.51,
        size_usd=500,
        score=75,
        source="Falcon",
        created_at=now,
    )
    signal.market = market
    signal.trader = trader
    db_session.add(signal)
    db_session.flush()

    decision = evaluate_alert_decision(db_session, signal, Settings())

    assert decision.tier == "ignore"
    assert "duplicate" in decision.reason


class _FakeChannel:
    name = "discord"

    def __init__(self) -> None:
        self.sent = 0

    def send(self, message):  # noqa: ANN001
        self.sent += 1
        return True, None

    def send_payload(self, payload):  # noqa: ANN001
        self.sent += 1
        return True, None


def _make_high_conviction_signal(db_session, *, outcome="Yes", score=82, created_at=None):
    now = created_at or datetime.utcnow()
    market = Market(
        slug=f"market-{outcome.lower()}",
        title=f"Market {outcome}",
        yes_price=0.51,
        no_price=0.49,
        volume_24h_usd=30_000,
        liquidity_usd=10_000,
    )
    trader = Trader(nickname=f"trader-{outcome}", wallet_address=f"0x{outcome.lower()}", trust_score=82)
    aligned = Trader(
        nickname=f"aligned-{outcome}",
        wallet_address=f"0xaligned{outcome.lower()}",
        trust_score=70,
    )
    db_session.add_all([market, trader, aligned])
    db_session.flush()
    db_session.add_all(
        [
            Trade(
                trader_id=trader.id,
                market_id=market.id,
                side="BUY",
                outcome=outcome,
                price=0.49,
                size_usd=2_000,
                source="Falcon",
                timestamp=now - timedelta(minutes=10),
            ),
            Trade(
                trader_id=aligned.id,
                market_id=market.id,
                side="BUY",
                outcome=outcome,
                price=0.50,
                size_usd=1_500,
                source="Falcon",
                timestamp=now,
            ),
        ]
    )
    signal = Signal(
        market_id=market.id,
        trader_id=trader.id,
        signal_type="trusted_wallet_entry",
        side="BUY",
        outcome=outcome,
        entry_price=0.50,
        size_usd=2_000,
        score=score,
        source="Falcon",
        created_at=now,
        reason="test",
    )
    signal.market = market
    signal.trader = trader
    db_session.add(signal)
    db_session.flush()
    return signal


def _dispatcher_with_fake_discord():
    dispatcher = AlertDispatcher()
    fake = _FakeChannel()
    dispatcher.channels = [fake]
    return dispatcher, fake


def test_dispatcher_skips_discord_when_action_is_avoid_chasing(db_session):
    signal = _make_high_conviction_signal(db_session)
    decision = evaluate_alert_decision(db_session, signal, Settings())
    decision = type(decision)(
        tier=decision.tier,
        context=decision.context,
        action="Avoid chasing",
        chase_risk=decision.chase_risk,
        reason=decision.reason,
    )
    import app.services.alerts as alerts_module
    original = alerts_module.evaluate_alert_decision
    alerts_module.evaluate_alert_decision = lambda db, sig, settings=None: decision
    db_session.flush()
    dispatcher, fake = _dispatcher_with_fake_discord()
    try:
        alerts = dispatcher.dispatch(db_session, signal)
        assert fake.sent == 0
        assert alerts[0].status == "skipped"
        assert "avoid chasing" in alerts[0].error
    finally:
        alerts_module.evaluate_alert_decision = original


def test_dispatcher_skips_same_market_side_outcome_in_last_hour(db_session):
    signal = _make_high_conviction_signal(db_session)
    previous = Signal(
        market_id=signal.market_id,
        trader_id=None,
        signal_type="trusted_wallet_entry",
        side=signal.side,
        outcome=signal.outcome,
        entry_price=0.5,
        size_usd=2_000,
        score=82,
        source="Falcon",
        created_at=datetime.utcnow() - timedelta(minutes=30),
    )
    db_session.add(previous)
    db_session.flush()
    db_session.add(
        Alert(
            signal_id=previous.id,
            channel="discord",
            status="sent",
            message="[discord_alert] sent",
            created_at=datetime.utcnow() - timedelta(minutes=30),
        )
    )
    db_session.flush()
    dispatcher, fake = _dispatcher_with_fake_discord()

    alerts = dispatcher.dispatch(db_session, signal)

    assert fake.sent == 0
    assert "same market + side + outcome" in alerts[0].error


def test_dispatcher_skips_opposite_binary_side(db_session):
    signal = _make_high_conviction_signal(db_session, outcome="No")
    previous = Signal(
        market_id=signal.market_id,
        trader_id=None,
        signal_type="trusted_wallet_entry",
        side="BUY",
        outcome="Yes",
        entry_price=0.5,
        size_usd=2_000,
        score=82,
        source="Falcon",
        created_at=datetime.utcnow() - timedelta(minutes=30),
    )
    db_session.add(previous)
    db_session.flush()
    db_session.add(
        Alert(
            signal_id=previous.id,
            channel="discord",
            status="sent",
            message="[discord_alert] sent",
            created_at=datetime.utcnow() - timedelta(minutes=30),
        )
    )
    db_session.flush()
    dispatcher, fake = _dispatcher_with_fake_discord()

    alerts = dispatcher.dispatch(db_session, signal)

    assert fake.sent == 0
    assert "opposite side" in alerts[0].error


def test_dispatcher_allows_market_summary_when_tier_upgrades(db_session):
    signal = _make_high_conviction_signal(db_session, score=90)
    previous = Signal(
        market_id=signal.market_id,
        trader_id=None,
        signal_type="trusted_wallet_entry",
        side="SELL",
        outcome="Other Team",
        entry_price=0.5,
        size_usd=2_000,
        score=82,
        source="Falcon",
        created_at=datetime.utcnow() - timedelta(minutes=30),
    )
    db_session.add(previous)
    db_session.flush()
    db_session.add(
        Alert(
            signal_id=previous.id,
            channel="discord",
            status="sent",
            message="[discord_alert] sent",
            created_at=datetime.utcnow() - timedelta(minutes=30),
        )
    )
    db_session.flush()
    dispatcher, fake = _dispatcher_with_fake_discord()

    alerts = dispatcher.dispatch(db_session, signal)

    assert fake.sent == 1
    assert alerts[0].status == "sent"


def test_dispatcher_skips_market_summary_without_tier_upgrade(db_session):
    signal = _make_high_conviction_signal(db_session)
    previous = Signal(
        market_id=signal.market_id,
        trader_id=None,
        signal_type="trusted_wallet_entry",
        side="SELL",
        outcome="Other Team",
        entry_price=0.5,
        size_usd=2_000,
        score=82,
        source="Falcon",
        created_at=datetime.utcnow() - timedelta(minutes=30),
    )
    db_session.add(previous)
    db_session.flush()
    db_session.add(
        Alert(
            signal_id=previous.id,
            channel="discord",
            status="sent",
            message="[high_conviction] sent",
            created_at=datetime.utcnow() - timedelta(minutes=30),
        )
    )
    db_session.flush()
    dispatcher, fake = _dispatcher_with_fake_discord()

    alerts = dispatcher.dispatch(db_session, signal)

    assert fake.sent == 0
    assert "market summary already sent" in alerts[0].error
