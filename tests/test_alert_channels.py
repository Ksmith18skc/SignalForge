from __future__ import annotations

from datetime import date, datetime, timedelta

import httpx

from app.config import Settings
from app.models import Alert, Market, Signal, Trade, Trader
from app.services.alerts import (
    AlertDispatcher,
    DiscordChannel,
    TelegramChannel,
    _discord_payload,
    _format_signal,
    _humanize_market,
    evaluate_alert_decision,
    event_date_from_slug,
    market_expiration_reason,
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
    # Polymarket events live at the matchup level; the -total-176pt5
    # suffix is line-specific and would 404. The alerter must strip it
    # from the URL even though the slug field still carries it.
    assert "market_url=https://polymarket.com/event/wnba-por-nyl-2026-05-25 " in message
    assert "polymarket.com/event/wnba-por-nyl-2026-05-25-total" not in message
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
    assert "Knicks +2.5 vs Cavs" in embed["title"]
    assert "Side:" in embed["description"]
    assert "BUY Knicks" in embed["description"]
    # Both raw $ amount AND implied probability annotation surface in the new
    # alpha embed; the operator should not have to mentally convert.
    assert "$3,420" in embed["description"]
    assert "0.47" in embed["description"] and "0.50" in embed["description"]
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


def test_dispatcher_sends_base_discord_alert(db_session):
    now = datetime.utcnow()
    market = Market(
        slug="base-discord-alert",
        title="Base Discord Alert",
        yes_price=0.51,
        volume_24h_usd=500,
        liquidity_usd=500,
    )
    trader = Trader(nickname="sharp", wallet_address="0xbase", trust_score=70)
    db_session.add_all([market, trader])
    db_session.flush()
    signal = Signal(
        market_id=market.id,
        trader_id=trader.id,
        signal_type="trusted_wallet_entry",
        side="BUY",
        outcome="Yes",
        entry_price=0.50,
        size_usd=2_000,
        score=75,
        source="Falcon",
        reason="base alert candidate",
        created_at=now,
    )
    signal.market = market
    signal.trader = trader
    db_session.add(signal)
    db_session.flush()
    dispatcher, fake = _dispatcher_with_fake_discord()

    alerts = dispatcher.dispatch(db_session, signal)

    assert fake.sent == 1
    assert alerts[0].status == "sent"


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


# ---------------------------------------------------------------------------
# Expired-market guard + alpha-embed regression tests
# ---------------------------------------------------------------------------


def test_event_date_from_slug_extracts_ymd():
    assert event_date_from_slug("mlb-mia-tor-2026-05-25-spread-home-1pt5") == date(2026, 5, 25)
    assert event_date_from_slug("wnba-por-nyl-2026-05-25-total-176pt5") == date(2026, 5, 25)


def test_event_date_from_slug_handles_missing_or_bad():
    assert event_date_from_slug(None) is None
    assert event_date_from_slug("no-date-here") is None
    assert event_date_from_slug("bogus-9999-99-99-bogus") is None


def test_market_expiration_reason_flags_past_slug_date(db_session):
    market = Market(slug="mlb-mia-tor-2026-05-25-spread-home-1pt5", title="m", yes_price=0.5)
    db_session.add(market)
    db_session.flush()
    signal = Signal(
        market_id=market.id,
        signal_type="trusted_wallet_entry",
        side="BUY",
        outcome="Miami",
        entry_price=0.5,
        size_usd=2_000,
        score=82,
        source="Falcon",
    )
    signal.market = market

    reason = market_expiration_reason(signal, now=datetime(2026, 5, 26, 12, 0, 0))
    assert reason is not None
    assert "2026-05-25" in reason


def test_market_expiration_reason_passes_today_event(db_session):
    market = Market(slug="mlb-mia-tor-2026-05-26-spread-home-1pt5", title="m", yes_price=0.5)
    db_session.add(market)
    db_session.flush()
    signal = Signal(
        market_id=market.id,
        signal_type="trusted_wallet_entry",
        side="BUY",
        outcome="Miami",
        entry_price=0.5,
        size_usd=2_000,
        score=82,
        source="Falcon",
    )
    signal.market = market

    reason = market_expiration_reason(signal, now=datetime(2026, 5, 26, 12, 0, 0))
    assert reason is None


def test_market_expiration_reason_allows_live_game_after_utc_rollover(db_session):
    """Regression: an evening Arizona game whose slug date is "today" must not
    be flagged expired just because UTC has crossed midnight into the next day.
    2026-05-27 02:30 UTC == 2026-05-26 19:30 Arizona, so a game dated
    2026-05-26 is still in-play."""
    market = Market(slug="nba-sas-okc-2026-05-26-spread-home-5pt5", title="m", yes_price=0.5)
    db_session.add(market)
    db_session.flush()
    signal = Signal(
        market_id=market.id,
        signal_type="trusted_wallet_entry",
        side="BUY",
        outcome="Spurs",
        entry_price=0.29,
        size_usd=2_000,
        score=82,
        source="Falcon",
    )
    signal.market = market

    reason = market_expiration_reason(signal, now=datetime(2026, 5, 27, 2, 30, 0))
    assert reason is None


def test_market_expiration_reason_flags_inactive_market(db_session):
    market = Market(slug="any-market", title="m", yes_price=0.5, is_active=False)
    db_session.add(market)
    db_session.flush()
    signal = Signal(
        market_id=market.id,
        signal_type="trusted_wallet_entry",
        side="BUY",
        outcome="Yes",
        entry_price=0.5,
        size_usd=2_000,
        score=82,
        source="Falcon",
    )
    signal.market = market

    reason = market_expiration_reason(signal)
    assert reason == "market is no longer active"


def test_alert_decision_rejects_expired_market(db_session):
    """The scanner can still emit signals after a game finishes (lag, late
    trade ingestion). The webhook must never fire for those events."""
    now = datetime.utcnow()
    # Two days back so the game is unambiguously past in Arizona terms,
    # regardless of where the UTC clock sits relative to the Arizona date.
    past_day = (now - timedelta(days=2)).date().isoformat()
    market = Market(
        slug=f"mlb-mia-tor-{past_day}-spread-home-1pt5",
        title=f"mlb-mia-tor-{past_day}-spread-home-1pt5",
        yes_price=0.37,
        volume_24h_usd=30_000,
        liquidity_usd=10_000,
    )
    trader = Trader(nickname="VeryLucky888", wallet_address="0xdef", trust_score=80)
    db_session.add_all([market, trader])
    db_session.flush()
    signal = Signal(
        market_id=market.id,
        trader_id=trader.id,
        signal_type="multi_wallet_consensus",
        side="BUY",
        outcome="Miami Marlins",
        entry_price=0.58,
        size_usd=9_800,
        score=72,
        source="Falcon",
        reason="4 watched wallets BUY Miami Marlins",
        created_at=now,
    )
    signal.market = market
    signal.trader = trader
    db_session.add(signal)
    db_session.flush()

    decision = evaluate_alert_decision(db_session, signal, Settings())

    assert decision.tier == "ignore"
    assert "event expired" in decision.reason


def test_alert_decision_rejects_market_end_date_in_past(db_session):
    """Even if the slug has no date, an `end_date` in the past should
    block the alert. Grace window is short enough for next-day cleanup."""
    now = datetime.utcnow()
    market = Market(
        slug="kalshi-presidential-winner",
        title="Will X win?",
        yes_price=0.5,
        end_date=now - timedelta(days=2),
    )
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
        size_usd=5_000,
        score=82,
        source="Falcon",
        created_at=now,
    )
    signal.market = market
    signal.trader = trader
    db_session.add(signal)
    db_session.flush()

    decision = evaluate_alert_decision(db_session, signal, Settings())

    assert decision.tier == "ignore"
    assert "expired" in decision.reason


def test_humanize_market_parses_mlb_spread_slug():
    market = Market(
        slug="mlb-mia-tor-2026-05-25-spread-home-1pt5",
        title="mlb-mia-tor-2026-05-25-spread-home-1pt5",
        platform="polymarket",
    )
    signal = Signal(
        market_id=1, signal_type="x", side="BUY",
        entry_price=0.5, size_usd=100, score=70, source="Falcon",
    )
    signal.market = market
    parts = _humanize_market(signal)
    assert parts["matchup"] == "MLB MIA @ TOR"
    assert parts["contract"] == "Spread Home 1.5"


def test_humanize_market_keeps_existing_human_title():
    market = Market(
        slug="knicks-cavs-spread",
        title="Knicks +2.5 vs Cavs",
        platform="polymarket",
    )
    signal = Signal(
        market_id=1, signal_type="x", side="BUY",
        entry_price=0.5, size_usd=100, score=70, source="Falcon",
    )
    signal.market = market
    parts = _humanize_market(signal)
    assert parts["matchup"] == "Knicks +2.5 vs Cavs"
    # Already human, no contract to extract.
    assert parts["contract"] == ""


def test_discord_embed_uses_humanized_title_and_implied_prob(db_session):
    """Alpha embed regression: title shows MLB matchup + contract, body
    shows entry price with implied probability so an operator can read it
    without translating Polymarket decimals."""
    now = datetime.utcnow()
    market = Market(
        slug=f"mlb-mia-tor-{now.date().isoformat()}-spread-home-1pt5",
        title=f"mlb-mia-tor-{now.date().isoformat()}-spread-home-1pt5",
        yes_price=0.37,
        volume_24h_usd=30_000,
        liquidity_usd=10_000,
        end_date=now + timedelta(hours=4),
    )
    traders = [
        Trader(nickname="VeryLucky888", wallet_address="0xa", trust_score=82),
        Trader(nickname="bananawoin", wallet_address="0xb", trust_score=70),
    ]
    db_session.add(market)
    db_session.add_all(traders)
    db_session.flush()
    db_session.add_all([
        Trade(
            trader_id=traders[0].id, market_id=market.id, side="BUY",
            outcome="Miami Marlins", price=0.58, size_usd=4_000,
            source="Falcon", timestamp=now - timedelta(minutes=20),
        ),
        Trade(
            trader_id=traders[1].id, market_id=market.id, side="BUY",
            outcome="Miami Marlins", price=0.62, size_usd=2_500,
            source="Falcon", timestamp=now - timedelta(minutes=5),
        ),
    ])
    signal = Signal(
        market_id=market.id, trader_id=traders[0].id,
        signal_type="multi_wallet_consensus", side="BUY",
        outcome="Miami Marlins", entry_price=0.58, size_usd=4_000,
        score=82, source="Falcon", created_at=now,
        reason="4 watched wallets aligned",
    )
    signal.market = market
    signal.trader = traders[0]
    db_session.add(signal)
    db_session.flush()

    decision = evaluate_alert_decision(db_session, signal, Settings())
    payload = _discord_payload(signal, decision)
    embed = payload["embeds"][0]

    # Title contains humanized matchup + contract, not the raw slug.
    assert "MLB MIA @ TOR" in embed["title"]
    assert "Spread Home 1.5" in embed["title"]
    # Description shows implied probability annotations.
    description = embed["description"]
    assert "Score:" in description
    assert "Smart money:" in description
    # 0.58 → 58.0%
    assert "58.0%" in description or "62.0%" in description
    # Footer retains the raw slug for forensics/dedup.
    assert embed["footer"]["text"].startswith("mlb-mia-tor-")
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
