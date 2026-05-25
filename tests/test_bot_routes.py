from __future__ import annotations

from datetime import date, datetime, timedelta

from sqlalchemy.orm import Session

from app.api.routes import bot_high_conviction, bot_status
from app.models import Alert, Market, Signal, Trade, Trader


def test_bot_status_counts_recent_rows(db_session: Session) -> None:
    market = Market(slug="status-market", title="Status Market")
    trader = Trader(nickname="status-trader", wallet_address="0xabc")
    db_session.add_all([market, trader])
    db_session.flush()
    signal = Signal(
        market_id=market.id,
        trader_id=trader.id,
        signal_type="trusted_wallet_entry",
        side="BUY",
        entry_price=0.5,
        size_usd=250,
        score=70,
        source="Falcon",
    )
    db_session.add(signal)
    db_session.flush()
    db_session.add_all(
        [
            Alert(signal_id=signal.id, channel="discord", status="sent", message="sent"),
            Alert(signal_id=signal.id, channel="discord", status="skipped", message="skipped"),
        ]
    )
    db_session.commit()

    payload = bot_status(db_session)

    assert payload["traders"] == 1
    assert payload["markets"] == 1
    assert payload["signals_24h"] == 1
    assert payload["discord_sent_24h"] == 1
    assert payload["discord_skipped_24h"] == 1


def test_bot_high_conviction_returns_filtered_candidates(db_session: Session) -> None:
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
    db_session.commit()

    payload = bot_high_conviction(db_session, limit=5, hours=24)

    assert payload["count"] == 1
    candidate = payload["signals"][0]
    assert candidate["tier"] == "high_conviction"
    assert candidate["market"] == "Knicks +2.5 vs Cavs"
    assert candidate["side"] == "BUY"
    assert candidate["outcome"] == "Knicks"
    assert candidate["total_tracked_size"] == 3420


def test_bot_high_conviction_returns_near_misses_and_event_date_filter(db_session: Session) -> None:
    now = datetime.utcnow()
    old_market = Market(
        slug="mlb-old-game-2026-05-24",
        title="Old Game",
        yes_price=0.5,
        volume_24h_usd=30_000,
    )
    current_market = Market(
        slug="mlb-current-game-2026-05-25",
        title="Current Game",
        yes_price=0.5,
        volume_24h_usd=30_000,
    )
    trader = Trader(nickname="near-miss", wallet_address="0xabc", trust_score=70)
    db_session.add_all([old_market, current_market, trader])
    db_session.flush()
    db_session.add_all(
        [
            Signal(
                market_id=old_market.id,
                trader_id=trader.id,
                signal_type="trusted_wallet_entry",
                side="BUY",
                outcome="Yes",
                entry_price=0.5,
                size_usd=500,
                score=64,
                source="Falcon",
                created_at=now,
            ),
            Signal(
                market_id=current_market.id,
                trader_id=trader.id,
                signal_type="trusted_wallet_entry",
                side="BUY",
                outcome="Yes",
                entry_price=0.5,
                size_usd=500,
                score=64,
                source="Falcon",
                created_at=now,
            ),
        ]
    )
    db_session.commit()

    payload = bot_high_conviction(
        db_session,
        limit=5,
        hours=24,
        event_date_from=date(2026, 5, 25),
    )

    assert payload["count"] == 0
    assert payload["event_date_from"] == "2026-05-25"
    assert len(payload["near_misses"]) == 1
    near_miss = payload["near_misses"][0]
    assert near_miss["market"] == "Current Game"
    assert near_miss["event_date"] == "2026-05-25"
    assert near_miss["failed_reason"] == "score below hard alert floor"
