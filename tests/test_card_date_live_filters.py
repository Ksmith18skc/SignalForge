from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select, update

from app.api.routes import list_alerts, list_signals
from app.config import get_settings
from app.models import Alert, Market, Signal, Trade, Trader
from app.providers.base import ProviderSource
from app.services import scanner


def _market(slug: str, title: str | None = None) -> Market:
    return Market(
        slug=slug,
        title=title or slug,
        category="sports",
        yes_price=0.52,
        no_price=0.48,
        liquidity_usd=25_000,
        volume_24h_usd=50_000,
        is_active=True,
    )


def _signal(db_session, market: Market, trader: Trader, *, generated_for_date: str | None) -> Signal:
    signal = Signal(
        market_id=market.id,
        trader_id=trader.id,
        signal_type="trusted_wallet_entry",
        side="BUY",
        outcome="Over",
        entry_price=0.52,
        size_usd=1_000,
        score=80,
        confidence=0.8,
        source="Falcon",
        reason=f"{trader.nickname} bought {market.slug}",
        generated_for_date=generated_for_date,
    )
    db_session.add(signal)
    db_session.flush()
    return signal


def test_may_27_dashboard_excludes_may_26_markets(db_session) -> None:
    trader = Trader(nickname="sharp", wallet_address="0xabc", trust_score=80)
    old_market = _market("mlb-old-game-2026-05-26-total")
    today_market = _market("mlb-today-game-2026-05-27-total")
    db_session.add_all([trader, old_market, today_market])
    db_session.flush()
    _signal(db_session, old_market, trader, generated_for_date="2026-05-26")
    _signal(db_session, today_market, trader, generated_for_date="2026-05-27")
    db_session.commit()

    rows = list_signals(
        db_session,
        limit=50,
        date="2026-05-27",
        active_only=True,
        exclude_resolved=True,
    )

    assert [row.market_slug for row in rows] == ["mlb-today-game-2026-05-27-total"]


def test_prior_day_alerts_hidden_from_today_view(db_session) -> None:
    trader = Trader(nickname="sharp", wallet_address="0xabc", trust_score=80)
    old_market = _market("nba-old-game-2026-05-26-spread")
    today_market = _market("nba-today-game-2026-05-27-spread")
    db_session.add_all([trader, old_market, today_market])
    db_session.flush()
    old_signal = _signal(db_session, old_market, trader, generated_for_date="2026-05-26")
    today_signal = _signal(db_session, today_market, trader, generated_for_date="2026-05-27")
    db_session.add_all(
        [
            Alert(
                signal_id=old_signal.id,
                channel="discord",
                status="sent",
                message="old",
                generated_for_date="2026-05-26",
            ),
            Alert(
                signal_id=today_signal.id,
                channel="discord",
                status="sent",
                message="today",
                generated_for_date="2026-05-27",
            ),
        ]
    )
    db_session.commit()

    rows = list_alerts(db_session, limit=50, date="2026-05-27")

    assert [row.message for row in rows] == ["today"]


def test_prior_day_positions_hidden_from_active_wallet_flow(db_session) -> None:
    trader = Trader(nickname="sharp", wallet_address="0xabc", trust_score=80)
    prior_market = _market("mlb-prior-game-2026-05-26-moneyline")
    today_market = _market("mlb-live-game-2026-05-27-moneyline")
    prior_market.is_active = True
    today_market.is_active = True
    db_session.add_all([trader, prior_market, today_market])
    db_session.flush()
    _signal(db_session, prior_market, trader, generated_for_date="2026-05-26")
    _signal(db_session, today_market, trader, generated_for_date="2026-05-27")
    db_session.commit()

    rows = list_signals(
        db_session,
        limit=50,
        date="2026-05-27",
        active_only=True,
        exclude_resolved=True,
    )

    assert len(rows) == 1
    assert rows[0].market_slug == "mlb-live-game-2026-05-27-moneyline"


def test_market_slug_date_fallback_works(db_session) -> None:
    trader = Trader(nickname="fallback", wallet_address="0xdef", trust_score=80)
    market = _market("mlb-slug-only-2026-05-27-total")
    db_session.add_all([trader, market])
    db_session.flush()
    signal = _signal(db_session, market, trader, generated_for_date=None)
    db_session.execute(
        update(Signal)
        .where(Signal.id == signal.id)
        .values(generated_for_date=None)
    )
    db_session.commit()

    stored = db_session.get(Signal, signal.id)
    assert stored is not None
    assert stored.generated_for_date is None

    rows = list_signals(
        db_session,
        limit=50,
        date="2026-05-27",
        active_only=True,
        exclude_resolved=True,
    )

    assert [row.market_slug for row in rows] == ["mlb-slug-only-2026-05-27-total"]


class _FakeProvider:
    source = ProviderSource.MOCK

    async def get_trader_stats(self, wallet: str) -> dict[str, Any]:
        return {"trust_score": 90, "source": self.source.value}

    async def get_trader_trades(self, wallet: str, limit: int = 50) -> list[dict[str, Any]]:
        return [
            {
                "market_slug": "mlb-prior-scan-2026-05-26-total",
                "market_title": "Prior scan",
                "category": "sports",
                "side": "YES",
                "outcome": "Over",
                "price": 0.5,
                "size_usd": 8_000,
                "timestamp": datetime.utcnow().isoformat(),
                "external_id": f"{wallet}-old",
                "source": self.source.value,
            },
            {
                "market_slug": "mlb-today-scan-2026-05-27-total",
                "market_title": "Today scan",
                "category": "sports",
                "side": "YES",
                "outcome": "Over",
                "price": 0.5,
                "size_usd": 8_000,
                "timestamp": datetime.utcnow().isoformat(),
                "external_id": f"{wallet}-today",
                "source": self.source.value,
            },
        ]

    async def get_market_data(self, market_slug: str) -> dict[str, Any]:
        return {
            "slug": market_slug,
            "title": market_slug,
            "category": "sports",
            "platform": "polymarket",
            "yes_price": 0.55,
            "no_price": 0.45,
            "liquidity_usd": 25_000,
            "volume_24h_usd": 50_000,
            "end_date": (datetime.utcnow() + timedelta(hours=4)).isoformat(),
            "is_active": True,
            "source": self.source.value,
        }

    async def get_orderbook(self, market_slug: str) -> dict[str, Any]:
        return {}

    async def get_cross_market_comparison(self, topic: str) -> list[dict[str, Any]]:
        return []

    async def get_sentiment_signals(self, market_slug: str) -> dict[str, Any]:
        return {}

    async def list_active_markets(self, limit: int = 25) -> list[dict[str, Any]]:
        return []


def test_scan_preserves_history_but_writes_today_rows(db_session, monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "signal_score_threshold", 0.0)
    fake = _FakeProvider()
    monkeypatch.setattr(
        "app.services.ingestion.build_providers",
        lambda: {"primary": fake, "polymarket": fake, "kalshi": fake, "mock": fake},
    )
    trader = Trader(nickname="scan-sharp", wallet_address="0xabc", trust_score=90)
    db_session.add(trader)
    db_session.commit()

    result = asyncio.run(scanner.run_scan_once(card_date="2026-05-27"))

    assert result.generated_for_date == "2026-05-27"
    assert result.markets_for_card_date == 1
    assert result.stale_markets_skipped == 1
    assert result.positions_written > 0

    trades = list(db_session.scalars(select(Trade)))
    assert {trade.market.slug for trade in trades} == {
        "mlb-prior-scan-2026-05-26-total",
        "mlb-today-scan-2026-05-27-total",
    }
    signals = list(db_session.scalars(select(Signal)))
    assert signals
    assert {signal.market.slug for signal in signals} == {"mlb-today-scan-2026-05-27-total"}
    assert {signal.generated_for_date for signal in signals} == {"2026-05-27"}
