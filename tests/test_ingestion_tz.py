"""Regression: provider-supplied ISO timestamps must not leak aware
datetimes onto Market.end_date / Trade.timestamp.

Symptom that prompted this test: a wallet scan failed with
``TypeError: can't compare offset-naive and offset-aware datetimes``
from ``alerts.is_market_done_at``, where the comparison was
``market.end_date + timedelta(...) < now`` and ``now`` was
``datetime.utcnow()`` (naive) while ``market.end_date`` had been set
from ``"...+00:00"`` by ``datetime.fromisoformat`` (aware).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Market, Signal, Trade, Trader
from app.providers.base import BaseProvider, ProviderSource
from app.services import alerts as alerts_module
from app.services import ingestion
from app.services.ingestion import _to_naive_utc


# ---------------------------------------------------------------------------
# Boundary helper
# ---------------------------------------------------------------------------


class TestToNaiveUtc:
    """The single funnel everything provider-side must go through."""

    def test_strips_tz_offset(self) -> None:
        out = _to_naive_utc("2026-05-28T19:27:26+00:00")
        assert out == datetime(2026, 5, 28, 19, 27, 26)
        assert out.tzinfo is None

    def test_strips_z_suffix(self) -> None:
        out = _to_naive_utc("2026-05-28T19:27:26Z")
        assert out == datetime(2026, 5, 28, 19, 27, 26)
        assert out.tzinfo is None

    def test_converts_non_utc_offset_to_utc(self) -> None:
        """A ``-05:00`` (Eastern) time must be shifted to UTC before
        the tz is stripped — otherwise we'd silently mis-stamp."""
        out = _to_naive_utc("2026-05-28T14:27:26-05:00")
        assert out == datetime(2026, 5, 28, 19, 27, 26)
        assert out.tzinfo is None

    def test_preserves_already_naive_datetime(self) -> None:
        naive = datetime(2026, 5, 28, 19, 27, 26)
        assert _to_naive_utc(naive) == naive

    def test_strips_tz_from_aware_datetime_object(self) -> None:
        aware = datetime(2026, 5, 28, 19, 27, 26, tzinfo=timezone.utc)
        out = _to_naive_utc(aware)
        assert out == datetime(2026, 5, 28, 19, 27, 26)
        assert out.tzinfo is None

    def test_none_passes_through(self) -> None:
        assert _to_naive_utc(None) is None

    def test_unparseable_string_returns_none(self) -> None:
        assert _to_naive_utc("not a date") is None


# ---------------------------------------------------------------------------
# End-to-end regression — the exact failure path from the bug report
# ---------------------------------------------------------------------------


class _StubProvider(BaseProvider):
    """Provider that returns timestamps with the ``+00:00`` offset that
    triggered the original bug."""

    source = ProviderSource.FALCON

    def __init__(self, trades: list[dict[str, Any]]) -> None:
        self._trades = trades

    async def get_trader_stats(self, wallet: str) -> dict[str, Any]:  # noqa: D401
        return {}

    async def get_trader_trades(self, wallet: str, limit: int = 50) -> list[dict[str, Any]]:
        return list(self._trades)

    async def get_market_data(self, market_slug: str) -> dict[str, Any]:
        return {"slug": market_slug, "title": market_slug}

    async def get_orderbook(self, market_slug: str) -> dict[str, Any]:
        return {"market_slug": market_slug, "bids": [], "asks": []}

    async def get_cross_market_comparison(self, topic: str) -> list[dict[str, Any]]:
        return []

    async def get_sentiment_signals(self, market_slug: str) -> dict[str, Any]:
        return {"market_slug": market_slug}


def _providers_with(trades: list[dict[str, Any]]) -> dict[str, BaseProvider]:
    stub = _StubProvider(trades)
    return {"primary": stub, "polymarket": stub, "kalshi": stub, "mock": stub}


def _make_trader(db: Session) -> Trader:
    t = Trader(nickname="wallet1", wallet_address="0x" + ("a" * 40), platform="polymarket")
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


def test_trade_timestamp_with_tz_offset_is_stored_naive(db_session: Session) -> None:
    """A provider timestamp like ``"...+00:00"`` must land on
    ``Trade.timestamp`` as a naive datetime.
    """
    trader = _make_trader(db_session)
    trades = [
        {
            "market_slug": "us-recession-2026",
            "market_title": "Will the US enter a recession in 2026?",
            "side": "YES",
            "price": 0.55,
            "size_usd": 1000.0,
            "external_id": "tz-aware-1",
            "source": "Falcon",
            "timestamp": "2026-05-28T19:27:26+00:00",
        }
    ]

    asyncio.run(
        ingestion.fetch_recent_trades(db_session, trader, _providers_with(trades))
    )
    db_session.commit()

    stored = db_session.scalar(select(Trade).where(Trade.trader_id == trader.id))
    assert stored is not None
    assert stored.timestamp.tzinfo is None
    assert stored.timestamp == datetime(2026, 5, 28, 19, 27, 26)


def test_market_end_date_with_tz_offset_does_not_break_alert_decision(
    db_session: Session,
) -> None:
    """The exact path that exploded the wallet scan.

    Reproduction:
      * Ingest a market whose ``end_date`` arrives as ``"...+00:00"``.
      * ``is_market_done_at`` then compares it against naive utcnow.
    Without the fix this raises:
      ``TypeError: can't compare offset-naive and offset-aware datetimes``.
    """
    market_data = {
        "slug": "mlb-tor-bal-2026-05-28-total-9pt5",
        "title": "MLB TOR @ BAL Total 9.5",
        "platform": "polymarket",
        "end_date": "2026-05-28T19:27:26+00:00",
    }
    market = ingestion.upsert_market_from_dict(db_session, market_data)
    db_session.commit()

    # End date must be stored naive — the entire downstream alert path
    # assumes naive utcnow comparisons.
    assert market.end_date is not None
    assert market.end_date.tzinfo is None

    # Now run the actual comparison that used to blow up. With the fix
    # in place this returns either None (market still open) or a string
    # reason (market done) — *not* a TypeError.
    trader = _make_trader(db_session)
    signal = Signal(
        market_id=market.id,
        trader_id=trader.id,
        signal_type="trusted_wallet_entry",
        side="BUY",
        outcome="Over",
        entry_price=0.55,
        size_usd=100.0,
        score=70.0,
        source="Falcon",
        reason="test",
    )
    signal.market = market
    signal.trader = trader

    # If the bug regresses, this call raises TypeError. The point of
    # this assertion is "no exception" — the precise return value
    # depends on grace_hours, so we just check the type rather than
    # locking ourselves to the current grace policy.
    result = alerts_module.market_expiration_reason(
        signal,
        now=datetime(2026, 5, 28, 12, 0, 0),  # 7h before end_date
    )
    assert result is None or isinstance(result, str)


def test_alert_decision_tolerates_legacy_aware_end_date_on_market(
    db_session: Session,
) -> None:
    """Belt-and-suspenders: even if a row written by an older build
    still holds an aware datetime, the alert decision must defensively
    strip the tz instead of letting the comparison raise.
    """
    trader = _make_trader(db_session)
    market = Market(
        slug="mlb-tor-bal-legacy",
        title="MLB legacy aware row",
        platform="polymarket",
    )
    # Bypass ingestion to simulate a row that escaped the boundary fix.
    market.end_date = datetime(2026, 5, 28, 19, 27, 26, tzinfo=timezone.utc)
    db_session.add(market)
    db_session.commit()

    signal = Signal(
        market_id=market.id,
        trader_id=trader.id,
        signal_type="trusted_wallet_entry",
        side="BUY",
        outcome="Over",
        entry_price=0.55,
        size_usd=100.0,
        score=70.0,
        source="Falcon",
        reason="test",
    )
    signal.market = market
    signal.trader = trader

    # Must not raise even though market.end_date is aware.
    result = alerts_module.market_expiration_reason(
        signal,
        now=datetime(2026, 5, 30, 0, 0, 0),  # well past the grace window
    )
    assert isinstance(result, str)
    assert "already passed" in result
