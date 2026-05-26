"""Regression tests for the Falcon external_id + rollback fix.

Background:
  Falcon trades carry long composite IDs that exceeded the old VARCHAR(128)
  cap on trades.external_id, raising StringDataRightTruncation; the failure
  then poisoned the Session with PendingRollbackError and killed every
  subsequent scanner iteration. These tests pin the new behavior:
    * TEXT column accepts arbitrarily long IDs
    * A failed trade insert rolls back to a savepoint, not the whole tx
    * /health surfaces ingestion_failures, db_rollbacks, last_ingestion_error
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

import app.services.ingestion_health as ingestion_health_module
from app.models import Market, Trade, Trader
from app.providers.base import BaseProvider, ProviderSource
from app.services import ingestion
from app.services.ingestion_health import get_ingestion_health, reset


@pytest.fixture(autouse=True)
def _reset_ingestion_health() -> None:
    reset()


def _make_trader(db: Session, *, nickname: str = "wallet1") -> Trader:
    trader = Trader(
        nickname=nickname,
        wallet_address="0x" + ("a" * 40),
        platform="polymarket",
    )
    db.add(trader)
    db.commit()
    db.refresh(trader)
    return trader


class _StubProvider(BaseProvider):
    """Returns whatever trades the test hands it."""

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


def test_long_external_id_inserts_without_truncation(db_session: Session) -> None:
    trader = _make_trader(db_session)
    long_id = "falcon-" + ("x" * 800)  # well past the old VARCHAR(128) cap

    trades = [
        {
            "market_slug": "us-recession-2026",
            "market_title": "Will the US enter a recession in 2026?",
            "side": "YES",
            "price": 0.55,
            "size_usd": 1000.0,
            "external_id": long_id,
            "source": "Falcon",
        }
    ]

    asyncio.run(
        ingestion.fetch_recent_trades(db_session, trader, _providers_with(trades))
    )
    db_session.commit()

    stored = db_session.scalar(select(Trade).where(Trade.trader_id == trader.id))
    assert stored is not None
    assert stored.external_id == long_id
    assert len(stored.external_id) > 128

    health = get_ingestion_health()
    assert health.trades_inserted == 1
    assert health.ingestion_failures == 0


def test_absurdly_large_external_id_is_truncated_and_counted(db_session: Session) -> None:
    trader = _make_trader(db_session)
    absurd_id = "z" * 5000  # > 2048 cap

    trades = [
        {
            "market_slug": "us-recession-2026",
            "market_title": "Will the US enter a recession in 2026?",
            "side": "YES",
            "price": 0.55,
            "size_usd": 1000.0,
            "external_id": absurd_id,
            "source": "Falcon",
        }
    ]

    asyncio.run(
        ingestion.fetch_recent_trades(db_session, trader, _providers_with(trades))
    )
    db_session.commit()

    stored = db_session.scalar(select(Trade).where(Trade.trader_id == trader.id))
    assert stored is not None
    assert len(stored.external_id) == 2048
    assert get_ingestion_health().trades_skipped_oversized == 1


def test_failed_insert_rolls_back_savepoint_and_loop_continues(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single bad trade must not poison the Session for later trades."""
    trader = _make_trader(db_session)
    trades = [
        {
            "market_slug": "us-recession-2026",
            "market_title": "Will the US enter a recession in 2026?",
            "side": "YES",
            "price": 0.55,
            "size_usd": 1000.0,
            "external_id": "good-1",
            "source": "Falcon",
        },
        {
            "market_slug": "fed-cuts-june-2026",
            "market_title": "Will the Fed cut rates at the June 2026 meeting?",
            "side": "NO",
            "price": 0.40,
            "size_usd": 500.0,
            "external_id": "boom-2",
            "source": "Falcon",
        },
        {
            "market_slug": "btc-150k-eoy-2026",
            "market_title": "Will BTC close above $150k on 2026-12-31?",
            "side": "YES",
            "price": 0.20,
            "size_usd": 2500.0,
            "external_id": "good-3",
            "source": "Falcon",
        },
    ]

    real_ingest = ingestion._ingest_one_trade

    def _maybe_explode(db, t, entry):
        if entry.get("external_id") == "boom-2":
            # Simulate StringDataRightTruncation / IntegrityError inside the
            # savepoint — must be a SQLAlchemyError subclass so the caller
            # records it as an ingestion failure.
            from sqlalchemy.exc import IntegrityError

            raise IntegrityError("simulated", params=None, orig=Exception("boom"))
        return real_ingest(db, t, entry)

    monkeypatch.setattr(ingestion, "_ingest_one_trade", _maybe_explode)

    new_trades = asyncio.run(
        ingestion.fetch_recent_trades(db_session, trader, _providers_with(trades))
    )
    db_session.commit()

    persisted = list(
        db_session.scalars(select(Trade).where(Trade.trader_id == trader.id))
    )
    persisted_ids = {t.external_id for t in persisted}
    assert persisted_ids == {"good-1", "good-3"}
    assert len(new_trades) == 2

    health = get_ingestion_health()
    assert health.ingestion_failures == 1
    assert health.db_rollbacks >= 1
    assert health.last_ingestion_error is not None
    assert health.trades_inserted == 2


def test_session_is_reusable_after_savepoint_failure(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a savepoint-only rollback, follow-up writes on the same Session
    must commit successfully — the original bug was that the Session entered
    PendingRollbackError and every later commit raised."""
    trader = _make_trader(db_session)

    real_ingest = ingestion._ingest_one_trade
    boom_once = {"fired": False}

    def _fail_once(db, t, entry):
        if not boom_once["fired"] and entry.get("external_id") == "first":
            boom_once["fired"] = True
            from sqlalchemy.exc import IntegrityError

            raise IntegrityError("simulated", params=None, orig=Exception("boom"))
        return real_ingest(db, t, entry)

    monkeypatch.setattr(ingestion, "_ingest_one_trade", _fail_once)

    trades = [
        {
            "market_slug": "us-recession-2026",
            "market_title": "x",
            "side": "YES",
            "price": 0.55,
            "size_usd": 1000.0,
            "external_id": "first",
            "source": "Falcon",
        },
    ]
    asyncio.run(
        ingestion.fetch_recent_trades(db_session, trader, _providers_with(trades))
    )

    # Session must still be usable for unrelated work — this was the failing
    # case in production where PendingRollbackError killed the next iteration.
    db_session.add(Market(slug="after-failure", title="After failure"))
    db_session.commit()

    assert (
        db_session.scalar(select(Market).where(Market.slug == "after-failure"))
        is not None
    )
    assert get_ingestion_health().ingestion_failures == 1


def test_health_endpoint_exposes_ingestion_counters() -> None:
    from app.api.routes import health

    ingestion_health_module.record_failure("simulated truncation")
    ingestion_health_module.record_rollback()
    ingestion_health_module.record_trade_inserted(3)
    ingestion_health_module.record_trade_skipped_oversized()

    payload = health()
    assert "ingestion" in payload
    block = payload["ingestion"]
    assert block["ingestion_failures"] >= 1
    assert block["db_rollbacks"] >= 1
    assert block["last_ingestion_error"] == "simulated truncation"
    assert block["last_ingestion_error_at"] is not None
    assert block["last_rollback_at"] is not None
    assert block["trades_inserted"] >= 3
    assert block["trades_skipped_oversized"] >= 1


def test_trade_external_id_column_is_text() -> None:
    """Pin the column type so we don't silently regress to VARCHAR again."""
    from sqlalchemy import Text

    from app.models import Trade

    column = Trade.__table__.c.external_id
    assert isinstance(column.type, Text)
