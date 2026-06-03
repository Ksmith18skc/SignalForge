"""Concurrency + per-call-timeout behaviour for the scan's network phases.

These guard the fix for scans timing out in ``enriching_traders``: wallet
enrichment + trade fetch must run concurrently (fast) and a single hung
provider call must not stall the whole scan.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.models import Market, Trade, Trader
from app.services import ingestion


class _SlowProvider:
    """Every call sleeps ``delay`` seconds, simulating a slow Falcon dyno."""

    def __init__(self, delay: float = 0.4):
        self.delay = delay

    async def get_trader_stats(self, key):
        await asyncio.sleep(self.delay)
        return {"trust_score": 60.0}

    async def get_trader_trades(self, key, limit=20):
        await asyncio.sleep(self.delay)
        return [{
            "market_slug": "mlb-nyy-oak-2026-05-30", "market_title": "NYY @ OAK",
            "side": "BUY", "outcome": "Over", "price": 0.5, "size_usd": 10.0,
            "external_id": f"{key}-1",
        }]

    async def get_market_data(self, slug):
        await asyncio.sleep(self.delay)
        return {"yes_price": 0.5, "no_price": 0.5, "liquidity_usd": 1000.0}


class _HangProvider(_SlowProvider):
    async def get_trader_stats(self, key):
        await asyncio.sleep(60)

    async def get_trader_trades(self, key, limit=20):
        await asyncio.sleep(60)
        return []


def _traders(n):
    out = []
    for i in range(n):
        t = Trader(nickname=f"w{i}", wallet_address=f"0x{i:040d}", platform="polymarket")
        t.id = i + 1
        out.append(t)
    return out


@pytest.mark.asyncio
async def test_gather_runs_concurrently_not_serially():
    providers = {"primary": _SlowProvider(0.4), "polymarket": _SlowProvider(0.4)}
    traders = _traders(10)
    fired: list[int] = []

    t0 = time.perf_counter()
    payloads = await ingestion.gather_trader_payloads(
        traders, providers, per_call_timeout=5.0, concurrency=8,
        on_wallet_done=lambda tr, trades: fired.append(tr.id),
    )
    elapsed = time.perf_counter() - t0

    # Serial would be 10 wallets * 0.4s ~= 4s+. Concurrent is well under 2s.
    assert elapsed < 2.0, f"gather was not concurrent ({elapsed:.1f}s)"
    assert len(payloads) == 10
    assert len(fired) == 10  # progress callback fired per wallet
    assert payloads[1]["trades"]


@pytest.mark.asyncio
async def test_per_call_timeout_caps_a_hung_wallet():
    providers = {"primary": _HangProvider(), "polymarket": _HangProvider()}
    t0 = time.perf_counter()
    payloads = await ingestion.gather_trader_payloads(
        _traders(3), providers, per_call_timeout=1.0, concurrency=8,
    )
    elapsed = time.perf_counter() - t0
    # Capped at ~1s by the per-call timeout, NOT the 60s hang.
    assert elapsed < 4.0, f"hung provider stalled the gather ({elapsed:.1f}s)"
    assert all(p["trades"] == [] for p in payloads.values())


def test_persist_trader_payload_writes_trades(db_session):
    trader = Trader(nickname="surf", wallet_address="0xabc", platform="polymarket")
    db_session.add(trader)
    db_session.flush()
    payload = {
        "stats": {"trust_score": 75.0},
        "poly": None,
        "trades": [{
            "market_slug": "mlb-nyy-oak-2026-05-30", "market_title": "NYY @ OAK",
            "side": "BUY", "outcome": "Over", "price": 0.5, "size_usd": 25.0,
            "external_id": "abc-1",
        }],
    }
    new_trades = ingestion.persist_trader_payload(db_session, trader, payload)
    assert len(new_trades) == 1
    assert trader.trust_score == 75.0
    assert db_session.query(Trade).count() == 1
    assert db_session.query(Market).filter_by(slug="mlb-nyy-oak-2026-05-30").count() == 1
