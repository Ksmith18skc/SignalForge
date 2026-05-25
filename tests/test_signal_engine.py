"""Tests for the signal generation engine."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from app.config import get_settings
from app.models import Market, MarketSnapshot, Trade, Trader
from app.services import signal_engine


def _seed(db, *, trust=90.0, win_rate=0.7) -> tuple[Trader, Market]:
    trader = Trader(
        nickname="alpha",
        wallet_address="0xabc",
        trust_score=trust,
        win_rate=win_rate,
        trader_rank=10,
    )
    market = Market(
        slug="m1",
        title="market 1",
        category="sports",
        yes_price=0.5,
        liquidity_usd=500_000.0,
    )
    db.add_all([trader, market])
    db.flush()
    return trader, market


def test_no_recent_trades_returns_nothing(db_session):
    signals = asyncio.run(signal_engine.generate_signals(db_session))
    assert signals == []


def test_trusted_wallet_entry_emitted(db_session, monkeypatch):
    monkeypatch.setattr(get_settings(), "signal_score_threshold", 0.0)
    trader, market = _seed(db_session)
    db_session.add(
        Trade(
            trader_id=trader.id,
            market_id=market.id,
            side="YES",
            price=0.45,
            size_usd=1_000,
            timestamp=datetime.utcnow(),
            source="Mock",
        )
    )
    db_session.flush()

    signals = asyncio.run(signal_engine.generate_signals(db_session))
    assert any(s.signal_type == "trusted_wallet_entry" for s in signals)


def test_multi_wallet_consensus_emitted(db_session, monkeypatch):
    monkeypatch.setattr(get_settings(), "signal_score_threshold", 0.0)
    t1, market = _seed(db_session)
    t2 = Trader(nickname="beta", wallet_address="0xdef", trust_score=85.0, win_rate=0.65)
    db_session.add(t2)
    db_session.flush()

    now = datetime.utcnow()
    db_session.add_all(
        [
            Trade(trader_id=t1.id, market_id=market.id, side="YES", price=0.5, size_usd=500, timestamp=now),
            Trade(trader_id=t2.id, market_id=market.id, side="YES", price=0.51, size_usd=700, timestamp=now),
        ]
    )
    db_session.flush()

    signals = asyncio.run(signal_engine.generate_signals(db_session))
    assert any(s.signal_type == "multi_wallet_consensus" for s in signals)


def test_size_threshold_emitted(db_session, monkeypatch):
    monkeypatch.setattr(get_settings(), "signal_score_threshold", 0.0)
    trader, market = _seed(db_session)
    db_session.add(
        Trade(
            trader_id=trader.id,
            market_id=market.id,
            side="NO",
            price=0.4,
            size_usd=10_000,  # well above the 5k threshold
            timestamp=datetime.utcnow(),
            source="Mock",
        )
    )
    db_session.flush()

    signals = asyncio.run(signal_engine.generate_signals(db_session))
    assert any(s.signal_type == "size_threshold" for s in signals)


def test_post_entry_price_move_emitted(db_session, monkeypatch):
    monkeypatch.setattr(get_settings(), "signal_score_threshold", 0.0)
    trader, market = _seed(db_session)
    db_session.add(
        Trade(
            trader_id=trader.id,
            market_id=market.id,
            side="YES",
            price=0.40,
            size_usd=500,
            timestamp=datetime.utcnow() - timedelta(hours=1),
            source="Mock",
        )
    )
    # Snapshot showing price moved 8pp in favor of the YES entry
    db_session.add(
        MarketSnapshot(
            market_id=market.id,
            yes_price=0.48,
            no_price=0.52,
            captured_at=datetime.utcnow(),
        )
    )
    db_session.flush()

    signals = asyncio.run(signal_engine.generate_signals(db_session))
    assert any(s.signal_type == "post_entry_price_move" for s in signals)


def test_old_trades_excluded_from_window(db_session, monkeypatch):
    monkeypatch.setattr(get_settings(), "signal_score_threshold", 0.0)
    trader, market = _seed(db_session)
    db_session.add(
        Trade(
            trader_id=trader.id,
            market_id=market.id,
            side="YES",
            price=0.5,
            size_usd=1_000,
            timestamp=datetime.utcnow() - timedelta(days=3),  # outside 24h window
            source="Mock",
        )
    )
    db_session.flush()

    signals = asyncio.run(signal_engine.generate_signals(db_session))
    assert signals == []


def test_signals_carry_required_metadata(db_session, monkeypatch):
    monkeypatch.setattr(get_settings(), "signal_score_threshold", 0.0)
    trader, market = _seed(db_session)
    db_session.add(
        Trade(
            trader_id=trader.id,
            market_id=market.id,
            side="YES",
            price=0.45,
            size_usd=2_000,
            timestamp=datetime.utcnow(),
            source="Mock",
        )
    )
    db_session.flush()

    signals = asyncio.run(signal_engine.generate_signals(db_session))
    assert signals
    sig = signals[0]
    assert sig.source in ("Falcon", "PolymarketAnalytics", "Polycopy", "Mock")
    assert sig.market_id == market.id
    assert sig.reason
    assert 0.0 <= sig.score <= 100.0
    assert 0.0 <= sig.confidence <= 1.0
    assert isinstance(sig.score_breakdown, dict)
    assert sig.score_breakdown["wallet360_confidence"] == pytest.approx(0.9)
    assert sig.score_breakdown["confidence_blend"] == pytest.approx(sig.confidence)
