"""Tests for the async regime-snapshot capture pipeline."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.models import (
    RegimeLearningStats,
    Signal,
    SignalLearningSnapshot,
    SignalRegimeSnapshot,
)
from app.providers.falcon import FalconResult
from app.services.falcon_learning import (
    capture_signal_attribution,
    record_signal_outcome,
)
from app.services.falcon_regime_capture import (
    _CaptureRequest,
    capture_batch,
)
from app.services.falcon_retraining import (
    lookup_regime_stats,
    recompute_regime_learning_stats,
)


# --- helpers ------------------------------------------------------------


def _signal(db_session, **overrides: Any) -> Signal:
    defaults = {
        "market_id": 1,
        "trader_id": None,
        "signal_type": "trusted_wallet_entry",
        "side": "BUY",
        "outcome": "YES",
        "entry_price": 0.5,
        "size_usd": 1000.0,
        "score": 70.0,
        "confidence": 0.7,
        "reason": "test",
        "source": "Falcon",
        "score_breakdown": {},
    }
    defaults.update(overrides)
    sig = Signal(**defaults)
    db_session.add(sig)
    db_session.flush()
    return sig


class _FalconStub:
    """Test double for FalconProvider that returns canned FalconResults
    and counts calls so retry behaviour is observable."""

    def __init__(
        self,
        *,
        candles: FalconResult | None = None,
        orderbook: FalconResult | None = None,
        trades: FalconResult | None = None,
        social: FalconResult | None = None,
        insights: FalconResult | None = None,
        candles_attempts_before_success: int = 0,
    ):
        self._candles = candles
        self._orderbook = orderbook
        self._trades = trades
        self._social = social
        self._insights = insights
        self._candle_remaining_failures = candles_attempts_before_success
        self.calls: dict[str, int] = {
            "candles": 0,
            "orderbook": 0,
            "trades": 0,
            "social_pulse": 0,
            "market_insights": 0,
        }

    async def fetch_polymarket_candles(self, **_: Any) -> FalconResult:
        self.calls["candles"] += 1
        if self._candle_remaining_failures > 0:
            self._candle_remaining_failures -= 1
            return FalconResult(agent_id=568, available=False, reason="transient")
        return self._candles or FalconResult(agent_id=568, available=False, reason="no candles")

    async def fetch_polymarket_orderbook(self, **_: Any) -> FalconResult:
        self.calls["orderbook"] += 1
        return self._orderbook or FalconResult(agent_id=572, available=False, reason="no book")

    async def fetch_polymarket_trades(self, **_: Any) -> FalconResult:
        self.calls["trades"] += 1
        return self._trades or FalconResult(agent_id=556, available=False, reason="no trades")

    async def fetch_social_pulse(self, **_: Any) -> FalconResult:
        self.calls["social_pulse"] += 1
        return self._social or FalconResult(agent_id=585, available=False, reason="no pulse")

    async def fetch_market_insights(self, **_: Any) -> FalconResult:
        self.calls["market_insights"] += 1
        return self._insights or FalconResult(agent_id=575, available=False, reason="no insights")


def _candle_rows(prices: list[float]) -> list[dict[str, Any]]:
    return [{"close": p} for p in prices]


# --- core capture --------------------------------------------------------


def test_capture_persists_immutable_snapshot(db_session):
    sig = _signal(db_session)
    falcon = _FalconStub(
        candles=FalconResult(
            agent_id=568, available=True,
            rows=_candle_rows([0.50, 0.51, 0.52, 0.54, 0.57, 0.59]),
        ),
        orderbook=FalconResult(
            agent_id=572, available=True,
            rows=[
                {"side": "bid", "size": 5000},
                {"side": "ask", "size": 1000},
            ],
        ),
        trades=FalconResult(
            agent_id=556, available=True,
            rows=[
                {"side": "BUY", "size_usd": 8000},
                {"side": "BUY", "size_usd": 6000},
                {"side": "SELL", "size_usd": 1000},
            ],
        ),
        social=FalconResult(
            agent_id=585, available=True, summary={"sentiment_score": 0.6},
        ),
        insights=FalconResult(
            agent_id=575, available=True, summary={"headline": "Test"},
        ),
    )

    summary = asyncio.run(capture_batch(
        db_session, falcon,
        [_CaptureRequest(signal_id=sig.id, market_slug="example-market")],
    ))

    assert summary.persisted == 1
    assert summary.skipped_existing == 0
    assert summary.partial == 0
    assert summary.failed == 0

    snap = db_session.get(SignalRegimeSnapshot, sig.id)
    assert snap is not None
    assert snap.enrichment_status == "complete"
    assert snap.steam_state in {"steady_steam", "late_steam"}
    assert snap.orderflow_state == "buy_pressure"
    assert snap.sentiment_state == "bullish"
    assert snap.regime_classification  # non-empty bucket label
    assert snap.market_price == pytest.approx(0.59, rel=1e-3)
    assert snap.line_velocity is not None and snap.line_velocity > 0
    assert snap.volatility_score is not None and snap.volatility_score >= 0
    assert snap.liquidity_score is not None
    assert snap.whale_activity_score is not None and snap.whale_activity_score > 0
    assert snap.components == {
        "candles": True,
        "orderbook": True,
        "trades": True,
        "social_pulse": True,
        "market_insights": True,
    }


def test_capture_is_immutable_on_repeat(db_session):
    sig = _signal(db_session)
    falcon = _FalconStub(
        candles=FalconResult(
            agent_id=568, available=True, rows=_candle_rows([0.5, 0.55, 0.6]),
        ),
    )
    asyncio.run(capture_batch(
        db_session, falcon,
        [_CaptureRequest(signal_id=sig.id, market_slug="x")],
    ))
    first = db_session.get(SignalRegimeSnapshot, sig.id)
    first_captured = first.captured_at
    first_classification = first.regime_classification

    # Second capture call must NOT update the row.
    second_falcon = _FalconStub(
        candles=FalconResult(
            agent_id=568, available=True, rows=_candle_rows([0.1, 0.05, 0.02]),
        ),
    )
    summary = asyncio.run(capture_batch(
        db_session, second_falcon,
        [_CaptureRequest(signal_id=sig.id, market_slug="x")],
    ))

    assert summary.persisted == 0
    assert summary.skipped_existing == 1
    db_session.expire_all()
    refreshed = db_session.get(SignalRegimeSnapshot, sig.id)
    assert refreshed.captured_at == first_captured
    assert refreshed.regime_classification == first_classification


def test_partial_falcon_failure_still_persists_snapshot(db_session):
    """Three agents fail, two succeed → snapshot persists with status=partial."""
    sig = _signal(db_session)
    falcon = _FalconStub(
        candles=FalconResult(
            agent_id=568, available=True,
            rows=_candle_rows([0.5, 0.51, 0.52]),
        ),
        trades=FalconResult(
            agent_id=556, available=True,
            rows=[{"side": "BUY", "size_usd": 100}],
        ),
        # orderbook, social, insights all return the default unavailable.
    )

    summary = asyncio.run(capture_batch(
        db_session, falcon,
        [_CaptureRequest(signal_id=sig.id, market_slug="x")],
    ))

    assert summary.persisted == 1
    assert summary.partial == 1
    snap = db_session.get(SignalRegimeSnapshot, sig.id)
    assert snap.enrichment_status == "partial"
    assert snap.components["candles"] is True
    assert snap.components["orderbook"] is False
    # Volatility derived from candles still landed despite other agent failures.
    assert snap.volatility_score is not None


def test_total_falcon_failure_still_persists_row(db_session):
    """Every agent fails → snapshot persists with status=failed and the
    errors array carries the per-agent reasons. Critical guarantee: no
    single Falcon endpoint can kill the snapshot."""
    sig = _signal(db_session)
    falcon = _FalconStub()  # everything defaults to unavailable

    summary = asyncio.run(capture_batch(
        db_session, falcon,
        [_CaptureRequest(signal_id=sig.id, market_slug="x")],
    ))

    assert summary.persisted == 1
    assert summary.failed >= 1
    snap = db_session.get(SignalRegimeSnapshot, sig.id)
    assert snap.enrichment_status == "failed"
    assert snap.errors  # non-empty
    assert snap.regime_classification  # classifier still emits a label


def test_retry_recovers_transient_candle_failure(db_session):
    """First two candle calls return unavailable; third succeeds. Retry
    logic should land us on the success and persist a complete snapshot."""
    sig = _signal(db_session)
    falcon = _FalconStub(
        candles=FalconResult(
            agent_id=568, available=True, rows=_candle_rows([0.5, 0.52, 0.55]),
        ),
        orderbook=FalconResult(
            agent_id=572, available=True,
            rows=[{"side": "bid", "size": 1000}, {"side": "ask", "size": 1000}],
        ),
        trades=FalconResult(
            agent_id=556, available=True,
            rows=[{"side": "BUY", "size_usd": 100}],
        ),
        social=FalconResult(agent_id=585, available=True, summary={"sentiment_score": 0.1}),
        insights=FalconResult(agent_id=575, available=True, summary={}),
        candles_attempts_before_success=1,
    )

    summary = asyncio.run(capture_batch(
        db_session, falcon,
        [_CaptureRequest(signal_id=sig.id, market_slug="x")],
        retry_count=2,
    ))

    assert summary.persisted == 1
    # We expected 2 attempts: one failure, then one success.
    assert falcon.calls["candles"] == 2
    snap = db_session.get(SignalRegimeSnapshot, sig.id)
    assert snap.enrichment_status == "complete"
    assert snap.components["candles"] is True


def test_retry_attempts_are_bounded(db_session):
    """If every retry fails the call count is still bounded by retry_count.
    No infinite loop, no resource explosion."""
    sig = _signal(db_session)
    falcon = _FalconStub()

    asyncio.run(capture_batch(
        db_session, falcon,
        [_CaptureRequest(signal_id=sig.id, market_slug="x")],
        retry_count=3,
    ))

    assert falcon.calls["candles"] == 3
    assert falcon.calls["orderbook"] == 3
    assert falcon.calls["trades"] == 3


# --- grading-driven regime learning -------------------------------------


def test_grading_updates_regime_stats(db_session):
    """Two graded signals share a regime classification → recompute_regime
    aggregates them into one RegimeLearningStats row."""
    # Build two graded signals, both classified as "sharp_low_vol".
    for i, (wlp, pnl, clv) in enumerate([("win", 1.0, 0.02), ("loss", -1.0, -0.01)]):
        sig = _signal(db_session, score=72.0)
        capture_signal_attribution(
            db_session,
            signal_id=sig.id,
            factors={"wallet_quality": 0.7},
            weights={"wallet_quality": 0.35},
            sport="basketball",
            market_type="trusted_wallet_entry",
            raw_score=72.0,
        )
        db_session.commit()
        record_signal_outcome(
            db_session, signal_id=sig.id,
            win_loss_push=wlp, realized_pnl=pnl, clv_points=clv,
        )
        db_session.add(SignalRegimeSnapshot(
            signal_id=sig.id,
            regime_classification="sharp_low_vol",
            steam_state="steady_steam",
            orderflow_state="balanced",
            sentiment_state="neutral",
            enrichment_status="complete",
        ))
        db_session.commit()

    summary = recompute_regime_learning_stats(db_session)
    assert summary.rows_examined == 2
    assert summary.rows_written >= 1

    stats = lookup_regime_stats(
        db_session,
        classification="sharp_low_vol",
        sport="basketball",
        market_type="trusted_wallet_entry",
        min_signals=1,
    )
    assert stats is not None
    assert stats["signals"] == 2
    assert stats["avg_roi"] == pytest.approx(0.0, abs=1e-9)
    assert stats["avg_clv"] == pytest.approx(0.005, abs=1e-9)
    assert stats["positive_clv_rate"] == pytest.approx(0.5, abs=1e-9)
    assert stats["win_rate"] == pytest.approx(0.5, abs=1e-9)


def test_lookup_regime_stats_respects_min_signals(db_session):
    """With only one graded signal in a regime, ``min_signals=5`` should
    return None — protects against drawing conclusions from small samples."""
    sig = _signal(db_session, score=72.0)
    capture_signal_attribution(
        db_session, signal_id=sig.id,
        factors={"wallet_quality": 0.7},
        weights={"wallet_quality": 0.35},
        sport="basketball", market_type="trusted_wallet_entry",
        raw_score=72.0,
    )
    db_session.commit()
    record_signal_outcome(
        db_session, signal_id=sig.id, win_loss_push="win",
        realized_pnl=1.0, clv_points=0.01,
    )
    db_session.add(SignalRegimeSnapshot(
        signal_id=sig.id,
        regime_classification="sharp_low_vol",
        enrichment_status="complete",
    ))
    db_session.commit()
    recompute_regime_learning_stats(db_session)

    stats = lookup_regime_stats(
        db_session, classification="sharp_low_vol",
        sport="basketball", market_type="trusted_wallet_entry",
        min_signals=5,
    )
    assert stats is None


# --- async fanout entrypoint -------------------------------------------


def test_capture_batch_skips_ungraded_when_snapshot_exists(db_session):
    """End-to-end: schedule path produces non-blocking task semantics."""
    sig = _signal(db_session)
    falcon = _FalconStub(
        candles=FalconResult(
            agent_id=568, available=True, rows=_candle_rows([0.5, 0.51, 0.52]),
        ),
    )

    # First batch persists.
    asyncio.run(capture_batch(
        db_session, falcon,
        [_CaptureRequest(signal_id=sig.id, market_slug="x")],
    ))
    snap = db_session.get(SignalRegimeSnapshot, sig.id)
    assert snap is not None
    captured_at = snap.captured_at

    # Second batch is a no-op even though the request is identical.
    summary = asyncio.run(capture_batch(
        db_session, _FalconStub(),
        [_CaptureRequest(signal_id=sig.id, market_slug="x")],
    ))
    assert summary.skipped_existing == 1
    db_session.expire_all()
    refreshed = db_session.get(SignalRegimeSnapshot, sig.id)
    assert refreshed.captured_at == captured_at
