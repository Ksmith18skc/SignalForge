"""Async regime-snapshot fanout invoked from the signal-emission lifecycle.

Design:

* **Non-blocking** — ``schedule_capture(signals)`` launches a fire-and-forget
  ``asyncio.create_task``. The signal engine returns immediately; this task
  may still be running.
* **Per-agent retries** — every Falcon call goes through ``_call_with_retry``
  (default 2 attempts, exponential backoff). Failures are logged per agent
  and recorded in the snapshot's ``errors`` array.
* **Partial-data tolerant** — a missing agent leaves its slice ``None`` but
  the snapshot still persists with ``enrichment_status='partial'``.
* **Immutable** — once a snapshot exists for a ``signal_id`` it is never
  overwritten. Re-running the capture for the same signal returns the
  existing row.

Snapshot semantics deliberately diverge from the older
``SignalRegimeFeatures`` table (which was mutable). The two coexist; new
code should write/read ``SignalRegimeSnapshot``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable

from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import SignalRegimeSnapshot
from app.providers.falcon import FalconProvider, FalconResult
from app.services.falcon_intelligence import (
    _candle_volatility,
    _line_velocity,
    _orderbook_imbalance,
    _sentiment_score,
    _trade_consensus,
    _to_float,
)

logger = logging.getLogger(__name__)


# --- task management ----------------------------------------------------


@dataclass
class _CaptureRequest:
    signal_id: int
    market_slug: str
    sport: str | None = None
    market_type: str | None = None
    same_side_wallets: int | None = None
    total_watched: int | None = None
    elite_disagreement_count: int = 0


@dataclass
class CaptureSummary:
    """Result of a capture batch — used by tests and the diagnostics endpoint."""

    requested: int = 0
    persisted: int = 0
    skipped_existing: int = 0
    partial: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "persisted": self.persisted,
            "skipped_existing": self.skipped_existing,
            "partial": self.partial,
            "failed": self.failed,
            "errors": self.errors[:20],
        }


# Mutable telemetry handle the diagnostics endpoint can read.
_last_summary: CaptureSummary | None = None
_inflight_tasks: set[asyncio.Task[Any]] = set()


def get_last_capture_summary() -> dict[str, Any] | None:
    return _last_summary.as_dict() if _last_summary else None


def inflight_count() -> int:
    return len([t for t in _inflight_tasks if not t.done()])


def schedule_capture(
    requests: list[dict[str, Any]],
    *,
    falcon: FalconProvider | None = None,
    retry_count: int = 2,
) -> asyncio.Task[CaptureSummary] | None:
    """Launch a background capture pass. Returns the task handle, or
    ``None`` when no event loop is running (caller is sync — happens in
    tests; just call ``capture_batch`` directly in that case).
    """
    if not requests:
        return None
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        return None

    typed: list[_CaptureRequest] = []
    for raw in requests:
        if not raw.get("market_slug"):
            continue
        typed.append(
            _CaptureRequest(
                signal_id=int(raw["signal_id"]),
                market_slug=str(raw["market_slug"]),
                sport=raw.get("sport"),
                market_type=raw.get("market_type"),
                same_side_wallets=raw.get("same_side_wallets"),
                total_watched=raw.get("total_watched"),
                elite_disagreement_count=int(raw.get("elite_disagreement_count") or 0),
            )
        )
    if not typed:
        return None

    task = loop.create_task(_run_capture_task(typed, falcon=falcon, retry_count=retry_count))
    _inflight_tasks.add(task)
    task.add_done_callback(_inflight_tasks.discard)
    return task


async def _run_capture_task(
    requests: list[_CaptureRequest],
    *,
    falcon: FalconProvider | None,
    retry_count: int,
) -> CaptureSummary:
    """Owned async entrypoint that opens its own DB session."""
    global _last_summary
    own_provider = falcon is None
    if falcon is None:
        from app.config import get_settings

        settings = get_settings()
        if not settings.has_falcon_credentials():
            summary = CaptureSummary(requested=len(requests), failed=len(requests))
            summary.errors.append("Falcon credentials missing — capture skipped")
            _last_summary = summary
            return summary
        falcon = FalconProvider(settings.falcon_api_key, settings.falcon_base_url)

    db: Session | None = None
    try:
        db = SessionLocal()
        summary = await capture_batch(db, falcon, requests, retry_count=retry_count)
        _last_summary = summary
        return summary
    except Exception as exc:  # noqa: BLE001 - never propagate to event loop
        logger.exception("falcon_regime_capture task crashed")
        summary = CaptureSummary(
            requested=len(requests),
            failed=len(requests),
            errors=[f"task crashed: {type(exc).__name__}: {exc}"],
        )
        _last_summary = summary
        return summary
    finally:
        if db is not None:
            db.close()
        # own_provider implies httpx client lifecycle managed inside calls.


# --- capture logic -------------------------------------------------------


async def _call_with_retry(
    coro_factory: Callable[[], Awaitable[FalconResult]],
    *,
    label: str,
    attempts: int = 2,
    base_delay: float = 0.5,
    errors: list[str] | None = None,
) -> FalconResult:
    """Run a single Falcon helper with bounded retries.

    Treats ``available=False`` as a soft failure worth retrying. Exceptions
    are logged and counted as failed attempts — they never propagate.
    """
    last_result: FalconResult | None = None
    for attempt in range(max(1, attempts)):
        try:
            result = await coro_factory()
        except Exception as exc:  # noqa: BLE001
            msg = f"{label} attempt {attempt + 1}: {type(exc).__name__}: {exc}"
            logger.warning(msg)
            if errors is not None:
                errors.append(msg)
            last_result = FalconResult(agent_id=0, available=False, reason=msg)
        else:
            if result.available:
                return result
            if errors is not None and result.reason:
                errors.append(f"{label} attempt {attempt + 1}: {result.reason}")
            last_result = result
        if attempt < attempts - 1:
            await asyncio.sleep(base_delay * (2 ** attempt))
    return last_result or FalconResult(agent_id=0, available=False, reason=f"{label} unavailable")


async def _gather_components(
    falcon: FalconProvider,
    market_slug: str,
    *,
    retry_count: int,
) -> tuple[dict[str, FalconResult], list[str]]:
    """Fan out across the four enrichment agents concurrently.

    Returns a dict keyed by component name ('candles', 'orderbook', 'trades',
    'social_pulse', 'market_insights') plus the collected error log.
    """
    errors: list[str] = []
    coros = {
        "candles": _call_with_retry(
            lambda: falcon.fetch_polymarket_candles(
                market_slug=market_slug, interval="1h", limit=24,
            ),
            label="candles", attempts=retry_count, errors=errors,
        ),
        "orderbook": _call_with_retry(
            lambda: falcon.fetch_polymarket_orderbook(market_slug=market_slug),
            label="orderbook", attempts=retry_count, errors=errors,
        ),
        "trades": _call_with_retry(
            lambda: falcon.fetch_polymarket_trades(market_slug=market_slug, limit=50),
            label="trades", attempts=retry_count, errors=errors,
        ),
        "social_pulse": _call_with_retry(
            lambda: falcon.fetch_social_pulse(market_slug=market_slug),
            label="social_pulse", attempts=retry_count, errors=errors,
        ),
        "market_insights": _call_with_retry(
            lambda: falcon.fetch_market_insights(market_slug=market_slug),
            label="market_insights", attempts=retry_count, errors=errors,
        ),
    }
    results = await asyncio.gather(*coros.values(), return_exceptions=True)
    payload: dict[str, FalconResult] = {}
    for name, result in zip(coros.keys(), results):
        if isinstance(result, Exception):
            errors.append(f"{name}: {type(result).__name__}: {result}")
            payload[name] = FalconResult(agent_id=0, available=False, reason=str(result))
        else:
            payload[name] = result
    return payload, errors


def _line_acceleration(candle_rows: list[dict[str, Any]]) -> float | None:
    closes = [
        _to_float(r.get("close") or r.get("c") or r.get("price")) for r in candle_rows
    ]
    closes = [c for c in closes if c is not None]
    if len(closes) < 4:
        return None
    # Acceleration ≈ change in line velocity across the two halves of the
    # window. Useful for detecting "late steam" vs steady drift.
    mid = len(closes) // 2
    v_first = abs(closes[mid - 1] - closes[0]) / max(mid, 1)
    v_second = abs(closes[-1] - closes[mid]) / max(len(closes) - mid, 1)
    return round(v_second - v_first, 6)


def _whale_activity_score(trade_rows: list[dict[str, Any]], threshold: float = 5_000.0) -> float | None:
    """Fraction of recent notional from trades above the whale threshold."""
    total = whale_total = 0.0
    for row in trade_rows:
        size = _to_float(row.get("size_usd") or row.get("notional") or row.get("size"))
        if size is None:
            continue
        total += size
        if size >= threshold:
            whale_total += size
    if total <= 0:
        return None
    return round(whale_total / total, 4)


def _liquidity_score(orderbook_rows: list[dict[str, Any]]) -> float | None:
    """Sum of bid+ask notional, log-normalised to [0, 1]."""
    total = 0.0
    for row in orderbook_rows:
        size = _to_float(row.get("size") or row.get("amount"))
        if size is None:
            continue
        total += size
    if total <= 0:
        return None
    import math

    # 1k → ~0.3, 10k → ~0.65, 100k+ → ~1.0
    return round(min(1.0, math.log10(max(total, 1.0)) / 6.0), 4)


def _steam_state(line_velocity: float | None, line_acceleration: float | None) -> str:
    if line_velocity is None and line_acceleration is None:
        return "unknown"
    accel = line_acceleration or 0.0
    velo = line_velocity or 0.0
    if accel > 0.005 and velo > 0.01:
        return "late_steam"
    if accel < -0.003 and velo > 0.005:
        return "fading_steam"
    if velo > 0.01:
        return "steady_steam"
    return "quiet"


def _orderflow_state(imbalance: float | None, whale: float | None) -> str:
    if imbalance is None and whale is None:
        return "unknown"
    if imbalance is None:
        imbalance = 0.0
    if imbalance > 0.3:
        return "buy_pressure"
    if imbalance < -0.3:
        return "sell_pressure"
    if (whale or 0.0) > 0.5:
        return "whale_dominated"
    return "balanced"


def _sentiment_state(sentiment: float | None) -> str:
    if sentiment is None:
        return "unknown"
    if sentiment >= 0.3:
        return "bullish"
    if sentiment <= -0.3:
        return "bearish"
    return "neutral"


def _candlestick_state(candles: list[dict[str, Any]]) -> str:
    closes = [
        _to_float(r.get("close") or r.get("c") or r.get("price")) for r in candles
    ]
    closes = [c for c in closes if c is not None]
    if len(closes) < 3:
        return "unknown"
    if closes[-1] > closes[0] * 1.02:
        return "uptrend"
    if closes[-1] < closes[0] * 0.98:
        return "downtrend"
    return "rangebound"


def _classify_regime(
    *,
    volatility_score: float | None,
    steam_state: str,
    orderflow_state: str,
    sentiment_state: str,
    consensus_concentration: float | None,
    elite_disagreement_count: int,
) -> str:
    """Compact regime label used as the learning bucket key.

    Six broad buckets so each carries enough sample to learn from. The
    classifier picks the highest-signal axis available; "unknown" is
    reserved for snapshots whose enrichment was empty.
    """
    vol = volatility_score or 0.0
    if elite_disagreement_count >= 1 and (consensus_concentration or 0.0) >= 0.7:
        return "crowded_with_elite_fade"
    if steam_state == "late_steam" and vol >= 0.02:
        return "late_steam_high_vol"
    if steam_state == "steady_steam" and vol < 0.015:
        return "sharp_low_vol"
    if orderflow_state == "buy_pressure" and sentiment_state == "bullish":
        return "aligned_buy_pressure"
    if orderflow_state == "sell_pressure" and sentiment_state == "bearish":
        return "aligned_sell_pressure"
    if steam_state == "quiet" and orderflow_state == "balanced":
        return "quiet_balanced"
    return "mixed"


# --- the public entrypoint ----------------------------------------------


async def capture_batch(
    db: Session,
    falcon: FalconProvider,
    requests: list[_CaptureRequest],
    *,
    retry_count: int = 2,
) -> CaptureSummary:
    """Persist one immutable snapshot per request.

    Existing snapshots are left untouched. Each request's fanout runs
    concurrently with the others (asyncio.gather) so the wall-clock for a
    batch is roughly the slowest single request, not the sum.
    """
    summary = CaptureSummary(requested=len(requests))

    async def _one(req: _CaptureRequest) -> None:
        existing = db.get(SignalRegimeSnapshot, req.signal_id)
        if existing is not None:
            summary.skipped_existing += 1
            return

        components, errors = await _gather_components(
            falcon, req.market_slug, retry_count=retry_count,
        )
        candles = components["candles"]
        orderbook = components["orderbook"]
        trades = components["trades"]
        social = components["social_pulse"]
        insights = components["market_insights"]

        component_flags = {k: v.available for k, v in components.items()}
        any_ok = any(component_flags.values())
        all_ok = all(component_flags.values())
        enrichment_status = (
            "complete" if all_ok
            else ("partial" if any_ok else "failed")
        )
        if enrichment_status == "partial":
            summary.partial += 1
        elif enrichment_status == "failed":
            summary.failed += 1
            summary.errors.extend(errors[:5])

        candle_rows = candles.rows if candles.available else []
        orderbook_rows = orderbook.rows if orderbook.available else []
        trade_rows = trades.rows if trades.available else []

        volatility = _candle_volatility(candle_rows) if candle_rows else None
        velocity = _line_velocity(candle_rows) if candle_rows else None
        acceleration = _line_acceleration(candle_rows) if candle_rows else None
        imbalance = _orderbook_imbalance(orderbook_rows) if orderbook_rows else None
        liquidity = _liquidity_score(orderbook_rows) if orderbook_rows else None
        consensus = _trade_consensus(trade_rows) if trade_rows else {"concentration": None, "dominant_side": None}
        whale = _whale_activity_score(trade_rows) if trade_rows else None
        sentiment_val = _sentiment_score(
            social.rows if social.available else [],
            social.summary if social.available else None,
        ) if (social.available) else None
        candlestick = _candlestick_state(candle_rows) if candle_rows else "unknown"

        steam = _steam_state(velocity, acceleration)
        flow = _orderflow_state(imbalance, whale)
        sentiment_label = _sentiment_state(sentiment_val)
        classification = _classify_regime(
            volatility_score=volatility,
            steam_state=steam,
            orderflow_state=flow,
            sentiment_state=sentiment_label,
            consensus_concentration=consensus.get("concentration"),
            elite_disagreement_count=req.elite_disagreement_count,
        )

        market_price: float | None = None
        if candle_rows:
            for row in reversed(candle_rows):
                price = _to_float(row.get("close") or row.get("c") or row.get("price"))
                if price is not None:
                    market_price = price
                    break

        conflict_flags = {
            "crowded_side": bool(
                req.same_side_wallets is not None
                and req.total_watched is not None
                and req.same_side_wallets >= max(3, int((req.total_watched or 0) * 0.5))
            ),
            "elite_disagreement": req.elite_disagreement_count >= 1,
        }

        snapshot = SignalRegimeSnapshot(
            signal_id=req.signal_id,
            captured_at=datetime.utcnow(),
            market_price=market_price,
            line_velocity=velocity,
            line_acceleration=acceleration,
            volatility_score=volatility,
            liquidity_score=liquidity,
            orderflow_state=flow,
            steam_state=steam,
            sentiment_state=sentiment_label,
            orderbook_imbalance=imbalance,
            consensus_concentration=consensus.get("concentration"),
            elite_disagreement_count=req.elite_disagreement_count,
            whale_activity_score=whale,
            candlestick_state=candlestick,
            conflict_flags=conflict_flags,
            regime_classification=classification,
            components=component_flags,
            enrichment_status=enrichment_status,
            errors=errors[:20],
            raw_payload_json={
                "candles_summary": {
                    "rows": len(candle_rows),
                    "first_close": candle_rows[0].get("close") if candle_rows else None,
                    "last_close": candle_rows[-1].get("close") if candle_rows else None,
                },
                "orderbook_summary": {
                    "rows": len(orderbook_rows),
                },
                "trades_summary": {
                    "rows": len(trade_rows),
                    "consensus": consensus,
                },
                "sentiment_summary": social.summary if social.available else None,
                "insights_summary": (
                    {k: v for k, v in (insights.summary or {}).items()
                     if k in ("headline", "category", "narrative", "regime")}
                    if insights.available else None
                ),
            },
        )
        db.add(snapshot)
        db.flush()
        summary.persisted += 1

    # Run requests concurrently — the gather makes the batch I/O-bound rather
    # than serial. Exceptions per request are caught by ``return_exceptions``.
    results = await asyncio.gather(
        *[_one(r) for r in requests], return_exceptions=True,
    )
    for outcome in results:
        if isinstance(outcome, Exception):
            summary.failed += 1
            summary.errors.append(f"{type(outcome).__name__}: {outcome}")
            logger.exception("regime capture per-signal task failed")
    db.commit()
    logger.info("regime capture batch: %s", summary.as_dict())
    return summary
