"""Lightweight in-process scheduler for Falcon learning recomputes.

Design constraints:

* **Render-safe** — must run on a single web dyno without external infra
  (no Celery, no cron). The web process owns the heartbeat task.
* **Non-blocking** — every tick runs inside its own ``asyncio.create_task``
  and never awaits on the request path.
* **Resilient** — provider outages, missing tables, empty data all degrade
  to "skipped this tick" rather than crashing the loop. The last result is
  surfaced via ``get_scheduler_status`` so the dashboard can show health.

The scheduler is opt-in. ``start_scheduler(app)`` is called from
``app.main`` on startup; tests/CLI invocations leave it dormant.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable

from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.services.falcon_retraining import run_full_retraining

logger = logging.getLogger(__name__)


@dataclass
class SchedulerStatus:
    enabled: bool = False
    last_tick_at: datetime | None = None
    last_success_at: datetime | None = None
    last_error: str | None = None
    ticks: int = 0
    successes: int = 0
    failures: int = 0
    recent_summaries: list[dict[str, Any]] = field(default_factory=list)
    interval_seconds: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "last_tick_at": self.last_tick_at.isoformat() if self.last_tick_at else None,
            "last_success_at": self.last_success_at.isoformat() if self.last_success_at else None,
            "last_error": self.last_error,
            "ticks": self.ticks,
            "successes": self.successes,
            "failures": self.failures,
            "recent_summaries": self.recent_summaries[-5:],
            "interval_seconds": self.interval_seconds,
        }


_status = SchedulerStatus()
_status_lock = asyncio.Lock()
_task: asyncio.Task[Any] | None = None


def get_scheduler_status() -> dict[str, Any]:
    """Snapshot for the diagnostics endpoint. Safe to call any time."""
    return _status.as_dict()


async def _record(success: bool, summary: dict[str, Any] | None, error: str | None) -> None:
    async with _status_lock:
        _status.ticks += 1
        _status.last_tick_at = datetime.utcnow()
        if success:
            _status.successes += 1
            _status.last_success_at = _status.last_tick_at
            _status.last_error = None
            if summary is not None:
                _status.recent_summaries.append({"at": _status.last_tick_at.isoformat(), **summary})
                _status.recent_summaries = _status.recent_summaries[-20:]
        else:
            _status.failures += 1
            _status.last_error = (error or "unknown")[:240]


async def run_one_tick(
    *,
    work: Callable[[Session], Awaitable[Any] | Any] | None = None,
) -> dict[str, Any]:
    """Execute one retraining pass. Default work is ``run_full_retraining``.

    Returns the summary dict on success, ``{"error": ...}`` on failure.
    Exceptions are caught — the scheduler must never propagate them up to
    the event loop.
    """
    db: Session | None = None
    try:
        db = SessionLocal()
        if work is None:
            summary = run_full_retraining(db).as_dict()
        else:
            result = work(db)
            if asyncio.iscoroutine(result):
                result = await result
            summary = result if isinstance(result, dict) else {"result": str(result)}
        await _record(True, summary, None)
        return summary
    except Exception as exc:  # noqa: BLE001
        logger.exception("falcon scheduler tick failed")
        await _record(False, None, f"{type(exc).__name__}: {exc}")
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        if db is not None:
            db.close()


async def _loop(interval_seconds: int) -> None:
    while True:
        try:
            await run_one_tick()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - run_one_tick already catches; defensive
            logger.exception("scheduler loop swallowed unexpected error")
        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            raise


def start_scheduler(*, interval_seconds: int = 1800) -> None:
    """Start the background loop. Idempotent — calling twice is a no-op."""
    global _task
    if _task is not None and not _task.done():
        return
    _status.enabled = True
    _status.interval_seconds = interval_seconds
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        return
    _task = loop.create_task(_loop(interval_seconds))


def stop_scheduler() -> None:
    """Cancel the background loop. Safe if never started."""
    global _task
    _status.enabled = False
    if _task is None:
        return
    _task.cancel()
    _task = None


# Convenience: callers that want a one-shot "kick the scheduler" without
# waiting on the heartbeat just await this directly.
async def trigger_recompute() -> dict[str, Any]:
    return await run_one_tick()
