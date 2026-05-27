"""Tests for the in-process Falcon recompute scheduler."""

from __future__ import annotations

import asyncio

import pytest

from app.services import falcon_scheduler


@pytest.fixture(autouse=True)
def _reset_scheduler_state():
    falcon_scheduler._status = falcon_scheduler.SchedulerStatus()
    yield
    falcon_scheduler.stop_scheduler()
    falcon_scheduler._status = falcon_scheduler.SchedulerStatus()


def test_run_one_tick_records_success(db_session):
    async def work(_db):
        return {"ok": True, "rows": 7}

    result = asyncio.run(falcon_scheduler.run_one_tick(work=work))

    assert result == {"ok": True, "rows": 7}
    status = falcon_scheduler.get_scheduler_status()
    assert status["ticks"] == 1
    assert status["successes"] == 1
    assert status["failures"] == 0
    assert status["last_success_at"] is not None


def test_run_one_tick_swallows_exceptions(db_session):
    async def work(_db):
        raise RuntimeError("boom")

    result = asyncio.run(falcon_scheduler.run_one_tick(work=work))

    assert "error" in result
    assert "RuntimeError" in result["error"]
    status = falcon_scheduler.get_scheduler_status()
    assert status["ticks"] == 1
    assert status["successes"] == 0
    assert status["failures"] == 1
    assert "RuntimeError" in (status["last_error"] or "")


def test_run_one_tick_handles_sync_callable(db_session):
    def work(_db):
        return {"sync": True}

    result = asyncio.run(falcon_scheduler.run_one_tick(work=work))
    assert result == {"sync": True}
