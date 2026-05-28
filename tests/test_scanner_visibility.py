"""Wallet-scan visibility + watchdog tests.

The dashboard polls ``scan_status()`` on every render — these tests are
the safety net behind the contract that:

* counters + per-wallet rows are exposed live as the scan runs,
* a stuck scan can never appear "running" longer than the wall-clock cap
  (defensive reap on every status read),
* the diagnostics dry-run never mutates the manual-scan status.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from app.services import scanner


@pytest.fixture(autouse=True)
def _reset_scan_status():
    """Each test starts (and ends) with a clean scan status — the
    module-level dict is shared, so cross-test leakage would otherwise
    bite us hard."""
    scanner.reset_scan_status()
    yield
    scanner.reset_scan_status()


# ---------------------------------------------------------------------------
# ScanProgress counters + per-wallet recording
# ---------------------------------------------------------------------------


def test_scan_progress_add_increments_named_counters() -> None:
    p = scanner.ScanProgress()
    p.add(wallets_scanned=3, raw_positions_found=7, api_errors=2)
    snap = p.snapshot()
    assert snap["wallets_scanned"] == 3
    assert snap["raw_positions_found"] == 7
    assert snap["api_errors"] == 2
    # Unknown keys are silently ignored — call sites shouldn't crash if
    # they bump a counter that hasn't been added to the dataclass yet.
    p.add(not_a_real_field=99)


def test_scan_progress_record_wallet_appends_then_updates() -> None:
    """First record_wallet for a given address appends; subsequent
    calls update the same row so a retry doesn't double-count.
    """
    p = scanner.ScanProgress()
    p.record_wallet(
        nickname="surf", address="0xabc", status="ok",
        raw_positions=4, active_positions=4, last_market="mlb-tor-bal-2026-05-28",
    )
    p.record_wallet(
        nickname="surf", address="0xabc", status="provider_error",
        raw_positions=4, active_positions=4, error="timeout",
    )
    rows = p.snapshot()["per_wallet"]
    assert len(rows) == 1
    assert rows[0]["status"] == "provider_error"
    assert rows[0]["error"] == "timeout"


def test_scan_progress_set_phase_updates_snapshot() -> None:
    p = scanner.ScanProgress()
    p.set_phase("fetching_trades")
    assert p.snapshot()["phase"] == "fetching_trades"


# ---------------------------------------------------------------------------
# Watchdog timeout: stuck scan → state=timeout, error explains why
# ---------------------------------------------------------------------------


def test_scan_status_reaps_expired_running_scan() -> None:
    """``scan_status()`` is the defensive escape hatch. If a watchdog
    thread crashed and left the status stuck on "running" past its
    timeout deadline, the next read must transition it to "timeout"
    on its own — that's the guarantee the dashboard relies on.
    """
    now = datetime.utcnow()
    past = (now - timedelta(seconds=10)).isoformat()
    scanner._manual_scan_status.update(  # noqa: SLF001
        {
            "state": "running",
            "started_at": (now - timedelta(seconds=300)).isoformat(),
            "timeout_at": past,
            "max_seconds": 180,
            "generated_for_date": "2026-05-28",
            "progress": scanner.ScanProgress(),
        }
    )

    status = scanner.scan_status()

    assert status["state"] == "timeout"
    assert "wall-clock cap" in (status.get("error") or "")
    assert status["finished_at"] is not None


def test_scan_status_does_not_reap_fresh_running_scan() -> None:
    """A scan that's still inside its deadline must NOT be reaped — the
    reaper's only job is to release the dashboard from a wedged status,
    not to interrupt healthy in-flight work.
    """
    now = datetime.utcnow()
    future = (now + timedelta(seconds=120)).isoformat()
    scanner._manual_scan_status.update(  # noqa: SLF001
        {
            "state": "running",
            "started_at": now.isoformat(),
            "timeout_at": future,
            "max_seconds": 180,
            "generated_for_date": "2026-05-28",
            "progress": scanner.ScanProgress(),
        }
    )

    status = scanner.scan_status()

    assert status["state"] == "running"


def test_scan_status_returns_serializable_progress_snapshot() -> None:
    """The dashboard pulls the status payload over HTTP and JSON-decodes
    it. The progress object must come back as a plain dict, not the
    ScanProgress instance (which carries an RLock that can't be
    serialized).
    """
    progress = scanner.ScanProgress()
    progress.add(wallets_loaded=3, wallets_scanned=2)
    progress.record_wallet(nickname="x", address="0x1", status="ok")
    scanner._manual_scan_status.update(  # noqa: SLF001
        {
            "state": "running",
            "started_at": datetime.utcnow().isoformat(),
            "timeout_at": (datetime.utcnow() + timedelta(seconds=60)).isoformat(),
            "progress": progress,
        }
    )

    status = scanner.scan_status()

    assert isinstance(status["progress"], dict)
    assert status["progress"]["wallets_loaded"] == 3
    assert status["progress"]["per_wallet"][0]["nickname"] == "x"


# ---------------------------------------------------------------------------
# trigger_manual_scan_background: timeout_at, idempotency, reset
# ---------------------------------------------------------------------------


class _NoopThread:
    """Drop-in for threading.Thread that never actually runs the target —
    lets us inspect the status payload set up *before* the scan starts.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def start(self) -> None:
        return None


def test_trigger_manual_scan_initializes_progress_and_timeout(monkeypatch) -> None:
    monkeypatch.setattr(scanner.threading, "Thread", _NoopThread)

    payload = scanner.trigger_manual_scan_background(
        card_date="2026-05-28", max_seconds=180,
    )

    assert payload["state"] == "running"
    assert payload["accepted"] is True
    assert payload["max_seconds"] == 180
    # timeout_at must land exactly ``max_seconds`` after started_at —
    # this is the contract the watchdog relies on.
    started = datetime.fromisoformat(str(payload["started_at"]))
    expires = datetime.fromisoformat(str(payload["timeout_at"]))
    assert abs((expires - started - timedelta(seconds=180)).total_seconds()) < 1.0
    # Progress is the live ScanProgress when read from the module dict,
    # but the status payload returns a snapshot dict.
    assert isinstance(payload["progress"], dict)


def test_trigger_manual_scan_is_idempotent_while_running(monkeypatch) -> None:
    monkeypatch.setattr(scanner.threading, "Thread", _NoopThread)

    first = scanner.trigger_manual_scan_background(card_date="2026-05-28")
    second = scanner.trigger_manual_scan_background(card_date="2026-05-28")

    assert first["accepted"] is True
    assert second["accepted"] is False
    assert "already running" in str(second["message"])


def test_reset_scan_status_drops_running_state() -> None:
    scanner._manual_scan_status.update(  # noqa: SLF001
        {"state": "running", "started_at": datetime.utcnow().isoformat()}
    )
    payload = scanner.reset_scan_status()
    assert payload["state"] == "idle"
    assert payload["progress"] is None


# ---------------------------------------------------------------------------
# Diagnostics dry-run
# ---------------------------------------------------------------------------


class _StubPrimary:
    def __init__(self, raw):
        self._raw = raw

    async def get_trader_trades(self, key, *, limit=20):
        return self._raw[:limit]


def test_run_scan_diagnostics_returns_sample_without_mutating_status(monkeypatch) -> None:
    """Diagnostics is a *probe* — it must never write to the manual scan
    status, otherwise the operator's pre-flight check would clobber a
    legitimate in-flight scan.
    """
    class _Trader:
        nickname = "surf"
        wallet_address = "0xabc"

    class _Session:
        def __enter__(self): return self
        def __exit__(self, *a): return None
        def close(self): return None
        def scalars(self, *_a, **_k):
            class _R:
                def __iter__(self_inner): return iter([_Trader()])
            return _R()

    raw = [
        {"external_id": "1", "market_slug": "mlb-tor-bal-2026-05-28-total-8pt5",
         "side": "BUY", "outcome": "Over", "price": 0.55, "size_usd": 100.0},
        {"external_id": "2", "market_slug": "mlb-nyy-kc-2026-05-28-total-9pt5",
         "side": "BUY", "outcome": "Under", "price": 0.50, "size_usd": 200.0},
    ]
    monkeypatch.setattr(scanner.ingestion, "build_providers", lambda: {"primary": _StubPrimary(raw)})
    monkeypatch.setattr(scanner.db_module, "SessionLocal", lambda: _Session())

    pre_status = scanner.scan_status()
    out = asyncio.run(scanner.run_scan_diagnostics(sample_size=2))
    post_status = scanner.scan_status()

    assert out["tracked_wallets"] == 1
    assert out["provider_reachable"] is True
    assert out["sample_count"] == 2
    assert out["sample_positions"][0]["market_slug"] == "mlb-tor-bal-2026-05-28-total-8pt5"
    # Manual-scan status untouched — that's the whole point of dry-run.
    assert pre_status["state"] == post_status["state"]


def test_run_scan_diagnostics_records_provider_error(monkeypatch) -> None:
    """When the primary provider raises, the diagnostics payload must
    capture the exception text so the operator sees *why* the dry-run
    failed — silent ``provider_reachable=False`` with no reason would
    be worse than failing loud.
    """
    class _Trader:
        nickname = "surf"
        wallet_address = "0xabc"

    class _Session:
        def close(self): return None
        def scalars(self, *_a, **_k):
            class _R:
                def __iter__(self_inner): return iter([_Trader()])
            return _R()

    class _BadPrimary:
        async def get_trader_trades(self, key, *, limit=20):
            raise RuntimeError("HTTP 429 rate-limited")

    monkeypatch.setattr(scanner.ingestion, "build_providers", lambda: {"primary": _BadPrimary()})
    monkeypatch.setattr(scanner.db_module, "SessionLocal", lambda: _Session())

    out = asyncio.run(scanner.run_scan_diagnostics(sample_size=5))

    assert out["provider_reachable"] is False
    assert "429" in (out["provider_error"] or "")
    assert out["sample_count"] == 0
