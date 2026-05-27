from __future__ import annotations

from app.api.routes import trigger_scan, trigger_scan_status
from app.services import scanner


class _NoopThread:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def start(self) -> None:
        return None


async def test_run_scan_endpoint_enqueues_background_job(monkeypatch) -> None:
    scanner._manual_scan_status.update(  # noqa: SLF001
        {
            "state": "idle",
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }
    )
    monkeypatch.setattr(scanner.threading, "Thread", _NoopThread)

    payload = await trigger_scan()

    assert "state" in payload
    assert payload["state"] == "running"
    assert payload["accepted"] is True

    status = trigger_scan_status()
    assert status["state"] == "running"
    assert status["started_at"] is not None

    scanner._manual_scan_status["state"] = "idle"  # noqa: SLF001
