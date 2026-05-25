"""Standalone scanner worker.

Use this when you don't want to run the FastAPI server but still want signals
generated on a fixed interval. It loops `run_scan_once` and sleeps for
`SCAN_INTERVAL_SECONDS` between passes.

    python -m scripts.run_worker
"""

from __future__ import annotations

import asyncio
import logging

from app.config import get_settings
from app.db import init_db
from app.services.scanner import run_scan_once
from app.utils.logging import configure_logging

logger = logging.getLogger(__name__)


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    init_db()

    interval = max(settings.scan_interval_seconds, 5)
    logger.info(
        "Worker starting — scan_interval=%ds, copy_mode=%s",
        interval,
        settings.default_copy_mode,
    )

    while True:
        try:
            result = await run_scan_once()
            logger.info(
                "scan done: signals=%d alerts=%d traders=%d markets=%d duration=%.2fs",
                result.new_signals,
                result.new_alerts,
                result.scanned_traders,
                result.scanned_markets,
                result.duration_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("scan failed: %s", exc)

        await asyncio.sleep(interval)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("worker stopped")
