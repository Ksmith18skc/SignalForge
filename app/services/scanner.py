"""Scanner: one full pass = enrich traders -> refresh markets -> generate signals -> dispatch alerts.

Run once via `run_scan_once(...)` (used by `POST /run-scan` and by the worker
loop), or schedule it on `scan_interval_seconds` via `start_background_scanner`.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import SessionLocal
from app.models import Market, Trader
from app.providers.falcon import begin_scan_window, end_scan_window, get_falcon_health
from app.schemas import ScanResult
from app.services import ingestion, ingestion_health, signal_engine
from app.services.alerts import AlertDispatcher

logger = logging.getLogger(__name__)


async def run_scan_once() -> ScanResult:
    """One full scan pass. Returns a structured result for the API."""
    started = time.perf_counter()
    settings = get_settings()
    providers = ingestion.build_providers()
    dispatcher = AlertDispatcher()

    # Reset Falcon per-scan counters so /health reflects this pass.
    begin_scan_window(settings.falcon_base_url)

    db: Session = SessionLocal()
    try:
        traders = list(db.scalars(select(Trader)))
        # 1. Enrich every watched trader.
        for trader in traders:
            try:
                await ingestion.enrich_trader(db, trader, providers)
            except SQLAlchemyError as exc:
                ingestion_health.record_failure(f"enrich_trader: {exc}")
                ingestion_health.safe_rollback(db)
                logger.warning("enrich_trader(%s) DB failure, rolled back: %s", trader.nickname, exc)
            except Exception as exc:  # noqa: BLE001
                logger.warning("enrich_trader(%s) failed: %s", trader.nickname, exc)

        # 2. Pull recent trades. One trader's bad data must not abort the rest.
        for trader in traders:
            try:
                await ingestion.fetch_recent_trades(db, trader, providers)
            except SQLAlchemyError as exc:
                ingestion_health.record_failure(f"fetch_recent_trades: {exc}")
                ingestion_health.safe_rollback(db)
                logger.warning(
                    "fetch_recent_trades(%s) DB failure, rolled back: %s",
                    trader.nickname, exc,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("fetch_recent_trades(%s) failed: %s", trader.nickname, exc)

        # 3. Discover + refresh markets.
        try:
            discovered = await ingestion.discover_markets(db, providers)
        except SQLAlchemyError as exc:
            ingestion_health.record_failure(f"discover_markets: {exc}")
            ingestion_health.safe_rollback(db)
            logger.warning("discover_markets DB failure, rolled back: %s", exc)
            discovered = []
        except Exception as exc:  # noqa: BLE001
            logger.warning("discover_markets failed: %s", exc)
            discovered = []

        existing_markets = list(db.scalars(select(Market)))
        refresh_targets = {market.id: market for market in [*discovered, *existing_markets]}
        for market in refresh_targets.values():
            try:
                await ingestion.refresh_market(db, market, providers)
            except SQLAlchemyError as exc:
                ingestion_health.record_failure(f"refresh_market: {exc}")
                ingestion_health.safe_rollback(db)
                logger.warning("refresh_market(%s) DB failure, rolled back: %s", market.slug, exc)
            except Exception as exc:  # noqa: BLE001
                logger.warning("refresh_market(%s) failed: %s", market.slug, exc)

        existing_markets = list(db.scalars(select(Market)))
        scanned_markets = len(existing_markets)

        # 4. Generate signals.
        try:
            new_signals = await signal_engine.generate_signals(db, providers)
        except SQLAlchemyError as exc:
            ingestion_health.record_failure(f"generate_signals: {exc}")
            ingestion_health.safe_rollback(db)
            logger.exception("generate_signals DB failure, rolled back")
            new_signals = []

        # 5. Dispatch alerts for each new signal.
        new_alerts = 0
        for sig in new_signals:
            try:
                alerts = dispatcher.dispatch(db, sig)
                new_alerts += len(alerts)
            except SQLAlchemyError as exc:
                ingestion_health.record_failure(f"dispatch_alert: {exc}")
                ingestion_health.safe_rollback(db)
                logger.warning("alert dispatch DB failure for signal=%s: %s", sig.id, exc)
            except Exception as exc:  # noqa: BLE001
                logger.warning("alert dispatch failed for signal=%s: %s", sig.id, exc)

        try:
            db.commit()
        except SQLAlchemyError as exc:
            ingestion_health.record_failure(f"scan_commit: {exc}")
            ingestion_health.safe_rollback(db)
            logger.exception("scan commit failed, rolled back")
            # Return what we know — the scanner loop must not crash.
            return ScanResult(
                scanned_markets=scanned_markets,
                scanned_traders=len(traders),
                new_signals=0,
                new_alerts=0,
                duration_seconds=round(time.perf_counter() - started, 3),
            )

        return ScanResult(
            scanned_markets=scanned_markets,
            scanned_traders=len(traders),
            new_signals=len(new_signals),
            new_alerts=new_alerts,
            duration_seconds=round(time.perf_counter() - started, 3),
        )
    except Exception:
        ingestion_health.safe_rollback(db)
        raise
    finally:
        db.close()
        # Snapshot Falcon health and log one summary line per scan (instead of
        # one warning per 404). Detail lives in /health.
        f_calls, f_ok = end_scan_window()
        if settings.has_falcon_credentials() and f_calls:
            if f_ok == 0:
                logger.warning(
                    "Falcon: 0/%d calls succeeded — falling back to mock. Last error: %s",
                    f_calls,
                    get_falcon_health().last_error,
                )
            elif f_ok < f_calls:
                logger.info(
                    "Falcon: %d/%d calls succeeded (mock fallback for the rest)",
                    f_ok, f_calls,
                )
            else:
                logger.info("Falcon: %d/%d calls succeeded (healthy)", f_ok, f_calls)


# --------------------------------------------------------------------------
# Background loop
# --------------------------------------------------------------------------


class BackgroundScanner:
    """Runs `run_scan_once` on an interval in a daemon thread.

    A dedicated thread with its own event loop keeps us decoupled from the
    FastAPI request loop, which is fine for the MVP — APScheduler can replace
    this when we need cron-style schedules.
    """

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="signalforge-scanner", daemon=True)
        self._thread.start()
        logger.info("Background scanner started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Background scanner stopped")

    def _run_loop(self) -> None:
        interval = max(get_settings().scan_interval_seconds, 5)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            while not self._stop.is_set():
                try:
                    result = loop.run_until_complete(run_scan_once())
                    logger.info(
                        "Scan @ %s: signals=%d alerts=%d markets=%d duration=%.2fs",
                        datetime.utcnow().isoformat(timespec="seconds"),
                        result.new_signals,
                        result.new_alerts,
                        result.scanned_markets,
                        result.duration_seconds,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Scan iteration failed: %s", exc)
                self._stop.wait(interval)
        finally:
            loop.close()


_singleton: BackgroundScanner | None = None


def get_background_scanner() -> BackgroundScanner:
    global _singleton
    if _singleton is None:
        _singleton = BackgroundScanner()
    return _singleton
