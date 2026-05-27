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

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import get_settings
from app import db as db_module
from app.models import Alert, Market, Signal, Trader
from app.providers.falcon import begin_scan_window, end_scan_window, get_falcon_health
from app.schemas import ScanResult
from app.services import ingestion, ingestion_health, signal_engine
from app.services.alerts import AlertDispatcher
from app.services.card_date import arizona_today, market_card_date

logger = logging.getLogger(__name__)


_manual_scan_lock = threading.RLock()
_manual_scan_status: dict[str, object] = {
    "state": "idle",
    "generated_for_date": None,
    "started_at": None,
    "finished_at": None,
    "result": None,
    "error": None,
}


async def run_scan_once(*, card_date: str | None = None) -> ScanResult:
    """One full scan pass. Returns a structured result for the API."""
    started = time.perf_counter()
    card_date = card_date or arizona_today()
    settings = get_settings()
    providers = ingestion.build_providers()
    dispatcher = AlertDispatcher()

    # Reset Falcon per-scan counters so /health reflects this pass.
    begin_scan_window(settings.falcon_base_url)

    db: Session = db_module.SessionLocal()
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
        refresh_targets = {
            market.id: market
            for market in [*discovered, *existing_markets]
            if market_card_date(market) == card_date
        }
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
        markets_for_card_date = sum(
            1 for market in existing_markets if market_card_date(market) == card_date
        )
        stale_markets_skipped = max(scanned_markets - markets_for_card_date, 0)
        preserved_prior_date_rows = _prior_date_row_count(db, card_date)

        # 4. Generate signals.
        try:
            new_signals = await signal_engine.generate_signals(
                db, providers, card_date=card_date
            )
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
                generated_for_date=card_date,
                markets_seen=scanned_markets,
                markets_for_card_date=markets_for_card_date,
                stale_markets_skipped=stale_markets_skipped,
                positions_written=0,
                alerts_written=0,
                preserved_prior_date_rows=preserved_prior_date_rows,
                reason="scan commit failed",
                scanned_markets=scanned_markets,
                scanned_traders=len(traders),
                new_signals=0,
                new_alerts=0,
                duration_seconds=round(time.perf_counter() - started, 3),
            )

        reason = "ok"
        if markets_for_card_date == 0:
            reason = "no current-card markets found"
        elif not new_signals:
            reason = "no current-card wallet flow found"
        return ScanResult(
            generated_for_date=card_date,
            markets_seen=scanned_markets,
            markets_for_card_date=markets_for_card_date,
            stale_markets_skipped=stale_markets_skipped,
            positions_written=len(new_signals),
            alerts_written=new_alerts,
            preserved_prior_date_rows=preserved_prior_date_rows,
            reason=reason,
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


def _prior_date_row_count(db: Session, card_date: str) -> int:
    signal_count = db.scalar(
        select(func.count())
        .select_from(Signal)
        .where(Signal.generated_for_date.is_not(None))
        .where(Signal.generated_for_date != card_date)
    ) or 0
    alert_count = db.scalar(
        select(func.count())
        .select_from(Alert)
        .where(Alert.generated_for_date.is_not(None))
        .where(Alert.generated_for_date != card_date)
    ) or 0
    return int(signal_count) + int(alert_count)


def trigger_manual_scan_background(*, card_date: str | None = None) -> dict[str, object]:
    """Start one scan in a daemon thread and return immediately.

    Render can time out long HTTP requests while Falcon/market enrichment is
    still working. This keeps `/run-scan` as an operator trigger without
    holding the request open for the full scan duration.
    """
    with _manual_scan_lock:
        if _manual_scan_status.get("state") == "running":
            return scan_status() | {"accepted": False, "message": "scan already running"}
        card_date = card_date or arizona_today()
        _manual_scan_status.update(
            {
                "state": "running",
                "generated_for_date": card_date,
                "started_at": datetime.utcnow().isoformat(),
                "finished_at": None,
                "result": None,
                "error": None,
            }
        )
        thread = threading.Thread(
            target=_run_manual_scan_thread,
            args=(card_date,),
            name="signalforge-manual-scan",
            daemon=True,
        )
        thread.start()
        return scan_status() | {"accepted": True, "message": "scan started"}


def scan_status() -> dict[str, object]:
    with _manual_scan_lock:
        return dict(_manual_scan_status)


def _run_manual_scan_thread(card_date: str | None = None) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(run_scan_once(card_date=card_date))
        payload = result.model_dump() if hasattr(result, "model_dump") else result.dict()
        with _manual_scan_lock:
            _manual_scan_status.update(
                {
                    "state": "finished",
                    "finished_at": datetime.utcnow().isoformat(),
                    "result": payload,
                    "generated_for_date": payload.get("generated_for_date"),
                    "error": None,
                }
            )
        logger.info(
            "Manual scan finished: signals=%s alerts=%s markets=%s duration=%ss",
            payload.get("new_signals"),
            payload.get("new_alerts"),
            payload.get("scanned_markets"),
            payload.get("duration_seconds"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Manual scan failed: %s", exc)
        with _manual_scan_lock:
            _manual_scan_status.update(
                {
                    "state": "failed",
                    "finished_at": datetime.utcnow().isoformat(),
                    "result": None,
                    "error": str(exc),
                }
            )
    finally:
        loop.close()


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
