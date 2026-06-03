"""Scanner: one full pass = enrich traders -> refresh markets -> generate signals -> dispatch alerts.

Run once via `run_scan_once(...)` (used by `POST /run-scan` and by the worker
loop), or schedule it on `scan_interval_seconds` via `start_background_scanner`.

Visibility contract (the dashboard reads ``scan_status()`` on every render):

* ``state`` ∈ ``idle | running | finished | failed | timeout``
* ``progress`` — counters updated as each stage completes
    (wallets_loaded, wallets_scanned, markets_checked, raw_positions_found,
    active_positions, positions_rejected_*, api_errors)
* ``per_wallet`` — one row per trader showing raw / active counts, last seen
    market, and any error the provider returned. Drives the per-wallet debug
    table on the dashboard.
* ``phase`` — current pipeline phase string
* ``timeout_at`` — when the watchdog will force-fail an otherwise-stuck scan
* ``summary`` — human-readable end-of-scan line ("12 wallets · 84 raw · 0 active. Reason: ...")

A 3-minute hard cap (``MAX_SCAN_SECONDS``) plus a defensive reap on every
``scan_status()`` read guarantees no scan ever appears running forever.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

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


# Wall-clock cap on a single scan. Render's worker dyno will already kill a
# stuck Python process eventually, but the watchdog gives us a clean failure
# reason ("Timed out after 180s") instead of a mysterious zombie status.
MAX_SCAN_SECONDS = int(os.environ.get("SIGNALFORGE_SCAN_MAX_SECONDS", "180"))


@dataclass
class ScanProgress:
    """Thread-safe counters + per-wallet diagnostics for the in-flight scan.

    Lives on ``_manual_scan_status["progress"]`` so the dashboard sees it
    update live without needing a separate poll endpoint. Counter writes
    take a brief lock; reads via ``snapshot()`` copy under the lock so the
    JSON serializer never sees a half-mutated dict.
    """

    wallets_loaded: int = 0
    wallets_scanned: int = 0
    markets_discovered: int = 0
    markets_checked: int = 0
    raw_positions_found: int = 0
    active_positions: int = 0
    positions_rejected_stale: int = 0
    positions_rejected_date_mismatch: int = 0
    positions_rejected_market_key_mismatch: int = 0
    api_errors: int = 0
    rate_limited: int = 0
    per_wallet: list[dict[str, Any]] = field(default_factory=list)
    phase: str = "starting"
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase
        logger.info("scan.phase=%s", phase)

    def add(self, **deltas: int) -> None:
        with self._lock:
            for key, delta in deltas.items():
                if hasattr(self, key) and not key.startswith("_"):
                    setattr(self, key, getattr(self, key) + int(delta))

    def record_wallet(
        self,
        *,
        nickname: str,
        address: str | None,
        status: str,
        raw_positions: int = 0,
        active_positions: int = 0,
        last_market: str | None = None,
        error: str | None = None,
    ) -> None:
        """Append (or update) the per-wallet row used by the dashboard's
        debug table. Idempotent on ``address`` so a retry doesn't double
        up rows.
        """
        with self._lock:
            existing = next(
                (
                    row for row in self.per_wallet
                    if row.get("address") == address
                ),
                None,
            )
            payload = {
                "nickname": nickname,
                "address": address,
                "status": status,
                "raw_positions": int(raw_positions),
                "active_positions": int(active_positions),
                "last_market": last_market,
                "error": error,
            }
            if existing is None:
                self.per_wallet.append(payload)
            else:
                existing.update(payload)
        logger.info(
            "scan.wallet name=%s addr=%s status=%s raw=%d active=%d last=%r err=%s",
            nickname, address, status, raw_positions, active_positions,
            last_market, error,
        )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "phase": self.phase,
                "wallets_loaded": self.wallets_loaded,
                "wallets_scanned": self.wallets_scanned,
                "markets_discovered": self.markets_discovered,
                "markets_checked": self.markets_checked,
                "raw_positions_found": self.raw_positions_found,
                "active_positions": self.active_positions,
                "positions_rejected_stale": self.positions_rejected_stale,
                "positions_rejected_date_mismatch": self.positions_rejected_date_mismatch,
                "positions_rejected_market_key_mismatch": self.positions_rejected_market_key_mismatch,
                "api_errors": self.api_errors,
                "rate_limited": self.rate_limited,
                "per_wallet": [dict(row) for row in self.per_wallet],
            }


_manual_scan_lock = threading.RLock()
_manual_scan_status: dict[str, object] = {
    "state": "idle",
    "generated_for_date": None,
    "started_at": None,
    "finished_at": None,
    "result": None,
    "error": None,
    "progress": None,
    "timeout_at": None,
    "summary": None,
}


def _build_summary(card_date: str, result: ScanResult | None, progress: ScanProgress | None) -> str:
    """Operator-friendly one-liner. Always names the *reason* for a 0-active
    outcome so the dashboard never just sits on a silent "no positions" state.
    """
    parts: list[str] = []
    if progress is not None:
        snap = progress.snapshot()
        parts.append(f"{snap['wallets_scanned']}/{snap['wallets_loaded']} wallets scanned")
        parts.append(f"{snap['raw_positions_found']} raw positions")
        parts.append(f"{snap['active_positions']} active after filters")
        if snap["positions_rejected_stale"]:
            parts.append(f"{snap['positions_rejected_stale']} rejected stale")
        if snap["positions_rejected_date_mismatch"]:
            parts.append(f"{snap['positions_rejected_date_mismatch']} rejected date-mismatch")
        if snap["positions_rejected_market_key_mismatch"]:
            parts.append(f"{snap['positions_rejected_market_key_mismatch']} rejected market-key-mismatch")
        if snap["api_errors"]:
            parts.append(f"{snap['api_errors']} API errors")
        if snap["rate_limited"]:
            parts.append(f"{snap['rate_limited']} rate-limit hits")
    if result is not None:
        reason = result.reason or "ok"
        parts.append(f"reason: {reason}")
    if not parts:
        return f"Wallet scan for {card_date}: no detail captured."
    return f"Wallet scan for {card_date}: " + " · ".join(parts)


async def run_scan_once(
    *,
    card_date: str | None = None,
    progress: ScanProgress | None = None,
) -> ScanResult:
    """One full scan pass. Returns a structured result for the API.

    ``progress`` is an optional ``ScanProgress`` recorder that the manual
    /run-scan path uses to expose live counters + per-wallet diagnostics
    to the dashboard. Background scans skip it (no UI consumer).
    """
    started = time.perf_counter()
    card_date = card_date or arizona_today()
    settings = get_settings()
    providers = ingestion.build_providers()
    dispatcher = AlertDispatcher()

    def _bump(**deltas: int) -> None:
        if progress is not None:
            progress.add(**deltas)

    def _phase(name: str) -> None:
        if progress is not None:
            progress.set_phase(name)

    # Reset Falcon per-scan counters so /health reflects this pass.
    begin_scan_window(settings.falcon_base_url)

    db: Session = db_module.SessionLocal()
    try:
        _phase("loading_traders")
        traders = list(db.scalars(select(Trader)))
        if progress is not None:
            progress.wallets_loaded = len(traders)
        logger.info("scan.start card_date=%s wallets_loaded=%d", card_date, len(traders))

        # 1+2. Load wallet stats + trades CONCURRENTLY (network), then persist
        # serially. Wallet enrichment is network-bound — one Falcon/Polymarket
        # round-trip per wallet — so running 11 wallets sequentially used to
        # blow the 180s cap and strand the scan in ``enriching_traders``
        # before signal generation ever ran. Fanning the calls out with a
        # per-call timeout turns ~33 serial round-trips into a few seconds of
        # parallel I/O, and the live counter climbs as each wallet lands.
        _phase("loading_wallet_data")

        def _on_wallet_done(trader: Trader, trades: list[dict[str, Any]]) -> None:
            # Fired from the gather as each wallet's network completes, so the
            # dashboard's "wallets scanned" counter moves in real time.
            _bump(wallets_scanned=1, raw_positions_found=len(trades))
            if progress is not None:
                last_market = None
                if trades:
                    last_market = trades[-1].get("market_slug") or trades[-1].get("market_title")
                progress.record_wallet(
                    nickname=trader.nickname or "?",
                    address=trader.wallet_address,
                    status="fetched",
                    raw_positions=len(trades),
                    active_positions=0,
                    last_market=last_market,
                )

        try:
            payloads = await ingestion.gather_trader_payloads(
                traders,
                providers,
                limit=20,
                per_call_timeout=float(settings.scan_provider_call_timeout_seconds),
                concurrency=int(settings.scan_wallet_concurrency),
                on_wallet_done=_on_wallet_done,
            )
        except Exception as exc:  # noqa: BLE001
            _bump(api_errors=1)
            logger.exception("wallet data gather failed: %s", exc)
            payloads = {}

        # Persist serially — fast, no network — so the single scanner Session
        # is never touched concurrently.
        _phase("persisting_wallet_data")
        for trader in traders:
            payload = payloads.get(trader.id)
            if payload is None:
                continue
            try:
                new_trades = ingestion.persist_trader_payload(db, trader, payload)
            except SQLAlchemyError as exc:
                ingestion_health.record_failure(f"persist_trader_payload: {exc}")
                ingestion_health.safe_rollback(db)
                _bump(api_errors=1)
                logger.warning(
                    "persist_trader_payload(%s) DB failure, rolled back: %s",
                    trader.nickname, exc,
                )
                continue
            _bump(active_positions=len(new_trades))
            if progress is not None:
                last_market = None
                if new_trades and new_trades[-1].market is not None:
                    last_market = new_trades[-1].market.slug or new_trades[-1].market.title
                progress.record_wallet(
                    nickname=trader.nickname or "?",
                    address=trader.wallet_address,
                    status="ok",
                    raw_positions=len(payload.get("trades") or []),
                    active_positions=len(new_trades),
                    last_market=last_market,
                )

        # 3. Discover + refresh markets.
        _phase("discovering_markets")
        try:
            discovered = await ingestion.discover_markets(db, providers)
        except SQLAlchemyError as exc:
            ingestion_health.record_failure(f"discover_markets: {exc}")
            ingestion_health.safe_rollback(db)
            _bump(api_errors=1)
            logger.warning("discover_markets DB failure, rolled back: %s", exc)
            discovered = []
        except Exception as exc:  # noqa: BLE001
            _bump(api_errors=1)
            logger.warning("discover_markets failed: %s", exc)
            discovered = []
        if progress is not None:
            progress.markets_discovered = len(discovered)

        # Refresh card-date markets with the SAME concurrent pattern: fan the
        # per-market price pulls out over the network, then snapshot serially.
        # A 130-market slate used to mean 130 sequential round-trips — the next
        # timeout waiting to happen once enrichment was no longer the cap.
        _phase("refreshing_markets")
        existing_markets = list(db.scalars(select(Market)))
        refresh_targets = {
            market.id: market
            for market in [*discovered, *existing_markets]
            if market_card_date(market) == card_date
        }
        try:
            market_data = await ingestion.gather_market_data(
                list(refresh_targets.values()),
                providers,
                per_call_timeout=float(settings.scan_provider_call_timeout_seconds),
                concurrency=int(settings.scan_wallet_concurrency),
                on_market_done=lambda _m: _bump(markets_checked=1),
            )
        except Exception as exc:  # noqa: BLE001
            _bump(api_errors=1)
            logger.exception("market data gather failed: %s", exc)
            market_data = {}
        for market_id, data in market_data.items():
            market = refresh_targets.get(market_id)
            if market is None:
                continue
            try:
                ingestion.persist_market_data(db, market, data)
            except SQLAlchemyError as exc:
                ingestion_health.record_failure(f"persist_market_data: {exc}")
                ingestion_health.safe_rollback(db)
                _bump(api_errors=1)
                logger.warning("persist_market_data(%s) DB failure, rolled back: %s", market.slug, exc)

        existing_markets = list(db.scalars(select(Market)))
        scanned_markets = len(existing_markets)
        markets_for_card_date = sum(
            1 for market in existing_markets if market_card_date(market) == card_date
        )
        stale_markets_skipped = max(scanned_markets - markets_for_card_date, 0)
        preserved_prior_date_rows = _prior_date_row_count(db, card_date)
        if progress is not None:
            progress.add(positions_rejected_date_mismatch=stale_markets_skipped)

        # 4. Generate signals.
        _phase("generating_signals")
        try:
            new_signals = await signal_engine.generate_signals(
                db, providers, card_date=card_date
            )
        except SQLAlchemyError as exc:
            ingestion_health.record_failure(f"generate_signals: {exc}")
            ingestion_health.safe_rollback(db)
            _bump(api_errors=1)
            logger.exception("generate_signals DB failure, rolled back")
            new_signals = []

        # 5. Dispatch alerts for each new signal.
        _phase("dispatching_alerts")
        new_alerts = 0
        for sig in new_signals:
            try:
                alerts = dispatcher.dispatch(db, sig)
                new_alerts += len(alerts)
            except SQLAlchemyError as exc:
                ingestion_health.record_failure(f"dispatch_alert: {exc}")
                ingestion_health.safe_rollback(db)
                _bump(api_errors=1)
                logger.warning("alert dispatch DB failure for signal=%s: %s", sig.id, exc)
            except Exception as exc:  # noqa: BLE001
                _bump(api_errors=1)
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


def trigger_manual_scan_background(
    *,
    card_date: str | None = None,
    max_seconds: int | None = None,
) -> dict[str, object]:
    """Start one scan in a daemon thread and return immediately.

    Render can time out long HTTP requests while Falcon/market enrichment is
    still working. This keeps ``/run-scan`` as an operator trigger without
    holding the request open for the full scan duration.

    A ``max_seconds`` watchdog (default ``MAX_SCAN_SECONDS``) guarantees
    that even if every layer below us hangs, the *frontend-visible*
    status never sits in "running" longer than the cap — see
    ``scan_status()`` for the reaping logic.
    """
    timeout = int(max_seconds or MAX_SCAN_SECONDS)
    with _manual_scan_lock:
        if _manual_scan_status.get("state") == "running":
            return scan_status() | {"accepted": False, "message": "scan already running"}
        card_date = card_date or arizona_today()
        started_at = datetime.utcnow()
        progress = ScanProgress()
        _manual_scan_status.update(
            {
                "state": "running",
                "generated_for_date": card_date,
                "started_at": started_at.isoformat(),
                "finished_at": None,
                "result": None,
                "error": None,
                "progress": progress,
                "timeout_at": (started_at + timedelta(seconds=timeout)).isoformat(),
                "max_seconds": timeout,
                "summary": None,
            }
        )
        thread = threading.Thread(
            target=_run_manual_scan_thread,
            args=(card_date, progress, timeout),
            name="signalforge-manual-scan",
            daemon=True,
        )
        thread.start()
        return scan_status() | {"accepted": True, "message": "scan started"}


def scan_status() -> dict[str, object]:
    """Return the live scan status with progress + per-wallet diagnostics.

    Defensive reap: if ``state == "running"`` but the watchdog deadline
    has passed, flip the status to ``timeout`` *before returning*. This
    is the safety net that guarantees the dashboard never sees a scan
    "running" for hours — even if the background thread is wedged in a
    blocking provider call.
    """
    with _manual_scan_lock:
        snapshot = dict(_manual_scan_status)
        progress_obj = snapshot.get("progress")
        if isinstance(progress_obj, ScanProgress):
            snapshot["progress"] = progress_obj.snapshot()
        # Defensive reap. The watchdog thread also flips this flag, but
        # we double-check on every read so a crashed watchdog can't leave
        # the dashboard frozen on "running".
        timeout_at_iso = snapshot.get("timeout_at")
        if (
            snapshot.get("state") == "running"
            and isinstance(timeout_at_iso, str)
        ):
            with suppress(ValueError):
                if datetime.utcnow() > datetime.fromisoformat(timeout_at_iso):
                    logger.warning(
                        "scan_status: reaping stuck scan (timeout_at=%s)",
                        timeout_at_iso,
                    )
                    _manual_scan_status.update(
                        {
                            "state": "timeout",
                            "finished_at": datetime.utcnow().isoformat(),
                            "error": (
                                f"Scan exceeded {snapshot.get('max_seconds')}s "
                                "wall-clock cap and was marked stale. Inspect "
                                "progress + per-wallet rows for the stuck stage."
                            ),
                            "summary": (
                                f"Scan for {snapshot.get('generated_for_date')} timed out. "
                                "Background thread may still be running; status cleared."
                            ),
                        }
                    )
                    # Refresh the snapshot we'll hand back so the caller
                    # sees the reaped state, not the pre-reap state.
                    snapshot = dict(_manual_scan_status)
                    progress_obj = snapshot.get("progress")
                    if isinstance(progress_obj, ScanProgress):
                        snapshot["progress"] = progress_obj.snapshot()
    return snapshot


def reset_scan_status() -> dict[str, object]:
    """Operator-facing reset. Returns the new (idle) status payload."""
    with _manual_scan_lock:
        _manual_scan_status.update(
            {
                "state": "idle",
                "started_at": None,
                "finished_at": None,
                "result": None,
                "error": None,
                "progress": None,
                "timeout_at": None,
                "summary": "Scan state cleared by operator.",
            }
        )
        return dict(_manual_scan_status)


def _run_manual_scan_thread(
    card_date: str | None,
    progress: ScanProgress,
    timeout_seconds: int,
) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            asyncio.wait_for(
                run_scan_once(card_date=card_date, progress=progress),
                timeout=float(timeout_seconds),
            )
        )
        payload = result.model_dump() if hasattr(result, "model_dump") else result.dict()
        with _manual_scan_lock:
            _manual_scan_status.update(
                {
                    "state": "finished",
                    "finished_at": datetime.utcnow().isoformat(),
                    "result": payload,
                    "generated_for_date": payload.get("generated_for_date"),
                    "error": None,
                    "summary": _build_summary(card_date or "?", result, progress),
                }
            )
        logger.info(
            "Manual scan finished: signals=%s alerts=%s markets=%s duration=%ss",
            payload.get("new_signals"),
            payload.get("new_alerts"),
            payload.get("scanned_markets"),
            payload.get("duration_seconds"),
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Manual scan exceeded %ss cap; marking timeout (phase=%s)",
            timeout_seconds, progress.phase,
        )
        with _manual_scan_lock:
            _manual_scan_status.update(
                {
                    "state": "timeout",
                    "finished_at": datetime.utcnow().isoformat(),
                    "result": None,
                    "error": (
                        f"Scan exceeded {timeout_seconds}s wall-clock cap "
                        f"(stuck in phase={progress.phase!r})."
                    ),
                    "summary": _build_summary(card_date or "?", None, progress)
                    + f" — timed out in phase={progress.phase!r}.",
                }
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
                    "summary": _build_summary(card_date or "?", None, progress)
                    + f" — failed: {exc}",
                }
            )
    finally:
        loop.close()


# --------------------------------------------------------------------------
# Diagnostics (dry-run)
# --------------------------------------------------------------------------


async def run_scan_diagnostics(*, sample_size: int = 5) -> dict[str, Any]:
    """Cheap pre-flight check the operator can run before a full scan.

    Confirms:
      * Tracked-wallet count from the DB
      * Primary provider is reachable
      * First ``sample_size`` raw positions for the first wallet (so the
        operator can verify the provider is returning real data)

    Never touches the manual-scan status; this is purely a probe.
    """
    out: dict[str, Any] = {
        "ran_at": datetime.utcnow().isoformat(),
        "tracked_wallets": 0,
        "provider_reachable": False,
        "provider_error": None,
        "sample_wallet": None,
        "sample_positions": [],
        "sample_count": 0,
    }
    providers = ingestion.build_providers()
    primary = providers.get("primary")
    db: Session = db_module.SessionLocal()
    try:
        traders = list(db.scalars(select(Trader)))
        out["tracked_wallets"] = len(traders)
        if not traders or primary is None:
            out["provider_error"] = (
                "no_traders" if not traders else "no_primary_provider"
            )
            return out
        first = traders[0]
        out["sample_wallet"] = {
            "nickname": first.nickname,
            "address": first.wallet_address,
        }
        try:
            raw = await primary.get_trader_trades(
                first.wallet_address or first.nickname,
                limit=int(sample_size or 5),
            )
        except Exception as exc:  # noqa: BLE001
            out["provider_error"] = f"{type(exc).__name__}: {exc}"
            return out
        out["provider_reachable"] = True
        # Strip to a shape that's safe to ship over JSON — provider
        # entries can carry unhashable types we don't want to surface.
        sample: list[dict[str, Any]] = []
        for entry in (raw or [])[: int(sample_size or 5)]:
            sample.append(
                {
                    "external_id": entry.get("external_id"),
                    "market_slug": entry.get("market_slug") or entry.get("slug"),
                    "market_title": entry.get("market_title") or entry.get("title"),
                    "side": entry.get("side"),
                    "outcome": entry.get("outcome"),
                    "price": entry.get("price"),
                    "size_usd": entry.get("size_usd"),
                    "timestamp": entry.get("timestamp"),
                }
            )
        out["sample_positions"] = sample
        out["sample_count"] = len(sample)
        return out
    finally:
        db.close()


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
