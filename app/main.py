"""FastAPI app entry point.

Startup contract: lifespan() must return immediately so Uvicorn can bind the
port before Render's port scan times out. Anything that can fail or block —
database DDL, watchlist seeding, the scanner loop, provider warmups — runs in
a background daemon thread fired *after* the app yields. Routes that need the
DB call ensure_db_initialized() lazily; until the bootstrap finishes they get
a fast 503 from /ready, but /health stays a pure liveness probe.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings
from app.utils.logging import configure_logging

logger = logging.getLogger(__name__)

_bootstrap_state: dict[str, object] = {
    "started_at": None,
    "finished_at": None,
    "db_ready": False,
    "watchlist_ready": False,
    "scanner_started": False,
    "error": None,
}
_bootstrap_lock = threading.Lock()
_bootstrap_thread: threading.Thread | None = None


def get_bootstrap_state() -> dict[str, object]:
    with _bootstrap_lock:
        return dict(_bootstrap_state)


def _set_bootstrap(**kwargs: object) -> None:
    with _bootstrap_lock:
        _bootstrap_state.update(kwargs)


def _background_bootstrap() -> None:
    """Heavy startup work — DB DDL, seeding, scanner — run off the event loop."""
    _set_bootstrap(started_at=time.time())
    settings = get_settings()

    # 1. Database metadata (create_all + idempotent ALTER TABLE migrations).
    try:
        logger.info("bootstrap: initializing database metadata")
        from app.db import ensure_db_initialized

        ensure_db_initialized()
        _set_bootstrap(db_ready=True)
        logger.info("bootstrap: database metadata ready")
    except Exception as exc:  # noqa: BLE001
        logger.exception("bootstrap: init_db failed — DB-backed routes will return errors")
        _set_bootstrap(error=f"init_db: {exc}", finished_at=time.time())
        return

    # 2. Watchlist seed (only if enabled). Failure here must not stop the scanner.
    if settings.auto_seed_watchlist:
        try:
            from app.db import SessionLocal
            from scripts.seed import seed_watchlist

            logger.info("bootstrap: seeding watchlist")
            db = SessionLocal()
            try:
                created, updated = seed_watchlist(db)
                db.commit()
                logger.info(
                    "bootstrap: watchlist seed complete (created=%d updated=%d)",
                    created,
                    updated,
                )
            except Exception:
                db.rollback()
                logger.exception("bootstrap: watchlist seed failed")
            finally:
                db.close()
            _set_bootstrap(watchlist_ready=True)
        except Exception:  # noqa: BLE001
            logger.exception("bootstrap: watchlist seed import/setup failed")

    # 3. Background scanner — itself spawns a daemon thread, so .start() is cheap.
    try:
        from app.services.scanner import get_background_scanner

        scanner = get_background_scanner()
        scanner.start()
        _set_bootstrap(scanner_started=True)
        logger.info("bootstrap: background scanner started")
    except Exception as exc:  # noqa: BLE001
        logger.exception("bootstrap: scanner start failed: %s", exc)
        _set_bootstrap(error=f"scanner: {exc}")

    if settings.enable_auto_trading:
        logger.warning(
            "ENABLE_AUTO_TRADING=True — MVP is not built for live trading. "
            "Flag is honored by risk.evaluate() but no order routing exists yet."
        )
    logger.info(
        "%s started in %s mode (auto_trading=%s, default_copy_mode=%s)",
        settings.app_name,
        settings.environment,
        settings.enable_auto_trading,
        settings.default_copy_mode,
    )
    _set_bootstrap(finished_at=time.time())
    logger.info("bootstrap: complete")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bootstrap_thread
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("startup begin")
    logger.info("routes registered: %d", len(app.routes))

    # Fire heavy work off the event loop so Uvicorn binds the port immediately.
    _bootstrap_thread = threading.Thread(
        target=_background_bootstrap,
        name="signalforge-bootstrap",
        daemon=True,
    )
    _bootstrap_thread.start()
    logger.info("startup complete (bootstrap continues in background)")

    try:
        yield
    finally:
        try:
            from app.services.scanner import get_background_scanner

            get_background_scanner().stop()
        except Exception:  # noqa: BLE001
            logger.exception("scanner stop failed during shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description=(
            "SignalForge — prediction market intelligence MVP. "
            "Alert-only by default. NOT FINANCIAL ADVICE."
        ),
        lifespan=lifespan,
    )
    origins = [origin.strip() for origin in settings.cors_allow_origins.split(",") if origin.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.include_router(router)

    # Last-resort liveness route registered directly on the app — if anything in
    # the router crashes at import time we still want Render's port scan to see
    # a 200. The router's /health takes precedence when both are reachable.
    @app.get("/_alive")
    def _alive() -> dict[str, object]:
        return {"ok": True, "status": "alive"}

    return app


app = create_app()
