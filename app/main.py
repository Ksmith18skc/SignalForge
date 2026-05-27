"""FastAPI app entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings
from app.db import SessionLocal, init_db
from app.services.scanner import get_background_scanner
from app.utils.logging import configure_logging
from scripts.seed import seed_watchlist

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Startup stage: settings loaded")
    logger.info("Startup stage: initializing database metadata")
    init_db()
    logger.info("Startup stage: database metadata ready")
    if settings.auto_seed_watchlist:
        logger.info("Startup stage: seeding watchlist")
        db = SessionLocal()
        try:
            created, updated = seed_watchlist(db)
            db.commit()
            logger.info(
                "Watchlist seed complete: created=%d updated=%d",
                created,
                updated,
            )
        except Exception:
            db.rollback()
            logger.exception("Watchlist seed failed")
            raise
        finally:
            db.close()
        logger.info("Startup stage: watchlist seed finished")
    logger.info(
        "%s starting in %s mode (auto_trading=%s, default_copy_mode=%s)",
        settings.app_name,
        settings.environment,
        settings.enable_auto_trading,
        settings.default_copy_mode,
    )
    if settings.enable_auto_trading:
        logger.warning(
            "ENABLE_AUTO_TRADING=True — MVP is not built for live trading. "
            "This flag is honored by risk.evaluate() but no order routing exists yet."
        )

    scanner = get_background_scanner()
    logger.info("Startup stage: starting background scanner")
    scanner.start()
    logger.info("Startup stage: application ready")
    try:
        yield
    finally:
        scanner.stop()


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
    return app


app = create_app()
