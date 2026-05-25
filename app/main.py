"""FastAPI app entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import get_settings
from app.db import init_db
from app.services.scanner import get_background_scanner
from app.utils.logging import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    init_db()
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
    scanner.start()
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
    app.include_router(router)
    return app


app = create_app()
