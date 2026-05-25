"""Logging setup. Single source of truth for log format and level."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Idempotent logging setup. Call once on app startup."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    # Replace any pre-existing handlers so output isn't duplicated.
    root.handlers = [handler]

    # Quiet the noisy per-request httpx logger — the scanner already prints a
    # single "Falcon: X/Y calls succeeded" summary per scan, which is what we
    # actually want to see in the console.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _CONFIGURED = True
