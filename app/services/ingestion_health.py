"""Thread-safe counters for ingestion failures and rollbacks.

These let `/health` surface "is the scanner actually persisting rows or are
inserts failing silently?" without scraping logs. Counters are process-local
and reset on restart; that's the right granularity for an MVP — anything
durable belongs in a metrics backend.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class IngestionHealth:
    ingestion_failures: int = 0
    db_rollbacks: int = 0
    last_ingestion_error: str | None = None
    last_ingestion_error_at: datetime | None = None
    last_rollback_at: datetime | None = None
    trades_inserted: int = 0
    trades_skipped_oversized: int = 0


_health = IngestionHealth()
_lock = threading.Lock()


def get_ingestion_health() -> IngestionHealth:
    with _lock:
        return IngestionHealth(**_health.__dict__)


def record_failure(error: str) -> None:
    with _lock:
        _health.ingestion_failures += 1
        _health.last_ingestion_error = error[:500]
        _health.last_ingestion_error_at = datetime.utcnow()


def record_rollback() -> None:
    with _lock:
        _health.db_rollbacks += 1
        _health.last_rollback_at = datetime.utcnow()


def record_trade_inserted(count: int = 1) -> None:
    with _lock:
        _health.trades_inserted += count


def record_trade_skipped_oversized() -> None:
    with _lock:
        _health.trades_skipped_oversized += 1


def reset() -> None:
    """Test helper — wipe counters between cases."""
    global _health
    with _lock:
        _health = IngestionHealth()


def safe_rollback(db) -> bool:  # noqa: ANN001 — Session, avoid import cycle
    """Roll the session back, record it, and swallow rollback errors.

    Returns True if rollback succeeded, False if it raised. After this call
    the session is safe to reuse for new transactions.
    """
    record_rollback()
    try:
        db.rollback()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.exception("db.rollback() itself failed: %s", exc)
        return False
