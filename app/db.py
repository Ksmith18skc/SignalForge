"""SQLAlchemy engine, session, and Base declarative class."""

from __future__ import annotations

import logging
import threading
from collections.abc import Generator

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

logger = logging.getLogger(__name__)

# Lazy-init guards. ensure_db_initialized() may be called from the bootstrap
# thread (during startup) and from request handlers (as a safety net if the
# bootstrap thread hasn't finished yet). The lock keeps create_all from racing.
_db_init_lock = threading.Lock()
_db_init_done = False
_db_init_error: Exception | None = None

_settings = get_settings()

# check_same_thread=False is required for SQLite when sharing across the
# FastAPI request handler thread and background scanner thread.
_is_sqlite = _settings.database_url.startswith("sqlite")
_is_postgres = _settings.database_url.startswith("postgres")
_connect_args = {"check_same_thread": False, "timeout": 30.0} if _is_sqlite else {}

engine = create_engine(
    _settings.database_url,
    connect_args=_connect_args,
    future=True,
)

if _is_sqlite:
    # WAL lets the scanner thread write while the API thread reads concurrently.
    # busy_timeout=30s gives any blocked writer time to retry instead of
    # raising "database is locked" immediately.
    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _conn_record):  # noqa: ANN001
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a scoped DB session.

    Calls ensure_db_initialized() so any request that arrives before the
    background bootstrap finishes still sees an initialized schema.
    """
    ensure_db_initialized()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def db_init_status() -> dict[str, object]:
    """Snapshot used by /ready to report whether the DB is ready yet."""
    return {
        "ready": _db_init_done,
        "error": str(_db_init_error) if _db_init_error else None,
    }


def ensure_db_initialized() -> None:
    """Run init_db() once, lazily and idempotently.

    Safe to call from many threads. After the first successful call this is a
    cheap flag check. If init_db() failed previously the cached error is
    re-raised so the caller can return a clean 5xx instead of running queries
    against a half-built schema.
    """
    global _db_init_done, _db_init_error
    if _db_init_done:
        return
    if _db_init_error is not None:
        raise _db_init_error
    with _db_init_lock:
        if _db_init_done:
            return
        if _db_init_error is not None:
            raise _db_init_error
        try:
            init_db()
            _db_init_done = True
        except Exception as exc:
            _db_init_error = exc
            logger.exception("ensure_db_initialized: init_db failed")
            raise


def init_db() -> None:
    """Create all tables. Safe to call repeatedly."""
    # Import models so they register on Base.metadata before create_all.
    from app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    if _is_sqlite:
        _add_sqlite_column_if_missing("trades", "outcome", "VARCHAR(64)")
        _add_sqlite_column_if_missing("signals", "outcome", "VARCHAR(64)")
        _add_sqlite_column_if_missing("signals", "generated_for_date", "VARCHAR(10)")
        _add_sqlite_column_if_missing("alerts", "generated_for_date", "VARCHAR(10)")
    if _is_postgres:
        _ensure_signal_date_columns()
        _ensure_postgres_trades_external_id_is_text()


def _add_sqlite_column_if_missing(table: str, column: str, ddl: str) -> None:
    inspector = inspect(engine)
    if table not in inspector.get_table_names():
        return
    if column in {col["name"] for col in inspector.get_columns(table)}:
        return
    try:
        with engine.begin() as conn:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
    except SQLAlchemyError as exc:
        if "duplicate column" in str(exc).lower() or "already exists" in str(exc).lower():
            logger.info("Column %s.%s already exists; skipping migration", table, column)
            return
        raise


def _ensure_signal_date_columns() -> None:
    """Backfill generated_for_date columns on existing deployments."""
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    dialect = engine.dialect.name
    if dialect == "sqlite":
        if "signals" in tables:
            _add_sqlite_column_if_missing("signals", "generated_for_date", "VARCHAR(10)")
        if "alerts" in tables:
            _add_sqlite_column_if_missing("alerts", "generated_for_date", "VARCHAR(10)")
        return
    if dialect.startswith("postgres"):
        statements = []
        if "signals" in tables:
            statements.append("ALTER TABLE signals ADD COLUMN IF NOT EXISTS generated_for_date VARCHAR(10)")
        if "alerts" in tables:
            statements.append("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS generated_for_date VARCHAR(10)")
        if not statements:
            return
        with engine.begin() as conn:
            for statement in statements:
                conn.execute(text(statement))


def _ensure_postgres_trades_external_id_is_text() -> None:
    """Widen trades.external_id to TEXT on existing Postgres deployments.

    Older deployments shipped with VARCHAR(128) and Falcon trade IDs can blow
    past that, causing StringDataRightTruncation. ALTER TABLE ... TYPE TEXT
    USING external_id::text is non-destructive: it preserves every existing
    row's value because the cast is a no-op widen. We probe information_schema
    first so this stays idempotent across restarts.
    """
    inspector = inspect(engine)
    if "trades" not in inspector.get_table_names():
        return
    try:
        with engine.begin() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT data_type, character_maximum_length
                    FROM information_schema.columns
                    WHERE table_name = 'trades' AND column_name = 'external_id'
                    """
                )
            ).first()
            if row is None:
                return
            data_type, max_len = row[0], row[1]
            # Already TEXT (no length) — nothing to do.
            if data_type == "text" and max_len is None:
                return
            logger.warning(
                "Migrating trades.external_id from %s(%s) to TEXT to fit Falcon IDs",
                data_type, max_len,
            )
            conn.execute(
                text(
                    "ALTER TABLE trades ALTER COLUMN external_id TYPE TEXT "
                    "USING external_id::text"
                )
            )
    except Exception as exc:  # noqa: BLE001
        # Don't crash app startup if the migration probe fails — log loudly and
        # let the column stay as it is; ingestion guards still cap inserts to a
        # safe length so the worst case is a truncation warning, not a crash.
        logger.exception("trades.external_id migration probe failed: %s", exc)
