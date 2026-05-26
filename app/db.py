"""SQLAlchemy engine, session, and Base declarative class."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

_settings = get_settings()

# check_same_thread=False is required for SQLite when sharing across the
# FastAPI request handler thread and background scanner thread.
_is_sqlite = _settings.database_url.startswith("sqlite")
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
    """FastAPI dependency that yields a scoped DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables. Safe to call repeatedly."""
    # Import models so they register on Base.metadata before create_all.
    from app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    if _is_sqlite:
        _add_sqlite_column_if_missing("trades", "outcome", "VARCHAR(64)")
        _add_sqlite_column_if_missing("signals", "outcome", "VARCHAR(64)")
        _add_sqlite_column_if_missing("mlb_edges", "opening_line", "FLOAT")
        _add_sqlite_column_if_missing("mlb_edges", "current_line", "FLOAT")
        _add_sqlite_column_if_missing("mlb_edges", "recommended_line", "FLOAT")
        _add_sqlite_column_if_missing("mlb_edges", "closing_line", "FLOAT")
        _add_sqlite_column_if_missing("mlb_edges", "closing_price", "FLOAT")
        _add_sqlite_column_if_missing("mlb_edges", "result", "VARCHAR(64)")
        _add_sqlite_column_if_missing("mlb_edges", "win_loss_push", "VARCHAR(8)")
        _add_sqlite_column_if_missing("mlb_edges", "implied_probability_at_entry", "FLOAT")
        _add_sqlite_column_if_missing("mlb_edges", "implied_probability_at_close", "FLOAT")
        _add_sqlite_column_if_missing("mlb_edges", "clv_points", "FLOAT")
        _add_sqlite_column_if_missing("mlb_edges", "clv_percent", "FLOAT")
        _add_sqlite_column_if_missing("mlb_edges", "roi_units", "FLOAT")
        _add_sqlite_column_if_missing("mlb_edges", "graded_at", "DATETIME")


def _add_sqlite_column_if_missing(table: str, column: str, ddl: str) -> None:
    inspector = inspect(engine)
    if table not in inspector.get_table_names():
        return
    if column in {col["name"] for col in inspector.get_columns(table)}:
        return
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
