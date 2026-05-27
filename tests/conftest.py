"""Shared pytest fixtures.

Each test gets a fresh in-memory SQLite database so tests stay isolated.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

import app.db as db_module
from app.db import Base


@pytest.fixture()
def db_session() -> Generator[Session, None, None]:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        future=True,
    )
    # Ensure all models register on metadata before create_all.
    from app import models  # noqa: F401
    from app import storage  # noqa: F401

    Base.metadata.create_all(engine)

    TestSession = sessionmaker(bind=engine, autoflush=False, future=True)

    # Swap the app's SessionLocal so any service code under test uses this DB.
    original_engine = db_module.engine
    original_session = db_module.SessionLocal
    db_module.engine = engine
    db_module.SessionLocal = TestSession

    session = TestSession()
    try:
        yield session
    finally:
        session.close()
        db_module.engine = original_engine
        db_module.SessionLocal = original_session
        Base.metadata.drop_all(engine)
        engine.dispose()
