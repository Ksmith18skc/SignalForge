from __future__ import annotations

from app.api.routes import _database_backend


def test_database_backend_classification_does_not_expose_url_parts():
    assert _database_backend("sqlite:///./signalforge.db") == "sqlite"
    assert _database_backend("postgresql://user:secret@example.com/db") == "postgres"
    assert _database_backend("postgres://user:secret@example.com/db") == "postgres"
    assert _database_backend("mysql://user:secret@example.com/db") == "unknown"
