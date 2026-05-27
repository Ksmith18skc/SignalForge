"""Persistence helpers that live outside the core ORM module.

Currently houses the personal P&L tracker schema (`pnl_store`). Importing
this package guarantees those models are registered on `Base.metadata`
before `create_all` runs at app startup.
"""

from app.storage import pnl_store  # noqa: F401
