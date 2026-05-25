"""Seed the SignalForge DB with the operator's Polymarket Analytics watchlist.

Run with:
    python -m scripts.seed
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select

from app.db import SessionLocal, init_db
from app.models import Trader
from app.utils.logging import configure_logging

logger = logging.getLogger(__name__)


# Seed list — wallet addresses backfilled where known; others may be discovered
# via Polymarket Analytics once that provider is wired up.
SEED_TRADERS: list[dict[str, Any]] = [
    {
        "nickname": "LaBradfordSmith22",
        "wallet_address": "0x9495425feeb0c250accb89275c97587011b19a27",
        "platform": "polymarket",
        "tags": ["sports", "macro"],
        "trust_score": 85.0,
        "category_strengths": {"sports": 0.9, "macro": 0.6},
    },
    {
        "nickname": "surfandturf",
        "platform": "polymarket",
        "tags": ["politics", "crypto"],
        "trust_score": 78.0,
        "category_strengths": {"politics": 0.85, "crypto": 0.7},
    },
    {
        "nickname": "HomeRunHazard",
        "platform": "polymarket",
        "tags": ["sports"],
        "trust_score": 80.0,
        "category_strengths": {"sports": 0.95},
    },
    {
        "nickname": "bananawoin",
        "platform": "polymarket",
        "tags": ["crypto"],
        "trust_score": 72.0,
        "category_strengths": {"crypto": 0.88},
    },
    {
        "nickname": "VeryLucky888",
        "platform": "polymarket",
        "tags": ["macro", "politics"],
        "trust_score": 70.0,
        "category_strengths": {"macro": 0.75, "politics": 0.7},
    },
    {
        "nickname": "Soarin22",
        "platform": "polymarket",
        "tags": ["sports", "entertainment"],
        "trust_score": 74.0,
        "category_strengths": {"sports": 0.8, "entertainment": 0.7},
    },
    {
        "nickname": "ewelmealt",
        "platform": "polymarket",
        "tags": ["macro"],
        "trust_score": 68.0,
    },
    {
        "nickname": "pinkblanket",
        "platform": "polymarket",
        "tags": ["politics"],
        "trust_score": 66.0,
    },
    {
        "nickname": "bambambole",
        "platform": "polymarket",
        "tags": ["sports"],
        "trust_score": 71.0,
    },
    {
        "nickname": "ooohhyeah",
        "platform": "polymarket",
        "tags": ["crypto", "macro"],
        "trust_score": 69.0,
    },
]


def seed() -> int:
    configure_logging()
    init_db()

    created = 0
    db = SessionLocal()
    try:
        for entry in SEED_TRADERS:
            existing = db.scalar(
                select(Trader).where(Trader.nickname == entry["nickname"])
            )
            if existing:
                logger.info("trader '%s' already exists — skipping", entry["nickname"])
                continue
            trader = Trader(**entry)
            db.add(trader)
            created += 1
            logger.info("seeded trader: %s", entry["nickname"])
        db.commit()
    finally:
        db.close()

    logger.info("Seed complete. Created %d new traders.", created)
    return created


if __name__ == "__main__":
    seed()
