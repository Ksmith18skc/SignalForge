"""Seed the SignalForge DB with the operator's Polymarket watchlist.

Run with:
    python -m scripts.seed
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select

from sqlalchemy.orm import Session

from app.db import SessionLocal, init_db
from app.models import Trader
from app.utils.logging import configure_logging

logger = logging.getLogger(__name__)


# Canonical watchlist. Keep this list to real wallets only; no placeholder
# traders or missing wallet addresses.
SEED_TRADERS: list[dict[str, Any]] = [
    {
        "nickname": "surfandturf",
        "wallet_address": "0x9f2fe025f84839ca81dd8e0338892605702d2ca8",
        "platform": "polymarket",
        "trust_score": 82.66,
        "trader_rank": 3681,
        "win_rate": 0.5373,
        "total_pnl": 263094.49,
        "seven_day_return": -0.0834,
        "total_positions": 367,
        "copy_mode": "alert_only",
        "tags": ["sports", "soccer"],
        "category_strengths": {"sports": 0.07, "politics": 0.76, "crypto": 0.31, "macro": 0.26},
    },
    {
        "nickname": "ooohhyeah",
        "wallet_address": "0x135098cf001826608023de690a2377d65d250841",
        "platform": "polymarket",
        "trust_score": 74.76,
        "trader_rank": 467,
        "win_rate": 0.5089,
        "total_pnl": 292105.25,
        "seven_day_return": 0.278,
        "total_positions": 253,
        "copy_mode": "alert_only",
        "tags": ["sports", "soccer"],
        "category_strengths": {"sports": 0.42, "politics": 0.57, "crypto": 0.41, "macro": 0.91},
    },
    {
        "nickname": "VeryLucky888",
        "wallet_address": "0x6d3c5bd13984b2de47c3a88ddc455309aab3d294",
        "platform": "polymarket",
        "trust_score": 73.41,
        "trader_rank": 3301,
        "win_rate": 0.4806,
        "total_pnl": 446087.64,
        "seven_day_return": 0.3179,
        "total_positions": 138,
        "copy_mode": "alert_only",
        "tags": ["sports", "soccer", "UFC"],
        "category_strengths": {"sports": 0.25, "politics": 0.23, "crypto": 0.44, "macro": 0.13},
    },
    {
        "nickname": "HomeRunHazard",
        "wallet_address": "0x5268527977f700f9bf9b6d5cd843859e4e70135d",
        "platform": "polymarket",
        "trust_score": 69.79,
        "trader_rank": 2911,
        "win_rate": 0.6647,
        "total_pnl": -42008.06,
        "seven_day_return": 0.3477,
        "total_positions": 76,
        "copy_mode": "alert_only",
        "tags": ["sports", "NHL", "MLB"],
        "category_strengths": {"sports": 0.7, "politics": 0.72, "crypto": 0.69, "macro": 0.55},
    },
    {
        "nickname": "bananawoin",
        "wallet_address": "0xbca08c1bc204a34f2fddbe47b438b9bd42ac9705",
        "platform": "polymarket",
        "trust_score": 69.64,
        "trader_rank": 334,
        "win_rate": 0.6838,
        "total_pnl": 350938.28,
        "seven_day_return": -0.1046,
        "total_positions": 90,
        "copy_mode": "alert_only",
        "tags": ["sports", "MLB"],
        "category_strengths": {"sports": 0.26, "politics": 0.48, "crypto": 0.81, "macro": 0.44},
    },
    {
        "nickname": "ewelmealt",
        "wallet_address": "0x07921379f7b31ef93da634b688b2fe36897db778",
        "platform": "polymarket",
        "trust_score": 68.7,
        "trader_rank": 298,
        "win_rate": 0.4791,
        "total_pnl": 37986.41,
        "seven_day_return": 0.2762,
        "total_positions": 496,
        "copy_mode": "alert_only",
        "tags": ["sports", "soccer"],
        "category_strengths": {"sports": 0.97, "politics": 0.77, "crypto": 0.87, "macro": 0.46},
    },
    {
        "nickname": "bambambole",
        "wallet_address": "0x087d5d7939d4757a4eb508c67028073eb900b872",
        "platform": "polymarket",
        "trust_score": 66.42,
        "trader_rank": 2583,
        "win_rate": 0.6282,
        "total_pnl": 124406.26,
        "seven_day_return": -0.0203,
        "total_positions": 232,
        "copy_mode": "alert_only",
        "tags": ["sports", "politics", "soccer"],
        "category_strengths": {"sports": 0.78, "politics": 0.39, "crypto": 0.25, "macro": 0.02},
    },
    {
        "nickname": "LaBradfordSmith22",
        "wallet_address": "0x9495425feeb0c250accb89275c97587011b19a27",
        "platform": "polymarket",
        "trust_score": 62.63,
        "trader_rank": 1011,
        "win_rate": 0.5202,
        "total_pnl": 313909.53,
        "seven_day_return": 0.1963,
        "total_positions": 356,
        "copy_mode": "alert_only",
        "tags": ["sports", "soccer", "NBA"],
        "category_strengths": {"sports": 0.29, "politics": 0.34, "crypto": 0.35, "macro": 0.14},
    },
    {
        "nickname": "beachboy4",
        "wallet_address": "0xc2e7800b5af46e6093872b177b7a5e7f0563be51",
        "platform": "polymarket",
        "trust_score": 53.8,
        "trader_rank": 1790,
        "win_rate": 0.6894,
        "total_pnl": 251298.69,
        "seven_day_return": 0.3369,
        "total_positions": 67,
        "copy_mode": "alert_only",
        "tags": ["sports", "soccer", "NHL"],
        "category_strengths": {"sports": 0.79, "politics": 0.28, "crypto": 0.49, "macro": 0.57},
    },
    {
        "nickname": "pinkblanket",
        "wallet_address": "0x0720803c7cb0d0c5a928787b3b7ea148c6831cdb",
        "platform": "polymarket",
        "trust_score": 51.91,
        "trader_rank": 4197,
        "win_rate": 0.4685,
        "total_pnl": 426160.7,
        "seven_day_return": 0.3853,
        "total_positions": 52,
        "copy_mode": "alert_only",
        "tags": ["sports", "NFL"],
        "category_strengths": {"sports": 0.97, "politics": 0.62, "crypto": 0.17, "macro": 0.11},
    },
    {
        "nickname": "Soarin22",
        "wallet_address": "0x84dbb7103982e3617704a2ed7d5b39691952aeeb",
        "platform": "polymarket",
        "trust_score": 45.69,
        "trader_rank": 4783,
        "win_rate": 0.732,
        "total_pnl": -6647.44,
        "seven_day_return": 0.3086,
        "total_positions": 434,
        "copy_mode": "alert_only",
        "tags": ["sports", "NFL", "PGA", "NBA"],
        "category_strengths": {"sports": 0.31, "politics": 0.69, "crypto": 0.24, "macro": 0.78},
    },
]


def seed_watchlist(db: Session) -> tuple[int, int]:
    created = 0
    updated = 0
    for entry in SEED_TRADERS:
        existing = db.scalar(
            select(Trader).where(Trader.nickname == entry["nickname"])
        )
        if existing:
            for key, value in entry.items():
                setattr(existing, key, value)
            updated += 1
            logger.info("updated trader: %s", entry["nickname"])
            continue

        trader = Trader(**entry)
        db.add(trader)
        created += 1
        logger.info("seeded trader: %s", entry["nickname"])
    return created, updated


def seed() -> int:
    configure_logging()
    init_db()

    db = SessionLocal()
    try:
        created, updated = seed_watchlist(db)
        db.commit()
    finally:
        db.close()

    logger.info("Seed complete. Created %d new traders. Updated %d traders.", created, updated)
    return created


if __name__ == "__main__":
    seed()
