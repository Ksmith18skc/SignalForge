"""Run the MLB edge engine and persist today's daily card."""

from __future__ import annotations

import argparse
import asyncio
import logging

from app.db import SessionLocal, init_db
from app.services.mlb_edge_engine import run_daily_mlb_edges


async def main_async(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SignalForge MLB daily card")
    parser.add_argument("--game-date", help="YYYY-MM-DD, defaults to today")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level)

    init_db()
    db = SessionLocal()
    try:
        result = await run_daily_mlb_edges(db, game_date=args.game_date)
        logging.info(
            "MLB daily card complete: date=%s games=%s edges=%s",
            result["date"],
            result["games"],
            result["edges"],
        )
        return 0
    except Exception:
        db.rollback()
        logging.exception("MLB daily card failed")
        return 1
    finally:
        db.close()


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
