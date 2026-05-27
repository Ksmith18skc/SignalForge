"""Backfill graded MLB history + recompute the learning layer.

Wires together the pieces that already exist so the dashboard's historical /
calibration card sections have real data:

  1. For each date in the window: ingest final scores + grade ungraded edges
     (`grade_mlb_results.run_async`). This is what `lookup_edge_score_band`
     reads directly for the edge cards.
  2. Best-effort: pull Wallet-360 / PnL for tracked wallets
     (`falcon_learning.backfill_tracked_wallets`) so wallet ROI/CLV + tiers
     are populated. Skipped gracefully when Falcon is unavailable.
  3. Recompute calibration bands, wallet tiers, and regime stats
     (`falcon_retraining.run_full_retraining`).

Idempotent: every step upserts, so re-running over the same window is safe.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date as date_cls, timedelta
from typing import Any

from sqlalchemy import select

from app.config import get_settings
from app.db import SessionLocal, init_db
from app.models import Trader
from app.providers.falcon import FalconProvider
from app.services.falcon_learning import backfill_tracked_wallets
from app.services.falcon_retraining import run_full_retraining

logger = logging.getLogger(__name__)


def _date_range(start: str, end: str) -> list[str]:
    s = date_cls.fromisoformat(start)
    e = date_cls.fromisoformat(end)
    if e < s:
        s, e = e, s
    out: list[str] = []
    cur = s
    while cur <= e:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


async def run_async(
    *,
    start_date: str,
    end_date: str,
    skip_wallets: bool = False,
) -> dict[str, Any]:
    from scripts.grade_mlb_results import run_async as grade_results_run

    init_db()
    db = SessionLocal()
    result: dict[str, Any] = {"dates": [], "grading": {}, "wallets": None, "retraining": None}
    try:
        for d in _date_range(start_date, end_date):
            grade = await grade_results_run(date=d, db=db)
            result["dates"].append(d)
            result["grading"][d] = grade

        if not skip_wallets:
            settings = get_settings()
            falcon = FalconProvider(settings.falcon_api_key, settings.falcon_base_url)
            wallets = [
                (t.wallet_address, t.nickname)
                for t in db.scalars(select(Trader).where(Trader.wallet_address.is_not(None)))
            ]
            try:
                summary = await backfill_tracked_wallets(db, falcon, wallets)
                db.commit()
                result["wallets"] = summary.as_dict() if hasattr(summary, "as_dict") else str(summary)
            except Exception as exc:  # noqa: BLE001 — Falcon may be offline
                db.rollback()
                logger.warning("Wallet backfill skipped: %s", exc)
                result["wallets"] = {"skipped": True, "reason": str(exc)}

        retrain = run_full_retraining(db)
        db.commit()
        result["retraining"] = retrain.as_dict() if hasattr(retrain, "as_dict") else str(retrain)
        result["status"] = "ok"
        return result
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill graded MLB history + learning layer.")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD (Arizona)")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD (Arizona)")
    parser.add_argument("--skip-wallets", action="store_true", help="Skip Falcon wallet backfill")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    out = asyncio.run(run_async(start_date=args.start, end_date=args.end, skip_wallets=args.skip_wallets))
    print(out)


if __name__ == "__main__":
    main()
