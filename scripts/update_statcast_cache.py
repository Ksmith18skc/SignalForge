"""Refresh cached Statcast summaries outside the FastAPI web process.

This worker is intentionally separate from request handlers because pybaseball
can allocate large pandas DataFrames and perform slow remote pulls.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Iterable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import SessionLocal, init_db
from app.models import BatterStatcastSummary, PitcherStatcastSummary
from app.providers.mlb_stats_api import MlbStatsApiProvider
from app.providers.pybaseball_provider import PyBaseballProvider

logger = logging.getLogger(__name__)


async def main_async(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level)
    init_db()
    end = date.fromisoformat(args.end_date) if args.end_date else date.today()
    last_n_days = args.last_n_days
    start = end - timedelta(days=last_n_days - 1)

    pyb = PyBaseballProvider()
    mlb = MlbStatsApiProvider()
    db = SessionLocal()
    try:
        pitcher_ids = set(args.pitcher_id or [])
        batter_ids = set(args.batter_id or [])

        if not pitcher_ids and not batter_ids:
            pitcher_ids.update(await _probable_pitcher_ids(mlb, end.isoformat()))

        pitcher_ids.update(_configured_player_ids())

        logger.info(
            "Refreshing Statcast cache for %d pitchers and %d batters (%s to %s)",
            len(pitcher_ids),
            len(batter_ids),
            start,
            end,
        )
        for player_id in sorted(pitcher_ids):
            await _refresh_pitcher(db, pyb, player_id, start, end, last_n_days)
        for player_id in sorted(batter_ids):
            await _refresh_batter(db, pyb, player_id, start, end, last_n_days)
        db.commit()
        logger.info("Statcast cache refresh complete")
        return 0
    except Exception:
        db.rollback()
        logger.exception("Statcast cache refresh failed")
        return 1
    finally:
        db.close()


async def _probable_pitcher_ids(mlb: MlbStatsApiProvider, game_date: str) -> set[int]:
    payload = await mlb.schedule(game_date=game_date)
    ids: set[int] = set()
    for day in payload.get("dates") or []:
        for game in day.get("games") or []:
            teams = game.get("teams") or {}
            for side in ("away", "home"):
                pitcher = ((teams.get(side) or {}).get("probablePitcher") or {})
                player_id = pitcher.get("id")
                if player_id:
                    ids.add(int(player_id))
    return ids


async def _refresh_pitcher(
    db: Session,
    pyb: PyBaseballProvider,
    player_id: int,
    start: date,
    end: date,
    last_n_days: int,
) -> None:
    records = await pyb.pitcher_statcast(
        player_id,
        start.isoformat(),
        end.isoformat(),
        limit=10_000,
    )
    summary = summarize_statcast(records, player_id=player_id, season=end.year, last_n_days=last_n_days)
    _upsert_summary(db, PitcherStatcastSummary, summary)


async def _refresh_batter(
    db: Session,
    pyb: PyBaseballProvider,
    player_id: int,
    start: date,
    end: date,
    last_n_days: int,
) -> None:
    records = await pyb.batter_statcast(
        player_id,
        start.isoformat(),
        end.isoformat(),
        limit=10_000,
    )
    summary = summarize_statcast(records, player_id=player_id, season=end.year, last_n_days=last_n_days)
    _upsert_summary(db, BatterStatcastSummary, summary)


def summarize_statcast(
    records: list[dict[str, Any]],
    *,
    player_id: int,
    season: int,
    last_n_days: int,
) -> dict[str, Any]:
    events = [str(row.get("events") or "").lower() for row in records]
    descriptions = [str(row.get("description") or "").lower() for row in records]
    games = {row.get("game_pk") for row in records if row.get("game_pk") is not None}
    player_name = _first_present(records, "player_name", "pitcher_name", "batter_name")
    strikeouts = sum(1 for event in events if event == "strikeout")
    walks = sum(1 for event in events if event in {"walk", "intent_walk", "hit_by_pitch"})
    plate_appearances = sum(1 for event in events if event)
    swings = sum(1 for desc in descriptions if _is_swing(desc))
    whiffs = sum(1 for desc in descriptions if desc in {"swinging_strike", "swinging_strike_blocked", "foul_tip"})
    chases = sum(1 for row, desc in zip(records, descriptions, strict=False) if _is_chase(row, desc))
    pitch_counts = _pitch_counts_by_game(records)

    return {
        "player_id": player_id,
        "player_name": player_name,
        "season": season,
        "last_n_days": last_n_days,
        "games": len(games),
        "innings_pitched": round(_estimated_outs(events) / 3, 2),
        "strikeouts": strikeouts,
        "walks": walks,
        "pitch_count_avg": round(sum(pitch_counts) / len(pitch_counts), 2) if pitch_counts else None,
        "strikeouts_per_start": round(strikeouts / len(games), 2) if games else None,
        "whiff_rate": round(whiffs / swings, 4) if swings else None,
        "chase_rate": round(chases / len(records), 4) if records else None,
        "k_rate": round(strikeouts / plate_appearances, 4) if plate_appearances else None,
        "updated_at": datetime.utcnow(),
        "source": "pybaseball_worker",
    }


def _upsert_summary(
    db: Session,
    model: type[PitcherStatcastSummary] | type[BatterStatcastSummary],
    values: dict[str, Any],
) -> None:
    existing = db.scalar(
        select(model).where(
            model.player_id == values["player_id"],
            model.season == values["season"],
            model.last_n_days == values["last_n_days"],
        )
    )
    if existing is None:
        db.add(model(**values))
        return
    for key, value in values.items():
        setattr(existing, key, value)


def _pitch_counts_by_game(records: Iterable[dict[str, Any]]) -> list[int]:
    counts: dict[Any, int] = {}
    for row in records:
        game = row.get("game_pk")
        if game is None:
            continue
        counts[game] = counts.get(game, 0) + 1
    return list(counts.values())


def _estimated_outs(events: Iterable[str]) -> int:
    out_events = {
        "strikeout",
        "field_out",
        "force_out",
        "grounded_into_double_play",
        "double_play",
        "fielders_choice_out",
        "sac_fly",
        "sac_bunt",
    }
    outs = 0
    for event in events:
        outs += 2 if event in {"grounded_into_double_play", "double_play"} else int(event in out_events)
    return outs


def _is_swing(description: str) -> bool:
    return description in {
        "swinging_strike",
        "swinging_strike_blocked",
        "foul",
        "foul_tip",
        "foul_bunt",
        "hit_into_play",
        "hit_into_play_no_out",
        "hit_into_play_score",
    }


def _is_chase(row: dict[str, Any], description: str) -> bool:
    zone = row.get("zone")
    try:
        zone_int = int(zone)
    except (TypeError, ValueError):
        return False
    return zone_int > 9 and _is_swing(description)


def _first_present(records: list[dict[str, Any]], *keys: str) -> str | None:
    for row in records:
        for key in keys:
            value = row.get(key)
            if value:
                return str(value)
    return None


def _configured_player_ids() -> set[int]:
    ids: set[int] = set()
    for part in get_settings().statcast_cache_player_ids.split(","):
        part = part.strip()
        if part:
            ids.add(int(part))
    return ids


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh cached Statcast summaries")
    parser.add_argument("--last-n-days", type=int, default=get_settings().statcast_cache_last_n_days)
    parser.add_argument("--end-date", help="YYYY-MM-DD, defaults to today")
    parser.add_argument("--pitcher-id", type=int, action="append", default=[])
    parser.add_argument("--batter-id", type=int, action="append", default=[])
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
