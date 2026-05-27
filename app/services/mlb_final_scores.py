"""Persisted MLB final-score ingestion.

Reads the StatsAPI schedule for a given Arizona card date, picks out
games that are flagged Final, and upserts a row per ``game_pk`` into
``mlb_final_scores``. Grading consumes this table first so the pipeline
survives a restart / cold cache: once a final is captured, it stays
captured.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Iterable

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import MlbFinalScore

logger = logging.getLogger(__name__)


def _final_state(status: dict[str, Any] | None) -> str | None:
    """Return a status string if the game is finished, else None."""
    if not isinstance(status, dict):
        return None
    abstract = str(status.get("abstractGameState") or "").lower()
    detailed = str(status.get("detailedState") or "")
    coded = str(status.get("codedGameState") or "").lower()
    if abstract == "final" or "final" in detailed.lower() or coded == "f":
        return detailed or "Final"
    return None


def _iter_schedule_games(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for day in payload.get("dates") or []:
        if not isinstance(day, dict):
            continue
        for game in day.get("games") or []:
            if isinstance(game, dict):
                yield game


def _extract_score(game: dict[str, Any]) -> dict[str, Any] | None:
    status_text = _final_state(game.get("status"))
    if not status_text:
        return None
    teams = game.get("teams") or {}
    home_node = teams.get("home") or {}
    away_node = teams.get("away") or {}
    try:
        home_score = int(home_node.get("score"))
        away_score = int(away_node.get("score"))
    except (TypeError, ValueError):
        return None
    game_pk = game.get("gamePk")
    if game_pk is None:
        return None
    home_team = ((home_node.get("team") or {}).get("name")
                 or (home_node.get("team") or {}).get("teamName") or "")
    away_team = ((away_node.get("team") or {}).get("name")
                 or (away_node.get("team") or {}).get("teamName") or "")
    return {
        "game_pk": int(game_pk),
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "status": status_text,
    }


def upsert_final_score(
    db: Session,
    *,
    game_pk: int,
    generated_for_date: str,
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
    status: str = "Final",
) -> MlbFinalScore:
    """Insert or update a single final-score row. Idempotent."""
    row = db.get(MlbFinalScore, int(game_pk))
    total = int(home_score) + int(away_score)
    if row is None:
        row = MlbFinalScore(
            game_pk=int(game_pk),
            generated_for_date=generated_for_date,
            home_team=home_team,
            away_team=away_team,
            home_score=int(home_score),
            away_score=int(away_score),
            total_runs=total,
            status=status,
            fetched_at=datetime.utcnow(),
        )
        db.add(row)
        return row
    row.generated_for_date = generated_for_date
    row.home_team = home_team
    row.away_team = away_team
    row.home_score = int(home_score)
    row.away_score = int(away_score)
    row.total_runs = total
    row.status = status
    row.fetched_at = datetime.utcnow()
    return row


def get_final_score(db: Session, game_pk: int) -> MlbFinalScore | None:
    return db.get(MlbFinalScore, int(game_pk))


def persisted_final_score_count(
    db: Session,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> int:
    query = select(func.count()).select_from(MlbFinalScore)
    if start_date:
        query = query.where(MlbFinalScore.generated_for_date >= start_date)
    if end_date:
        query = query.where(MlbFinalScore.generated_for_date <= end_date)
    return int(db.execute(query).scalar() or 0)


async def ingest_final_scores_for_date(
    db: Session,
    mlb: Any,
    *,
    date: str,
) -> dict[str, Any]:
    """Fetch the schedule for ``date`` and upsert any final games found.

    Returns a structured summary so callers can show counts in the UI.
    """
    try:
        payload = await mlb.schedule(game_date=date, hydrate="linescore")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Final-score ingestion failed for %s: %s", date, exc)
        return {
            "date": date,
            "games_seen": 0,
            "finals_found": 0,
            "upserted": 0,
            "error": str(exc),
        }

    upserted = 0
    games_seen = 0
    finals_found = 0
    for game in _iter_schedule_games(payload):
        games_seen += 1
        score = _extract_score(game)
        if score is None:
            continue
        finals_found += 1
        upsert_final_score(
            db,
            generated_for_date=date,
            **score,
        )
        upserted += 1
    db.commit()
    logger.info(
        "MLB final-score ingestion: date=%s games_seen=%d finals_found=%d upserted=%d",
        date, games_seen, finals_found, upserted,
    )
    return {
        "date": date,
        "games_seen": games_seen,
        "finals_found": finals_found,
        "upserted": upserted,
    }
