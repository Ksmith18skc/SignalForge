"""Grade resolved MLB edges using MLB StatsAPI game results."""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import SessionLocal, init_db
from app.models import MlbEdge, MlbGame
from app.providers.mlb_stats_api import MlbStatsApiProvider
from app.services.mlb_final_scores import (
    get_final_score,
    ingest_final_scores_for_date,
)
from app.services.mlb_performance import grade_edge

logger = logging.getLogger(__name__)


async def run_async(
    *,
    date: str | None = None,
    db: Session | None = None,
    mlb: Any | None = None,
    skip_ingestion: bool = False,
) -> dict[str, Any]:
    """Grade ungraded edges. Optionally filter by game_date (Arizona).

    Order of operations:
      1. If ``date`` is provided, ingest final scores for that date into the
         persisted ``mlb_final_scores`` table (idempotent upsert). This makes
         the pipeline redeploy-safe: once a final is captured, it stays
         captured.
      2. For each candidate edge, prefer the persisted final-score row and
         only fall back to a live ``linescore`` call when the row is missing.
      3. Pitcher-K markets always need the live boxscore (per-pitcher stats),
         since the persisted score table only stores team totals.
    """
    if db is None:
        init_db()
        db = SessionLocal()
        owns_db = True
    else:
        owns_db = False
    if mlb is None:
        mlb = MlbStatsApiProvider()

    counts = {
        "graded": 0,
        "skipped_not_final": 0,
        "skipped_no_outcome": 0,
        "failed": 0,
    }
    sources = {"persisted": 0, "live": 0}
    candidate_count = 0
    final_count = 0
    ingestion: dict[str, Any] | None = None
    reason: str | None = None
    try:
        query = (
            select(MlbEdge, MlbGame)
            .join(MlbGame, MlbGame.game_pk == MlbEdge.game_pk)
            .where(MlbEdge.win_loss_push.is_(None))
        )
        if date:
            query = query.where(MlbGame.game_date == date)
        rows = list(db.execute(query).all())
        candidate_count = len(rows)

        if not skip_ingestion:
            # Always ingest the schedule(s) covering the candidate edges so
            # `mlb_final_scores` is populated before grading runs. Without
            # this, multi-day windows (date=None from the dashboard) would
            # skip ingestion entirely and force every edge down the live
            # fallback path — where `_is_final` can't reliably tell that
            # the game is over (game.game_status is stale, the linescore
            # endpoint carries no terminal status).
            if date:
                dates_to_ingest = [date]
            else:
                dates_to_ingest = sorted({
                    g.game_date for _, g in rows if g.game_date
                })
            summaries: list[dict[str, Any]] = []
            total_upserted = 0
            for ingest_date in dates_to_ingest:
                try:
                    summary = await ingest_final_scores_for_date(
                        db, mlb, date=ingest_date,
                    )
                except Exception as exc:  # noqa: BLE001
                    # Ingestion failure is non-fatal — we can still grade
                    # from persisted rows captured by an earlier run.
                    logger.warning(
                        "Pre-grade ingestion failed for %s: %s", ingest_date, exc,
                    )
                    summary = {"date": ingest_date, "error": str(exc)}
                summaries.append(summary)
                try:
                    total_upserted += int(summary.get("upserted") or 0)
                except (TypeError, ValueError):
                    pass
            if date and summaries:
                ingestion = summaries[0]
            elif summaries:
                ingestion = {
                    "dates": [s.get("date") for s in summaries],
                    "upserted": total_upserted,
                    "summaries": summaries,
                }

        if candidate_count == 0:
            reason = (
                f"No ungraded edge snapshots found for {date}."
                if date else "No ungraded edge snapshots found."
            )
            logger.info("MLB grading: %s", reason)
            return _result(counts, sources, candidate_count, final_count, date, reason, ingestion)

        for edge, game in rows:
            try:
                outcome, source = await _grade_one(db, edge, game, mlb)
                if outcome is None:
                    if source == "not_final":
                        counts["skipped_not_final"] += 1
                    else:
                        counts["skipped_no_outcome"] += 1
                    continue
                final_count += 1
                sources[source] += 1
                result, wlp = outcome
                grade_edge(edge, result=result, win_loss_push=wlp)
                counts["graded"] += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed grading edge=%s game=%s: %s", edge.id, game.game_pk, exc)
                counts["failed"] += 1
        db.commit()
        logger.info(
            "MLB grading: candidates=%d final=%d graded=%d source_persisted=%d "
            "source_live=%d skipped_not_final=%d skipped_no_outcome=%d failed=%d date=%s",
            candidate_count, final_count, counts["graded"], sources["persisted"],
            sources["live"], counts["skipped_not_final"], counts["skipped_no_outcome"],
            counts["failed"], date,
        )
        if counts["graded"] == 0 and reason is None:
            if final_count == 0:
                reason = (
                    f"Found {candidate_count} ungraded edges but no games are final yet"
                    + (f" for {date}." if date else ".")
                )
            else:
                reason = (
                    f"Final scores available for {final_count} game(s) but no edges "
                    "could be graded (missing recommended_line or unmatched pitcher)."
                )
        return _result(counts, sources, candidate_count, final_count, date, reason, ingestion)
    finally:
        if owns_db:
            db.close()


async def _grade_one(
    db: Session,
    edge: MlbEdge,
    game: MlbGame,
    mlb: Any,
) -> tuple[tuple[str, str] | None, str]:
    """Grade one edge and report which source supplied the score.

    Returns ``(outcome, source)`` where source is one of:
      * ``persisted`` — graded from ``mlb_final_scores``
      * ``live`` — graded from a live StatsAPI call
      * ``not_final`` — no final state available
      * ``no_outcome`` — final available but no graded outcome could be derived
    """
    # 1) Try persisted score first (only useful for game-total edges; pitcher
    #    K grading still needs the live boxscore for strikeout counts).
    if edge.edge_type == "game_total":
        stored = get_final_score(db, game.game_pk)
        if stored is not None:
            outcome = _grade_total_from_persisted(edge, stored)
            return (outcome, "persisted" if outcome else "no_outcome")

    # 2) Fall back to live API. Use the persisted final-score table as the
    #    source of truth for "is this game over": ingestion only writes a
    #    row when the schedule reports a Final status, so a row's presence
    #    is the most reliable terminal-state signal we have. The
    #    `game.game_status` column is a snapshot from edge-generation time
    #    and is typically stale ("Scheduled").
    if not _is_final(db, game):
        return (None, "not_final")
    linescore = await mlb.linescore(game.game_pk)
    if edge.edge_type == "game_total":
        outcome = _grade_total(edge, linescore)
    elif edge.edge_type == "pitcher_strikeouts":
        boxscore = await mlb.boxscore(game.game_pk)
        outcome = _grade_pitcher_k(edge, boxscore)
    else:
        outcome = None
    return (outcome, "live" if outcome else "no_outcome")


def _result(
    counts: dict[str, int],
    sources: dict[str, int],
    candidates: int,
    finals: int,
    date: str | None,
    reason: str | None,
    ingestion: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": counts["failed"] == 0,
        "date": date,
        "candidates": candidates,
        "finals_found": finals,
        "graded": counts["graded"],
        "graded_from_persisted": sources["persisted"],
        "graded_from_live": sources["live"],
        "skipped_not_final": counts["skipped_not_final"],
        "skipped_no_outcome": counts["skipped_no_outcome"],
        "failed": counts["failed"],
    }
    if ingestion is not None:
        payload["ingestion"] = ingestion
    if reason:
        payload["reason"] = reason
    return payload


def _grade_total_from_persisted(edge: MlbEdge, score) -> tuple[str, str] | None:
    if edge.recommended_line is None:
        return None
    total = int(score.total_runs)
    if abs(total - edge.recommended_line) < 0.0001:
        wlp = "push"
    elif edge.side.lower() == "over":
        wlp = "win" if total > edge.recommended_line else "loss"
    else:
        wlp = "win" if total < edge.recommended_line else "loss"
    return f"Final total {total} runs", wlp


async def main_async(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade final MLB edge results")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD game date filter")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level)

    result = await run_async(date=args.date)
    return 0 if result.get("ok") else 1


def _is_final(db: Session, game: MlbGame) -> bool:
    state = (game.game_status or "").lower()
    if "final" in state:
        return True
    # The linescore endpoint does not carry game status — `currentInningOrdinal`
    # is "1st"/"9th"/etc, never "Final". Use the persisted final-score row
    # (written by ingestion when the schedule reports Final) as the canonical
    # terminal-state check instead.
    return get_final_score(db, int(game.game_pk)) is not None


def _grade_total(edge: MlbEdge, linescore: dict[str, Any]) -> tuple[str, str] | None:
    teams = linescore.get("teams") or {}
    home_runs = _runs((teams.get("home") or {}))
    away_runs = _runs((teams.get("away") or {}))
    if home_runs is None or away_runs is None or edge.recommended_line is None:
        return None
    total = home_runs + away_runs
    if abs(total - edge.recommended_line) < 0.0001:
        wlp = "push"
    elif edge.side.lower() == "over":
        wlp = "win" if total > edge.recommended_line else "loss"
    else:
        wlp = "win" if total < edge.recommended_line else "loss"
    return f"Final total {total} runs", wlp


def _grade_pitcher_k(edge: MlbEdge, boxscore: dict[str, Any]) -> tuple[str, str] | None:
    pitcher_name = edge.market.split(" Over ")[0].split(" Under ")[0].strip().lower()
    strikeouts = _pitcher_strikeouts(boxscore, pitcher_name)
    if strikeouts is None or edge.recommended_line is None:
        return None
    if abs(strikeouts - edge.recommended_line) < 0.0001:
        wlp = "push"
    elif edge.side.lower() == "over":
        wlp = "win" if strikeouts > edge.recommended_line else "loss"
    else:
        wlp = "win" if strikeouts < edge.recommended_line else "loss"
    return f"{pitcher_name.title()} {strikeouts} strikeouts", wlp


def _pitcher_strikeouts(boxscore: dict[str, Any], pitcher_name: str) -> int | None:
    teams = boxscore.get("teams") or {}
    for side in ("away", "home"):
        players = ((teams.get(side) or {}).get("players") or {})
        for player in players.values():
            person = player.get("person") or {}
            name = str(person.get("fullName") or "").lower()
            if name != pitcher_name:
                continue
            pitching = ((player.get("stats") or {}).get("pitching") or {})
            value = pitching.get("strikeOuts")
            return int(value) if value is not None else None
    return None


def _runs(team_node: dict[str, Any]) -> int | None:
    runs = team_node.get("runs")
    try:
        return int(runs)
    except (TypeError, ValueError):
        return None


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
