"""Grade resolved MLB edges using MLB StatsAPI game results."""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any

from sqlalchemy import select

from app.db import SessionLocal, init_db
from app.models import MlbEdge, MlbGame
from app.providers.mlb_stats_api import MlbStatsApiProvider
from app.services.mlb_performance import grade_edge

logger = logging.getLogger(__name__)


async def main_async(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade final MLB edge results")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level)

    init_db()
    mlb = MlbStatsApiProvider()
    db = SessionLocal()
    counts = {"graded": 0, "skipped": 0, "failed": 0}
    try:
        rows = list(
            db.execute(
                select(MlbEdge, MlbGame)
                .join(MlbGame, MlbGame.game_pk == MlbEdge.game_pk)
                .where(MlbEdge.win_loss_push.is_(None))
            ).all()
        )
        for edge, game in rows:
            try:
                linescore = await mlb.linescore(game.game_pk)
                if not _is_final(linescore, game):
                    counts["skipped"] += 1
                    continue
                if edge.edge_type == "game_total":
                    outcome = _grade_total(edge, linescore)
                elif edge.edge_type == "pitcher_strikeouts":
                    boxscore = await mlb.boxscore(game.game_pk)
                    outcome = _grade_pitcher_k(edge, boxscore)
                else:
                    outcome = None
                if outcome is None:
                    counts["skipped"] += 1
                    continue
                result, wlp = outcome
                grade_edge(edge, result=result, win_loss_push=wlp)
                counts["graded"] += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed grading edge=%s game=%s: %s", edge.id, game.game_pk, exc)
                counts["failed"] += 1
        db.commit()
        logger.info("MLB grading: graded=%d skipped=%d failed=%d", counts["graded"], counts["skipped"], counts["failed"])
        return 0 if counts["failed"] == 0 else 1
    finally:
        db.close()


def _is_final(linescore: dict[str, Any], game: MlbGame) -> bool:
    state = (game.game_status or "").lower()
    if "final" in state:
        return True
    current_inning_state = str(linescore.get("currentInningOrdinal") or "").lower()
    return "final" in current_inning_state


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
