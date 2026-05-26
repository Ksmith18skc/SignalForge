"""Update MLB edge closing lines shortly before or after game start."""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import SessionLocal, init_db
from app.models import MlbEdge, MlbGame
from app.providers.odds_api import OddsApiProvider
from app.services.mlb_edge_engine import _best_effort_odds_payload
from app.services.mlb_odds_analysis import analyze_game_totals
from app.services.mlb_prop_odds import consensus_for_pitcher, normalize_pitcher_strikeout_props
from app.services.mlb_performance import update_closing_line_fields

logger = logging.getLogger(__name__)


async def main_async(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Update MLB edge closing lines")
    parser.add_argument("--window-minutes", type=int, default=30)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level)

    init_db()
    settings = get_settings()
    odds = OddsApiProvider(settings.odds_api_key, settings.odds_api_base_url, settings.odds_bookmakers)
    db = SessionLocal()
    counts = {"updated": 0, "skipped": 0, "failed": 0}
    try:
        for edge, game in _candidate_edges(db, args.window_minutes):
            try:
                payload = await _best_effort_odds_payload(odds, _game_dict(game))
                analysis = _analysis_for_edge(edge, payload)
                closing_line, closing_price = _closing_values(edge, analysis)
                if closing_line is None and closing_price is None:
                    counts["skipped"] += 1
                    continue
                update_closing_line_fields(edge, closing_line=closing_line, closing_price=closing_price)
                counts["updated"] += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed closing-line update for edge=%s: %s", edge.id, exc)
                counts["failed"] += 1
        db.commit()
        logger.info("MLB closing lines: updated=%d skipped=%d failed=%d", counts["updated"], counts["skipped"], counts["failed"])
        return 0 if counts["failed"] == 0 else 1
    finally:
        db.close()


def _candidate_edges(db: Session, window_minutes: int) -> list[tuple[MlbEdge, MlbGame]]:
    cutoff = datetime.utcnow() + timedelta(minutes=window_minutes)
    return list(
        db.execute(
            select(MlbEdge, MlbGame)
            .join(MlbGame, MlbGame.game_pk == MlbEdge.game_pk)
            .where(
                MlbEdge.win_loss_push.is_(None),
                MlbGame.start_time.is_not(None),
                MlbGame.start_time <= cutoff,
            )
        ).all()
    )


def _analysis_for_edge(edge: MlbEdge, payload: dict[str, Any] | None) -> dict[str, Any]:
    if edge.edge_type == "pitcher_strikeouts":
        pitcher_name = edge.market.split(" Over ")[0].split(" Under ")[0]
        return consensus_for_pitcher(normalize_pitcher_strikeout_props(payload), pitcher_name)
    return analyze_game_totals(payload)


def _closing_values(edge: MlbEdge, analysis: dict[str, Any]) -> tuple[float | None, float | None]:
    side = edge.side.lower()
    line = analysis.get("line") if edge.edge_type == "pitcher_strikeouts" else analysis.get("consensus_total_line")
    price = analysis.get(f"best_{side}_price")
    return _num(line), _num(price)


def _game_dict(game: MlbGame) -> dict[str, Any]:
    return {
        "game_pk": game.game_pk,
        "away_team": game.away_team,
        "home_team": game.home_team,
    }


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
