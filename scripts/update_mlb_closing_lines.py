"""Update MLB edge closing lines shortly before or after game start."""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import SessionLocal, init_db
from app.models import MlbEdge, MlbGame
from app.providers.odds_api import OddsApiProvider
from app.services import odds_cache
from app.services.mlb_odds_analysis import analyze_game_totals
from app.services.mlb_prop_odds import consensus_for_pitcher, normalize_pitcher_strikeout_props
from app.services.mlb_performance import update_closing_line_fields

logger = logging.getLogger(__name__)


async def run_async(
    *,
    window_minutes: int = 30,
    date: str | None = None,
    db: Session | None = None,
    odds: Any | None = None,
) -> dict[str, Any]:
    """Refresh closing lines for ungraded edges and return a structured result.

    When ``date`` is provided, only edges whose linked game is on that date are
    considered. Returns counts plus a `reason` when zero rows are updated so
    the dashboard can show something more useful than a generic "OK".
    """
    if db is None:
        init_db()
        db = SessionLocal()
        owns_db = True
    else:
        owns_db = False
    if odds is None:
        settings = get_settings()
        odds = OddsApiProvider(
            settings.odds_api_key, settings.odds_api_base_url, settings.odds_bookmakers,
        )

    counts = {"updated": 0, "skipped": 0, "failed": 0}
    candidates: list[tuple[MlbEdge, MlbGame]] = []
    games_refreshed = 0
    reason: str | None = None
    try:
        candidates = _candidate_edges(db, window_minutes, date=date)
        if not candidates:
            reason = (
                f"No ungraded edge snapshots near start time for {date}."
                if date else
                "No ungraded edge snapshots near start time. Run an MLB edge scan "
                "before game day to enable closing-line capture."
            )
            logger.info("MLB closing lines: %s", reason)
            return _result(counts, candidates, games_refreshed, date, reason)

        for game_date, games in _games_by_date(candidates).items():
            await odds_cache.refresh_mlb_odds_cache(
                db, odds, games, game_date=game_date, force=True,
            )
            games_refreshed += len(games)

        for edge, game in candidates:
            try:
                payload = _cached_payload_for_game(db, game)
                analysis = _analysis_for_edge(edge, payload)
                closing_line, closing_price = _closing_values(edge, analysis)
                if closing_line is None and closing_price is None:
                    counts["skipped"] += 1
                    continue
                update_closing_line_fields(
                    edge, closing_line=closing_line, closing_price=closing_price,
                )
                counts["updated"] += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed closing-line update for edge=%s: %s", edge.id, exc)
                counts["failed"] += 1
        db.commit()
        logger.info(
            "MLB closing lines: candidates=%d games_refreshed=%d updated=%d skipped=%d "
            "failed=%d date=%s",
            len(candidates), games_refreshed, counts["updated"], counts["skipped"],
            counts["failed"], date,
        )
        if counts["updated"] == 0 and reason is None:
            reason = (
                f"Found {len(candidates)} candidate edge(s) but no closing lines were "
                "available from upstream odds. Try again after odds-api refresh."
            )
        return _result(counts, candidates, games_refreshed, date, reason)
    finally:
        if owns_db:
            db.close()


def _result(
    counts: dict[str, int],
    candidates: list[tuple[MlbEdge, MlbGame]],
    games_refreshed: int,
    date: str | None,
    reason: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": counts["failed"] == 0,
        "date": date,
        "candidates": len(candidates),
        "games_refreshed": games_refreshed,
        "closing_lines_updated": counts["updated"],
        "skipped": counts["skipped"],
        "failed": counts["failed"],
    }
    if reason:
        payload["reason"] = reason
    return payload


async def main_async(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Update MLB edge closing lines")
    parser.add_argument("--window-minutes", type=int, default=30)
    parser.add_argument("--date", default=None, help="YYYY-MM-DD game date filter")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level)

    result = await run_async(window_minutes=args.window_minutes, date=args.date)
    return 0 if result.get("ok") else 1


def _candidate_edges(
    db: Session,
    window_minutes: int,
    *,
    date: str | None = None,
) -> list[tuple[MlbEdge, MlbGame]]:
    cutoff = datetime.utcnow() + timedelta(minutes=window_minutes)
    query = (
        select(MlbEdge, MlbGame)
        .join(MlbGame, MlbGame.game_pk == MlbEdge.game_pk)
        .where(
            MlbEdge.win_loss_push.is_(None),
            MlbGame.start_time.is_not(None),
            MlbGame.start_time <= cutoff,
        )
    )
    if date:
        query = query.where(MlbGame.game_date == date)
    return list(db.execute(query).all())


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
        "game_date": game.game_date,
        "away_team": game.away_team,
        "home_team": game.home_team,
    }


def _games_by_date(
    candidates: list[tuple[MlbEdge, MlbGame]],
) -> dict[str, list[dict[str, Any]]]:
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for _edge, game in candidates:
        key = (game.game_date, game.game_pk)
        if key in seen:
            continue
        seen.add(key)
        by_date[game.game_date].append(_game_dict(game))
    return by_date


def _cached_payload_for_game(db: Session, game: MlbGame) -> dict[str, Any] | None:
    """Cache-only lookup. Closing-line passes reuse the events list cached
    by the refresh above; if no match is found we serve None and the
    consumer reports `skipped`."""
    results, _ = odds_cache.matches_for_games(
        db, [_game_dict(game)], game_date=game.game_date,
    )
    if not results:
        return None
    match = results[0]
    if not match.matched_event_id:
        return None
    return odds_cache.get_cached_event_odds(db, match.matched_event_id)


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
