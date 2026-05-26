"""Daily MLB edge engine orchestration."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from sqlalchemy import delete, desc, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import (
    MlbDailyCard,
    MlbEdge,
    MlbEdgeFactor,
    MlbGame,
    MlbGameEnvironmentSnapshot,
    MlbOddsSnapshot,
    MlbPitcherPropSnapshot,
)
from app.providers.mlb_stats_api import MlbStatsApiProvider
from app.providers.odds_api import OddsApiProvider
from app.providers.weather_api import WeatherApiProvider
from app.services.mlb_edge import statcast_context
from app.services.mlb_edge_scoring import edge_to_dict
from app.services.mlb_environment import score_environment
from app.services.mlb_odds_analysis import analyze_game_totals, analyze_pitcher_k_props
from app.services.mlb_pitcher_k_model import pitcher_k_edges
from app.services.mlb_totals_model import total_edges

logger = logging.getLogger(__name__)


async def run_daily_mlb_edges(db: Session, *, game_date: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    card_date = game_date or settings.mlb_edge_default_game_date or date.today().isoformat()
    mlb = MlbStatsApiProvider()
    weather = WeatherApiProvider(settings.weather_api_key, settings.weather_api_base_url)
    odds = OddsApiProvider(settings.odds_api_key, settings.odds_api_base_url, settings.odds_bookmakers)

    games = await _load_games(mlb, card_date)
    db.execute(delete(MlbEdgeFactor).where(MlbEdgeFactor.edge_id.in_(select(MlbEdge.id).where(MlbEdge.generated_for_date == card_date))))
    db.execute(delete(MlbEdge).where(MlbEdge.generated_for_date == card_date))

    created_edges: list[MlbEdge] = []
    for game in games:
        _upsert_game(db, game)
        env = await _environment_for_game(db, weather, game)
        totals_analysis = await _odds_for_game(db, odds, game)
        for edge_payload in total_edges(game=game, odds_analysis=totals_analysis, environment=env):
            created_edges.append(_persist_edge(db, edge_payload, card_date))

        for pitcher in _pitchers(game):
            prop = await _pitcher_prop_for_game(db, odds, game, pitcher)
            statcast = statcast_context(
                db,
                player_id=pitcher.get("id") or 0,
                player_type="pitcher",
                season=int(card_date[:4]),
                last_n_days=settings.statcast_cache_last_n_days,
            )
            for edge_payload in pitcher_k_edges(
                game=game,
                pitcher=pitcher,
                prop_analysis=prop,
                statcast_context=statcast,
                environment=env,
            ):
                created_edges.append(_persist_edge(db, edge_payload, card_date))

    card = _build_daily_card(db, card_date)
    db.commit()
    return {
        "date": card_date,
        "games": len(games),
        "edges": len(created_edges),
        "daily_card": _card_to_dict(card),
    }


def latest_daily_card(db: Session, *, card_date: str | None = None) -> dict[str, Any] | None:
    query = select(MlbDailyCard)
    if card_date:
        query = query.where(MlbDailyCard.card_date == card_date)
    card = db.scalar(query.order_by(desc(MlbDailyCard.card_date)).limit(1))
    return _card_to_dict(card) if card else None


def edges_for_date(db: Session, *, card_date: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    target = card_date or date.today().isoformat()
    edges = list(
        db.scalars(
            select(MlbEdge)
            .where(MlbEdge.generated_for_date == target)
            .order_by(desc(MlbEdge.score))
            .limit(limit)
        )
    )
    return [edge_to_dict(edge) for edge in edges]


def discord_ready_summary(edge: MlbEdge) -> str:
    reasons = "\n".join(f"- {reason}" for reason in (edge.reasons or [])[:3])
    warnings = "\n".join(f"- {warning}" for warning in (edge.warnings or [])[:3])
    warning_block = f"\nWarnings:\n{warnings}" if warnings else ""
    return (
        f"MLB EDGE - {edge.edge_type.replace('_', ' ').title()}\n\n"
        f"Market: {edge.market}\n"
        f"Best line: {edge.side.title()} {edge.line if edge.line is not None else '?'} "
        f"at {edge.best_price if edge.best_price is not None else '?'} on {edge.best_book or 'N/A'}\n"
        f"Score: {edge.score:.0f}\n"
        f"Confidence: {edge.confidence.title()}\n"
        f"Chase risk: {edge.chase_risk.title()}\n\n"
        f"Why:\n{reasons or '- Not enough supporting reasons'}{warning_block}\n\n"
        f"Action: {edge.action}"
    )


async def _load_games(mlb: MlbStatsApiProvider, card_date: str) -> list[dict[str, Any]]:
    payload = await mlb.schedule(game_date=card_date)
    games: list[dict[str, Any]] = []
    for day in payload.get("dates") or []:
        for raw in day.get("games") or []:
            games.append(_normalize_game(raw, card_date))
    return games


def _normalize_game(raw: dict[str, Any], card_date: str) -> dict[str, Any]:
    teams = raw.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    home_pitcher = home.get("probablePitcher") or {}
    away_pitcher = away.get("probablePitcher") or {}
    venue = raw.get("venue") or {}
    status = raw.get("status") or {}
    start = _parse_dt(raw.get("gameDate"))
    home_team = ((home.get("team") or {}).get("name")) or "Home"
    away_team = ((away.get("team") or {}).get("name")) or "Away"
    return {
        "game_pk": int(raw.get("gamePk")),
        "game_date": card_date,
        "home_team": home_team,
        "away_team": away_team,
        "venue": venue.get("name"),
        "probable_home_pitcher": home_pitcher.get("fullName"),
        "probable_home_pitcher_id": home_pitcher.get("id"),
        "probable_away_pitcher": away_pitcher.get("fullName"),
        "probable_away_pitcher_id": away_pitcher.get("id"),
        "game_status": status.get("detailedState") or status.get("abstractGameState"),
        "start_time": start,
        "weather_location_query": venue.get("name") or f"{home_team} stadium",
    }


async def _environment_for_game(
    db: Session,
    weather: WeatherApiProvider,
    game: dict[str, Any],
) -> dict[str, Any]:
    raw_weather: dict[str, Any] = {}
    weather_fields: dict[str, Any] = {}
    try:
        hour = game["start_time"].hour if game.get("start_time") else None
        raw_weather = await weather.baseball_weather(
            game["weather_location_query"],
            game_date=game["game_date"],
            hour=hour,
        )
        weather_fields = raw_weather.get("weather") or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Weather lookup failed for %s: %s", game.get("game_pk"), exc)
    env = score_environment(weather_fields)
    db.add(
        MlbGameEnvironmentSnapshot(
            game_pk=game["game_pk"],
            temperature_score=env["temperature_score"],
            wind_score=env["wind_score"],
            humidity_score=env["humidity_score"],
            precipitation_risk=env["precipitation_risk"],
            park_factor=env["park_factor"],
            run_environment_score=env["run_environment_score"],
            under_environment_score=env["under_environment_score"],
            k_environment_score=env["k_environment_score"],
            warnings=env["warnings"],
            raw_weather=raw_weather,
        )
    )
    return env


async def _odds_for_game(db: Session, odds: OddsApiProvider, game: dict[str, Any]) -> dict[str, Any]:
    payload = await _best_effort_odds_payload(odds, game)
    analysis = analyze_game_totals(payload)
    db.add(
        MlbOddsSnapshot(
            game_pk=game["game_pk"],
            sportsbook_event_id=str(payload.get("id")) if payload else None,
            consensus_total_line=analysis.get("consensus_total_line"),
            best_over_price=analysis.get("best_over_price"),
            best_over_book=analysis.get("best_over_book"),
            best_under_price=analysis.get("best_under_price"),
            best_under_book=analysis.get("best_under_book"),
            consensus_price=analysis.get("consensus_price"),
            line_disagreement=analysis.get("line_disagreement") or 0.0,
            book_count=analysis.get("book_count") or 0,
            stale_book_candidates=analysis.get("stale_book_candidates") or [],
            movement_direction=analysis.get("movement_direction"),
            steam_velocity=analysis.get("steam_velocity"),
            rows=analysis.get("rows") or [],
        )
    )
    return analysis


async def _pitcher_prop_for_game(
    db: Session,
    odds: OddsApiProvider,
    game: dict[str, Any],
    pitcher: dict[str, Any],
) -> dict[str, Any]:
    payload = await _best_effort_odds_payload(odds, game)
    analysis = analyze_pitcher_k_props(payload, pitcher_name=pitcher.get("name"))
    db.add(
        MlbPitcherPropSnapshot(
            game_pk=game["game_pk"],
            pitcher_id=pitcher.get("id"),
            pitcher_name=pitcher.get("name"),
            line=analysis.get("line"),
            best_over_price=analysis.get("best_over_price"),
            best_over_book=analysis.get("best_over_book"),
            best_under_price=analysis.get("best_under_price"),
            best_under_book=analysis.get("best_under_book"),
            consensus_price=analysis.get("consensus_price"),
            line_disagreement=analysis.get("line_disagreement") or 0.0,
            book_count=analysis.get("book_count") or 0,
            movement_direction=analysis.get("movement_direction"),
            steam_velocity=analysis.get("steam_velocity"),
            rows=analysis.get("rows") or [],
        )
    )
    return analysis


async def _best_effort_odds_payload(odds: OddsApiProvider, game: dict[str, Any]) -> dict[str, Any] | None:
    try:
        query = f"{game['away_team']} {game['home_team']}"
        events = await odds.search_events(query)
        if not events:
            return None
        event_id = events[0].get("id")
        if not event_id:
            return None
        return await odds.odds(event_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Odds lookup failed for %s: %s", game.get("game_pk"), exc)
        return None


def _persist_edge(db: Session, payload: dict[str, Any], card_date: str) -> MlbEdge:
    edge = MlbEdge(
        game_pk=payload["game_pk"],
        edge_type=payload["edge_type"],
        market=payload["market"],
        side=payload["side"],
        line=payload.get("line"),
        best_book=payload.get("best_book"),
        best_price=payload.get("best_price"),
        consensus_price=payload.get("consensus_price"),
        score=payload["score"],
        confidence=payload["confidence"],
        action=payload["action"],
        chase_risk=payload["chase_risk"],
        reasons=payload.get("reasons") or [],
        warnings=payload.get("warnings") or [],
        data_sources_used=payload.get("data_sources_used") or [],
        factors=payload.get("factors") or {},
        generated_for_date=card_date,
    )
    db.add(edge)
    db.flush()
    for name, value in (edge.factors or {}).items():
        db.add(MlbEdgeFactor(edge_id=edge.id, name=name, value=float(value), weight=0.0))
    return edge


def _build_daily_card(db: Session, card_date: str) -> MlbDailyCard:
    edges = list(
        db.scalars(
            select(MlbEdge)
            .where(MlbEdge.generated_for_date == card_date)
            .order_by(desc(MlbEdge.score))
        )
    )
    totals = [edge_to_dict(e) for e in edges if e.edge_type == "game_total" and e.score >= 65][:5]
    props = [edge_to_dict(e) for e in edges if e.edge_type == "pitcher_strikeouts" and e.score >= 65][:5]
    near = [edge_to_dict(e) for e in edges if 55 <= e.score < 65][:5]
    passes = [edge_to_dict(e) for e in edges if e.score < 55][:20]
    summary = {
        "edge_count": len(edges),
        "high_confidence": sum(1 for e in edges if e.confidence == "high"),
        "with_warnings": sum(1 for e in edges if e.warnings),
        "missing_odds": sum(1 for e in edges if any("odds" in w.lower() for w in (e.warnings or []))),
    }
    card = db.scalar(select(MlbDailyCard).where(MlbDailyCard.card_date == card_date))
    values = {
        "top_game_totals": totals,
        "top_pitcher_strikeouts": props,
        "near_misses": near,
        "pass_list": passes,
        "data_quality_summary": summary,
        "updated_at": datetime.utcnow(),
    }
    if card is None:
        card = MlbDailyCard(card_date=card_date, **values)
        db.add(card)
    else:
        for key, value in values.items():
            setattr(card, key, value)
    return card


def _upsert_game(db: Session, game: dict[str, Any]) -> None:
    existing = db.scalar(select(MlbGame).where(MlbGame.game_pk == game["game_pk"]))
    if existing is None:
        db.add(MlbGame(**game))
        return
    for key, value in game.items():
        setattr(existing, key, value)


def _card_to_dict(card: MlbDailyCard | None) -> dict[str, Any] | None:
    if card is None:
        return None
    return {
        "id": card.id,
        "card_date": card.card_date,
        "top_game_totals": card.top_game_totals or [],
        "top_pitcher_strikeouts": card.top_pitcher_strikeouts or [],
        "near_misses": card.near_misses or [],
        "pass_list": card.pass_list or [],
        "data_quality_summary": card.data_quality_summary or {},
        "created_at": card.created_at.isoformat() if card.created_at else None,
        "updated_at": card.updated_at.isoformat() if card.updated_at else None,
    }


def _pitchers(game: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"id": game.get("probable_away_pitcher_id"), "name": game.get("probable_away_pitcher"), "team": game.get("away_team")},
        {"id": game.get("probable_home_pitcher_id"), "name": game.get("probable_home_pitcher"), "team": game.get("home_team")},
    ]


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None
