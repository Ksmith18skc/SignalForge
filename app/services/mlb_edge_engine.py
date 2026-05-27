"""Daily MLB edge engine orchestration."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from sqlalchemy import delete, desc, func, select
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
    PitcherPropOddsSnapshot,
)
from app.providers.mlb_stats_api import MlbStatsApiProvider
from app.providers.odds_api import OddsApiProvider
from app.providers.weather_api import WeatherApiProvider
from app.services import odds_cache
from app.services.mlb_edge import statcast_context
from app.services.mlb_edge_scoring import edge_to_dict
from app.services.mlb_environment import score_environment
from app.services.mlb_odds_analysis import analyze_game_totals
from app.services.mlb_odds_matching import MatchResult
from app.services.mlb_pitcher_k_model import pitcher_k_edges
from app.services.mlb_prop_odds import consensus_for_pitcher, names_match, normalize_pitcher_strikeout_props
from app.services.mlb_totals_model import total_edges

logger = logging.getLogger(__name__)


# Odds-API sport/league params for MLB. Kept here (not config) because they're
# vendor identifiers, not user preferences.
ODDS_MLB_SPORT = "baseball"
ODDS_MLB_LEAGUE = "MLB"


async def run_daily_mlb_edges(db: Session, *, game_date: str | None = None) -> dict[str, Any]:
    from app.services.mlb_performance import arizona_today

    settings = get_settings()
    card_date = game_date or settings.mlb_edge_default_game_date or arizona_today()
    mlb = MlbStatsApiProvider()
    weather = WeatherApiProvider(settings.weather_api_key, settings.weather_api_base_url)
    odds = OddsApiProvider(settings.odds_api_key, settings.odds_api_base_url, settings.odds_bookmakers)

    games = await _load_games(mlb, card_date)
    # Snapshots for OTHER dates are never touched — only this card_date's
    # rows are recomputed. We surface both counts so the UI can show the
    # user "X snapshots written for today, Y preserved from prior dates."
    preserved_snapshots = int(
        db.scalar(
            select(func.count())
            .select_from(MlbEdge)
            .where(MlbEdge.generated_for_date != card_date)
        )
        or 0
    )
    db.execute(delete(MlbEdgeFactor).where(MlbEdgeFactor.edge_id.in_(select(MlbEdge.id).where(MlbEdge.generated_for_date == card_date))))
    db.execute(delete(MlbEdge).where(MlbEdge.generated_for_date == card_date))

    # Refresh the centralized odds cache ONCE for this slate. This is the only
    # place we hit Odds-API live — every other consumer reads from
    # odds_snapshots. Concurrent callers coalesce via odds_cache's lock.
    refresh = await odds_cache.refresh_mlb_odds_cache(
        db, odds, games, game_date=card_date, force=False,
    )
    match_results, unmatched_events = odds_cache.matches_for_games(
        db, games, game_date=card_date,
    )
    matches_by_game: dict[int, MatchResult] = {m.game_pk: m for m in match_results}
    logger.info(
        "MLB scan summary: games=%d odds_events=%d matched=%d unmatched_games=%d "
        "unmatched_events=%d refresh=%s",
        len(games),
        refresh.events_fetched,
        sum(1 for m in match_results if m.matched_event_id),
        sum(1 for m in match_results if not m.matched_event_id),
        len(unmatched_events),
        refresh.reason,
    )

    created_edges: list[MlbEdge] = []
    totals_count = 0
    pitcher_k_count = 0
    for game in games:
        _upsert_game(db, game)
        env = await _environment_for_game(db, weather, game)
        match = matches_by_game.get(int(game["game_pk"]))
        payload = _resolve_cached_payload(db, match)
        totals_analysis = await _odds_for_game(db, game, payload)
        if totals_analysis.get("book_count") and totals_analysis.get("is_valid", True):
            totals_count += 1
        if not totals_analysis.get("is_valid", True):
            logger.warning(
                "Skipping totals edges for game_pk=%s: %s",
                game.get("game_pk"),
                totals_analysis.get("validation_reason"),
            )
        else:
            for edge_payload in total_edges(game=game, odds_analysis=totals_analysis, environment=env):
                created_edges.append(_persist_edge(db, edge_payload, card_date))

        for pitcher in _pitchers(game):
            prop = await _pitcher_prop_for_game(db, game, pitcher, payload)
            if prop.get("book_count") and prop.get("is_valid", True):
                pitcher_k_count += 1
            if not prop.get("is_valid", True):
                logger.info(
                    "Skipping pitcher K edge due to invalid market: game=%s pitcher=%s reason=%s",
                    game.get("game_pk"),
                    pitcher.get("name"),
                    prop.get("validation_reason"),
                )
                continue
            if prop.get("line") is None or not prop.get("rows"):
                logger.info(
                    "Skipping pitcher K edge without valid prop line: game=%s pitcher=%s warnings=%s",
                    game.get("game_pk"),
                    pitcher.get("name"),
                    prop.get("warnings"),
                )
                continue
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
    logger.info(
        "MLB edge run: date=%s games=%d odds_events=%d events_with_totals=%d "
        "events_with_pitcher_props=%d edges=%d odds_calls=%d cache_hits=%d",
        card_date, len(games), refresh.events_fetched, totals_count, pitcher_k_count,
        len(created_edges), refresh.odds_calls,
        odds_cache.get_odds_cache_health().cache_hits,
    )
    return {
        "date": card_date,
        "generated_for_date": card_date,
        "games": len(games),
        "odds_events": refresh.events_fetched,
        "events_with_totals": totals_count,
        "events_with_pitcher_props": pitcher_k_count,
        "edges": len(created_edges),
        "snapshots_written": len(created_edges),
        "snapshots_preserved_from_prior_dates": preserved_snapshots,
        "odds_refresh": refresh.as_dict(),
        "daily_card": _card_to_dict(card),
    }


def latest_daily_card(db: Session, *, card_date: str | None = None) -> dict[str, Any] | None:
    query = select(MlbDailyCard)
    if card_date:
        query = query.where(MlbDailyCard.card_date == card_date)
    card = db.scalar(query.order_by(desc(MlbDailyCard.card_date)).limit(1))
    return _card_to_dict(card) if card else None


def edges_for_date(db: Session, *, card_date: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    from app.services.mlb_performance import arizona_today

    target = card_date or arizona_today()
    edges = list(
        db.scalars(
            select(MlbEdge)
            .where(MlbEdge.generated_for_date == target)
            .where(MlbEdge.is_valid.is_(True))
            .order_by(desc(MlbEdge.score))
            .limit(limit)
        )
    )
    return [edge_to_dict(edge) for edge in edges]


def discord_ready_summary(edge: MlbEdge) -> str:
    if not edge.is_valid:
        return ""
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


async def _odds_for_game(
    db: Session,
    game: dict[str, Any],
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    analysis = analyze_game_totals(payload)
    payload_source = (payload or {}).get("source") or "odds_api"
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
            source=payload_source,
        )
    )
    return analysis


async def _pitcher_prop_for_game(
    db: Session,
    game: dict[str, Any],
    pitcher: dict[str, Any],
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    prop_lines = normalize_pitcher_strikeout_props(payload)
    payload_source = (payload or {}).get("source") or "odds_api"
    for line in prop_lines:
        if pitcher.get("name") and not names_match(line.player_name, pitcher.get("name") or ""):
            continue
        db.add(
            PitcherPropOddsSnapshot(
                game_pk=game["game_pk"],
                sportsbook_event_id=str(payload.get("id")) if payload else None,
                player_name=line.player_name,
                matched_pitcher_name=pitcher.get("name") if pitcher.get("name") else None,
                line=line.line,
                over_price=line.over_price,
                under_price=line.under_price,
                sportsbook=line.sportsbook,
                timestamp=line.timestamp,
                raw=line.raw,
                source=payload_source,
            )
        )
    analysis = consensus_for_pitcher(prop_lines, pitcher.get("name") or "")
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
            source=payload_source,
        )
    )
    return analysis


def _resolve_cached_payload(
    db: Session,
    match: MatchResult | None,
) -> dict[str, Any] | None:
    """Pure cache read — never touches the live API.

    `refresh_mlb_odds_cache` should have populated the cache earlier in
    `run_daily_mlb_edges`; if it didn't (rate-limited + no stale row), we
    return None and the consumer emits a "missing odds" warning.
    """
    if match is None or not match.matched_event_id:
        if match is not None:
            logger.info(
                "Skipping odds for game_pk=%s: %s",
                match.game_pk, match.reason,
            )
        return None
    return odds_cache.get_cached_event_odds(db, match.matched_event_id)


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
        opening_line=payload.get("line"),
        current_line=payload.get("line"),
        recommended_line=payload.get("line"),
        implied_probability_at_entry=_implied_probability(payload.get("best_price")),
        normalized_market_name=payload.get("normalized_market_name"),
        market_scope=payload.get("market_scope"),
        is_valid=payload.get("is_valid", True),
        validation_reason=payload.get("validation_reason"),
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
            .where(MlbEdge.is_valid.is_(True))
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


def _implied_probability(price: Any) -> float | None:
    try:
        decimal_price = float(price)
    except (TypeError, ValueError):
        return None
    if decimal_price <= 1:
        return None
    return round(1 / decimal_price, 4)
