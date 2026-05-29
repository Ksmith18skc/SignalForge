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
from app.services.mlb_edge_scoring import (
    PITCHER_K_WEIGHTS,
    TOTAL_WEIGHTS,
    classify_edge,
    edge_to_dict,
    watchlist_sort_key,
)

# Maps an edge_type to the weight map its score was built from, so persisted
# MlbEdgeFactor rows carry the real weight (not a 0.0 placeholder).
_WEIGHTS_FOR_EDGE_TYPE = {
    "game_total": TOTAL_WEIGHTS,
    "pitcher_strikeouts": PITCHER_K_WEIGHTS,
}
from app.services.mlb_environment import score_environment
from app.services.mlb_odds_analysis import analyze_game_totals
from app.services.mlb_odds_matching import MatchResult
from app.services.mlb_pitcher_k_model import pitcher_k_edges
from app.services.mlb_prop_odds import consensus_for_pitcher, names_match, normalize_pitcher_strikeout_props
from app.services.mlb_totals_model import total_edges

logger = logging.getLogger(__name__)


# Odds-API sport/league params for MLB. Kept here as compatibility exports for
# debug routes/tests; the cache owns the vendor request shape.
ODDS_MLB_SPORT = odds_cache.ODDS_MLB_SPORT
ODDS_MLB_LEAGUE = odds_cache.ODDS_MLB_LEAGUE


async def run_daily_mlb_edges(
    db: Session,
    *,
    game_date: str | None = None,
    force_stale: bool = False,
) -> dict[str, Any]:
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
    # Refresh the centralized odds cache ONCE for this slate. This is the only
    # place we hit Odds-API live — every other consumer reads from
    # odds_snapshots. Concurrent callers coalesce via odds_cache's lock.
    refresh = await odds_cache.refresh_mlb_odds_cache(
        db, odds, games, game_date=card_date, force=False,
    )
    match_results, unmatched_events = odds_cache.matches_for_games(
        db, games, game_date=card_date, fallback_stale=force_stale,
    )
    matches_by_game: dict[int, MatchResult] = {m.game_pk: m for m in match_results}
    diagnostics = _initial_diagnostics(db, games, match_results, card_date=card_date)
    if (
        games
        and diagnostics["fresh_odds_snapshots_found"] == 0
        and (diagnostics["markets_matched"] > 0 or not diagnostics["events_list_fresh"])
        and not force_stale
    ):
        reason = "Odds cache stale; refresh required before edge scan."
        logger.warning("%s diagnostics=%s refresh=%s", reason, diagnostics, refresh.as_dict())
        return {
            "date": card_date,
            "generated_for_date": card_date,
            "status": "blocked",
            "reason": reason,
            "games": len(games),
            "odds_events": refresh.events_fetched,
            "events_with_totals": 0,
            "events_with_pitcher_props": 0,
            "edges": 0,
            "snapshots_written": 0,
            "snapshots_preserved_from_prior_dates": preserved_snapshots,
            "odds_refresh": refresh.as_dict(),
            "diagnostics": diagnostics,
            "daily_card": latest_daily_card(db, card_date=card_date),
        }

    db.execute(delete(MlbEdgeFactor).where(MlbEdgeFactor.edge_id.in_(select(MlbEdge.id).where(MlbEdge.generated_for_date == card_date))))
    db.execute(delete(MlbEdge).where(MlbEdge.generated_for_date == card_date))
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

    # Rolling per-side performance drives the side-underperformance
    # penalty applied inside _total_edge. Looked up once per scan rather
    # than per edge — the value is window-level, not per-game.
    from app.services.mlb_performance import recent_side_performance

    rolling_side_perf = recent_side_performance(db, today=card_date, edge_type="game_total")
    side_penalty_points = {
        side: float((rolling_side_perf.get("sides") or {}).get(side, {}).get("penalty_points") or 0.0)
        for side in ("over", "under")
    }

    # Pitcher-K diagnostics — populated as the engine walks each game so
    # the dashboard can show stage-by-stage funnel collapse instead of
    # "No qualifying edges."
    from app.services.ballparkpal_integration import k_projection_index
    from app.services.mlb_pitcher_k_diagnostics import PitcherKDiagnostics
    from app.services.mlb_pitcher_k_fallback import (
        FALLBACK_SOURCE,
        build_fallback_prop_analysis,
        is_fallback_payload,
        name_matches_loose,
        normalize_pitcher_name,
    )

    k_diag = PitcherKDiagnostics()
    bpp_strikeout_rows = k_projection_index(db, slate_date=card_date) or {}
    k_diag.strikeout_projections_loaded = len(bpp_strikeout_rows)

    created_edges: list[MlbEdge] = []
    totals_count = 0
    pitcher_k_count = 0
    skipped_no_threshold = 0
    for game in games:
        _upsert_game(db, game)
        env = await _environment_for_game(db, weather, game)
        match = matches_by_game.get(int(game["game_pk"]))
        if match is None or not match.matched_event_id:
            diagnostics["skipped_missing_odds"] += 1
            continue
        state = odds_cache.event_odds_cache_state(db, match.matched_event_id)
        if not state.get("fresh") and not force_stale:
            diagnostics["skipped_stale_odds"] += 1
            continue
        payload = _resolve_cached_payload(db, match, fallback_stale=force_stale)
        if payload is None:
            diagnostics["skipped_missing_odds"] += 1
            continue
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
            for edge_payload in total_edges(
                game=game,
                odds_analysis=totals_analysis,
                environment=env,
                side_penalty_points=side_penalty_points,
            ):
                _apply_wallet_flow(db, edge_payload, game, card_date)
                if (edge_payload.get("score") or 0) < 65:
                    skipped_no_threshold += 1
                created_edges.append(_persist_edge(db, edge_payload, card_date))

        game_pitchers = _pitchers(game)
        if game_pitchers:
            k_diag.games_with_pitchers += 1
        for pitcher in game_pitchers:
            pitcher_name = pitcher.get("name") or ""
            prop = await _pitcher_prop_for_game(db, game, pitcher, payload)
            sportsbook_rows = prop.get("rows") if prop else []
            # The sportsbook side of the funnel — counted whether or not
            # the prop ended up usable. We need to see "we saw 12 K-prop
            # rows but matched 0 to our pitcher" as distinct from "the
            # cache was empty in the first place."
            k_diag.sportsbook_pitcher_k_props_loaded += len(sportsbook_rows or [])

            sportsbook_usable = (
                prop.get("is_valid", True)
                and prop.get("line") is not None
                and bool(sportsbook_rows)
            )
            if sportsbook_usable:
                k_diag.pitcher_names_matched_sportsbook += 1
            elif sportsbook_rows is not None:
                k_diag.pitcher_names_unmatched_sportsbook += 1
                k_diag.add_unmatched_example(
                    pitcher_name=pitcher_name, source="sportsbook",
                    reason=(prop.get("validation_reason") or "no matching prop line"),
                )

            # BallparkPal fallback. Either the sportsbook prop didn't
            # cover this pitcher, or the prop is unusable. Look up the
            # pitcher in the strikeout cache and synthesize the same
            # prop_analysis shape so pitcher_k_edges still runs.
            if not sportsbook_usable and bpp_strikeout_rows:
                bpp_row = _find_bpp_row_for_pitcher(
                    pitcher_name, bpp_strikeout_rows,
                )
                if bpp_row is not None:
                    fallback = build_fallback_prop_analysis(
                        pitcher_name=pitcher_name, bpp_row=bpp_row,
                    )
                    if fallback is not None:
                        prop = fallback
                        sportsbook_usable = True
                        k_diag.pitcher_names_matched_ballparkpal += 1
                        k_diag.candidates_built_from_ballparkpal_fallback += 1
                        k_diag.add_fallback_example({
                            "pitcher_name": pitcher_name,
                            "projected_k": fallback.get("ballparkpal_projected_k"),
                            "line": fallback.get("line"),
                            "over_price": fallback.get("best_over_price"),
                        })
                else:
                    k_diag.pitcher_names_unmatched_ballparkpal += 1
                    k_diag.add_unmatched_example(
                        pitcher_name=pitcher_name, source="ballparkpal",
                        reason="pitcher not in Strikeout Center cache",
                    )

            if prop.get("book_count") and prop.get("is_valid", True):
                pitcher_k_count += 1
            if not prop.get("is_valid", True):
                logger.info(
                    "Skipping pitcher K edge due to invalid market: game=%s pitcher=%s reason=%s",
                    game.get("game_pk"),
                    pitcher_name,
                    prop.get("validation_reason"),
                )
                k_diag.candidates_rejected_missing_odds += 1
                continue
            if prop.get("line") is None or not prop.get("rows"):
                logger.info(
                    "Skipping pitcher K edge without valid prop line: game=%s pitcher=%s warnings=%s",
                    game.get("game_pk"),
                    pitcher_name,
                    prop.get("warnings"),
                )
                k_diag.candidates_rejected_missing_odds += 1
                continue

            k_diag.lines_matched_against_props += 1
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
                k_diag.candidates_built_total += 1
                if is_fallback_payload(prop):
                    # Tag the persisted edge so the dashboard card can
                    # render "BallparkPal fallback odds" instead of the
                    # legacy sportsbook label. Stored inside ``factors``
                    # because that field already round-trips JSON-safely
                    # through ``edge_to_dict`` — no schema migration.
                    factors_payload = edge_payload.setdefault("factors", {})
                    factors_payload["odds_source"] = FALLBACK_SOURCE
                    factors_payload["ballparkpal_projected_k"] = prop.get(
                        "ballparkpal_projected_k"
                    )
                    edge_payload["odds_source"] = FALLBACK_SOURCE
                    edge_payload["ballparkpal_projected_k"] = prop.get(
                        "ballparkpal_projected_k"
                    )
                else:
                    k_diag.candidates_built_from_sportsbook += 1
                if (edge_payload.get("score") or 0) < 65:
                    skipped_no_threshold += 1
                created_edges.append(_persist_edge(db, edge_payload, card_date))

    card = _build_daily_card(db, card_date, pitcher_k_diagnostics=k_diag)
    diagnostics["prop_snapshots_found"] = pitcher_k_count
    diagnostics["edges_generated"] = len(created_edges)
    diagnostics["skipped_no_threshold"] = skipped_no_threshold
    diagnostics["pitcher_k"] = k_diag.to_dict()
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
        "diagnostics": diagnostics,
        "daily_card": _card_to_dict(card),
    }


def latest_daily_card(db: Session, *, card_date: str | None = None) -> dict[str, Any] | None:
    query = select(MlbDailyCard)
    if card_date:
        query = query.where(MlbDailyCard.card_date == card_date)
    card = db.scalar(query.order_by(desc(MlbDailyCard.card_date)).limit(1))
    return _card_to_dict(card) if card else None


def _initial_diagnostics(
    db: Session,
    games: list[dict[str, Any]],
    match_results: list[MatchResult],
    *,
    card_date: str,
) -> dict[str, Any]:
    matched = [m for m in match_results if m.matched_event_id]
    events_state = odds_cache.events_list_cache_state(db, game_date=card_date)
    fresh_odds = 0
    stale_odds = 0
    missing_odds = 0
    for match in matched:
        state = odds_cache.event_odds_cache_state(db, match.matched_event_id or "")
        if state["state"] == "fresh":
            fresh_odds += 1
        elif state["state"] == "stale":
            stale_odds += 1
        else:
            missing_odds += 1
    return {
        "mlb_games_today": len(games),
        "events_list_fresh": bool(events_state.get("fresh")),
        "events_list_age_minutes": events_state.get("age_minutes"),
        "odds_snapshots_found": fresh_odds + stale_odds,
        "fresh_odds_snapshots_found": fresh_odds,
        "prop_snapshots_found": 0,
        "markets_matched": len(matched),
        "edges_generated": 0,
        "skipped_missing_odds": missing_odds + max(len(games) - len(matched), 0),
        "skipped_stale_odds": stale_odds,
        "skipped_no_threshold": 0,
    }


def edges_for_date(db: Session, *, card_date: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    from app.services.mlb_performance import arizona_today

    target = card_date or arizona_today()
    edges = list(
        db.scalars(
            select(MlbEdge)
            .where(MlbEdge.generated_for_date == target)
            .where(MlbEdge.is_valid.is_(True))
            .order_by(desc(MlbEdge.score))
        )
    )
    payloads = [edge_to_dict(edge) for edge in edges]
    payloads.sort(key=watchlist_sort_key, reverse=True)
    return payloads[:limit]


def discord_ready_summary(edge: MlbEdge) -> str:
    if not edge.is_valid:
        return ""
    reasons = "\n".join(f"- {reason}" for reason in (edge.reasons or [])[:3])
    warnings = "\n".join(f"- {warning}" for warning in (edge.warnings or [])[:3])
    warning_block = f"\nWarnings:\n{warnings}" if warnings else ""
    prediction = edge.prediction_score if edge.prediction_score is not None else edge.score
    execution = f"{edge.execution_score:.0f}" if edge.execution_score is not None else "N/A"
    return (
        f"MLB EDGE - {edge.edge_type.replace('_', ' ').title()}\n\n"
        f"Market: {edge.market}\n"
        f"Best line: {edge.side.title()} {edge.line if edge.line is not None else '?'} "
        f"at {edge.best_price if edge.best_price is not None else '?'} on {edge.best_book or 'N/A'}\n"
        f"Prediction score: {prediction:.0f}\n"
        f"Execution score: {execution}\n"
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
    *,
    fallback_stale: bool = False,
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
    return odds_cache.get_cached_event_odds(
        db,
        match.matched_event_id,
        fallback_stale=fallback_stale,
    )


def _apply_wallet_flow(
    db: Session,
    payload: dict[str, Any],
    game: dict[str, Any],
    card_date: str,
) -> None:
    """Join tracked-wallet activity to a game-level edge.

    Attaches the ``wallet_context`` payload, then recomputes
    ``prediction_score`` with the now-known wallet_alignment factor.
    The legacy ``score`` field still gets a bounded ``confidence_adjustment``
    bump so the Pricing Edge tab keeps working unchanged.
    """
    from app.services.mlb_totals_model import recompute_with_wallet_alignment
    from app.services.wallet_flow import build_wallet_context

    try:
        context = build_wallet_context(
            db,
            edge=payload,
            home_team=game.get("home_team"),
            away_team=game.get("away_team"),
            card_date=card_date,
        )
    except Exception as exc:  # noqa: BLE001 — enrichment is best-effort
        logger.warning("wallet-flow enrichment failed for game_pk=%s: %s", game.get("game_pk"), exc)
        return

    payload["wallet_context"] = context

    # 1) Refresh the prediction axis with the real wallet_alignment. This
    # also re-runs penalties and re-classifies on prediction_score.
    if payload.get("edge_type") == "game_total":
        recompute_with_wallet_alignment(payload, wallet_context=context)

    # 2) Preserve the legacy adjustment pathway so the legacy_score /
    # Pricing Edge tab still reflect the wallet bump. The new prediction
    # score is wallet-aware via the factor itself, not via this hack.
    adjustment = float(context.get("confidence_adjustment") or 0.0)
    if not adjustment:
        return
    base_legacy = float(payload.get("legacy_score") or payload.get("score") or 0.0)
    new_legacy = round(max(0.0, min(95.0, base_legacy + adjustment)), 2)
    payload["legacy_score"] = new_legacy
    payload["score"] = new_legacy


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
        wallet_context=payload.get("wallet_context") or None,
        score_contributions=payload.get("score_contributions") or None,
        generated_for_date=card_date,
        opening_line=payload.get("line"),
        current_line=payload.get("line"),
        recommended_line=payload.get("line"),
        implied_probability_at_entry=_implied_probability(payload.get("best_price")),
        normalized_market_name=payload.get("normalized_market_name"),
        market_scope=payload.get("market_scope"),
        is_valid=payload.get("is_valid", True),
        validation_reason=payload.get("validation_reason"),
        # Persist the at-scan-time projection so the calibration report
        # has a real projected-vs-actual residual to score against after
        # grading. Pitcher-K edges don't emit a projection yet so the
        # field stays NULL there.
        model_projected_total=payload.get("model_projected_total"),
        # Dual-score refactor columns. Pitcher-K edges leave the
        # prediction/execution pair NULL until that side of the model
        # gets its own factor set wired up.
        prediction_score=payload.get("prediction_score"),
        execution_score=payload.get("execution_score"),
        legacy_score=payload.get("legacy_score") or payload.get("score"),
        prediction_breakdown=payload.get("prediction_breakdown"),
        execution_breakdown=payload.get("execution_breakdown"),
        cheap_price_trap=payload.get("cheap_price_trap"),
    )
    db.add(edge)
    db.flush()
    contributions = payload.get("score_contributions") or {}
    weights = _WEIGHTS_FOR_EDGE_TYPE.get(payload["edge_type"], {})
    for name, value in (edge.factors or {}).items():
        db.add(
            MlbEdgeFactor(
                edge_id=edge.id,
                name=name,
                value=float(value),
                weight=float(weights.get(name, 0.0)),
                note=(f"{contributions[name]:+.2f} pts" if name in contributions else None),
            )
        )
    return edge


def _build_daily_card(
    db: Session,
    card_date: str,
    *,
    pitcher_k_diagnostics: Any | None = None,
) -> MlbDailyCard:
    from app.services.mlb_pitcher_k_diagnostics import (
        K_EDGE_WATCHLIST_FLOOR,
        classify_k_edge_magnitude,
    )
    from app.services.mlb_pitcher_k_fallback import k_edge_from_fallback

    edges = list(
        db.scalars(
            select(MlbEdge)
            .where(MlbEdge.generated_for_date == card_date)
            .where(MlbEdge.is_valid.is_(True))
            .order_by(desc(MlbEdge.score))
        )
    )
    payloads = [edge_to_dict(e) for e in edges]

    def _decision_score(payload: dict[str, Any]) -> float:
        value = payload.get("prediction_score")
        if value is None:
            value = payload.get("score")
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _pitcher_k_edge_magnitude(payload: dict[str, Any]) -> float:
        # Read projected_k either from the top-level marker (during the
        # same scan) or from the persisted ``factors`` dict (when the
        # daily card is regenerated from edges already in the DB).
        factors_payload = payload.get("factors") or {}
        projected = (
            payload.get("ballparkpal_projected_k")
            or factors_payload.get("ballparkpal_projected_k")
            or factors_payload.get("projected_k")
        )
        line = payload.get("line")
        if projected is None or line is None:
            return 0.0
        try:
            return abs(float(projected) - float(line))
        except (TypeError, ValueError):
            return 0.0

    totals_all = sorted(
        [p for p in payloads if p.get("edge_type") == "game_total"],
        key=watchlist_sort_key,
        reverse=True,
    )
    # Pitcher-K edges use K-edge magnitude (|projected_k − line|) for
    # banding, NOT the composite score threshold. K markets move in
    # tighter ranges than game totals, so a 0.5-K edge is meaningful
    # even when the score sits below 65. Composite score is preserved
    # as the SECOND sort key so high-conviction cards still float up.
    props_all = sorted(
        [p for p in payloads if p.get("edge_type") == "pitcher_strikeouts"],
        key=lambda p: (
            _pitcher_k_edge_magnitude(p),
            float(p.get("score") or 0.0),
        ),
        reverse=True,
    )
    totals = [p for p in totals_all if _decision_score(p) >= 65][:5]
    props = [
        p for p in props_all
        if _pitcher_k_edge_magnitude(p) >= K_EDGE_WATCHLIST_FLOOR
    ][:5]
    near = [
        p for p in sorted(payloads, key=watchlist_sort_key, reverse=True)
        if 55 <= _decision_score(p) < 65
    ][:5]
    passes = [
        p for p in sorted(payloads, key=watchlist_sort_key, reverse=True)
        if _decision_score(p) < 55
    ][:20]
    summary = {
        "edge_count": len(edges),
        "high_confidence": sum(1 for e in edges if e.confidence == "high"),
        "with_warnings": sum(1 for e in edges if e.warnings),
        "missing_odds": sum(1 for e in edges if any("odds" in w.lower() for w in (e.warnings or []))),
    }
    if pitcher_k_diagnostics is not None:
        # Tally final placements (cards / watchlist) AFTER selection so
        # the operator sees the real funnel — not what would have been
        # promoted if the band were wider.
        for payload in props_all:
            band = classify_k_edge_magnitude(
                _pitcher_k_edge_magnitude(payload)
                * (1 if payload.get("side") == "over" else -1)
            )
            if band == "strong":
                pitcher_k_diagnostics.candidates_promoted_strong += 1
            elif band == "candidate":
                pitcher_k_diagnostics.candidates_promoted_candidate += 1
            elif band == "watchlist":
                pitcher_k_diagnostics.candidates_promoted_watchlist += 1
            else:
                pitcher_k_diagnostics.candidates_rejected_by_threshold += 1
        pitcher_k_diagnostics.cards_rendered = len(props)
        # Persist counters on the card so the dashboard can render them
        # without re-running the engine.
        summary["pitcher_k"] = pitcher_k_diagnostics.to_dict()
        summary["pitcher_k_empty_state_message"] = (
            pitcher_k_diagnostics.empty_state_message()
        )
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


def _find_bpp_row_for_pitcher(
    pitcher_name: str | None,
    bpp_index: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Loose-match a pitcher to a BallparkPal Strikeout Center row.

    The BPP cache index is keyed by ``"".join(c for c in name.lower() if
    c.isalnum())``, which already drops spaces / punctuation. We probe
    that exact form first, then fall back to the looser
    ``name_matches_loose`` matcher (first-initial + last name) so
    ``"J. Ryan"`` finds ``"Joe Ryan"`` without us re-indexing.
    """
    from app.services.mlb_pitcher_k_fallback import (
        name_matches_loose,
        normalize_pitcher_name,
    )

    if not pitcher_name or not bpp_index:
        return None
    direct_key = "".join(
        ch for ch in str(pitcher_name).lower() if ch.isalnum()
    )
    if direct_key and direct_key in bpp_index:
        return bpp_index[direct_key]
    # Fallback: walk the index and use loose matching. Cheap — the BPP
    # strikeout cache is at most ~60 rows on a full slate.
    for indexed_name, row in bpp_index.items():
        candidate = row.get("pitcher") or indexed_name
        if name_matches_loose(candidate, pitcher_name):
            return row
    return None


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
