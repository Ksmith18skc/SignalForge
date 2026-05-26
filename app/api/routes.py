"""HTTP API routes."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from urllib.parse import unquote, urlparse

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import delete, desc, func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.models import Alert, Market, Position, Signal, Trade, Trader
from app.providers.falcon import FalconProvider, get_falcon_health
from app.providers.mlb_stats_api import MlbStatsApiError, MlbStatsApiProvider
from app.providers.odds_api import OddsApiError, OddsApiProvider
from app.providers.pybaseball_provider import PyBaseballError, PyBaseballProvider
from app.providers.weather_api import WeatherApiError, WeatherApiProvider
from app.schemas import (
    AlertOut,
    DashboardSummary,
    MarketOut,
    ScanResult,
    SignalOut,
    TraderCreate,
    TraderOut,
    WatchlistHealth,
)
from app.services.scanner import run_scan_once
from app.services.alerts import DISCORD_TIERS, evaluate_alert_decision

router = APIRouter()

_SLUG_DATE_RE = re.compile(r"(20\d{2})-(\d{2})-(\d{2})")


# ---------------------------- health ----------------------------------------


@router.get("/health")
def health() -> dict[str, object]:
    s = get_settings()
    falcon_health = get_falcon_health()
    # "Configured" means a key is set. "Healthy" means calls are actually
    # succeeding — these can disagree (wrong base URL, expired key, etc.).
    return {
        "status": "ok",
        "app": s.app_name,
        "environment": s.environment,
        "default_copy_mode": s.default_copy_mode,
        "auto_trading_enabled": s.enable_auto_trading,
        "providers": {
            "falcon": {
                "configured": s.has_falcon_credentials(),
                "healthy": falcon_health.healthy,
                "calls": falcon_health.calls,
                "successes": falcon_health.successes,
                "success_rate": round(falcon_health.success_rate, 3),
                "last_status_code": falcon_health.last_status_code,
                "last_error": falcon_health.last_error,
                "last_endpoint": falcon_health.last_endpoint,
                "last_scan_at": (
                    falcon_health.last_scan_at.isoformat()
                    if falcon_health.last_scan_at
                    else None
                ),
                "last_scan_calls": falcon_health.last_scan_calls,
                "last_scan_successes": falcon_health.last_scan_successes,
                "base_url": falcon_health.base_url or s.falcon_base_url,
            },
            "polymarket": {"configured": s.has_polymarket_credentials()},
            "kalshi": {"configured": s.has_kalshi_credentials()},
            "odds_api": {
                "configured": s.has_odds_api_credentials(),
                "base_url": s.odds_api_base_url,
                "bookmakers": [b.strip() for b in s.odds_bookmakers.split(",") if b.strip()],
            },
            "mlb_stats_api": {
                "configured": s.mlb_stats_enabled,
                "requires_api_key": False,
            },
            "weather_api": {
                "configured": s.has_weather_api_credentials(),
                "base_url": s.weather_api_base_url,
            },
            "pybaseball": {
                "configured": s.pybaseball_enabled,
                "requires_api_key": False,
            },
        },
        "alerts": {
            "console": {"configured": True},
            "discord": {"configured": bool(s.discord_webhook_url)},
            "telegram": {
                "configured": bool(s.telegram_bot_token and s.telegram_chat_id),
                "has_bot_token": bool(s.telegram_bot_token),
                "has_chat_id": bool(s.telegram_chat_id),
            },
            "email": {
                "configured": bool(
                    s.alert_email_to
                    and s.smtp_host
                    and (s.alert_email_from or s.smtp_username)
                ),
                "has_recipient": bool(s.alert_email_to),
                "has_smtp_host": bool(s.smtp_host),
                "has_from_address": bool(s.alert_email_from or s.smtp_username),
            },
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


# ---------------------------- traders ---------------------------------------


@router.get("/traders", response_model=list[TraderOut])
def list_traders(db: Session = Depends(get_db)) -> list[Trader]:
    return list(db.scalars(select(Trader).order_by(Trader.trust_score.desc())))


@router.post("/traders", response_model=TraderOut, status_code=status.HTTP_201_CREATED)
def create_trader(payload: TraderCreate, db: Session = Depends(get_db)) -> Trader:
    existing = db.scalar(select(Trader).where(Trader.nickname == payload.nickname))
    if existing:
        raise HTTPException(status_code=409, detail=f"trader '{payload.nickname}' already exists")

    if payload.wallet_address:
        existing_wallet = db.scalar(
            select(Trader).where(Trader.wallet_address == payload.wallet_address)
        )
        if existing_wallet:
            raise HTTPException(
                status_code=409,
                detail=f"wallet '{payload.wallet_address}' already exists",
            )

    trader = Trader(**payload.model_dump())
    # MVP guard: never allow live copy_mode through the API.
    if trader.copy_mode == "live" and not get_settings().enable_auto_trading:
        trader.copy_mode = "alert_only"

    db.add(trader)
    db.commit()
    db.refresh(trader)
    return trader


@router.delete(
    "/traders/{trader_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    response_class=Response,
)
def delete_trader(trader_id: int, db: Session = Depends(get_db)) -> None:
    trader = db.get(Trader, trader_id)
    if trader is None:
        raise HTTPException(status_code=404, detail=f"trader {trader_id} not found")

    signal_ids = list(
        db.scalars(select(Signal.id).where(Signal.trader_id == trader_id))
    )
    if signal_ids:
        db.execute(delete(Alert).where(Alert.signal_id.in_(signal_ids)))
        db.execute(delete(Signal).where(Signal.id.in_(signal_ids)))

    db.execute(delete(Position).where(Position.trader_id == trader_id))
    db.execute(delete(Trade).where(Trade.trader_id == trader_id))
    db.delete(trader)
    db.commit()
    return None


# ---------------------------- markets ---------------------------------------


@router.get("/markets", response_model=list[MarketOut])
def list_markets(db: Session = Depends(get_db), limit: int = 50) -> list[Market]:
    return list(
        db.scalars(
            select(Market).where(Market.is_active.is_(True)).limit(limit)
        )
    )


# ---------------------------- signals ---------------------------------------


def _enrich_signal(signal: Signal) -> SignalOut:
    base = SignalOut.model_validate(signal)
    if signal.trader:
        base.wallet = signal.trader.wallet_address
        base.trader_nickname = signal.trader.nickname
    if signal.market:
        base.market_title = signal.market.title
        base.market_slug = signal.market.slug
        base.market_platform = signal.market.platform
    return base


@router.get("/signals", response_model=list[SignalOut])
def list_signals(db: Session = Depends(get_db), limit: int = 50) -> list[SignalOut]:
    signals = list(
        db.scalars(select(Signal).order_by(desc(Signal.created_at)).limit(limit))
    )
    return [_enrich_signal(s) for s in signals]


# ---------------------------- alerts ----------------------------------------


@router.get("/alerts", response_model=list[AlertOut])
def list_alerts(db: Session = Depends(get_db), limit: int = 50) -> list[Alert]:
    return list(db.scalars(select(Alert).order_by(desc(Alert.created_at)).limit(limit)))


# ---------------------------- sportsbook odds -------------------------------


def _odds_provider() -> OddsApiProvider:
    s = get_settings()
    return OddsApiProvider(s.odds_api_key, s.odds_api_base_url, s.odds_bookmakers)


def _odds_error(exc: Exception) -> HTTPException:
    if isinstance(exc, OddsApiError):
        return HTTPException(status_code=400, detail=str(exc))
    if hasattr(exc, "response"):
        response = exc.response  # type: ignore[attr-defined]
        return HTTPException(
            status_code=response.status_code,
            detail=response.text[:500],
        )
    return HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}")


@router.get("/odds/sports")
async def odds_sports() -> list[dict[str, object]]:
    try:
        return await _odds_provider().sports()
    except Exception as exc:  # noqa: BLE001
        raise _odds_error(exc) from exc


@router.get("/odds/bookmakers")
async def odds_bookmakers() -> list[dict[str, object]]:
    try:
        return await _odds_provider().bookmakers()
    except Exception as exc:  # noqa: BLE001
        raise _odds_error(exc) from exc


@router.get("/odds/bookmakers/selected")
async def odds_selected_bookmakers() -> object:
    try:
        return await _odds_provider().selected_bookmakers()
    except Exception as exc:  # noqa: BLE001
        raise _odds_error(exc) from exc


@router.get("/odds/leagues")
async def odds_leagues(sport: str) -> list[dict[str, object]]:
    try:
        return await _odds_provider().leagues(sport)
    except Exception as exc:  # noqa: BLE001
        raise _odds_error(exc) from exc


@router.get("/odds/events")
async def odds_events(
    sport: str,
    league: str | None = None,
    status: str | None = "pending,live",
    bookmaker: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict[str, object]]:
    try:
        return await _odds_provider().events(
            sport,
            league=league,
            status=status,
            bookmaker=bookmaker,
            date_from=date_from,
            date_to=date_to,
        )
    except Exception as exc:  # noqa: BLE001
        raise _odds_error(exc) from exc


@router.get("/odds/events/live")
async def odds_live_events(sport: str | None = None) -> list[dict[str, object]]:
    try:
        return await _odds_provider().live_events(sport)
    except Exception as exc:  # noqa: BLE001
        raise _odds_error(exc) from exc


@router.get("/odds/events/search")
async def odds_search_events(query: str) -> list[dict[str, object]]:
    if len(query.strip()) < 3:
        raise HTTPException(status_code=400, detail="query must be at least 3 characters")
    try:
        return await _odds_provider().search_events(query)
    except Exception as exc:  # noqa: BLE001
        raise _odds_error(exc) from exc


@router.get("/odds/events/{event_id}")
async def odds_event(event_id: str) -> dict[str, object]:
    try:
        return await _odds_provider().event(event_id)
    except Exception as exc:  # noqa: BLE001
        raise _odds_error(exc) from exc


@router.get("/odds/compare")
async def odds_compare(
    event_id: str,
    bookmakers: str | None = None,
    market: str | None = None,
    side: str | None = None,
    line: float | None = None,
) -> dict[str, object]:
    """Compare sportsbook lines across books for one event."""
    try:
        return await _odds_provider().compare_lines(
            event_id,
            bookmakers=bookmakers,
            market=market,
            side=side,
            line=line,
        )
    except Exception as exc:  # noqa: BLE001
        raise _odds_error(exc) from exc


@router.get("/odds/movements")
async def odds_movements(
    event_id: str,
    bookmaker: str,
    market: str,
    market_line: float | None = None,
) -> dict[str, object]:
    try:
        return await _odds_provider().odds_movements(event_id, bookmaker, market, market_line)
    except Exception as exc:  # noqa: BLE001
        raise _odds_error(exc) from exc


# ---------------------------- MLB StatsAPI ----------------------------------


def _mlb_provider() -> MlbStatsApiProvider:
    if not get_settings().mlb_stats_enabled:
        raise HTTPException(status_code=503, detail="MLB StatsAPI integration is disabled")
    return MlbStatsApiProvider()


def _mlb_error(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, MlbStatsApiError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}")


@router.get("/mlb/schedule")
async def mlb_schedule(
    game_date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    team_id: int | None = None,
    season: int | None = None,
) -> dict[str, object]:
    """Return MLB schedule data, including probable pitchers when available."""
    try:
        return await _mlb_provider().schedule(
            game_date=game_date,
            start_date=start_date,
            end_date=end_date,
            team_id=team_id,
            season=season,
        )
    except Exception as exc:  # noqa: BLE001
        raise _mlb_error(exc) from exc


@router.get("/mlb/games/live")
async def mlb_live_scores(game_date: str | None = None) -> dict[str, object]:
    """Return compact MLB game status and live score summaries."""
    try:
        return await _mlb_provider().live_scores(game_date=game_date)
    except Exception as exc:  # noqa: BLE001
        raise _mlb_error(exc) from exc


@router.get("/mlb/games/{game_pk}")
async def mlb_game(game_pk: int) -> dict[str, object]:
    try:
        return await _mlb_provider().game(game_pk)
    except Exception as exc:  # noqa: BLE001
        raise _mlb_error(exc) from exc


@router.get("/mlb/games/{game_pk}/linescore")
async def mlb_linescore(game_pk: int) -> dict[str, object]:
    try:
        return await _mlb_provider().linescore(game_pk)
    except Exception as exc:  # noqa: BLE001
        raise _mlb_error(exc) from exc


@router.get("/mlb/games/{game_pk}/boxscore")
async def mlb_boxscore(game_pk: int) -> dict[str, object]:
    try:
        return await _mlb_provider().boxscore(game_pk)
    except Exception as exc:  # noqa: BLE001
        raise _mlb_error(exc) from exc


@router.get("/mlb/games/{game_pk}/lineups")
async def mlb_lineups(game_pk: int) -> dict[str, object]:
    try:
        return await _mlb_provider().lineups(game_pk)
    except Exception as exc:  # noqa: BLE001
        raise _mlb_error(exc) from exc


@router.get("/mlb/probable-pitchers")
async def mlb_probable_pitchers(game_date: str | None = None) -> dict[str, object]:
    try:
        return await _mlb_provider().probable_pitchers(game_date=game_date)
    except Exception as exc:  # noqa: BLE001
        raise _mlb_error(exc) from exc


@router.get("/mlb/teams")
async def mlb_teams(season: int | None = None) -> dict[str, object]:
    try:
        return await _mlb_provider().teams(season=season)
    except Exception as exc:  # noqa: BLE001
        raise _mlb_error(exc) from exc


@router.get("/mlb/teams/{team_id}/stats")
async def mlb_team_stats(
    team_id: int,
    season: int,
    group: str = "hitting",
    stats: str = "season",
    game_type: str = "R",
) -> dict[str, object]:
    try:
        return await _mlb_provider().team_stats(
            team_id,
            season=season,
            group=group,
            stats=stats,
            game_type=game_type,
        )
    except Exception as exc:  # noqa: BLE001
        raise _mlb_error(exc) from exc


@router.get("/mlb/players/{person_id}/stats")
async def mlb_player_stats(
    person_id: int,
    season: int,
    group: str = "hitting",
    stats: str = "season",
    game_type: str = "R",
) -> dict[str, object]:
    try:
        return await _mlb_provider().player_stats(
            person_id,
            season=season,
            group=group,
            stats=stats,
            game_type=game_type,
        )
    except Exception as exc:  # noqa: BLE001
        raise _mlb_error(exc) from exc


@router.get("/mlb/history")
async def mlb_history(
    start_date: str,
    end_date: str,
    team_id: int | None = None,
) -> dict[str, object]:
    try:
        return await _mlb_provider().historical_games(
            start_date=start_date,
            end_date=end_date,
            team_id=team_id,
        )
    except Exception as exc:  # noqa: BLE001
        raise _mlb_error(exc) from exc


# ---------------------------- weather ---------------------------------------


def _weather_provider() -> WeatherApiProvider:
    s = get_settings()
    return WeatherApiProvider(s.weather_api_key, s.weather_api_base_url)


def _weather_error(exc: Exception) -> HTTPException:
    if isinstance(exc, WeatherApiError):
        return HTTPException(status_code=400, detail=str(exc))
    if hasattr(exc, "response"):
        response = exc.response  # type: ignore[attr-defined]
        return HTTPException(status_code=response.status_code, detail=response.text[:500])
    return HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}")


@router.get("/weather/current")
async def weather_current(q: str) -> dict[str, object]:
    try:
        return await _weather_provider().current(q)
    except Exception as exc:  # noqa: BLE001
        raise _weather_error(exc) from exc


@router.get("/weather/forecast")
async def weather_forecast(
    q: str,
    days: int = 1,
    dt: str | None = None,
    hour: int | None = None,
) -> dict[str, object]:
    try:
        return await _weather_provider().forecast(q, days=days, dt=dt, hour=hour)
    except Exception as exc:  # noqa: BLE001
        raise _weather_error(exc) from exc


@router.get("/weather/history")
async def weather_history(
    q: str,
    dt: str,
    hour: int | None = None,
    end_dt: str | None = None,
) -> dict[str, object]:
    try:
        return await _weather_provider().history(q, dt=dt, hour=hour, end_dt=end_dt)
    except Exception as exc:  # noqa: BLE001
        raise _weather_error(exc) from exc


@router.get("/weather/baseball")
async def weather_baseball(
    q: str,
    game_date: str | None = None,
    hour: int | None = None,
) -> dict[str, object]:
    """Return compact weather fields that matter for baseball run totals."""
    try:
        return await _weather_provider().baseball_weather(q, game_date=game_date, hour=hour)
    except Exception as exc:  # noqa: BLE001
        raise _weather_error(exc) from exc


# ---------------------------- pybaseball ------------------------------------


def _pybaseball_provider() -> PyBaseballProvider:
    if not get_settings().pybaseball_enabled:
        raise HTTPException(status_code=503, detail="pybaseball integration is disabled")
    return PyBaseballProvider()


def _pybaseball_error(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, PyBaseballError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}")


@router.get("/baseball/player-lookup")
async def baseball_player_lookup(first: str, last: str) -> list[dict[str, object]]:
    try:
        return await _pybaseball_provider().player_lookup(first, last)
    except Exception as exc:  # noqa: BLE001
        raise _pybaseball_error(exc) from exc


@router.get("/baseball/statcast")
async def baseball_statcast(
    start_dt: str,
    end_dt: str | None = None,
    limit: int = 500,
) -> list[dict[str, object]]:
    try:
        return await _pybaseball_provider().statcast(start_dt, end_dt, limit=limit)
    except Exception as exc:  # noqa: BLE001
        raise _pybaseball_error(exc) from exc


@router.get("/baseball/pitchers/{pitcher_id}/statcast")
async def baseball_pitcher_statcast(
    pitcher_id: int,
    start_dt: str,
    end_dt: str | None = None,
    limit: int = 500,
) -> list[dict[str, object]]:
    try:
        return await _pybaseball_provider().pitcher_statcast(
            pitcher_id,
            start_dt,
            end_dt,
            limit=limit,
        )
    except Exception as exc:  # noqa: BLE001
        raise _pybaseball_error(exc) from exc


@router.get("/baseball/batters/{batter_id}/statcast")
async def baseball_batter_statcast(
    batter_id: int,
    start_dt: str,
    end_dt: str | None = None,
    limit: int = 500,
) -> list[dict[str, object]]:
    try:
        return await _pybaseball_provider().batter_statcast(
            batter_id,
            start_dt,
            end_dt,
            limit=limit,
        )
    except Exception as exc:  # noqa: BLE001
        raise _pybaseball_error(exc) from exc


@router.get("/baseball/pitching-stats")
async def baseball_pitching_stats(
    start_season: int,
    end_season: int | None = None,
) -> list[dict[str, object]]:
    try:
        return await _pybaseball_provider().pitching_stats(start_season, end_season)
    except Exception as exc:  # noqa: BLE001
        raise _pybaseball_error(exc) from exc


@router.get("/baseball/batting-stats")
async def baseball_batting_stats(
    start_season: int,
    end_season: int | None = None,
) -> list[dict[str, object]]:
    try:
        return await _pybaseball_provider().batting_stats(start_season, end_season)
    except Exception as exc:  # noqa: BLE001
        raise _pybaseball_error(exc) from exc


@router.get("/baseball/players/{player_id}/splits")
async def baseball_player_splits(
    player_id: str,
    season: int | None = None,
    pitching_splits: bool = False,
    limit: int = 500,
) -> list[dict[str, object]]:
    try:
        return await _pybaseball_provider().player_splits(
            player_id,
            season=season,
            pitching_splits=pitching_splits,
            limit=limit,
        )
    except Exception as exc:  # noqa: BLE001
        raise _pybaseball_error(exc) from exc


# ---------------------------- bot helpers -----------------------------------


def _market_url_for_signal(signal: Signal) -> str | None:
    market = signal.market
    if not market or not market.slug:
        return None
    if market.platform and market.platform.lower() == "kalshi":
        return f"https://kalshi.com/markets/{market.slug.upper()}"
    return f"https://polymarket.com/event/{market.slug}"


def _market_url(market: Market) -> str:
    if market.platform and market.platform.lower() == "kalshi":
        return f"https://kalshi.com/markets/{market.slug.upper()}"
    return f"https://polymarket.com/event/{market.slug}"


def _extract_market_slug(market_url: str) -> str:
    parsed = urlparse(market_url.strip())
    path = parsed.path if parsed.scheme else market_url
    parts = [unquote(part).strip() for part in path.split("/") if part.strip()]
    if not parts:
        raise HTTPException(status_code=400, detail="Could not parse market URL")

    lowered = [part.lower() for part in parts]
    for marker in ("event", "markets", "market"):
        if marker in lowered:
            idx = lowered.index(marker)
            if idx + 1 < len(parts):
                return parts[idx + 1].lower()

    return parts[-1].lower()


def _event_date_for_market(market: Market | None) -> date | None:
    slug = market.slug if market else ""
    match = _SLUG_DATE_RE.search(slug)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def _trader_url_for_signal(signal: Signal) -> str | None:
    trader = signal.trader
    if not (trader and trader.wallet_address):
        return None
    return f"https://polymarketanalytics.com/traders/{trader.wallet_address}"


def _event_date_for_signal(signal: Signal) -> date | None:
    return _event_date_for_market(signal.market)


def _signal_bot_payload(signal: Signal, tier: str) -> dict[str, object]:
    event_date = _event_date_for_signal(signal)
    return {
        "id": signal.id,
        "tier": tier,
        "score": round(signal.score, 2),
        "confidence": round(signal.confidence, 4),
        "market": signal.market.title if signal.market else f"market#{signal.market_id}",
        "market_slug": signal.market.slug if signal.market else None,
        "market_url": _market_url_for_signal(signal),
        "event_date": event_date.isoformat() if event_date else None,
        "trader": signal.trader.nickname if signal.trader else None,
        "trader_url": _trader_url_for_signal(signal),
        "side": signal.side,
        "outcome": signal.outcome,
        "entry_price": signal.entry_price,
        "size_usd": signal.size_usd,
        "reason": signal.reason,
        "created_at": signal.created_at.isoformat() if signal.created_at else None,
    }


@router.get("/bot/status")
def bot_status(db: Session = Depends(get_db)) -> dict[str, object]:
    s = get_settings()
    falcon_health = get_falcon_health()
    recent_cutoff = datetime.utcnow() - timedelta(hours=24)
    return {
        "status": "ok",
        "environment": s.environment,
        "traders": db.scalar(select(func.count(Trader.id))) or 0,
        "markets": db.scalar(select(func.count(Market.id))) or 0,
        "signals_24h": db.scalar(select(func.count(Signal.id)).where(Signal.created_at >= recent_cutoff)) or 0,
        "discord_sent_24h": db.scalar(
            select(func.count(Alert.id)).where(
                Alert.channel == "discord",
                Alert.status == "sent",
                Alert.created_at >= recent_cutoff,
            )
        ) or 0,
        "discord_skipped_24h": db.scalar(
            select(func.count(Alert.id)).where(
                Alert.channel == "discord",
                Alert.status == "skipped",
                Alert.created_at >= recent_cutoff,
            )
        ) or 0,
        "falcon": {
            "configured": s.has_falcon_credentials(),
            "healthy": falcon_health.healthy,
            "success_rate": round(falcon_health.success_rate, 3),
            "last_scan_calls": falcon_health.last_scan_calls,
            "last_scan_successes": falcon_health.last_scan_successes,
            "last_error": falcon_health.last_error,
            "last_scan_at": (
                falcon_health.last_scan_at.isoformat()
                if falcon_health.last_scan_at
                else None
            ),
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/bot/high-conviction")
def bot_high_conviction(
    db: Session = Depends(get_db),
    limit: int = 5,
    hours: int = 24,
    event_date_from: date | None = None,
) -> dict[str, object]:
    limit = max(1, min(limit, 10))
    hours = max(1, min(hours, 168))
    since = datetime.utcnow() - timedelta(hours=hours)
    candidates = list(
        db.scalars(
            select(Signal)
            .where(Signal.created_at >= since)
            .order_by(desc(Signal.score), desc(Signal.created_at))
            .limit(100)
        )
    )

    matches: list[dict[str, object]] = []
    near_misses: list[dict[str, object]] = []
    for signal in candidates:
        event_date = _event_date_for_signal(signal)
        if event_date_from and event_date and event_date < event_date_from:
            continue
        decision = evaluate_alert_decision(db, signal)
        if decision.tier in {"high_conviction", "possible_entry"}:
            payload = _signal_bot_payload(signal, decision.tier)
            payload.update(
                {
                    "action": decision.action,
                    "chase_risk": decision.chase_risk,
                    "total_tracked_size": round(decision.context.total_tracked_size, 2),
                    "trader_count": decision.context.trader_count,
                    "current_price": decision.context.current_price,
                    "entry_price_min": decision.context.entry_price_min,
                    "entry_price_max": decision.context.entry_price_max,
                }
            )
            matches.append(payload)
        elif len(near_misses) < 3:
            payload = _signal_bot_payload(signal, decision.tier)
            payload.update(
                {
                    "failed_reason": decision.reason,
                    "action": decision.action,
                    "chase_risk": decision.chase_risk,
                    "total_tracked_size": round(decision.context.total_tracked_size, 2),
                    "trader_count": decision.context.trader_count,
                    "current_price": decision.context.current_price,
                    "entry_price_min": decision.context.entry_price_min,
                    "entry_price_max": decision.context.entry_price_max,
                }
            )
            near_misses.append(payload)
        if len(matches) >= limit:
            break

    return {
        "count": len(matches),
        "hours": hours,
        "event_date_from": event_date_from.isoformat() if event_date_from else None,
        "discord_tiers": sorted(DISCORD_TIERS),
        "signals": matches,
        "near_misses": near_misses,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/bot/search")
def bot_search_market(
    market_url: str,
    db: Session = Depends(get_db),
    limit: int = 10,
) -> dict[str, object]:
    limit = max(1, min(limit, 25))
    slug = _extract_market_slug(market_url)
    market = db.scalar(
        select(Market).where(func.lower(Market.slug) == slug.lower())
    )
    if market is None:
        raise HTTPException(
            status_code=404,
            detail=f"Market '{slug}' has not been seen by SignalForge yet",
        )

    trades = list(
        db.scalars(
            select(Trade)
            .where(Trade.market_id == market.id)
            .order_by(Trade.timestamp.desc())
        )
    )
    signals = list(
        db.scalars(
            select(Signal)
            .where(Signal.market_id == market.id)
            .order_by(desc(Signal.score), desc(Signal.confidence), desc(Signal.created_at))
        )
    )
    signals_by_key: dict[tuple[int | None, str | None, str | None], Signal] = {}
    for signal in signals:
        key = (signal.trader_id, signal.side, signal.outcome)
        signals_by_key.setdefault(key, signal)

    grouped: dict[tuple[int, str, str | None], list[Trade]] = {}
    for trade in trades:
        grouped.setdefault((trade.trader_id, trade.side, trade.outcome), []).append(trade)

    positions: list[dict[str, object]] = []
    for (trader_id, side, outcome), grouped_trades in grouped.items():
        trader = grouped_trades[0].trader
        total_size = sum(t.size_usd or 0 for t in grouped_trades)
        weighted_price_sum = sum((t.price or 0) * (t.size_usd or 0) for t in grouped_trades)
        avg_entry = weighted_price_sum / total_size if total_size > 0 else None
        first_trade = min(grouped_trades, key=lambda t: t.timestamp)
        last_trade = max(grouped_trades, key=lambda t: t.timestamp)
        signal = signals_by_key.get((trader_id, side, outcome))
        if signal is None:
            signal = signals_by_key.get((trader_id, side, None))

        score = signal.score if signal else trader.trust_score if trader else 0.0
        confidence = signal.confidence if signal else 0.0
        reason = signal.reason if signal else f"{len(grouped_trades)} trade(s) found for this market"
        trader_url = (
            f"https://polymarketanalytics.com/traders/{trader.wallet_address}"
            if trader and trader.wallet_address
            else None
        )

        positions.append(
            {
                "market": market.title,
                "market_slug": market.slug,
                "market_url": _market_url(market),
                "event_date": (
                    _event_date_for_market(market).isoformat()
                    if _event_date_for_market(market)
                    else None
                ),
                "trader": trader.nickname if trader else f"trader#{trader_id}",
                "trader_url": trader_url,
                "wallet": trader.wallet_address if trader else None,
                "side": side,
                "outcome": outcome,
                "avg_entry_price": round(avg_entry, 4) if avg_entry is not None else None,
                "total_size_usd": round(total_size, 2),
                "trade_count": len(grouped_trades),
                "first_trade_at": first_trade.timestamp.isoformat() if first_trade.timestamp else None,
                "last_trade_at": last_trade.timestamp.isoformat() if last_trade.timestamp else None,
                "score": round(score, 2),
                "confidence": round(confidence, 4),
                "reason": reason,
            }
        )

    positions.sort(
        key=lambda p: (
            float(p.get("score") or 0),
            float(p.get("confidence") or 0),
            float(p.get("total_size_usd") or 0),
        ),
        reverse=True,
    )

    return {
        "market": market.title,
        "market_slug": market.slug,
        "market_url": _market_url(market),
        "event_date": (
            _event_date_for_market(market).isoformat()
            if _event_date_for_market(market)
            else None
        ),
        "count": len(positions[:limit]),
        "positions": positions[:limit],
        "timestamp": datetime.utcnow().isoformat(),
    }


# ---------------------------- run-scan --------------------------------------


@router.post("/run-scan", response_model=ScanResult)
async def trigger_scan() -> ScanResult:
    return await run_scan_once()


# ---------------------------- falcon-test -----------------------------------

# LaBradfordSmith22 from the seeded watchlist — a known-good wallet to probe.
_DEFAULT_TEST_WALLET = "0x9495425feeb0c250accb89275c97587011b19a27"


@router.get("/falcon-test")
async def falcon_test(
    wallet: str = _DEFAULT_TEST_WALLET,
    agent_id: int = FalconProvider.AGENT_WALLET_360,
    window_days: int = 3,
) -> dict[str, object]:
    """Probe a single Falcon agent and return the raw response.

    Useful for verifying the API key works and discovering the actual response
    shape so the parser in FalconProvider can be tightened. Defaults to
    agent_id=581 (Wallet 360) against LaBradfordSmith22's wallet.
    """
    s = get_settings()
    if not s.has_falcon_credentials():
        raise HTTPException(
            status_code=400,
            detail="SIGNALFORGE_FALCON_API_KEY is not set",
        )

    falcon = FalconProvider(s.falcon_api_key, s.falcon_base_url)
    raw = await falcon.query_agent(
        agent_id,
        params={"proxy_wallet": wallet, "window_days": str(window_days)},
    )
    health = get_falcon_health()
    return {
        "agent_id": agent_id,
        "wallet": wallet,
        "window_days": window_days,
        "ok": raw is not None,
        "raw_response": raw,
        "falcon_health": {
            "last_status_code": health.last_status_code,
            "last_error": health.last_error,
            "last_endpoint": health.last_endpoint,
            "last_agent_id": health.last_agent_id,
        },
    }


# ---------------------------- dashboard -------------------------------------


@router.get("/dashboard-summary", response_model=DashboardSummary)
def dashboard_summary(db: Session = Depends(get_db)) -> DashboardSummary:
    active_signals_raw = list(
        db.scalars(
            select(Signal).order_by(desc(Signal.score), desc(Signal.created_at)).limit(10)
        )
    )
    active_signals = [_enrich_signal(s) for s in active_signals_raw]

    top_traders = list(
        db.scalars(
            select(Trader).order_by(desc(Trader.trust_score)).limit(10)
        )
    )

    # "Highest conviction" markets = those carrying the highest-scored signals.
    top_market_ids_query = (
        select(Signal.market_id, func.max(Signal.score).label("max_score"))
        .group_by(Signal.market_id)
        .order_by(desc("max_score"))
        .limit(10)
    )
    top_market_ids = [row[0] for row in db.execute(top_market_ids_query).all()]
    if top_market_ids:
        markets = list(db.scalars(select(Market).where(Market.id.in_(top_market_ids))))
        # preserve order from the score query
        order = {mid: idx for idx, mid in enumerate(top_market_ids)}
        markets.sort(key=lambda m: order.get(m.id, 999))
    else:
        markets = list(db.scalars(select(Market).limit(10)))

    recent_alerts = list(
        db.scalars(select(Alert).order_by(desc(Alert.created_at)).limit(10))
    )

    # Watchlist health: how complete is our trader enrichment?
    all_traders = list(db.scalars(select(Trader)))
    enriched = [t for t in all_traders if t.total_pnl is not None or t.trader_rank is not None]
    win_rates = [t.win_rate for t in all_traders if t.win_rate is not None]
    health = WatchlistHealth(
        total_traders=len(all_traders),
        enriched_traders=len(enriched),
        avg_trust_score=round(
            sum(t.trust_score for t in all_traders) / max(len(all_traders), 1), 2
        ),
        avg_win_rate=round(sum(win_rates) / len(win_rates), 4) if win_rates else None,
        enabled_for_copy=sum(1 for t in all_traders if t.copy_enabled),
    )

    # Simulated PnL is a placeholder — the MVP doesn't execute trades.
    simulated_pnl = 0.0

    return DashboardSummary(
        active_signals=active_signals,
        top_traders=top_traders,
        highest_conviction_markets=markets,
        recent_alerts=recent_alerts,
        simulated_pnl_usd=simulated_pnl,
        watchlist_health=health,
    )
