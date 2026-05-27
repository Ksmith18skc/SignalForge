"""Centralized Odds-API cache for MLB.

The Odds-API.io free plan rate-limits aggressively. Before this module every
consumer (edge engine, closing-line updater, debug endpoints) called the API
directly, which both burned the quota in minutes AND made all MLB edges show
line=null when 429s hit.

The flow now is:
    1. ONE caller per slate runs `refresh_mlb_odds_cache(db, game_date=...)`.
       That issues at most:
         - 1 GET /events     (lists every MLB event for the date)
         - N GET /odds       (one per matched event; usually 8-15 games/day)
       Each response is persisted to `odds_snapshots` with a TTL.
    2. Every consumer reads via `get_cached_*` helpers — never the live API.
    3. On a 429 (or any other upstream error) we fall back to whatever stale
       row is still in the table; edges keep getting generated.

A process-level threading.Lock ensures concurrent refresh attempts (e.g. the
scanner thread + a /run-scan HTTP request) coalesce into one upstream fetch.
Stale concurrent callers just wait briefly and then read the freshly-cached
rows — no duplicate calls, no thundering herd.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import date as date_cls, datetime, time, timedelta, timezone
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import OddsSnapshot, ProviderHealthState
from app.providers.odds_api import OddsApiError, OddsApiProvider, OddsApiRateLimited
from app.providers.sportsgameodds import (
    SportsGameOddsError,
    SportsGameOddsProvider,
    SportsGameOddsRateLimited,
    reset_cooldown,
)
from app.services.card_date import TZ_ARIZONA
from app.services.mlb_odds_matching import MatchResult, match_all_games
from app.utils.redaction import sanitize_text

logger = logging.getLogger(__name__)

# --- TTLs (aggressive — free-plan friendly) ---------------------------------

# Full-game totals don't move fast enough mid-afternoon to justify <5 min.
MLB_TOTALS_TTL = timedelta(minutes=8)
# Pitcher props are stickier — books re-post them once an hour at most.
MLB_PITCHER_PROPS_TTL = timedelta(minutes=12)
# The events list (slate roster) is essentially immutable once the day starts.
MLB_EVENTS_LIST_TTL = timedelta(minutes=30)

# When a cached row's TTL has passed but the upstream is unreachable (or
# rate-limited), we keep serving the row up to this absolute age cap. Beyond
# this we give up and report "missing odds" — the data is too stale to trust.
MLB_STALE_GRACE = timedelta(hours=2)

# Odds-API.io v3 uses broad sport keys. MLB is filtered by matching the
# returned baseball events to MLB StatsAPI games.
ODDS_MLB_SPORT = "baseball"
ODDS_MLB_LEAGUE: str | None = None
ODDS_API_PROVIDER = "Odds-API"
SGO_PROVIDER = "SportsGameOdds"

MARKET_TYPE_EVENTS_LIST = "events_list"
MARKET_TYPE_EVENT_ODDS = "event_odds"


# --- metrics ----------------------------------------------------------------


@dataclass
class OddsCacheHealth:
    live_api_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avoided_api_calls: int = 0
    rate_limited_count: int = 0
    last_rate_limited_at: datetime | None = None
    last_refresh_at: datetime | None = None
    last_refresh_event_count: int = 0
    last_refresh_skipped_reason: str | None = None
    stale_fallbacks: int = 0
    refresh_errors: int = 0
    last_refresh_error: str | None = None

    def hourly_call_rate(self, window: timedelta = timedelta(hours=1)) -> float:
        """Crude rate estimate — only meaningful if metrics aren't reset."""
        # We don't keep a timeseries; this just reports the raw counter so
        # operators can compare counts over time.
        return float(self.live_api_calls)


@dataclass
class ProviderHealth:
    name: str
    enabled: bool
    last_success_at: datetime | None = None
    last_error: str | None = None
    last_error_at: datetime | None = None
    events_fetched: int = 0
    totals_found: int = 0
    pitcher_props_found: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "last_success_at": self.last_success_at.isoformat() if self.last_success_at else None,
            "last_error": self.last_error,
            "last_error_at": self.last_error_at.isoformat() if self.last_error_at else None,
            "events_fetched": self.events_fetched,
            "totals_found": self.totals_found,
            "pitcher_props_found": self.pitcher_props_found,
        }


@dataclass
class ProvidersHealth:
    primary: ProviderHealth
    backup: ProviderHealth
    last_provider_used: str | None = None
    last_errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary.as_dict(),
            "backup": self.backup.as_dict(),
            "last_provider_used": self.last_provider_used,
            "last_errors": list(self.last_errors),
        }


_health = OddsCacheHealth()
_metrics_lock = threading.Lock()
# Provider diagnostics (primary Odds-API, backup SportsGameOdds).
_providers_lock = threading.Lock()
_providers_health = ProvidersHealth(
    primary=ProviderHealth(name="Odds-API", enabled=True),
    backup=ProviderHealth(name="SportsGameOdds", enabled=False),
)
# One refresh at a time, full stop. Threading.Lock (not asyncio.Lock) so
# concurrent callers on different event loops are also serialized.
_refresh_lock = threading.Lock()


def get_odds_cache_health() -> OddsCacheHealth:
    with _metrics_lock:
        return OddsCacheHealth(**_health.__dict__)


def get_odds_provider_health(db: Session | None = None) -> dict[str, Any]:
    with _providers_lock:
        payload = _providers_health.as_dict()
    if db is not None:
        persisted = provider_health_snapshot(db)
        for key, state in persisted.items():
            if key == ODDS_API_PROVIDER:
                payload["primary"].update(state)
            elif key == SGO_PROVIDER:
                payload["backup"].update(state)
        payload["providers"] = list(persisted.values())
    return payload


def reset_metrics() -> None:
    """Test helper."""
    global _health
    with _metrics_lock:
        _health = OddsCacheHealth()
    with _providers_lock:
        _providers_health.primary = ProviderHealth(name="Odds-API", enabled=True)
        _providers_health.backup = ProviderHealth(name="SportsGameOdds", enabled=False)
        _providers_health.last_provider_used = None
        _providers_health.last_errors = []
    reset_cooldown()


def _record(**deltas: Any) -> None:
    with _metrics_lock:
        for key, value in deltas.items():
            if isinstance(value, (int, float)):
                setattr(_health, key, getattr(_health, key) + value)
            else:
                setattr(_health, key, value)


def _record_provider_event(
    *,
    provider: str,
    enabled: bool | None = None,
    events_fetched: int | None = None,
    totals_found: int | None = None,
    pitcher_props_found: int | None = None,
    last_success: bool = False,
    error: str | None = None,
) -> None:
    with _providers_lock:
        target = _providers_health.primary if provider == ODDS_API_PROVIDER else _providers_health.backup
        if enabled is not None:
            target.enabled = enabled
        if events_fetched is not None:
            target.events_fetched = events_fetched
        if totals_found is not None:
            target.totals_found = totals_found
        if pitcher_props_found is not None:
            target.pitcher_props_found = pitcher_props_found
        if last_success:
            target.last_success_at = datetime.utcnow()
        if error:
            target.last_error = sanitize_text(error, limit=700)
            target.last_error_at = datetime.utcnow()
            _providers_health.last_errors.append(sanitize_text(error, limit=700))


def _record_last_provider_used(name: str | None) -> None:
    with _providers_lock:
        _providers_health.last_provider_used = name


def provider_health_snapshot(db: Session) -> dict[str, dict[str, Any]]:
    states = list(db.scalars(select(ProviderHealthState)))
    now = datetime.utcnow()
    out: dict[str, dict[str, Any]] = {}
    for state in states:
        status = "ok" if state.last_success_at and not state.last_error_at else "unknown"
        if state.cooldown_until and state.cooldown_until > now:
            status = "cooldown"
        elif state.last_error_at and (
            not state.last_success_at or state.last_error_at >= state.last_success_at
        ):
            status = "error"
        out[state.provider] = {
            "provider": state.provider,
            "enabled": state.enabled,
            "last_success_at": state.last_success_at.isoformat() if state.last_success_at else None,
            "last_error": state.last_error,
            "last_error_at": state.last_error_at.isoformat() if state.last_error_at else None,
            "cooldown_until": state.cooldown_until.isoformat() if state.cooldown_until else None,
            "recent_failures": state.recent_failures,
            "last_status_code": state.last_status_code,
            "last_successful_strategy": state.last_successful_strategy,
            "last_refresh_event_count": state.last_refresh_event_count,
            "refresh_errors": state.refresh_errors,
            "status": status,
        }
    return out


def _record_provider_health(
    db: Session,
    provider: str,
    *,
    enabled: bool | None = None,
    success: bool = False,
    error: str | None = None,
    status_code: int | None = None,
    cooldown_until: datetime | None = None,
    strategy: str | None = None,
    events_fetched: int | None = None,
) -> None:
    state = db.get(ProviderHealthState, provider)
    if state is None:
        state = ProviderHealthState(provider=provider, enabled=enabled if enabled is not None else True)
        db.add(state)
    if enabled is not None:
        state.enabled = enabled
    state.updated_at = datetime.utcnow()
    if success:
        state.last_success_at = state.updated_at
        state.recent_failures = 0
        state.last_error = None
        state.last_status_code = status_code or 200
        state.cooldown_until = None
        if strategy:
            state.last_successful_strategy = strategy
        if events_fetched is not None:
            state.last_refresh_event_count = events_fetched
    if error:
        state.last_error_at = state.updated_at
        state.last_error = sanitize_text(error, limit=700)
        state.last_status_code = status_code
        if state.recent_failures is None:
            state.recent_failures = 0
        state.recent_failures += 1
        if state.refresh_errors is None:
            state.refresh_errors = 0
        state.refresh_errors += 1
    if cooldown_until is not None:
        state.cooldown_until = cooldown_until
    db.flush()


def _provider_on_cooldown(db: Session, provider: str) -> ProviderHealthState | None:
    state = db.get(ProviderHealthState, provider)
    if state and state.cooldown_until and state.cooldown_until > datetime.utcnow():
        return state
    return None


# --- read paths -------------------------------------------------------------


def _events_list_key(game_date: str) -> str:
    """Synthetic event_id for the per-day events list row."""
    return f"_events_{game_date}"


def _sgo_provider() -> SportsGameOddsProvider | None:
    settings = get_settings()
    enabled = bool(settings.sgo_enabled)
    _record_provider_event(provider=SGO_PROVIDER, enabled=enabled)
    if not enabled:
        return None
    return SportsGameOddsProvider(settings.sgo_api_key, settings.sgo_base_url)


def _load_snapshot(
    db: Session,
    *,
    sport: str,
    event_id: str,
    market_type: str,
) -> OddsSnapshot | None:
    return db.scalar(
        select(OddsSnapshot)
        .where(
            OddsSnapshot.sport == sport,
            OddsSnapshot.event_id == event_id,
            OddsSnapshot.market_type == market_type,
        )
        .order_by(OddsSnapshot.fetched_at.desc())
        .limit(1)
    )


def _snapshot_is_fresh(snap: OddsSnapshot | None, now: datetime | None = None) -> bool:
    if snap is None:
        return False
    now = now or datetime.utcnow()
    return snap.expires_at > now


def _snapshot_is_within_grace(snap: OddsSnapshot | None, now: datetime | None = None) -> bool:
    if snap is None:
        return False
    now = now or datetime.utcnow()
    return (now - snap.fetched_at) <= MLB_STALE_GRACE


def get_cached_events_list(
    db: Session,
    *,
    game_date: str,
    fallback_stale: bool = True,
) -> list[dict[str, Any]] | None:
    snap = _load_snapshot(
        db, sport=ODDS_MLB_SPORT, event_id=_events_list_key(game_date),
        market_type=MARKET_TYPE_EVENTS_LIST,
    )
    if snap is None:
        _record(cache_misses=1)
        return None
    if _snapshot_is_fresh(snap):
        _record(cache_hits=1, avoided_api_calls=1)
        return list(snap.payload or [])
    if fallback_stale and _snapshot_is_within_grace(snap):
        _record(cache_hits=1, stale_fallbacks=1, avoided_api_calls=1)
        return list(snap.payload or [])
    _record(cache_misses=1)
    return None


def get_cached_event_odds(
    db: Session,
    event_id: str,
    *,
    fallback_stale: bool = True,
) -> dict[str, Any] | None:
    snap = _load_snapshot(
        db, sport=ODDS_MLB_SPORT, event_id=event_id, market_type=MARKET_TYPE_EVENT_ODDS,
    )
    if snap is None:
        _record(cache_misses=1)
        return None
    if _snapshot_is_fresh(snap):
        _record(cache_hits=1, avoided_api_calls=1)
        return dict(snap.payload or {})
    if fallback_stale and _snapshot_is_within_grace(snap):
        _record(cache_hits=1, stale_fallbacks=1, avoided_api_calls=1)
        return dict(snap.payload or {})
    _record(cache_misses=1)
    return None


def events_list_cache_state(db: Session, *, game_date: str) -> dict[str, Any]:
    snap = _load_snapshot(
        db,
        sport=ODDS_MLB_SPORT,
        event_id=_events_list_key(game_date),
        market_type=MARKET_TYPE_EVENTS_LIST,
    )
    return _cache_state_dict(snap)


def event_odds_cache_state(db: Session, event_id: str) -> dict[str, Any]:
    snap = _load_snapshot(
        db,
        sport=ODDS_MLB_SPORT,
        event_id=event_id,
        market_type=MARKET_TYPE_EVENT_ODDS,
    )
    return _cache_state_dict(snap)


def _cache_state_dict(snap: OddsSnapshot | None) -> dict[str, Any]:
    now = datetime.utcnow()
    if snap is None:
        return {
            "state": "missing",
            "fresh": False,
            "age_minutes": None,
            "fetched_at": None,
            "expires_at": None,
        }
    age_minutes = int((now - snap.fetched_at).total_seconds() / 60)
    fresh = snap.expires_at > now
    return {
        "state": "fresh" if fresh else "stale",
        "fresh": fresh,
        "age_minutes": age_minutes,
        "fetched_at": snap.fetched_at.isoformat() if snap.fetched_at else None,
        "expires_at": snap.expires_at.isoformat() if snap.expires_at else None,
    }


def get_cached_mlb_totals(
    db: Session,
    *,
    event_id: str,
    fallback_stale: bool = True,
) -> dict[str, Any]:
    """Return parsed game-totals analysis for one event_id, cache-only."""
    from app.services.mlb_odds_analysis import analyze_game_totals

    payload = get_cached_event_odds(db, event_id, fallback_stale=fallback_stale)
    return analyze_game_totals(payload)


def get_cached_pitcher_props(
    db: Session,
    *,
    event_id: str,
    pitcher_name: str | None = None,
    fallback_stale: bool = True,
) -> dict[str, Any]:
    """Return parsed pitcher-strikeout consensus for one event_id, cache-only."""
    from app.services.mlb_prop_odds import consensus_for_pitcher, normalize_pitcher_strikeout_props

    payload = get_cached_event_odds(db, event_id, fallback_stale=fallback_stale)
    lines = normalize_pitcher_strikeout_props(payload)
    if not pitcher_name:
        # Return raw line list when no pitcher specified — callers can filter.
        return {"rows": [_line_to_dict_inline(line) for line in lines]}
    return consensus_for_pitcher(lines, pitcher_name)


def _line_to_dict_inline(line: Any) -> dict[str, Any]:
    return {
        "player_name": line.player_name,
        "line": line.line,
        "over_price": line.over_price,
        "under_price": line.under_price,
        "sportsbook": line.sportsbook,
    }


# --- write path -------------------------------------------------------------


def _upsert_snapshot(
    db: Session,
    *,
    sport: str,
    event_id: str,
    market_type: str,
    payload: Any,
    ttl: timedelta,
    sportsbook: str | None = None,
) -> OddsSnapshot:
    """Replace any prior row for this (sport, event_id, market_type, book)."""
    db.execute(
        delete(OddsSnapshot).where(
            OddsSnapshot.sport == sport,
            OddsSnapshot.event_id == event_id,
            OddsSnapshot.market_type == market_type,
            OddsSnapshot.sportsbook.is_(sportsbook) if sportsbook is None
            else OddsSnapshot.sportsbook == sportsbook,
        )
    )
    now = datetime.utcnow()
    snap = OddsSnapshot(
        sport=sport,
        event_id=event_id,
        market_type=market_type,
        sportsbook=sportsbook,
        payload=payload,
        fetched_at=now,
        expires_at=now + ttl,
    )
    db.add(snap)
    db.flush()
    return snap


# --- refresh path -----------------------------------------------------------


@dataclass
class RefreshResult:
    """Outcome of a refresh call — exposed via /mlb/debug/odds-cache."""

    refreshed: bool
    reason: str
    game_date: str
    events_fetched: int = 0
    events_cached: int = 0
    matched_games: int = 0
    odds_calls: int = 0
    odds_cached: int = 0
    rate_limited: int = 0
    errors: list[str] = field(default_factory=list)
    provider: str | None = None
    strategy_attempted: list[str] = field(default_factory=list)
    strategy_successful: str | None = None
    parsed_error_reason: str | None = None
    provider_cooldowns: dict[str, str | None] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "refreshed": self.refreshed,
            "reason": self.reason,
            "game_date": self.game_date,
            "events_fetched": self.events_fetched,
            "events_cached": self.events_cached,
            "matched_games": self.matched_games,
            "odds_calls": self.odds_calls,
            "odds_cached": self.odds_cached,
            "rate_limited": self.rate_limited,
            "errors": [sanitize_text(err, limit=700) for err in self.errors],
            "provider": self.provider,
            "strategy_attempted": list(self.strategy_attempted),
            "strategy_successful": self.strategy_successful,
            "parsed_error_reason": self.parsed_error_reason,
            "provider_cooldowns": dict(self.provider_cooldowns),
        }


def _events_list_fresh(db: Session, game_date: str) -> bool:
    snap = _load_snapshot(
        db, sport=ODDS_MLB_SPORT, event_id=_events_list_key(game_date),
        market_type=MARKET_TYPE_EVENTS_LIST,
    )
    return _snapshot_is_fresh(snap)


def odds_api_event_window(game_date: str) -> tuple[str, str]:
    """Return the Arizona card-date window as UTC RFC3339 timestamps."""
    card = date_cls.fromisoformat(game_date)
    start_local = datetime.combine(card, time.min, tzinfo=TZ_ARIZONA)
    end_local = start_local + timedelta(days=1) - timedelta(seconds=1)
    return _rfc3339_utc(start_local), _rfc3339_utc(end_local)


def _rfc3339_utc(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def odds_api_event_strategies(game_date: str) -> list[dict[str, Any]]:
    """Bounded request fallback plan for Odds-API events."""
    date_from, date_to = odds_api_event_window(game_date)
    return [
        {
            "name": "full_iso_window",
            "sport": ODDS_MLB_SPORT,
            "league": ODDS_MLB_LEAGUE,
            "date_from": date_from,
            "date_to": date_to,
            "limit": 200,
        },
        {
            "name": "date_only_window",
            "sport": ODDS_MLB_SPORT,
            "league": ODDS_MLB_LEAGUE,
            "date_from": game_date,
            "date_to": game_date,
            "limit": 200,
        },
        {
            "name": "no_window_local_filter",
            "sport": ODDS_MLB_SPORT,
            "league": ODDS_MLB_LEAGUE,
            "limit": 200,
        },
        {
            "name": "provider_native_defaults",
            "sport": ODDS_MLB_SPORT,
            "league": ODDS_MLB_LEAGUE,
        },
    ]


async def _fetch_odds_api_events_with_fallback(
    db: Session,
    odds: OddsApiProvider,
    result: RefreshResult,
    *,
    game_date: str,
) -> list[dict[str, Any]] | None:
    last_error: str | None = None
    for strategy in odds_api_event_strategies(game_date):
        name = str(strategy["name"])
        result.strategy_attempted.append(name)
        try:
            events_payload = await odds.events(
                str(strategy["sport"]),
                league=strategy.get("league"),
                date_from=strategy.get("date_from"),
                date_to=strategy.get("date_to"),
                limit=strategy.get("limit"),
            )
            _record(live_api_calls=1)
            events = _filter_events_for_card_date(events_payload or [], game_date)
            result.strategy_successful = name
            result.provider = ODDS_API_PROVIDER
            _record_provider_event(
                provider=ODDS_API_PROVIDER,
                events_fetched=len(events),
                last_success=True,
            )
            _record_provider_health(
                db,
                ODDS_API_PROVIDER,
                enabled=True,
                success=True,
                strategy=name,
                events_fetched=len(events),
            )
            return events
        except OddsApiRateLimited as exc:
            _record(rate_limited_count=1, last_rate_limited_at=datetime.utcnow())
            msg = f"events 429 via {name}: {exc}"
            result.errors.append(msg)
            result.rate_limited += 1
            result.parsed_error_reason = "rate_limited"
            _record_provider_event(provider=ODDS_API_PROVIDER, error=msg)
            _record_provider_health(
                db,
                ODDS_API_PROVIDER,
                enabled=True,
                error=msg,
                status_code=429,
            )
            return None
        except OddsApiError as exc:
            msg = f"events failed via {name}: {type(exc).__name__}: {exc}"
            last_error = msg
            status_code = getattr(exc, "status_code", None)
            result.errors.append(msg)
            result.parsed_error_reason = _parse_provider_error_reason(exc)
            _record(refresh_errors=1, last_refresh_error=msg)
            _record_provider_event(provider=ODDS_API_PROVIDER, error=msg)
            _record_provider_health(
                db,
                ODDS_API_PROVIDER,
                enabled=True,
                error=msg,
                status_code=status_code,
            )
            if status_code == 400:
                continue
            return None
        except Exception as exc:  # noqa: BLE001
            msg = f"events failed via {name}: {type(exc).__name__}: {sanitize_text(exc)}"
            last_error = msg
            result.errors.append(msg)
            result.parsed_error_reason = "upstream_error"
            _record(refresh_errors=1, last_refresh_error=msg)
            _record_provider_event(provider=ODDS_API_PROVIDER, error=msg)
            _record_provider_health(db, ODDS_API_PROVIDER, enabled=True, error=msg)
            return None
    if last_error:
        _record(last_refresh_error=last_error)
    return None


def _filter_events_for_card_date(events: list[dict[str, Any]], game_date: str) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        event_date = _event_card_date(event)
        if event_date is None or event_date == game_date:
            filtered.append(event)
    return filtered


def _event_card_date(event: dict[str, Any]) -> str | None:
    raw = (
        event.get("date")
        or event.get("startTime")
        or event.get("startsAt")
        or (event.get("status") or {}).get("startsAt")
    )
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return str(raw)[:10] if len(str(raw)) >= 10 else None
    if parsed.tzinfo is None:
        return parsed.date().isoformat()
    return parsed.astimezone(TZ_ARIZONA).date().isoformat()


def _parse_provider_error_reason(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    if status == 400:
        return "bad_request"
    if status == 401:
        return "unauthorized"
    if status == 403:
        return "forbidden"
    if status == 429:
        return "rate_limited"
    return "upstream_error"


async def refresh_mlb_odds_cache(
    db: Session,
    odds: OddsApiProvider,
    games: list[dict[str, Any]],
    *,
    game_date: str,
    force: bool = False,
) -> RefreshResult:
    """Pull MLB events + per-event odds and persist them to odds_snapshots.

    Coalesces concurrent callers via a non-blocking threading.Lock — if another
    caller already holds the refresh lock we return immediately with
    refreshed=False (the cache rows the other caller is writing will be
    available on next read).

    Set `force=True` to bypass the freshness check (operator-triggered refresh,
    e.g. POST /mlb/edges/run).
    """
    result = RefreshResult(refreshed=False, reason="not attempted", game_date=game_date)

    primary_totals = 0
    primary_props = 0
    backup_totals = 0
    backup_props = 0
    primary_events = 0
    backup_events = 0
    used_providers: set[str] = set()
    events: list[dict[str, Any]] = []

    sgo_provider: SportsGameOddsProvider | None = None
    sgo_events_raw: list[dict[str, Any]] | None = None
    sgo_events_norm: list[dict[str, Any]] | None = None
    sgo_events_by_id: dict[str, dict[str, Any]] = {}
    sgo_match_by_game: dict[int, dict[str, Any]] = {}
    sgo_loaded = False

    async def _load_sgo_events() -> list[dict[str, Any]] | None:
        nonlocal sgo_provider, sgo_events_raw, sgo_events_norm, sgo_events_by_id, sgo_loaded
        if sgo_loaded:
            return sgo_events_norm
        sgo_loaded = True
        cooldown_state = _provider_on_cooldown(db, SGO_PROVIDER)
        if cooldown_state is not None:
            until = cooldown_state.cooldown_until.isoformat() if cooldown_state.cooldown_until else None
            msg = f"SportsGameOdds rate-limited until {until}"
            result.errors.append(msg)
            result.provider_cooldowns[SGO_PROVIDER] = until
            _record_provider_event(provider=SGO_PROVIDER, enabled=True, error=msg)
            return None
        sgo_provider = _sgo_provider()
        if not sgo_provider:
            _record_provider_health(db, SGO_PROVIDER, enabled=False)
            return None
        try:
            sgo_events_raw = await sgo_provider.fetch_mlb_events_with_odds(game_date)
        except SportsGameOddsRateLimited as exc:
            msg = f"sgo events 429: {exc}"
            result.errors.append(msg)
            result.rate_limited += 1
            cooldown_until = getattr(exc, "cooldown_until", None)
            if cooldown_until:
                result.provider_cooldowns[SGO_PROVIDER] = cooldown_until.isoformat()
            _record(rate_limited_count=1, last_rate_limited_at=datetime.utcnow())
            _record_provider_event(provider=SGO_PROVIDER, error=msg)
            _record_provider_health(
                db,
                SGO_PROVIDER,
                enabled=True,
                error=msg,
                status_code=429,
                cooldown_until=cooldown_until,
            )
            return None
        except SportsGameOddsError as exc:
            msg = f"sgo events failed: {type(exc).__name__}: {exc}"
            result.errors.append(msg)
            _record_provider_event(provider=SGO_PROVIDER, error=msg)
            _record_provider_health(
                db,
                SGO_PROVIDER,
                enabled=True,
                error=msg,
                status_code=getattr(exc, "status_code", None),
            )
            return None
        except Exception as exc:  # noqa: BLE001
            msg = f"sgo events failed: {type(exc).__name__}: {sanitize_text(exc)}"
            result.errors.append(msg)
            _record_provider_event(provider=SGO_PROVIDER, error=msg)
            _record_provider_health(db, SGO_PROVIDER, enabled=True, error=msg)
            return None
        sgo_events_raw = sgo_events_raw or []
        sgo_events_norm = [sgo_provider.normalize_event(e) for e in sgo_events_raw]
        sgo_events_by_id = {str(e.get("eventID") or ""): e for e in sgo_events_raw}
        _record_provider_event(
            provider=SGO_PROVIDER,
            events_fetched=len(sgo_events_norm),
            last_success=True,
        )
        _record_provider_health(
            db,
            SGO_PROVIDER,
            enabled=True,
            success=True,
            events_fetched=len(sgo_events_norm),
        )
        return sgo_events_norm

    def _payload_empty(payload: dict[str, Any] | None) -> bool:
        if not payload or not isinstance(payload, dict):
            return True
        books = payload.get("bookmakers") or {}
        return not isinstance(books, dict) or not books

    def _payload_market_flags(payload: dict[str, Any] | None) -> tuple[bool, bool]:
        from app.services.mlb_odds_analysis import analyze_game_totals
        from app.services.mlb_prop_odds import normalize_pitcher_strikeout_props

        if not payload:
            return False, False
        totals = analyze_game_totals(payload).get("book_count") or 0
        props = len(normalize_pitcher_strikeout_props(payload))
        return totals > 0, props > 0

    async def _sgo_payload_for_game(
        game_pk: int,
        *,
        override_event_id: str | None = None,
    ) -> dict[str, Any] | None:
        nonlocal sgo_match_by_game
        events_norm = await _load_sgo_events()
        if not events_norm or not sgo_provider:
            return None
        if not sgo_match_by_game:
            matches, _unmatched = match_all_games(games, events_norm)
            for match in matches:
                if match.matched_event_id:
                    event = sgo_events_by_id.get(str(match.matched_event_id))
                    if event:
                        sgo_match_by_game[match.game_pk] = event
        event = sgo_match_by_game.get(game_pk)
        if not event:
            return None
        return sgo_provider.normalize_event_odds(event, override_event_id=override_event_id)

    if not _refresh_lock.acquire(blocking=False):
        result.reason = "another refresh in progress; reusing cache"
        _record(last_refresh_skipped_reason=result.reason)
        return result
    try:
        if not force and _events_list_fresh(db, game_date):
            result.reason = "cache fresh; skipping refresh"
            _record(last_refresh_skipped_reason=result.reason)
            return result

        # ---- 1. events list ----
        events_payload = await _fetch_odds_api_events_with_fallback(
            db,
            odds,
            result,
            game_date=game_date,
        )

        events_source = ODDS_API_PROVIDER
        if not isinstance(events_payload, list) or not events_payload:
            # Couldn't refresh events; if a stale list exists we proceed with
            # IT for matching odds-per-event below. Otherwise we have nothing
            # actionable and bail.
            sgo_events = await _load_sgo_events()
            if sgo_events:
                events = sgo_events
                events_source = SGO_PROVIDER
                result.provider = SGO_PROVIDER
                backup_events = len(events)
                result.reason = "events list refreshed (SportsGameOdds fallback)"
            else:
                stale = get_cached_events_list(db, game_date=game_date, fallback_stale=True)
                if not stale:
                    result.reason = "events fetch failed and no cached events"
                    _record(last_refresh_skipped_reason=result.reason)
                    db.commit()
                    return result
                events = stale
                result.reason = "events fetch failed; serving stale events"
        else:
            events = events_payload
            primary_events = len(events)
            _upsert_snapshot(
                db,
                sport=ODDS_MLB_SPORT,
                event_id=_events_list_key(game_date),
                market_type=MARKET_TYPE_EVENTS_LIST,
                payload=events,
                ttl=MLB_EVENTS_LIST_TTL,
            )
            result.events_cached = 1
            result.refreshed = True
            result.reason = "events list refreshed"
        if events_source == SGO_PROVIDER:
            _upsert_snapshot(
                db,
                sport=ODDS_MLB_SPORT,
                event_id=_events_list_key(game_date),
                market_type=MARKET_TYPE_EVENTS_LIST,
                payload=events,
                ttl=MLB_EVENTS_LIST_TTL,
            )
            result.events_cached = 1
            result.refreshed = True
        result.events_fetched = len(events)

        # ---- 2. per-event odds ----
        match_results, _unmatched_events = match_all_games(games, events)
        matched = [m for m in match_results if m.matched_event_id]
        result.matched_games = len(matched)

        for match in matched:
            event_id = match.matched_event_id or ""
            payload: dict[str, Any] | None = None
            provider_used = ODDS_API_PROVIDER
            if events_source == SGO_PROVIDER:
                if not sgo_provider:
                    continue
                raw_event = sgo_events_by_id.get(event_id)
                if not raw_event:
                    continue
                payload = sgo_provider.normalize_event_odds(raw_event)
                provider_used = SGO_PROVIDER
            else:
                try:
                    payload = await odds.odds(event_id)
                    _record(live_api_calls=1)
                    result.odds_calls += 1
                except OddsApiRateLimited as exc:
                    _record(rate_limited_count=1, last_rate_limited_at=datetime.utcnow())
                    msg = f"odds({event_id}) 429: {exc}"
                    result.errors.append(msg)
                    result.rate_limited += 1
                    _record_provider_event(provider=ODDS_API_PROVIDER, error=msg)
                    _record_provider_health(
                        db,
                        ODDS_API_PROVIDER,
                        enabled=True,
                        error=msg,
                        status_code=429,
                    )
                    payload = None
                except OddsApiError as exc:
                    msg = f"odds({event_id}) failed: {type(exc).__name__}: {exc}"
                    _record(refresh_errors=1, last_refresh_error=msg)
                    result.errors.append(msg)
                    _record_provider_event(
                        provider=ODDS_API_PROVIDER,
                        error=msg,
                    )
                    _record_provider_health(
                        db,
                        ODDS_API_PROVIDER,
                        enabled=True,
                        error=msg,
                        status_code=getattr(exc, "status_code", None),
                    )
                    payload = None
                except Exception as exc:  # noqa: BLE001
                    msg = f"odds({event_id}) failed: {type(exc).__name__}: {sanitize_text(exc)}"
                    _record(refresh_errors=1, last_refresh_error=msg)
                    result.errors.append(msg)
                    _record_provider_event(provider=ODDS_API_PROVIDER, error=msg)
                    _record_provider_health(db, ODDS_API_PROVIDER, enabled=True, error=msg)
                    payload = None

                totals_ok, props_ok = _payload_market_flags(payload)
                if _payload_empty(payload) or not totals_ok or not props_ok:
                    sgo_payload = await _sgo_payload_for_game(match.game_pk, override_event_id=event_id)
                    if sgo_payload:
                        payload = sgo_payload
                        provider_used = SGO_PROVIDER

            if not payload:
                continue

            totals_ok, props_ok = _payload_market_flags(payload)
            if provider_used == ODDS_API_PROVIDER:
                primary_totals += int(totals_ok)
                primary_props += int(props_ok)
            else:
                backup_totals += int(totals_ok)
                backup_props += int(props_ok)
            used_providers.add(provider_used)

            _upsert_snapshot(
                db,
                sport=ODDS_MLB_SPORT,
                event_id=event_id,
                market_type=MARKET_TYPE_EVENT_ODDS,
                payload=payload,
                # One TTL for the raw payload; the totals/props lookups use
                # the same cached row, so we pick the shorter MLB_TOTALS_TTL
                # to keep totals fresh enough.
                ttl=MLB_TOTALS_TTL,
            )
            result.odds_cached += 1

        db.commit()
        if events_source == SGO_PROVIDER:
            backup_events = len(events)
        _record_provider_event(
            provider=ODDS_API_PROVIDER,
            events_fetched=primary_events,
            totals_found=primary_totals,
            pitcher_props_found=primary_props,
        )
        _record_provider_event(
            provider=SGO_PROVIDER,
            events_fetched=backup_events,
            totals_found=backup_totals,
            pitcher_props_found=backup_props,
        )
        if used_providers == {SGO_PROVIDER}:
            _record_last_provider_used(SGO_PROVIDER)
        elif used_providers == {ODDS_API_PROVIDER}:
            _record_last_provider_used(ODDS_API_PROVIDER)
        elif used_providers:
            _record_last_provider_used("Mixed")
        _record(
            last_refresh_at=datetime.utcnow(),
            last_refresh_event_count=result.events_fetched,
            last_refresh_skipped_reason=None,
        )
        if result.errors and not result.refreshed:
            _mark_refresh_failed_stale(db, game_date=game_date, events=events)
            db.commit()
        logger.info(
            "Odds cache refresh: date=%s events=%d matched=%d odds_calls=%d cached=%d rate_limited=%d",
            game_date, result.events_fetched, result.matched_games,
            result.odds_calls, result.odds_cached, result.rate_limited,
        )
        return result
    finally:
        _refresh_lock.release()


def _mark_refresh_failed_stale(
    db: Session,
    *,
    game_date: str,
    events: list[dict[str, Any]],
) -> None:
    """Expire the current slate cache rows without deleting them."""
    now = datetime.utcnow() - timedelta(seconds=1)
    db.execute(
        update(OddsSnapshot)
        .where(
            OddsSnapshot.sport == ODDS_MLB_SPORT,
            OddsSnapshot.event_id == _events_list_key(game_date),
            OddsSnapshot.market_type == MARKET_TYPE_EVENTS_LIST,
        )
        .values(expires_at=now)
    )
    for event in events:
        event_id = str((event or {}).get("id") or "").strip()
        if not event_id:
            continue
        db.execute(
            update(OddsSnapshot)
            .where(
                OddsSnapshot.sport == ODDS_MLB_SPORT,
                OddsSnapshot.event_id == event_id,
                OddsSnapshot.market_type == MARKET_TYPE_EVENT_ODDS,
            )
            .values(expires_at=now)
        )


# --- mapping helpers --------------------------------------------------------


def matches_for_games(
    db: Session,
    games: list[dict[str, Any]],
    *,
    game_date: str,
    fallback_stale: bool = True,
) -> tuple[list[MatchResult], list[dict[str, Any]]]:
    """Match games to the cached events list. Empty list if no cache exists."""
    events = get_cached_events_list(db, game_date=game_date, fallback_stale=fallback_stale)
    if not events:
        return [], []
    return match_all_games(games, events)


# --- debug snapshot ---------------------------------------------------------


def cache_summary(db: Session) -> dict[str, Any]:
    """Inventory + freshness summary for /mlb/debug/odds-cache."""
    rows = list(db.scalars(select(OddsSnapshot)))
    now = datetime.utcnow()
    fresh = sum(1 for r in rows if r.expires_at > now)
    stale = len(rows) - fresh
    by_type: dict[str, int] = {}
    for row in rows:
        by_type[row.market_type] = by_type.get(row.market_type, 0) + 1
    oldest = min((r.fetched_at for r in rows), default=None)
    newest = max((r.fetched_at for r in rows), default=None)
    health = get_odds_cache_health()
    return {
        "rows": len(rows),
        "fresh": fresh,
        "stale": stale,
        "by_market_type": by_type,
        "oldest_fetched_at": oldest.isoformat() if oldest else None,
        "newest_fetched_at": newest.isoformat() if newest else None,
        "metrics": {
            "live_api_calls": health.live_api_calls,
            "cache_hits": health.cache_hits,
            "cache_misses": health.cache_misses,
            "avoided_api_calls": health.avoided_api_calls,
            "stale_fallbacks": health.stale_fallbacks,
            "rate_limited_count": health.rate_limited_count,
            "last_rate_limited_at": (
                health.last_rate_limited_at.isoformat()
                if health.last_rate_limited_at else None
            ),
            "refresh_errors": health.refresh_errors,
            "last_refresh_error": health.last_refresh_error,
            "last_refresh_at": (
                health.last_refresh_at.isoformat()
                if health.last_refresh_at else None
            ),
            "last_refresh_event_count": health.last_refresh_event_count,
            "last_refresh_skipped_reason": health.last_refresh_skipped_reason,
            "hourly_call_rate": health.hourly_call_rate(),
        },
        "provider_health": list(provider_health_snapshot(db).values()),
    }
