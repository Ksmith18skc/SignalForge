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
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models import OddsSnapshot
from app.providers.odds_api import OddsApiError, OddsApiProvider, OddsApiRateLimited
from app.services.mlb_odds_matching import MatchResult, match_all_games

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

ODDS_MLB_SPORT = "baseball"
ODDS_MLB_LEAGUE = "MLB"

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


_health = OddsCacheHealth()
_metrics_lock = threading.Lock()
# One refresh at a time, full stop. Threading.Lock (not asyncio.Lock) so
# concurrent callers on different event loops are also serialized.
_refresh_lock = threading.Lock()


def get_odds_cache_health() -> OddsCacheHealth:
    with _metrics_lock:
        return OddsCacheHealth(**_health.__dict__)


def reset_metrics() -> None:
    """Test helper."""
    global _health
    with _metrics_lock:
        _health = OddsCacheHealth()


def _record(**deltas: Any) -> None:
    with _metrics_lock:
        for key, value in deltas.items():
            if isinstance(value, (int, float)):
                setattr(_health, key, getattr(_health, key) + value)
            else:
                setattr(_health, key, value)


# --- read paths -------------------------------------------------------------


def _events_list_key(game_date: str) -> str:
    """Synthetic event_id for the per-day events list row."""
    return f"_events_{game_date}"


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
            "errors": list(self.errors),
        }


def _events_list_fresh(db: Session, game_date: str) -> bool:
    snap = _load_snapshot(
        db, sport=ODDS_MLB_SPORT, event_id=_events_list_key(game_date),
        market_type=MARKET_TYPE_EVENTS_LIST,
    )
    return _snapshot_is_fresh(snap)


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
        try:
            events_payload = await odds.events(
                ODDS_MLB_SPORT,
                league=ODDS_MLB_LEAGUE,
                date_from=game_date,
                date_to=game_date,
            )
            _record(live_api_calls=1)
        except OddsApiRateLimited as exc:
            _record(rate_limited_count=1, last_rate_limited_at=datetime.utcnow())
            result.errors.append(f"events 429: {exc}")
            result.rate_limited += 1
            events_payload = None
        except Exception as exc:  # noqa: BLE001
            _record(refresh_errors=1, last_refresh_error=f"events: {exc}")
            result.errors.append(f"events failed: {type(exc).__name__}: {exc}")
            events_payload = None

        if not isinstance(events_payload, list):
            # Couldn't refresh events; if a stale list exists we proceed with
            # IT for matching odds-per-event below. Otherwise we have nothing
            # actionable and bail.
            stale = get_cached_events_list(db, game_date=game_date, fallback_stale=True)
            if not stale:
                result.reason = "events fetch failed and no cached events"
                _record(last_refresh_skipped_reason=result.reason)
                return result
            events = stale
            result.reason = "events fetch failed; serving stale events"
        else:
            events = events_payload
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
        result.events_fetched = len(events)

        # ---- 2. per-event odds ----
        match_results, _unmatched_events = match_all_games(games, events)
        matched = [m for m in match_results if m.matched_event_id]
        result.matched_games = len(matched)

        for match in matched:
            event_id = match.matched_event_id or ""
            try:
                payload = await odds.odds(event_id)
                _record(live_api_calls=1)
                result.odds_calls += 1
            except OddsApiRateLimited as exc:
                _record(rate_limited_count=1, last_rate_limited_at=datetime.utcnow())
                result.errors.append(f"odds({event_id}) 429: {exc}")
                result.rate_limited += 1
                # Leave whatever stale row exists — readers will fall back.
                continue
            except Exception as exc:  # noqa: BLE001
                _record(refresh_errors=1, last_refresh_error=f"odds({event_id}): {exc}")
                result.errors.append(f"odds({event_id}) failed: {type(exc).__name__}: {exc}")
                continue
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
        _record(
            last_refresh_at=datetime.utcnow(),
            last_refresh_event_count=result.events_fetched,
            last_refresh_skipped_reason=None,
        )
        if not result.refreshed and result.odds_cached:
            result.refreshed = True
        logger.info(
            "Odds cache refresh: date=%s events=%d matched=%d odds_calls=%d cached=%d rate_limited=%d",
            game_date, result.events_fetched, result.matched_games,
            result.odds_calls, result.odds_cached, result.rate_limited,
        )
        return result
    finally:
        _refresh_lock.release()


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
    }
