"""Centralized Odds-API cache: reuse, stale fallback, 429 recovery, expiration.

Pins the behavior that all MLB consumers serve from one shared Postgres-backed
cache so we never burn the free-plan rate limit on duplicate fetches.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import select

from app.models import OddsSnapshot
from app.providers.odds_api import OddsApiRateLimited
from app.services import odds_cache
from app.services.odds_cache import (
    MARKET_TYPE_EVENT_ODDS,
    MARKET_TYPE_EVENTS_LIST,
    MLB_EVENTS_LIST_TTL,
    MLB_TOTALS_TTL,
    ODDS_MLB_SPORT,
    _events_list_key,
    _upsert_snapshot,
    cache_summary,
    get_cached_event_odds,
    get_cached_events_list,
    get_cached_mlb_totals,
    get_cached_pitcher_props,
    get_odds_cache_health,
    refresh_mlb_odds_cache,
    reset_metrics,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_metrics_between_tests() -> None:
    reset_metrics()


def _events() -> list[dict[str, Any]]:
    return [
        {"id": "ev_a", "home": "New York Yankees", "away": "Boston Red Sox"},
        {"id": "ev_b", "home": "Los Angeles Dodgers", "away": "San Francisco Giants"},
    ]


def _odds_for(event_id: str) -> dict[str, Any]:
    return {
        "id": event_id,
        "bookmakers": {
            "DraftKings": [
                {
                    "name": "Totals",
                    "updatedAt": "2026-05-25T18:00:00Z",
                    "odds": [{"hdp": 8.5, "over": 1.91, "under": 1.95}],
                }
            ],
        },
    }


def _games() -> list[dict[str, Any]]:
    return [
        {"game_pk": 1, "home_team": "New York Yankees", "away_team": "Boston Red Sox"},
        {"game_pk": 2, "home_team": "Los Angeles Dodgers", "away_team": "San Francisco Giants"},
    ]


class _StubOdds:
    """Tracks every call so tests can assert the cache is short-circuiting."""

    def __init__(
        self,
        events_payload: Any = None,
        odds_payload_factory=None,
        events_raises: Exception | None = None,
        odds_raises: Exception | None = None,
    ) -> None:
        self.events_payload = events_payload if events_payload is not None else _events()
        self.odds_payload_factory = odds_payload_factory or _odds_for
        self.events_raises = events_raises
        self.odds_raises = odds_raises
        self.events_calls = 0
        self.events_args: list[tuple[Any, dict[str, Any]]] = []
        self.odds_calls = 0

    async def events(self, sport, **kwargs):
        self.events_calls += 1
        self.events_args.append((sport, dict(kwargs)))
        if self.events_raises is not None:
            raise self.events_raises
        return self.events_payload

    async def odds(self, event_id):
        self.odds_calls += 1
        if self.odds_raises is not None:
            raise self.odds_raises
        return self.odds_payload_factory(event_id)


# --------------------------------------------------------------------------
# Refresh budget: at most 1 events call + N odds calls per slate.
# --------------------------------------------------------------------------


def test_refresh_uses_minimal_api_calls(db_session) -> None:
    odds = _StubOdds()

    result = asyncio.run(
        refresh_mlb_odds_cache(db_session, odds, _games(), game_date="2026-05-25")
    )

    # 1 events call + 1 odds call per matched game.
    assert odds.events_calls == 1
    assert odds.odds_calls == 2
    assert result.refreshed is True
    assert result.events_fetched == 2
    assert result.matched_games == 2
    assert result.odds_cached == 2

    # Cache is populated.
    cached_events = get_cached_events_list(db_session, game_date="2026-05-25")
    assert cached_events is not None and len(cached_events) == 2
    assert get_cached_event_odds(db_session, "ev_a") is not None


def test_refresh_requests_current_odds_api_mlb_params(db_session) -> None:
    odds = _StubOdds()

    asyncio.run(refresh_mlb_odds_cache(db_session, odds, _games(), game_date="2026-05-25"))

    assert odds.events_args == [
        (
            "mlb",
            {
                "league": None,
                "date_from": "2026-05-25T07:00:00Z",
                "date_to": "2026-05-26T06:59:59Z",
            },
        )
    ]


def test_consecutive_refreshes_within_ttl_reuse_cache(db_session) -> None:
    odds = _StubOdds()

    asyncio.run(refresh_mlb_odds_cache(db_session, odds, _games(), game_date="2026-05-25"))
    asyncio.run(refresh_mlb_odds_cache(db_session, odds, _games(), game_date="2026-05-25"))

    # Second refresh sees a fresh events row and bails — zero extra calls.
    assert odds.events_calls == 1
    assert odds.odds_calls == 2


def test_forced_refresh_bypasses_freshness(db_session) -> None:
    odds = _StubOdds()
    asyncio.run(refresh_mlb_odds_cache(db_session, odds, _games(), game_date="2026-05-25"))
    asyncio.run(
        refresh_mlb_odds_cache(
            db_session, odds, _games(), game_date="2026-05-25", force=True
        )
    )
    assert odds.events_calls == 2
    assert odds.odds_calls == 4


# --------------------------------------------------------------------------
# Stale fallback
# --------------------------------------------------------------------------


def test_get_cached_event_odds_returns_stale_within_grace(db_session) -> None:
    payload = _odds_for("ev_stale")
    snap = _upsert_snapshot(
        db_session,
        sport=ODDS_MLB_SPORT,
        event_id="ev_stale",
        market_type=MARKET_TYPE_EVENT_ODDS,
        payload=payload,
        ttl=timedelta(minutes=5),
    )
    # Make it visibly stale (10 minutes past expiry).
    snap.expires_at = datetime.utcnow() - timedelta(minutes=10)
    snap.fetched_at = datetime.utcnow() - timedelta(minutes=20)
    db_session.commit()

    cached = get_cached_event_odds(db_session, "ev_stale", fallback_stale=True)
    assert cached is not None and cached["id"] == "ev_stale"
    assert get_odds_cache_health().stale_fallbacks == 1

    none = get_cached_event_odds(db_session, "ev_stale", fallback_stale=False)
    assert none is None


def test_get_cached_event_odds_drops_when_past_grace(db_session) -> None:
    snap = _upsert_snapshot(
        db_session,
        sport=ODDS_MLB_SPORT,
        event_id="ev_too_old",
        market_type=MARKET_TYPE_EVENT_ODDS,
        payload=_odds_for("ev_too_old"),
        ttl=timedelta(minutes=5),
    )
    snap.fetched_at = datetime.utcnow() - timedelta(hours=6)
    snap.expires_at = datetime.utcnow() - timedelta(hours=5)
    db_session.commit()

    assert get_cached_event_odds(db_session, "ev_too_old") is None


# --------------------------------------------------------------------------
# 429 / rate-limit recovery
# --------------------------------------------------------------------------


def test_refresh_records_rate_limit_and_skips_event(db_session) -> None:
    # First populate one event so we can verify it survives a 429 on a refresh.
    odds = _StubOdds()
    asyncio.run(refresh_mlb_odds_cache(db_session, odds, _games(), game_date="2026-05-25"))

    # Now simulate a 429 on the next events fetch.
    rate_limited = OddsApiRateLimited("rate-limited", retry_after=1.0)
    odds_429 = _StubOdds(events_raises=rate_limited)

    result = asyncio.run(
        refresh_mlb_odds_cache(
            db_session, odds_429, _games(), game_date="2026-05-25", force=True
        )
    )

    assert result.rate_limited >= 1
    health = get_odds_cache_health()
    assert health.rate_limited_count >= 1
    assert health.last_rate_limited_at is not None
    # Cache from the earlier successful refresh is still readable.
    assert get_cached_event_odds(db_session, "ev_a") is not None


def test_refresh_falls_back_to_stale_events_when_429s(db_session) -> None:
    """Even if the events fetch 429s, we still try per-event odds for the
    games we know about from the previous slate cache."""
    # Seed events cache.
    asyncio.run(
        refresh_mlb_odds_cache(db_session, _StubOdds(), _games(), game_date="2026-05-25")
    )

    # New refresh: events 429s, odds calls succeed (mid-day re-fetch case).
    odds_429_events_only = _StubOdds(events_raises=OddsApiRateLimited("rl"))

    result = asyncio.run(
        refresh_mlb_odds_cache(
            db_session,
            odds_429_events_only,
            _games(),
            game_date="2026-05-25",
            force=True,
        )
    )

    # We still attempted per-event odds via the stale events list.
    assert odds_429_events_only.odds_calls == 2
    assert "events fetch failed" in result.reason or "stale" in result.reason


def test_provider_retries_429_then_raises_rate_limited(monkeypatch) -> None:
    """OddsApiProvider should retry transient 429s before raising."""
    from app.providers.odds_api import OddsApiProvider

    sleep_calls: list[float] = []

    async def _no_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr("app.providers.odds_api.asyncio.sleep", _no_sleep)

    class _Resp429:
        status_code = 429
        headers = {"Retry-After": "0.1"}

        def raise_for_status(self):
            raise AssertionError("should not be called for 429")

        def json(self):
            return {}

    class _Client:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, params=None):
            return _Resp429()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    provider = OddsApiProvider("key", "https://api.odds-api.io/v3", "DraftKings")

    with pytest.raises(OddsApiRateLimited) as exc:
        asyncio.run(provider._get("/events", {"sport": "baseball"}, max_retries=2))

    # 2 retries between 3 attempts → 2 sleeps.
    assert len(sleep_calls) == 2
    assert exc.value.retry_after == 0.1


# --------------------------------------------------------------------------
# Concurrency: lock coalesces simultaneous refresh attempts.
# --------------------------------------------------------------------------


def test_concurrent_refresh_attempts_coalesce(db_session, monkeypatch) -> None:
    """When the lock is held, a second caller returns immediately rather than
    issuing a duplicate upstream call."""
    # Pre-acquire the lock to simulate an in-flight refresh.
    assert odds_cache._refresh_lock.acquire(blocking=False) is True
    try:
        odds = _StubOdds()
        result = asyncio.run(
            refresh_mlb_odds_cache(
                db_session, odds, _games(), game_date="2026-05-25", force=True
            )
        )
        assert result.refreshed is False
        assert "another refresh in progress" in result.reason
        assert odds.events_calls == 0
        assert odds.odds_calls == 0
    finally:
        odds_cache._refresh_lock.release()


# --------------------------------------------------------------------------
# Parsed lookups
# --------------------------------------------------------------------------


def test_get_cached_mlb_totals_returns_analysis_from_cache(db_session) -> None:
    _upsert_snapshot(
        db_session,
        sport=ODDS_MLB_SPORT,
        event_id="ev_a",
        market_type=MARKET_TYPE_EVENT_ODDS,
        payload=_odds_for("ev_a"),
        ttl=MLB_TOTALS_TTL,
    )
    db_session.commit()
    analysis = get_cached_mlb_totals(db_session, event_id="ev_a")
    assert analysis["book_count"] == 1
    assert analysis["consensus_total_line"] == 8.5


def test_get_cached_pitcher_props_returns_empty_when_cache_missing(db_session) -> None:
    result = get_cached_pitcher_props(db_session, event_id="ev_unknown", pitcher_name="X")
    assert result["line"] is None
    assert "No pitcher strikeout props" in result["warnings"][0]


# --------------------------------------------------------------------------
# Expiration + cache_summary
# --------------------------------------------------------------------------


def test_cache_summary_buckets_fresh_vs_stale(db_session) -> None:
    fresh = _upsert_snapshot(
        db_session,
        sport=ODDS_MLB_SPORT,
        event_id="ev_fresh",
        market_type=MARKET_TYPE_EVENT_ODDS,
        payload={},
        ttl=timedelta(minutes=10),
    )
    stale = _upsert_snapshot(
        db_session,
        sport=ODDS_MLB_SPORT,
        event_id="ev_stale",
        market_type=MARKET_TYPE_EVENT_ODDS,
        payload={},
        ttl=timedelta(minutes=10),
    )
    stale.expires_at = datetime.utcnow() - timedelta(minutes=1)
    db_session.commit()

    summary = cache_summary(db_session)
    assert summary["rows"] >= 2
    assert summary["fresh"] >= 1
    assert summary["stale"] >= 1
    assert MARKET_TYPE_EVENT_ODDS in summary["by_market_type"]


def test_cache_summary_includes_metrics(db_session) -> None:
    odds = _StubOdds()
    asyncio.run(refresh_mlb_odds_cache(db_session, odds, _games(), game_date="2026-05-25"))
    # Cache hit
    get_cached_event_odds(db_session, "ev_a")

    summary = cache_summary(db_session)
    metrics = summary["metrics"]
    assert metrics["live_api_calls"] >= 3  # 1 events + 2 odds
    assert metrics["cache_hits"] >= 1
    assert metrics["avoided_api_calls"] >= 1


# --------------------------------------------------------------------------
# Snapshot replacement semantics
# --------------------------------------------------------------------------


def test_upsert_replaces_existing_snapshot(db_session) -> None:
    _upsert_snapshot(
        db_session,
        sport=ODDS_MLB_SPORT,
        event_id="ev_x",
        market_type=MARKET_TYPE_EVENT_ODDS,
        payload={"version": 1},
        ttl=timedelta(minutes=5),
    )
    _upsert_snapshot(
        db_session,
        sport=ODDS_MLB_SPORT,
        event_id="ev_x",
        market_type=MARKET_TYPE_EVENT_ODDS,
        payload={"version": 2},
        ttl=timedelta(minutes=5),
    )
    db_session.commit()
    rows = list(
        db_session.scalars(
            select(OddsSnapshot).where(OddsSnapshot.event_id == "ev_x")
        )
    )
    assert len(rows) == 1
    assert rows[0].payload == {"version": 2}
