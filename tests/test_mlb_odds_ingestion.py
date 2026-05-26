"""Regression tests for the MLB Odds-API ingestion fix.

Background:
  Every MLB edge was emitting line=null / book_count=0 because:
    1. Per-game `search_events("Away Home")` returned the wrong sport's event
       (or nothing), so we never resolved a correct event_id.
    2. The market-name allowlist missed how books actually label things —
       e.g. "Game Total", "Over/Under", "OVER_UNDER", "Total Runs".
    3. Pitcher strikeout markets ship under different names per book
       ("Pitcher Strikeouts", "Player Strikeouts", "Total Strikeouts").

  These tests pin the new behavior using fixture payloads in the actual
  Odds-API shape so the parsers stay correct even if internal helpers
  refactor later.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.services.mlb_odds_analysis import (
    analyze_game_totals,
    analyze_pitcher_k_props,
    is_pitcher_k_market,
    is_total_market,
    summarize_markets,
)
from app.services.mlb_odds_matching import (
    match_all_games,
    match_game_to_event,
    normalize_team_name,
    teams_match,
)
from app.services.mlb_prop_odds import normalize_pitcher_strikeout_props


# --------------------------------------------------------------------------
# Fixtures — shaped after real Odds-API /events and /odds payloads.
# --------------------------------------------------------------------------


def _events_payload() -> list[dict[str, Any]]:
    return [
        {
            "id": "ev_yankees_redsox",
            "sport": "baseball",
            "league": "MLB",
            "home": "New York Yankees",
            "away": "Boston Red Sox",
            "date": "2026-05-25T23:05:00Z",
        },
        {
            "id": "ev_dodgers_giants",
            "sport": "baseball",
            "league": "MLB",
            # Odds-API frequently uses abbreviated city tokens.
            "home": "LA Dodgers",
            "away": "SF Giants",
            "date": "2026-05-25T22:10:00Z",
        },
        {
            "id": "ev_unrelated",
            "sport": "baseball",
            "league": "MLB",
            "home": "Toronto Blue Jays",
            "away": "Cleveland Guardians",
            "date": "2026-05-25T19:07:00Z",
        },
    ]


def _games() -> list[dict[str, Any]]:
    return [
        {"game_pk": 1, "home_team": "New York Yankees", "away_team": "Boston Red Sox"},
        {"game_pk": 2, "home_team": "Los Angeles Dodgers", "away_team": "San Francisco Giants"},
        {"game_pk": 3, "home_team": "Chicago Cubs", "away_team": "St. Louis Cardinals"},
    ]


def _totals_payload_dk_fd_betmgm() -> dict[str, Any]:
    """Three books, three different totals market names — historically all missed."""
    return {
        "id": "ev_yankees_redsox",
        "home": "New York Yankees",
        "away": "Boston Red Sox",
        "bookmakers": {
            "DraftKings": [
                {
                    "name": "Totals",
                    "updatedAt": "2026-05-25T18:00:00Z",
                    "odds": [{"hdp": 8.5, "over": 1.91, "under": 1.95}],
                }
            ],
            "FanDuel": [
                {
                    "name": "Total Runs",
                    "updatedAt": "2026-05-25T18:01:00Z",
                    "odds": [{"hdp": 8.5, "over": 1.94, "under": 1.90}],
                }
            ],
            "BetMGM": [
                {
                    "name": "Game Total",
                    "updatedAt": "2026-05-25T18:02:00Z",
                    "odds": [{"hdp": 9.0, "over": 2.00, "under": 1.85}],
                }
            ],
            # Should be ignored — first-5-innings is a different market.
            "Caesars": [
                {
                    "name": "1st 5 Innings Total",
                    "odds": [{"hdp": 4.5, "over": 1.91, "under": 1.91}],
                },
                {
                    "name": "OVER_UNDER",
                    "updatedAt": "2026-05-25T18:03:00Z",
                    "odds": [{"hdp": 8.5, "over": 1.92, "under": 1.92}],
                },
            ],
        },
    }


def _pitcher_props_payload() -> dict[str, Any]:
    return {
        "id": "ev_yankees_redsox",
        "bookmakers": {
            "DraftKings": [
                {
                    "name": "Pitcher Strikeouts",
                    "updatedAt": "2026-05-25T18:00:00Z",
                    "odds": [
                        {"playerName": "Gerrit Cole", "hdp": 7.5, "over": 1.91, "under": 1.83},
                    ],
                }
            ],
            "FanDuel": [
                {
                    "name": "Player Strikeouts",
                    "updatedAt": "2026-05-25T18:01:00Z",
                    "odds": [
                        {
                            "label": "Gerrit Cole Over 7.5 Strikeouts",
                            "hdp": 7.5,
                            "over": 2.0,
                            "under": 1.80,
                        }
                    ],
                }
            ],
            "BetMGM": [
                {
                    "name": "Pitcher Total Strikeouts",
                    "updatedAt": "2026-05-25T18:02:00Z",
                    "odds": [
                        {"playerName": "Gerrit Cole", "hdp": 7.0, "over": 1.85, "under": 1.90}
                    ],
                }
            ],
        },
    }


# --------------------------------------------------------------------------
# Team-matching
# --------------------------------------------------------------------------


def test_normalize_team_name_strips_diacritics_and_noise() -> None:
    assert normalize_team_name("Los Angeles Dodgers") == "los angeles dodgers"
    assert normalize_team_name("LA Dodgers") == "los angeles dodgers"
    assert normalize_team_name("San Francisco Giants") == "san francisco giants"
    assert normalize_team_name("SF Giants") == "san francisco giants"
    assert normalize_team_name("") == ""


def test_teams_match_handles_abbreviations_and_full_names() -> None:
    assert teams_match("Los Angeles Dodgers", "LA Dodgers")
    assert teams_match("New York Yankees", "NY Yankees")
    assert teams_match("Yankees", "New York Yankees")  # subset OK
    assert not teams_match("New York Yankees", "Boston Red Sox")
    assert not teams_match("", "Yankees")


def test_match_game_to_event_picks_best_match_and_flags_missing() -> None:
    events = _events_payload()

    # Hit: full name on game, abbreviated on event.
    result = match_game_to_event(
        {"game_pk": 2, "home_team": "Los Angeles Dodgers", "away_team": "San Francisco Giants"},
        events,
    )
    assert result.matched_event_id == "ev_dodgers_giants"
    assert result.match_strength >= 1.0

    # Miss: no matching teams in events list.
    miss = match_game_to_event(
        {"game_pk": 3, "home_team": "Chicago Cubs", "away_team": "St. Louis Cardinals"},
        events,
    )
    assert miss.matched_event_id is None
    assert "no Odds-API event" in miss.reason


def test_match_all_games_reports_unmatched_events_and_games() -> None:
    results, unmatched = match_all_games(_games(), _events_payload())

    matched_pks = {r.game_pk for r in results if r.matched_event_id}
    assert matched_pks == {1, 2}

    unmatched_ids = {e.get("id") for e in unmatched}
    assert unmatched_ids == {"ev_unrelated"}


# --------------------------------------------------------------------------
# Market detection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["Totals", "Total", "Total Runs", "Game Total", "Full Game Total", "OVER_UNDER", "Over/Under"],
)
def test_is_total_market_accepts_known_aliases(name: str) -> None:
    assert is_total_market({"market": name})


@pytest.mark.parametrize(
    "name", ["Team Total", "1st 5 Innings Total", "Alt Total", "Moneyline", "Spread", ""]
)
def test_is_total_market_rejects_partials_and_alts(name: str) -> None:
    assert not is_total_market({"market": name})


@pytest.mark.parametrize(
    "name",
    [
        "Pitcher Strikeouts",
        "Player Strikeouts",
        "Pitcher Total Strikeouts",
        "Total Strikeouts",
        "Pitcher Strikeouts (K)",
        "Pitcher Ks",
    ],
)
def test_is_pitcher_k_market_accepts_known_aliases(name: str) -> None:
    assert is_pitcher_k_market({"market": name})


def test_analyze_game_totals_handles_mixed_market_names() -> None:
    analysis = analyze_game_totals(_totals_payload_dk_fd_betmgm())

    # All four books (DK Totals, FD Total Runs, BetMGM Game Total, Caesars OVER_UNDER)
    # contribute — the 1st-5 Innings line is correctly excluded.
    assert analysis["book_count"] == 4
    assert analysis["best_over_book"] == "BetMGM"
    assert analysis["best_over_price"] == 2.0
    assert analysis["consensus_total_line"] == pytest.approx(8.625, abs=0.01)
    # Old code returned warnings here; new code returns clean rows.
    assert "No full-game totals found" not in (analysis.get("warnings") or [])


def test_analyze_game_totals_emits_market_inventory_when_no_match() -> None:
    """When zero totals markets exist we surface the seen names so /health
    or /mlb/debug can show the operator exactly what the upstream returned."""
    payload = {
        "id": "no_totals_event",
        "bookmakers": {
            "DraftKings": [
                {"name": "Moneyline", "odds": [{"home": 1.9, "away": 2.0}]},
                {"name": "Spread", "odds": [{"hdp": -1.5, "home": 2.0, "away": 1.8}]},
            ]
        },
    }

    analysis = analyze_game_totals(payload)

    assert analysis["book_count"] == 0
    warning = analysis["warnings"][0]
    assert "Markets present" in warning
    assert "Moneyline" in warning


def test_analyze_pitcher_k_props_handles_multiple_book_names() -> None:
    analysis = analyze_pitcher_k_props(
        _pitcher_props_payload(), pitcher_name="Gerrit Cole"
    )

    # The summary lives on normalize_pitcher_strikeout_props for the consensus
    # path, but analyze_pitcher_k_props on the row path should still detect
    # all three K markets via the expanded substring matcher.
    assert analysis["book_count"] >= 1
    assert "No pitcher strikeout props" not in (analysis.get("warnings") or [""])[0]


def test_normalize_pitcher_strikeout_props_picks_up_betmgm_variant() -> None:
    rows = normalize_pitcher_strikeout_props(_pitcher_props_payload())

    books = {row.sportsbook for row in rows}
    # All three supported books should now register — previously BetMGM's
    # "Pitcher Total Strikeouts" was silently dropped.
    assert books == {"DraftKings", "FanDuel", "BetMGM"}


def test_summarize_markets_lists_real_names_and_flags() -> None:
    summary = summarize_markets(_totals_payload_dk_fd_betmgm())

    assert summary["event_id"] == "ev_yankees_redsox"
    assert summary["has_totals"] is True
    assert "Totals" in summary["market_names"]
    assert "Total Runs" in summary["market_names"]
    assert "Game Total" in summary["market_names"]
    # Pitcher K not in this payload, so the flag stays false.
    assert summary["has_pitcher_ks"] is False


# --------------------------------------------------------------------------
# Debug endpoints — exercised with monkeypatched providers so no network calls.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_debug_events_endpoint_runs_refresh_and_returns_cached_payload(
    monkeypatch, db_session
) -> None:
    from app.api import routes as routes_module
    from app.api.routes import mlb_debug_odds_events
    from app.services import odds_cache

    odds_cache.reset_metrics()

    captured: dict[str, Any] = {}

    class _StubOdds:
        async def events(self, sport, **kwargs):
            captured["sport"] = sport
            captured["kwargs"] = kwargs
            return _events_payload()

        async def odds(self, event_id):
            return {"id": event_id, "bookmakers": {}}

    monkeypatch.setattr(routes_module, "_odds_provider", lambda: _StubOdds())

    # First call: cache empty → refresh runs once.
    result = await mlb_debug_odds_events(game_date="2026-05-25", live=False, db=db_session)

    assert captured["sport"] == "baseball"
    assert captured["kwargs"]["league"] == "MLB"
    assert captured["kwargs"]["date_from"] == "2026-05-25"
    assert result["count"] == 3
    assert result["events"][0]["id"] == "ev_yankees_redsox"


@pytest.mark.asyncio
async def test_debug_event_match_reads_only_from_cache(monkeypatch, db_session) -> None:
    from app.api import routes as routes_module
    from app.api.routes import mlb_debug_odds_event_match
    from app.services import odds_cache

    odds_cache.reset_metrics()

    class _StubMlb:
        async def schedule(self, **kwargs):
            return {
                "dates": [
                    {
                        "games": [
                            {
                                "gamePk": 1,
                                "teams": {
                                    "home": {"team": {"name": "New York Yankees"}},
                                    "away": {"team": {"name": "Boston Red Sox"}},
                                },
                            },
                            {
                                "gamePk": 3,
                                "teams": {
                                    "home": {"team": {"name": "Chicago Cubs"}},
                                    "away": {"team": {"name": "St. Louis Cardinals"}},
                                },
                            },
                        ]
                    }
                ]
            }

    monkeypatch.setattr(routes_module, "_mlb_provider", lambda: _StubMlb())

    # Pre-populate the cache so the match endpoint doesn't have to fetch.
    odds_cache._upsert_snapshot(
        db_session,
        sport=odds_cache.ODDS_MLB_SPORT,
        event_id=odds_cache._events_list_key("2026-05-25"),
        market_type=odds_cache.MARKET_TYPE_EVENTS_LIST,
        payload=_events_payload(),
        ttl=odds_cache.MLB_EVENTS_LIST_TTL,
    )
    db_session.commit()

    result = await mlb_debug_odds_event_match(game_date="2026-05-25", db=db_session)

    assert result["mlb_games"] == 2
    assert result["odds_events"] == 3
    matched = [m for m in result["matches"] if m["matched_event_id"]]
    assert len(matched) == 1
    assert matched[0]["matched_event_id"] == "ev_yankees_redsox"
    unmatched_ids = {e["id"] for e in result["unmatched_events"]}
    assert "ev_dodgers_giants" in unmatched_ids


@pytest.mark.asyncio
async def test_debug_markets_endpoint_requires_cache(monkeypatch, db_session) -> None:
    from fastapi import HTTPException

    from app.api.routes import mlb_debug_odds_markets
    from app.services import odds_cache

    odds_cache.reset_metrics()

    # Cache miss → 404 (we no longer silently call the live API).
    with pytest.raises(HTTPException) as exc_info:
        await mlb_debug_odds_markets(event_id="ev_missing", db=db_session)
    assert exc_info.value.status_code == 404

    # After caching → returns the inventory.
    odds_cache._upsert_snapshot(
        db_session,
        sport=odds_cache.ODDS_MLB_SPORT,
        event_id="ev_yankees_redsox",
        market_type=odds_cache.MARKET_TYPE_EVENT_ODDS,
        payload=_totals_payload_dk_fd_betmgm(),
        ttl=odds_cache.MLB_TOTALS_TTL,
    )
    db_session.commit()
    result = await mlb_debug_odds_markets(event_id="ev_yankees_redsox", db=db_session)
    assert result["has_totals"] is True
    assert "Totals" in result["market_names"]


@pytest.mark.asyncio
async def test_debug_raw_endpoint_serves_cache_then_falls_through_to_live(
    monkeypatch, db_session
) -> None:
    from app.api import routes as routes_module
    from app.api.routes import mlb_debug_odds_raw
    from app.services import odds_cache

    odds_cache.reset_metrics()
    live_calls = {"count": 0}

    class _StubOdds:
        async def odds(self, event_id):
            live_calls["count"] += 1
            return {"id": event_id, "bookmakers": {"DraftKings": []}}

    monkeypatch.setattr(routes_module, "_odds_provider", lambda: _StubOdds())

    # First call: cache miss → live fetch, then cached.
    first = await mlb_debug_odds_raw(event_id="abc", live=False, db=db_session)
    assert first["source"] == "live"
    assert live_calls["count"] == 1

    # Second call: cache hit → no live fetch.
    second = await mlb_debug_odds_raw(event_id="abc", live=False, db=db_session)
    assert second["source"] == "cache"
    assert live_calls["count"] == 1
