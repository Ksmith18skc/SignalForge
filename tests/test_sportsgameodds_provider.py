"""SportsGameOdds normalization and odds-cache fallback tests."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.providers.sportsgameodds import SportsGameOddsProvider
from app.services import odds_cache
from app.services.mlb_prop_odds import normalize_pitcher_strikeout_props
from app.services.odds_cache import refresh_mlb_odds_cache


def _sgo_event() -> dict[str, Any]:
    return {
        "eventID": "sgo_ev_a",
        "leagueID": "MLB",
        "sportID": "BASEBALL",
        "teams": {
            "home": {"names": {"long": "New York Yankees"}},
            "away": {"names": {"long": "Boston Red Sox"}},
        },
        "status": {"startsAt": "2026-05-25T19:05:00Z"},
        "players": {
            "PITCHER_1": {"name": "Gerrit Cole"},
        },
        "odds": {
            "runs-all-game-ou-over": {
                "statID": "runs",
                "statEntityID": "all",
                "periodID": "game",
                "betTypeID": "ou",
                "sideID": "over",
                "marketName": "Total Runs",
                "byBookmaker": {
                    "draftkings": {
                        "odds": "-110",
                        "overUnder": "8.5",
                        "lastUpdatedAt": "2026-05-25T18:00:00Z",
                        "available": True,
                    }
                },
            },
            "runs-all-game-ou-under": {
                "statID": "runs",
                "statEntityID": "all",
                "periodID": "game",
                "betTypeID": "ou",
                "sideID": "under",
                "marketName": "Total Runs",
                "byBookmaker": {
                    "draftkings": {
                        "odds": "-112",
                        "overUnder": "8.5",
                        "lastUpdatedAt": "2026-05-25T18:01:00Z",
                        "available": True,
                    }
                },
            },
            "strikeouts-PITCHER_1-game-ou-over": {
                "statID": "strikeouts",
                "statEntityID": "PITCHER_1",
                "playerID": "PITCHER_1",
                "periodID": "game",
                "betTypeID": "ou",
                "sideID": "over",
                "marketName": "Pitcher Strikeouts",
                "byBookmaker": {
                    "fanduel": {
                        "odds": "+105",
                        "overUnder": "5.5",
                        "lastUpdatedAt": "2026-05-25T18:02:00Z",
                        "available": True,
                    }
                },
            },
            "strikeouts-PITCHER_1-game-ou-under": {
                "statID": "strikeouts",
                "statEntityID": "PITCHER_1",
                "playerID": "PITCHER_1",
                "periodID": "game",
                "betTypeID": "ou",
                "sideID": "under",
                "marketName": "Pitcher Strikeouts",
                "byBookmaker": {
                    "fanduel": {
                        "odds": "-125",
                        "overUnder": "5.5",
                        "lastUpdatedAt": "2026-05-25T18:02:30Z",
                        "available": True,
                    }
                },
            },
        },
    }


def _events() -> list[dict[str, Any]]:
    return [
        {"id": "ev_a", "home": "New York Yankees", "away": "Boston Red Sox"},
    ]


def _games() -> list[dict[str, Any]]:
    return [
        {"game_pk": 1, "home_team": "New York Yankees", "away_team": "Boston Red Sox"},
    ]


def test_sgo_normalize_event_odds_includes_totals_and_pitcher_props() -> None:
    provider = SportsGameOddsProvider("key", "https://api.sportsgameodds.com/v2")

    payload = provider.normalize_event_odds(_sgo_event())

    assert payload["bookmakers"]
    totals = payload["bookmakers"]["DraftKings"][0]
    assert totals["name"] == "Totals"
    assert totals["odds"][0]["hdp"] == 8.5
    prop_lines = normalize_pitcher_strikeout_props(payload)
    assert prop_lines
    assert prop_lines[0].player_name == "Gerrit Cole"


def test_refresh_falls_back_to_sgo_for_missing_props(db_session, monkeypatch) -> None:
    odds_cache.reset_metrics()

    class _StubOdds:
        async def events(self, sport, **kwargs):
            return _events()

        async def odds(self, event_id):
            return {
                "id": event_id,
                "bookmakers": {
                    "DraftKings": [
                        {
                            "name": "Totals",
                            "updatedAt": "2026-05-25T18:00:00Z",
                            "odds": [{"hdp": 8.5, "over": 1.91, "under": 1.95}],
                        }
                    ]
                },
            }

    provider = SportsGameOddsProvider("key", "https://api.sportsgameodds.com/v2")

    async def _fake_get(path, params=None, max_retries=1):
        return {"success": True, "data": [_sgo_event()]}

    monkeypatch.setattr(provider, "_get", _fake_get)
    monkeypatch.setattr(odds_cache, "_sgo_provider", lambda: provider)

    result = asyncio.run(
        refresh_mlb_odds_cache(db_session, _StubOdds(), _games(), game_date="2026-05-25", force=True)
    )

    assert result.odds_cached == 1
    payload = odds_cache.get_cached_event_odds(db_session, "ev_a")
    assert payload
    assert payload.get("source") == "SportsGameOdds"
    props = odds_cache.get_cached_pitcher_props(db_session, event_id="ev_a", pitcher_name="Gerrit Cole")
    assert props.get("line") is not None
