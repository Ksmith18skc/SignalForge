from __future__ import annotations

import pytest

from app.providers.mlb_stats_api import MlbStatsApiProvider


class FakeStatsApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get(self, endpoint: str, params: dict[str, object]) -> dict[str, object]:
        self.calls.append((endpoint, params))
        if endpoint == "schedule":
            return {
                "dates": [
                    {
                        "games": [
                            {
                                "gamePk": 123,
                                "gameDate": "2026-05-25T23:05:00Z",
                                "status": {
                                    "abstractGameState": "Live",
                                    "detailedState": "In Progress",
                                },
                                "teams": {
                                    "away": {
                                        "score": 3,
                                        "team": {"name": "New York Yankees"},
                                        "probablePitcher": {"fullName": "Away Starter"},
                                    },
                                    "home": {
                                        "score": 4,
                                        "team": {"name": "Kansas City Royals"},
                                        "probablePitcher": {"fullName": "Home Starter"},
                                    },
                                },
                            }
                        ]
                    }
                ]
            }
        if endpoint == "game_boxscore":
            return {
                "teams": {
                    "away": {
                        "team": {"name": "Away"},
                        "players": {
                            "ID1": {
                                "battingOrder": "200",
                                "person": {"id": 1, "fullName": "Second Batter"},
                                "position": {"abbreviation": "2B"},
                            },
                            "ID2": {
                                "battingOrder": "100",
                                "person": {"id": 2, "fullName": "Leadoff"},
                                "position": {"abbreviation": "CF"},
                            },
                        },
                    },
                    "home": {"team": {"name": "Home"}, "players": {}},
                }
            }
        return {"endpoint": endpoint, "params": params}


@pytest.mark.asyncio
async def test_live_scores_summarizes_schedule_games():
    fake = FakeStatsApi()
    provider = MlbStatsApiProvider(fake)

    payload = await provider.live_scores(game_date="2026-05-25")

    assert payload["date"] == "2026-05-25"
    assert payload["games"][0]["game_pk"] == 123
    assert payload["games"][0]["status"] == "In Progress"
    assert payload["games"][0]["away_score"] == 3
    assert payload["games"][0]["home_probable_pitcher"] == "Home Starter"
    assert fake.calls[0][0] == "schedule"
    assert fake.calls[0][1]["sportId"] == 1


@pytest.mark.asyncio
async def test_lineups_extracts_batting_order_from_boxscore():
    provider = MlbStatsApiProvider(FakeStatsApi())

    payload = await provider.lineups(123)

    away = payload["away"]["batting_order"]
    assert [row["name"] for row in away] == ["Leadoff", "Second Batter"]
    assert away[0]["position"] == "CF"
