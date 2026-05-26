from __future__ import annotations

import math

import pandas as pd
import pytest

from app.providers.pybaseball_provider import PyBaseballProvider, dataframe_records


class FakePyBaseball:
    def __init__(self) -> None:
        self.calls = []

    def statcast_pitcher(self, start_dt, end_dt, pitcher_id):
        self.calls.append(("statcast_pitcher", start_dt, end_dt, pitcher_id))
        return pd.DataFrame(
            [
                {"pitch_type": "FF", "release_speed": 97.1, "game_date": pd.Timestamp("2026-05-25")},
                {"pitch_type": "SL", "release_speed": math.nan, "game_date": pd.Timestamp("2026-05-25")},
            ]
        )

    def get_splits(self, player_id, year=None, pitching_splits=False):
        self.calls.append(("get_splits", player_id, year, pitching_splits))
        return pd.DataFrame([{"Split": "vs RHP", "PA": 100, "OPS": 0.812}])


def test_dataframe_records_is_json_safe():
    rows = dataframe_records(pd.DataFrame([{"a": math.nan, "b": pd.Timestamp("2026-05-25")}]))

    assert rows == [{"a": None, "b": "2026-05-25T00:00:00"}]


@pytest.mark.asyncio
async def test_pitcher_statcast_uses_pybaseball_function():
    fake = FakePyBaseball()
    provider = PyBaseballProvider(fake)

    rows = await provider.pitcher_statcast(123, "2026-05-01", "2026-05-25")

    assert rows[0]["pitch_type"] == "FF"
    assert rows[1]["release_speed"] is None
    assert fake.calls[0] == ("statcast_pitcher", "2026-05-01", "2026-05-25", 123)


@pytest.mark.asyncio
async def test_player_splits_uses_get_splits():
    fake = FakePyBaseball()
    provider = PyBaseballProvider(fake)

    rows = await provider.player_splits("judgeaa01", season=2026)

    assert rows[0]["Split"] == "vs RHP"
    assert fake.calls[0] == ("get_splits", "judgeaa01", 2026, False)
