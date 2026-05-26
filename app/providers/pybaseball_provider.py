"""pybaseball provider for Statcast and historical baseball trends."""

from __future__ import annotations

import asyncio
import math
from typing import Any


class PyBaseballError(RuntimeError):
    """Raised for pybaseball configuration or upstream failures."""


class PyBaseballProvider:
    """Async wrapper around pybaseball's synchronous data pulls."""

    def __init__(self, pybaseball_module: Any | None = None) -> None:
        if pybaseball_module is not None:
            self._pybaseball = pybaseball_module
            return
        try:
            import pybaseball  # type: ignore[import-not-found]
        except ImportError as exc:
            raise PyBaseballError(
                "pybaseball is not installed. Run `pip install -r requirements.txt`."
            ) from exc
        self._pybaseball = pybaseball

    async def _call(self, func_name: str, *args: Any, **kwargs: Any) -> Any:
        func = getattr(self._pybaseball, func_name, None)
        if func is None:
            raise PyBaseballError(f"pybaseball.{func_name} is unavailable")
        return await asyncio.to_thread(func, *args, **kwargs)

    async def player_lookup(self, first: str, last: str) -> list[dict[str, Any]]:
        df = await self._call("playerid_lookup", last, first)
        return dataframe_records(df)

    async def statcast(
        self,
        start_dt: str,
        end_dt: str | None = None,
        *,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        df = await self._call("statcast", start_dt, end_dt or start_dt)
        return dataframe_records(df, limit=limit)

    async def pitcher_statcast(
        self,
        pitcher_id: int,
        start_dt: str,
        end_dt: str | None = None,
        *,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        df = await self._call("statcast_pitcher", start_dt, end_dt or start_dt, pitcher_id)
        return dataframe_records(df, limit=limit)

    async def batter_statcast(
        self,
        batter_id: int,
        start_dt: str,
        end_dt: str | None = None,
        *,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        df = await self._call("statcast_batter", start_dt, end_dt or start_dt, batter_id)
        return dataframe_records(df, limit=limit)

    async def pitching_stats(self, start_season: int, end_season: int | None = None) -> list[dict[str, Any]]:
        df = await self._call("pitching_stats", start_season, end_season or start_season)
        return dataframe_records(df)

    async def batting_stats(self, start_season: int, end_season: int | None = None) -> list[dict[str, Any]]:
        df = await self._call("batting_stats", start_season, end_season or start_season)
        return dataframe_records(df)

    async def player_splits(
        self,
        player_id: str,
        *,
        season: int | None = None,
        pitching_splits: bool = False,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        df = await self._call(
            "get_splits",
            player_id,
            year=season,
            pitching_splits=pitching_splits,
        )
        if isinstance(df, tuple):
            df = df[0]
        return dataframe_records(df, limit=limit)


def dataframe_records(value: Any, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Convert pandas-like DataFrames into JSON-safe record dictionaries."""
    if hasattr(value, "head") and limit is not None:
        value = value.head(limit)
    if hasattr(value, "replace"):
        try:
            value = value.replace({float("inf"): None, float("-inf"): None})
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "where") and hasattr(value, "notnull"):
        try:
            value = value.where(value.notnull(), None)
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "to_dict"):
        records = value.to_dict(orient="records")
    elif isinstance(value, list):
        records = value
    else:
        return []
    return [_json_safe_record(row) for row in records if isinstance(row, dict)]


def _json_safe_record(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            out[str(key)] = None
        elif hasattr(value, "isoformat"):
            out[str(key)] = value.isoformat()
        else:
            out[str(key)] = value
    return out
