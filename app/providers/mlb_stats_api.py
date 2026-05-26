"""MLB StatsAPI provider for game context, scores, lineups, and stats."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any


class MlbStatsApiError(RuntimeError):
    """Raised for MLB StatsAPI configuration or upstream failures."""


class MlbStatsApiProvider:
    """Async wrapper around the synchronous MLB-StatsAPI `statsapi` package."""

    def __init__(self, statsapi_module: Any | None = None) -> None:
        if statsapi_module is not None:
            self._statsapi = statsapi_module
            return
        try:
            import statsapi  # type: ignore[import-not-found]
        except ImportError as exc:
            raise MlbStatsApiError(
                "MLB-StatsAPI is not installed. Run `pip install -r requirements.txt`."
            ) from exc
        self._statsapi = statsapi

    async def _call(self, func_name: str, *args: Any, **kwargs: Any) -> Any:
        func = getattr(self._statsapi, func_name, None)
        if func is None:
            raise MlbStatsApiError(f"statsapi.{func_name} is unavailable")
        return await asyncio.to_thread(func, *args, **kwargs)

    async def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        return await self._call("get", endpoint, _clean_params(params or {}))

    async def schedule(
        self,
        *,
        game_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        team_id: int | None = None,
        season: int | None = None,
        hydrate: str | None = "probablePitcher(note),linescore",
    ) -> dict[str, Any]:
        params = {
            "sportId": 1,
            "date": game_date,
            "startDate": start_date,
            "endDate": end_date,
            "teamId": team_id,
            "season": season,
            "hydrate": hydrate,
        }
        return await self._get("schedule", params)

    async def game(self, game_pk: int) -> dict[str, Any]:
        return await self._get("game", {"gamePk": game_pk})

    async def linescore(self, game_pk: int) -> dict[str, Any]:
        return await self._get("game_linescore", {"gamePk": game_pk})

    async def boxscore(self, game_pk: int) -> dict[str, Any]:
        return await self._get("game_boxscore", {"gamePk": game_pk})

    async def probable_pitchers(self, *, game_date: str | None = None) -> dict[str, Any]:
        payload = await self.schedule(game_date=game_date or date.today().isoformat())
        return {
            "date": game_date or date.today().isoformat(),
            "games": [
                _probable_pitcher_summary(game)
                for game in _iter_schedule_games(payload)
            ],
        }

    async def live_scores(self, *, game_date: str | None = None) -> dict[str, Any]:
        payload = await self.schedule(game_date=game_date or date.today().isoformat())
        return {
            "date": game_date or date.today().isoformat(),
            "games": [_game_summary(game) for game in _iter_schedule_games(payload)],
        }

    async def lineups(self, game_pk: int) -> dict[str, Any]:
        payload = await self.boxscore(game_pk)
        teams = payload.get("teams") or {}
        return {
            "game_pk": game_pk,
            "away": _lineup_for_team(teams.get("away") or {}),
            "home": _lineup_for_team(teams.get("home") or {}),
            "raw": payload,
        }

    async def teams(self, *, season: int | None = None) -> dict[str, Any]:
        return await self._get("teams", {"sportId": 1, "season": season})

    async def team_stats(
        self,
        team_id: int,
        *,
        season: int,
        group: str = "hitting",
        stats: str = "season",
        game_type: str = "R",
    ) -> dict[str, Any]:
        return await self._get(
            "team_stats",
            {
                "teamId": team_id,
                "season": season,
                "group": group,
                "stats": stats,
                "gameType": game_type,
            },
        )

    async def player_stats(
        self,
        person_id: int,
        *,
        season: int,
        group: str = "hitting",
        stats: str = "season",
        game_type: str = "R",
    ) -> dict[str, Any]:
        return await self._get(
            "stats",
            {
                "personId": person_id,
                "season": season,
                "group": group,
                "stats": stats,
                "gameType": game_type,
                "sportIds": 1,
            },
        )

    async def historical_games(
        self,
        *,
        start_date: str,
        end_date: str,
        team_id: int | None = None,
    ) -> dict[str, Any]:
        return await self.schedule(start_date=start_date, end_date=end_date, team_id=team_id)


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if value not in (None, "", [])}


def _iter_schedule_games(payload: dict[str, Any]) -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = []
    for day in payload.get("dates") or []:
        if not isinstance(day, dict):
            continue
        for game in day.get("games") or []:
            if isinstance(game, dict):
                games.append(game)
    return games


def _team_name(team_node: dict[str, Any]) -> str | None:
    team = team_node.get("team") if isinstance(team_node, dict) else None
    if isinstance(team, dict):
        return team.get("name") or team.get("teamName")
    return None


def _pitcher_name(team_node: dict[str, Any]) -> str | None:
    pitcher = team_node.get("probablePitcher") if isinstance(team_node, dict) else None
    if isinstance(pitcher, dict):
        return pitcher.get("fullName") or pitcher.get("name")
    return None


def _score(game: dict[str, Any], side: str) -> int | None:
    teams = game.get("teams") or {}
    node = teams.get(side) or {}
    return node.get("score") if isinstance(node, dict) else None


def _game_summary(game: dict[str, Any]) -> dict[str, Any]:
    teams = game.get("teams") or {}
    status = game.get("status") or {}
    return {
        "game_pk": game.get("gamePk"),
        "game_date": game.get("gameDate"),
        "status": status.get("detailedState") or status.get("abstractGameState"),
        "abstract_state": status.get("abstractGameState"),
        "away_team": _team_name(teams.get("away") or {}),
        "home_team": _team_name(teams.get("home") or {}),
        "away_score": _score(game, "away"),
        "home_score": _score(game, "home"),
        "away_probable_pitcher": _pitcher_name(teams.get("away") or {}),
        "home_probable_pitcher": _pitcher_name(teams.get("home") or {}),
    }


def _probable_pitcher_summary(game: dict[str, Any]) -> dict[str, Any]:
    summary = _game_summary(game)
    return {
        "game_pk": summary["game_pk"],
        "game_date": summary["game_date"],
        "status": summary["status"],
        "away_team": summary["away_team"],
        "home_team": summary["home_team"],
        "away_probable_pitcher": summary["away_probable_pitcher"],
        "home_probable_pitcher": summary["home_probable_pitcher"],
    }


def _lineup_for_team(team_payload: dict[str, Any]) -> dict[str, Any]:
    players = team_payload.get("players") or {}
    starters: list[dict[str, Any]] = []
    for player_key, player in players.items():
        if not isinstance(player, dict):
            continue
        batting_order = player.get("battingOrder")
        if not batting_order:
            continue
        person = player.get("person") or {}
        position = player.get("position") or {}
        starters.append(
            {
                "player_key": player_key,
                "batting_order": int(batting_order),
                "person_id": person.get("id"),
                "name": person.get("fullName"),
                "position": position.get("abbreviation") or position.get("name"),
            }
        )
    starters.sort(key=lambda row: row["batting_order"])
    return {
        "team": (team_payload.get("team") or {}).get("name"),
        "batting_order": starters,
    }
