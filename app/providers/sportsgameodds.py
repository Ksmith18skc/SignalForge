"""SportsGameOdds (SGO) provider for MLB odds fallback."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class SportsGameOddsError(RuntimeError):
    """Raised for SportsGameOdds configuration or upstream failures."""


class SportsGameOddsRateLimited(SportsGameOddsError):
    """Raised when SportsGameOdds returns HTTP 429."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class SportsGameOddsProvider:
    """Async client for https://api.sportsgameodds.com/v2."""

    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        timeout: float = 20.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _auth_headers(self) -> dict[str, str]:
        if not self._api_key:
            raise SportsGameOddsError("SIGNALFORGE_SGO_API_KEY is not configured")
        return {"x-api-key": self._api_key}

    async def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        max_retries: int = 1,
    ) -> Any:
        query = dict(params or {})
        delay = 1.0
        last_retry_after: float | None = None
        for attempt in range(max_retries + 1):
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    f"{self._base_url}{path}",
                    params=query,
                    headers=self._auth_headers(),
                )
            if response.status_code != 429:
                response.raise_for_status()
                return response.json()
            retry_after_raw = response.headers.get("Retry-After")
            try:
                last_retry_after = float(retry_after_raw) if retry_after_raw else None
            except (TypeError, ValueError):
                last_retry_after = None
            if attempt >= max_retries:
                break
            sleep_for = min(last_retry_after or delay, 8.0)
            logger.warning(
                "SportsGameOdds 429 on %s (attempt %d/%d) - sleeping %.1fs",
                path, attempt + 1, max_retries + 1, sleep_for,
            )
            await asyncio.sleep(sleep_for)
            delay *= 2
        raise SportsGameOddsRateLimited(
            f"SportsGameOdds returned 429 on {path} after {max_retries + 1} attempts",
            retry_after=last_retry_after,
        )

    async def fetch_mlb_events_with_odds(self, game_date: str) -> list[dict[str, Any]]:
        """Return MLB events with odds data filtered to the requested date."""
        payload = await self._get(
            "/events",
            {
                "leagueID": "MLB",
                "oddsAvailable": "true",
                "limit": 200,
            },
        )
        if not isinstance(payload, dict):
            raise SportsGameOddsError("Unexpected SportsGameOdds response")
        if payload.get("success") is False:
            raise SportsGameOddsError(str(payload.get("error") or "SGO request failed"))
        data = payload.get("data") or []
        if not isinstance(data, list):
            return []
        return [
            event for event in data
            if _event_date(event) == game_date
        ]

    def normalize_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Map a SportsGameOdds event to the Odds-API-compatible schema."""
        return {
            "id": str(event.get("eventID") or ""),
            "home": _team_name((event.get("teams") or {}).get("home")),
            "away": _team_name((event.get("teams") or {}).get("away")),
            "date": _event_start_time(event),
            "sport": "baseball",
            "league": "MLB",
        }

    def normalize_event_odds(
        self,
        event: dict[str, Any],
        *,
        override_event_id: str | None = None,
    ) -> dict[str, Any]:
        """Normalize SportsGameOdds odds into the Odds-API payload shape."""
        event_id = override_event_id or str(event.get("eventID") or "")
        home = _team_name((event.get("teams") or {}).get("home"))
        away = _team_name((event.get("teams") or {}).get("away"))
        start_time = _event_start_time(event)
        players = event.get("players") or {}
        odds = event.get("odds") or {}

        market_entries: dict[tuple[str, str | None, float, str], dict[str, Any]] = {}
        updated_at: dict[tuple[str, str | None, float, str], str | None] = {}

        for odd in odds.values():
            if not isinstance(odd, dict):
                continue
            bet_type = str(odd.get("betTypeID") or "").lower()
            period = str(odd.get("periodID") or "").lower()
            side = str(odd.get("sideID") or "").lower()
            stat_id = str(odd.get("statID") or "").lower()
            stat_entity = str(odd.get("statEntityID") or "").lower()
            market_name = str(odd.get("marketName") or "").lower()
            player_id = str(odd.get("playerID") or odd.get("statEntityID") or "")

            if bet_type != "ou" or period not in {"game", "full", "fullgame", "full_game"}:
                continue
            market_kind, player_name = _market_kind(
                stat_id=stat_id,
                stat_entity=stat_entity,
                market_name=market_name,
                player_id=player_id,
                players=players,
            )
            if not market_kind:
                continue

            for book_id, book_data in (odd.get("byBookmaker") or {}).items():
                if not isinstance(book_data, dict):
                    continue
                if book_data.get("available") is False:
                    continue
                book = _bookmaker_name(str(book_id))
                line = _line_value(book_data, odd)
                if line is None:
                    continue
                price = _to_decimal(book_data.get("odds") or book_data.get("bookOdds"))
                if price is None:
                    continue
                entry_key = (book, market_kind, player_name, line)
                entry = market_entries.setdefault(
                    entry_key,
                    {
                        "hdp": line,
                        "over": None,
                        "under": None,
                        "player": player_name,
                    },
                )
                if side == "over":
                    entry["over"] = price
                elif side == "under":
                    entry["under"] = price
                updated_at[entry_key] = _best_timestamp(
                    updated_at.get(entry_key),
                    book_data.get("lastUpdatedAt") or odd.get("lastUpdatedAt"),
                )

        bookmakers: dict[str, list[dict[str, Any]]] = {}
        for (book, market_kind, _player, _line), entry in market_entries.items():
            if entry.get("over") is None and entry.get("under") is None:
                continue
            market_name = "Totals" if market_kind == "totals" else "Pitcher Strikeouts"
            market = {
                "name": market_name,
                "updatedAt": updated_at.get((book, market_kind, entry.get("player"), entry.get("hdp"))),
                "odds": [entry],
            }
            bookmakers.setdefault(book, []).append(market)

        return {
            "id": event_id,
            "home": home,
            "away": away,
            "date": start_time,
            "bookmakers": bookmakers,
            "source": "SportsGameOdds",
        }


def _event_start_time(event: dict[str, Any]) -> str | None:
    status = event.get("status") or {}
    return (
        status.get("startsAt")
        or event.get("startsAt")
        or event.get("startTime")
    )


def _event_date(event: dict[str, Any]) -> str | None:
    start = _event_start_time(event)
    if not start:
        return None
    return str(start)[:10]


def _team_name(team: dict[str, Any] | None) -> str:
    if not team:
        return ""
    names = team.get("names") or {}
    for key in ("long", "medium", "short"):
        value = names.get(key)
        if value:
            return str(value)
    return str(team.get("name") or team.get("teamName") or "")


def _player_name(player_id: str, players: dict[str, Any]) -> str | None:
    if not player_id:
        return None
    player = players.get(player_id) or {}
    name = player.get("name")
    if name:
        return str(name)
    first = player.get("firstName") or ""
    last = player.get("lastName") or ""
    full = f"{first} {last}".strip()
    return full or None


def _market_kind(
    *,
    stat_id: str,
    stat_entity: str,
    market_name: str,
    player_id: str,
    players: dict[str, Any],
) -> tuple[str | None, str | None]:
    if _is_total_market(stat_id, stat_entity, market_name):
        return "totals", None
    if "strikeout" in stat_id or "strikeout" in market_name:
        player_name = _player_name(player_id, players)
        if player_name:
            return "pitcher_k", player_name
    return None, None


def _is_total_market(stat_id: str, stat_entity: str, market_name: str) -> bool:
    if stat_entity in {"home", "away"}:
        return False
    if any(token in stat_id for token in ("runs", "points", "score")):
        return True
    return "total" in market_name and "team" not in market_name


def _bookmaker_name(bookmaker_id: str) -> str:
    key = "".join(ch for ch in bookmaker_id.lower() if ch.isalpha())
    mapping = {
        "draftkings": "DraftKings",
        "fanduel": "FanDuel",
        "betmgm": "BetMGM",
        "caesars": "Caesars",
        "espnbet": "ESPNBet",
    }
    return mapping.get(key, bookmaker_id.title())


def _line_value(book_data: dict[str, Any], odd: dict[str, Any]) -> float | None:
    for key in ("overUnder", "bookOverUnder"):
        value = book_data.get(key)
        if value is not None:
            return _to_float(value)
    for key in ("bookOverUnder", "fairOverUnder", "overUnder"):
        value = odd.get(key)
        if value is not None:
            return _to_float(value)
    return None


def _best_timestamp(current: str | None, candidate: Any) -> str | None:
    if not candidate:
        return current
    if not current:
        return str(candidate)
    try:
        curr_dt = datetime.fromisoformat(str(current).replace("Z", "+00:00"))
        cand_dt = datetime.fromisoformat(str(candidate).replace("Z", "+00:00"))
    except ValueError:
        return current
    return str(candidate) if cand_dt > curr_dt else current


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_decimal(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num >= 100:
        return round(1 + num / 100, 4)
    if num <= -100:
        return round(1 + 100 / abs(num), 4)
    if num > 0:
        return float(num)
    return None
