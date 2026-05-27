"""SportsGameOdds (SGO) provider for MLB odds fallback."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Any

import httpx

from app.config import get_settings
from app.utils.redaction import redact_headers, redact_url, sanitize_text

logger = logging.getLogger(__name__)

_cooldown_until: datetime | None = None


class SportsGameOddsError(RuntimeError):
    """Raised for SportsGameOdds configuration or upstream failures."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        sanitized_url: str | None = None,
        error_body_preview: str | None = None,
    ) -> None:
        super().__init__(sanitize_text(message))
        self.status_code = status_code
        self.sanitized_url = sanitized_url
        self.error_body_preview = error_body_preview


class SportsGameOddsRateLimited(SportsGameOddsError):
    """Raised when SportsGameOdds returns HTTP 429."""

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        cooldown_until: datetime | None = None,
        sanitized_url: str | None = None,
    ) -> None:
        super().__init__(message, status_code=429, sanitized_url=sanitized_url)
        self.retry_after = retry_after
        self.cooldown_until = cooldown_until


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

    def preview_events_request(
        self,
        *,
        params: dict[str, Any] | None = None,
        include_auth: bool = True,
    ) -> dict[str, Any]:
        request_params = dict(params or {})
        request_url = httpx.URL(f"{self._base_url}/events", params=request_params)
        return {
            "method": "GET",
            "url": redact_url(str(request_url)),
            "params": request_params,
            "headers": redact_headers(self._auth_headers()) if include_auth else {},
        }

    def mlb_events_params(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
        odds_available: bool = True,
        include_alt_lines: bool = True,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "leagueID": self._validate_mlb_league("MLB"),
            "oddsAvailable": "true" if odds_available else "false",
            "includeAltLines": "true" if include_alt_lines else "false",
        }
        if limit is not None:
            params["limit"] = self._validate_page_size(limit)
        if cursor:
            params["cursor"] = str(cursor)
        return params

    async def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        max_retries: int = 1,
    ) -> Any:
        _raise_if_in_cooldown(path)
        query = dict(params or {})
        delay = 1.0
        last_retry_after: float | None = None
        last_url: str | None = None
        for attempt in range(max_retries + 1):
            limits = httpx.Limits(max_connections=2, max_keepalive_connections=1)
            try:
                client_factory = httpx.AsyncClient(timeout=self._timeout, limits=limits)
            except TypeError:
                client_factory = httpx.AsyncClient(timeout=self._timeout)
            async with client_factory as client:
                response = await client.get(
                    f"{self._base_url}{path}",
                    params=query,
                    headers=self._auth_headers(),
                )
            last_url = _response_url(response, fallback=f"{self._base_url}{path}")
            if response.status_code != 429:
                if response.status_code >= 400:
                    body = _response_text(response)
                    raise SportsGameOddsError(
                        (
                            f"SportsGameOdds HTTP {response.status_code} on {path}; "
                            f"url={redact_url(last_url)}; headers={redact_headers(self._auth_headers())}; "
                            f"body={sanitize_text(body, limit=300)}"
                        ),
                        status_code=response.status_code,
                        sanitized_url=redact_url(last_url),
                        error_body_preview=sanitize_text(body, limit=500),
                    )
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
        cooldown_until = _set_cooldown(last_retry_after)
        raise SportsGameOddsRateLimited(
            f"SportsGameOdds returned 429 on {path} after {max_retries + 1} attempts",
            retry_after=last_retry_after,
            cooldown_until=cooldown_until,
            sanitized_url=redact_url(last_url),
        )

    async def fetch_mlb_events_with_odds(self, game_date: str) -> list[dict[str, Any]]:
        """Return MLB events with odds data filtered to the requested date."""
        settings = get_settings()
        page_limit = max(1, min(int(settings.sgo_page_limit or 200), 200))
        max_pages = max(1, min(int(settings.sgo_max_pages or 3), 5))
        cursor: str | None = None
        events: list[dict[str, Any]] = []
        for _page in range(max_pages):
            params = self.mlb_events_params(limit=page_limit, cursor=cursor)
            payload = await self._get("/events", params)
            if not isinstance(payload, dict):
                raise SportsGameOddsError("Unexpected SportsGameOdds response")
            if payload.get("success") is False:
                raise SportsGameOddsError(str(payload.get("error") or "SGO request failed"))
            data = payload.get("data") or []
            if isinstance(data, list):
                events.extend(event for event in data if isinstance(event, dict))
            cursor = _next_cursor(payload)
            if not cursor:
                break
        return [
            event for event in events
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

    @staticmethod
    def _validate_mlb_league(league_id: str) -> str:
        league = str(league_id or "").strip().upper()
        if league != "MLB":
            raise SportsGameOddsError("SportsGameOdds MLB requests must use leagueID=MLB")
        return league

    @staticmethod
    def _validate_page_size(limit: int) -> int:
        try:
            value = int(limit)
        except (TypeError, ValueError) as exc:
            raise SportsGameOddsError("limit must be an integer") from exc
        if value < 1:
            raise SportsGameOddsError("limit must be >= 1")
        return min(value, 200)


def _event_start_time(event: dict[str, Any]) -> str | None:
    status = event.get("status") or {}
    return (
        status.get("startsAt")
        or event.get("startsAt")
        or event.get("startTime")
    )


def cooldown_until() -> datetime | None:
    return _cooldown_until


def reset_cooldown() -> None:
    global _cooldown_until
    _cooldown_until = None


def _raise_if_in_cooldown(path: str) -> None:
    until = _cooldown_until
    if until is None:
        return
    now = datetime.utcnow()
    if until <= now:
        reset_cooldown()
        return
    retry_after = max((until - now).total_seconds(), 0.0)
    raise SportsGameOddsRateLimited(
        f"SportsGameOdds cooldown active until {until.isoformat()}",
        retry_after=retry_after,
        cooldown_until=until,
        sanitized_url=path,
    )


def _set_cooldown(retry_after: float | None) -> datetime:
    global _cooldown_until
    seconds = retry_after if retry_after and retry_after > 0 else _default_cooldown_seconds()
    _cooldown_until = datetime.utcnow() + timedelta(seconds=float(seconds))
    return _cooldown_until


def _default_cooldown_seconds() -> int:
    raw = os.getenv("SPORTS_GAME_ODDS_COOLDOWN_SECONDS")
    if raw:
        try:
            return max(int(raw), 1)
        except ValueError:
            pass
    return max(int(get_settings().sports_game_odds_cooldown_seconds or 300), 1)


def _next_cursor(payload: dict[str, Any]) -> str | None:
    for key in ("nextCursor", "next_cursor", "cursor"):
        value = payload.get(key)
        if value:
            return str(value)
    meta = payload.get("meta") or payload.get("metadata") or {}
    if isinstance(meta, dict):
        for key in ("nextCursor", "next_cursor", "cursor"):
            value = meta.get(key)
            if value:
                return str(value)
    return None


def _response_url(response: Any, *, fallback: str) -> str:
    url = getattr(response, "url", None)
    if url:
        return str(url)
    request = getattr(response, "request", None)
    if request is not None and getattr(request, "url", None):
        return str(request.url)
    return fallback


def _response_text(response: Any) -> str:
    try:
        return str(response.text)
    except Exception:  # noqa: BLE001
        return ""


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
