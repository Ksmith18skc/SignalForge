"""Odds-API.io provider for sportsbook odds and line comparison."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class OddsApiError(RuntimeError):
    """Raised for Odds-API configuration or upstream failures."""


class OddsApiRateLimited(OddsApiError):
    """Raised when Odds-API returns HTTP 429.

    Callers (especially the centralized cache) should fall back to stale
    data rather than failing edge generation. The exception carries the
    Retry-After header (seconds) when present so consumers can pace retries.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class OddsApiProvider:
    """Small async client for https://api.odds-api.io/v3."""

    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        bookmakers: str | list[str] | None = None,
        timeout: float = 15.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._bookmakers = _split_csv(bookmakers)

    @property
    def default_bookmakers(self) -> list[str]:
        return self._bookmakers

    def _auth_params(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._api_key:
            raise OddsApiError("SIGNALFORGE_ODDS_API_KEY is not configured")
        out = dict(params or {})
        out["apiKey"] = self._api_key
        return out

    async def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        auth: bool = True,
        max_retries: int = 2,
    ) -> Any:
        """GET with 429-aware exponential backoff.

        On the free Odds-API plan, 429s are common. We retry up to `max_retries`
        times with exponential delays (capped at 8s), preferring the upstream's
        Retry-After header when present. After the final attempt we raise
        OddsApiRateLimited so callers can serve stale cache instead of crashing.
        """
        query = self._auth_params(params) if auth else params
        delay = 1.0
        last_status: int | None = None
        last_retry_after: float | None = None
        for attempt in range(max_retries + 1):
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(f"{self._base_url}{path}", params=query)
            if response.status_code != 429:
                response.raise_for_status()
                return response.json()
            # 429 — back off.
            last_status = 429
            retry_after_raw = response.headers.get("Retry-After")
            try:
                retry_after = float(retry_after_raw) if retry_after_raw else None
            except (TypeError, ValueError):
                retry_after = None
            last_retry_after = retry_after
            if attempt >= max_retries:
                break
            sleep_for = min(retry_after or delay, 8.0)
            logger.warning(
                "Odds-API 429 on %s (attempt %d/%d) — sleeping %.1fs",
                path, attempt + 1, max_retries + 1, sleep_for,
            )
            await asyncio.sleep(sleep_for)
            delay *= 2
        raise OddsApiRateLimited(
            f"Odds-API returned 429 on {path} after {max_retries + 1} attempts",
            retry_after=last_retry_after,
        )

    async def sports(self) -> list[dict[str, Any]]:
        return await self._get("/sports", auth=False)

    async def bookmakers(self) -> list[dict[str, Any]]:
        return await self._get("/bookmakers", auth=False)

    async def selected_bookmakers(self) -> Any:
        return await self._get("/bookmakers/selected")

    async def leagues(self, sport: str) -> list[dict[str, Any]]:
        return await self._get("/leagues", {"sport": sport})

    async def events(
        self,
        sport: str,
        *,
        league: str | None = None,
        status: str | None = None,
        bookmaker: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict[str, Any]]:
        params = _clean_params(
            {
                "sport": sport,
                "league": league,
                "status": status,
                "bookmaker": bookmaker,
                "from": date_from,
                "to": date_to,
            }
        )
        return await self._get("/events", params)

    async def live_events(self, sport: str | None = None) -> list[dict[str, Any]]:
        return await self._get("/events/live", _clean_params({"sport": sport}))

    async def search_events(self, query: str) -> list[dict[str, Any]]:
        return await self._get("/events/search", {"query": query})

    async def event(self, event_id: str | int) -> dict[str, Any]:
        return await self._get(f"/events/{event_id}")

    async def odds(self, event_id: str | int, bookmakers: str | list[str] | None = None) -> dict[str, Any]:
        books = _bookmaker_param(bookmakers or self._bookmakers)
        if not books:
            raise OddsApiError("At least one bookmaker is required for odds lookup")
        return await self._get("/odds", {"eventId": event_id, "bookmakers": books})

    async def odds_multi(
        self,
        event_ids: list[str | int] | str,
        bookmakers: str | list[str] | None = None,
    ) -> list[dict[str, Any]]:
        ids = event_ids if isinstance(event_ids, str) else ",".join(str(e) for e in event_ids[:10])
        books = _bookmaker_param(bookmakers or self._bookmakers)
        if not books:
            raise OddsApiError("At least one bookmaker is required for odds lookup")
        return await self._get("/odds/multi", {"eventIds": ids, "bookmakers": books})

    async def odds_movements(
        self,
        event_id: str | int,
        bookmaker: str,
        market: str,
        market_line: float | None = None,
    ) -> dict[str, Any]:
        params = _clean_params(
            {
                "eventId": event_id,
                "bookmaker": bookmaker,
                "market": market,
                "marketLine": market_line,
            }
        )
        return await self._get("/odds/movements", params)

    async def compare_lines(
        self,
        event_id: str | int,
        *,
        bookmakers: str | list[str] | None = None,
        market: str | None = None,
        side: str | None = None,
        line: float | None = None,
    ) -> dict[str, Any]:
        odds_payload = await self.odds(event_id, bookmakers)
        rows = normalize_odds_lines(odds_payload)
        if market:
            rows = [r for r in rows if str(r["market"]).lower() == market.lower()]
        if line is not None:
            rows = [r for r in rows if r.get("line") == line]

        best_by_outcome = best_prices(rows, side)
        return {
            "event": {
                "id": odds_payload.get("id"),
                "home": odds_payload.get("home"),
                "away": odds_payload.get("away"),
                "date": odds_payload.get("date"),
                "status": odds_payload.get("status"),
                "sport": odds_payload.get("sport"),
                "league": odds_payload.get("league"),
            },
            "filters": {"market": market, "side": side, "line": line},
            "bookmakers": list((odds_payload.get("bookmakers") or {}).keys()),
            "rows": rows,
            "best_by_outcome": best_by_outcome,
        }


def _split_csv(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in value.split(",") if part.strip()]


def _bookmaker_param(value: str | list[str] | None) -> str:
    return ",".join(_split_csv(value))


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if value not in (None, "", [])}


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_odds_lines(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten Odds-API bookmaker market payloads into comparable line rows."""
    rows: list[dict[str, Any]] = []
    urls = payload.get("urls") or {}
    for bookmaker, markets in (payload.get("bookmakers") or {}).items():
        if not isinstance(markets, list):
            continue
        for market in markets:
            if not isinstance(market, dict):
                continue
            market_name = market.get("name")
            updated_at = market.get("updatedAt")
            odds_list = market.get("odds") or []
            if not isinstance(odds_list, list):
                continue
            for odds in odds_list:
                if not isinstance(odds, dict):
                    continue
                outcomes: dict[str, float] = {}
                direct_links: dict[str, str] = {}
                for key, value in odds.items():
                    if key in {"hdp", "max", "label"} or key.endswith("DirectLink"):
                        if key.endswith("DirectLink") and isinstance(value, str):
                            direct_links[key.removesuffix("DirectLink")] = value
                        continue
                    parsed = _to_float(value)
                    if parsed is not None:
                        outcomes[key] = parsed
                rows.append(
                    {
                        "bookmaker": bookmaker,
                        "bookmaker_url": urls.get(bookmaker),
                        "market": market_name,
                        "line": _to_float(odds.get("hdp")),
                        "label": odds.get("label") or market.get("label"),
                        "updated_at": updated_at,
                        "outcomes": outcomes,
                        "direct_links": direct_links,
                        "max": _to_float(odds.get("max")),
                    }
                )
    return rows


def best_prices(rows: list[dict[str, Any]], side: str | None = None) -> dict[str, dict[str, Any]]:
    """Return the highest decimal price for each outcome, or one requested side."""
    requested = side.lower() if side else None
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        outcomes = row.get("outcomes") or {}
        if not isinstance(outcomes, dict):
            continue
        for outcome, price in outcomes.items():
            outcome_key = str(outcome).lower()
            if requested and outcome_key != requested:
                continue
            current = best.get(outcome_key)
            if current is None or float(price) > float(current["price"]):
                best[outcome_key] = {
                    "outcome": outcome,
                    "price": price,
                    "bookmaker": row.get("bookmaker"),
                    "market": row.get("market"),
                    "line": row.get("line"),
                    "label": row.get("label"),
                    "direct_link": (row.get("direct_links") or {}).get(outcome),
                }
    return best
