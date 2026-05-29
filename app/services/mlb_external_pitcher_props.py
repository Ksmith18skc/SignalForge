"""Optional Kalshi/Polymarket enrichment for MLB pitcher strikeout props.

This module never decides whether a Pitcher K card exists. BallparkPal
Strikeout Center rows are the primary source; external markets only add
a URL/price when a same-pitcher, same-date, same-line market is found.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.services.mlb_pitcher_k_fallback import name_matches_loose
from app.services.wallet_market_resolver import market_url_for

logger = logging.getLogger(__name__)

_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_LINE_RE = re.compile(r"\b(?:over|under)\s+(\d+(?:\.\d+)?)\b", re.I)
_K_LINE_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:strikeouts?|ks?)\b", re.I)
_SIDE_RE = re.compile(r"\b(over|under)\b", re.I)
_TITLE_NAME_RE = re.compile(
    r"(?:will\s+)?([A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){1,3})"
    r".{0,60}?\b(?:over|under)\b.{0,20}?\d+(?:\.\d+)?"
    r".{0,20}?(?:strikeouts?|ks?)",
    re.I,
)


async def fetch_external_pitcher_k_props(
    *,
    game_date: str,
    providers: dict[str, Any] | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return normalized external pitcher-K markets, if providers have any.

    Network/API failures are swallowed by the caller, but this function
    also isolates each provider so one bad upstream does not hide rows
    from another.
    """
    if providers is None:
        try:
            from app.services.ingestion import build_providers

            providers = build_providers()
        except Exception as exc:  # noqa: BLE001
            logger.debug("external pitcher-K providers unavailable: %s", exc)
            return []

    query = f"MLB pitcher strikeouts {game_date}"
    out: list[dict[str, Any]] = []

    primary = providers.get("primary") if isinstance(providers, dict) else None
    out.extend(
        await _fetch_falcon_markets(
            primary, method_name="fetch_polymarket_markets",
            platform="polymarket", query=query, game_date=game_date, limit=limit,
        )
    )
    out.extend(
        await _fetch_falcon_markets(
            primary, method_name="fetch_kalshi_markets",
            platform="kalshi", query=query, game_date=game_date, limit=limit,
        )
    )

    for key, platform in (("polymarket", "polymarket"), ("kalshi", "kalshi")):
        provider = providers.get(key) if isinstance(providers, dict) else None
        if provider is None or not hasattr(provider, "list_active_markets"):
            continue
        try:
            rows = await provider.list_active_markets(limit=limit)
        except Exception as exc:  # noqa: BLE001
            logger.debug("%s pitcher-K market list failed: %s", platform, exc)
            continue
        out.extend(
            normalized
            for row in rows or []
            if (normalized := normalize_external_pitcher_k_market(
                row, platform=platform, default_date=game_date,
            ))
        )

    return _dedupe_markets(out)


def match_external_pitcher_k_prop(
    *,
    pitcher_name: str | None,
    game_date: str,
    line: Any,
    markets: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Find a same-pitcher/date/line external market for a BPP row."""
    try:
        target_line = float(line)
    except (TypeError, ValueError):
        return None
    for market in markets or []:
        market_date = market.get("event_date")
        if market_date and str(market_date) != str(game_date):
            continue
        try:
            market_line = float(market.get("line"))
        except (TypeError, ValueError):
            continue
        if abs(market_line - target_line) > 0.05:
            continue
        if name_matches_loose(market.get("pitcher_name"), pitcher_name):
            return dict(market)
    return None


def normalize_external_pitcher_k_market(
    row: dict[str, Any],
    *,
    platform: str,
    default_date: str,
) -> dict[str, Any] | None:
    """Normalize one raw Kalshi/Polymarket row into a K-prop shape."""
    if not isinstance(row, dict):
        return None
    text = _market_text(row)
    if "strikeout" not in text.lower() and not re.search(r"\bks?\b", text, re.I):
        return None

    pitcher = _first_text(row, "pitcher_name", "pitcher", "player_name", "player", "participant")
    if not pitcher:
        match = _TITLE_NAME_RE.search(text)
        pitcher = match.group(1).strip() if match else None
    pitcher = _clean_pitcher_name(pitcher)
    line = _first_number(row, "line", "strikeout_line", "over_line", "point", "hdp")
    if line is None:
        line = _line_from_text(text)
    if not pitcher or line is None:
        return None

    event_date = _date_from_row(row) or _date_from_text(text)
    if event_date and event_date != default_date:
        return None
    side = _side_from_row(row) or _side_from_text(text)
    price = _price_from_row(row)
    slug = _first_text(row, "market_slug", "slug", "ticker", "id")
    url = _first_text(row, "market_url", "source_url", "url")
    if not url and slug:
        url = market_url_for(slug, platform)

    return {
        "platform": platform,
        "pitcher_name": pitcher,
        "event_date": event_date or default_date,
        "line": float(line),
        "side": side,
        "price": price,
        "implied_probability": price,
        "market_url": url,
        "market_slug": slug,
        "raw": row,
    }


async def _fetch_falcon_markets(
    provider: Any,
    *,
    method_name: str,
    platform: str,
    query: str,
    game_date: str,
    limit: int,
) -> list[dict[str, Any]]:
    if provider is None or not hasattr(provider, method_name):
        return []
    try:
        result = await getattr(provider, method_name)(query=query, limit=limit)
    except Exception as exc:  # noqa: BLE001
        logger.debug("%s pitcher-K search failed: %s", platform, exc)
        return []
    rows = []
    if getattr(result, "rows", None):
        rows.extend(result.rows or [])
    if getattr(result, "summary", None):
        rows.append(result.summary)
    return [
        normalized
        for row in rows
        if (normalized := normalize_external_pitcher_k_market(
            row, platform=platform, default_date=game_date,
        ))
    ]


def _market_text(row: dict[str, Any]) -> str:
    keys = ("title", "market_title", "question", "name", "slug", "market_slug", "ticker")
    return " ".join(str(row.get(key) or "") for key in keys)


def _first_text(row: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return None


def _first_number(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _line_from_text(text: str) -> float | None:
    for pattern in (_LINE_RE, _K_LINE_RE):
        match = pattern.search(text)
        if match:
            try:
                return float(match.group(1))
            except (TypeError, ValueError):
                return None
    return None


def _date_from_row(row: dict[str, Any]) -> str | None:
    for key in ("event_date", "game_date", "date", "end_date", "close_time"):
        value = row.get(key)
        if not value:
            continue
        match = _DATE_RE.search(str(value))
        if match:
            return match.group(1)
    return None


def _date_from_text(text: str) -> str | None:
    match = _DATE_RE.search(text)
    return match.group(1) if match else None


def _side_from_row(row: dict[str, Any]) -> str | None:
    raw = _first_text(row, "side", "outcome", "contract", "position")
    if not raw:
        return None
    raw_lower = raw.lower()
    if "over" in raw_lower:
        return "over"
    if "under" in raw_lower:
        return "under"
    return None


def _side_from_text(text: str) -> str | None:
    match = _SIDE_RE.search(text)
    return match.group(1).lower() if match else None


def _price_from_row(row: dict[str, Any]) -> float | None:
    price = _first_number(
        row,
        "side_price",
        "yes_price",
        "price",
        "last_price",
        "best_yes_price",
        "bid",
        "ask",
    )
    if price is None:
        return None
    if 1 < price <= 100:
        return round(price / 100.0, 4)
    if 0 <= price <= 1:
        return round(price, 4)
    return None


def _clean_pitcher_name(value: str | None) -> str | None:
    if not value:
        return None
    stopwords = {"record", "throw", "throws", "get", "have", "post"}
    parts = [part for part in str(value).split() if part.lower() not in stopwords]
    cleaned = " ".join(parts).strip()
    return cleaned or None


def _dedupe_markets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, float]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("platform") or ""),
            str(row.get("market_slug") or row.get("market_url") or ""),
            str(row.get("pitcher_name") or "").lower(),
            float(row.get("line") or 0.0),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out
