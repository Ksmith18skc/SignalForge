"""Arizona card-date helpers shared by live dashboard flows."""

from __future__ import annotations

import re
from datetime import date as date_cls, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo


TZ_ARIZONA = ZoneInfo("America/Phoenix")

_SLUG_DATE_RE = re.compile(r"(20\d{2})[-_](\d{2})[-_](\d{2})")


def arizona_today() -> str:
    """Today's card date in Arizona/MST as ISO YYYY-MM-DD."""
    return datetime.now(TZ_ARIZONA).date().isoformat()


def arizona_yesterday() -> str:
    """Yesterday's card date in Arizona/MST as ISO YYYY-MM-DD."""
    return (datetime.now(TZ_ARIZONA).date() - timedelta(days=1)).isoformat()


def arizona_window(days: int, *, today: str | None = None) -> tuple[str, str]:
    """Inclusive ISO date window of ``days`` ending on the Arizona card date."""
    end_iso = today or arizona_today()
    end = date_cls.fromisoformat(end_iso)
    start = end - timedelta(days=max(int(days) - 1, 0))
    return start.isoformat(), end.isoformat()


def parse_slug_date(slug: str | None) -> date_cls | None:
    """Extract YYYY-MM-DD from a market slug when present."""
    if not slug:
        return None
    match = _SLUG_DATE_RE.search(slug)
    if not match:
        return None
    try:
        return date_cls(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def parse_iso_date(value: Any) -> date_cls | None:
    """Best-effort ISO date parser for strings/datetimes."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date_cls):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    try:
        return date_cls.fromisoformat(text[:10])
    except ValueError:
        return None


def market_card_date(market: Any) -> str | None:
    """Return the best card date for a market from slug, then end_date."""
    if market is None:
        return None
    slug_date = parse_slug_date(getattr(market, "slug", None))
    if slug_date is not None:
        return slug_date.isoformat()
    end_date = parse_iso_date(getattr(market, "end_date", None))
    return end_date.isoformat() if end_date else None
