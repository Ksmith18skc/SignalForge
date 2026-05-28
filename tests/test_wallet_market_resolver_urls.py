"""URL + internal-key builders in ``wallet_market_resolver``.

These guard against the recurring "click landed on a 404 line-specific
event page" bug: Polymarket events live at the matchup level, not the
per-line level, so any time we render a clickable URL we must strip
the ``-total-9pt5`` / ``-spread-home-1pt5`` / ``-moneyline`` suffix from
the slug. The line-specific identifier still has a use — joining MLB
edges to wallet trades — but it lives in ``internal_market_key`` and
must never leak into a URL.
"""

from __future__ import annotations

import pytest

from app.services.wallet_market_resolver import (
    internal_market_key,
    market_url_for,
    parse_market_slug,
    polymarket_event_slug,
    polymarket_event_url,
)


# ---------------------------------------------------------------------------
# polymarket_event_slug — input → expected event-level slug
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        # Total / spread / moneyline suffixes get stripped.
        ("mlb-tor-bal-2026-05-28-total-9pt5", "mlb-tor-bal-2026-05-28"),
        ("mlb-hou-tex-2026-05-28-total-8pt5", "mlb-hou-tex-2026-05-28"),
        ("mlb-tor-bal-2026-05-28-spread-home-1pt5", "mlb-tor-bal-2026-05-28"),
        ("mlb-tor-bal-2026-05-28-moneyline", "mlb-tor-bal-2026-05-28"),
        # Already-event-level slugs pass through unchanged.
        ("mlb-tor-bal-2026-05-28", "mlb-tor-bal-2026-05-28"),
        # Non-MLB sports follow the same suffix-stripping rules.
        ("nba-nyk-cle-2026-05-25-total-216pt5", "nba-nyk-cle-2026-05-25"),
        ("wnba-por-nyl-2026-05-25-total-176pt5", "wnba-por-nyl-2026-05-25"),
    ],
)
def test_polymarket_event_slug_strips_market_suffix(raw: str, expected: str) -> None:
    assert polymarket_event_slug(raw) == expected


@pytest.mark.parametrize("blank", [None, ""])
def test_polymarket_event_slug_handles_blank(blank) -> None:
    assert polymarket_event_slug(blank) is None


def test_polymarket_event_slug_passes_through_unrecognized_layout() -> None:
    """Human-readable slugs ("will-it-rain") have no recognized date,
    so we'd rather link to the (possibly wrong) original than fabricate
    something that's definitely wrong."""
    assert polymarket_event_slug("will-it-rain-2025") == "will-it-rain-2025"


# ---------------------------------------------------------------------------
# polymarket_event_url — full URL with captured-URL preference
# ---------------------------------------------------------------------------


def test_polymarket_event_url_strips_line_suffix_in_url() -> None:
    """The spec example. ``-total-9pt5`` must never appear in the URL."""
    url = polymarket_event_url("mlb-tor-bal-2026-05-28-total-9pt5")
    assert url == "https://polymarket.com/event/mlb-tor-bal-2026-05-28"
    assert "total-9pt5" not in (url or "")


def test_polymarket_event_url_prefers_captured_url() -> None:
    """If Polymarket scrape already captured the real event URL, we
    trust it verbatim — the platform sometimes changes URL schemes and
    a captured URL is authoritative."""
    captured = "https://polymarket.com/event/mlb-tor-bal-special-2026-05-28"
    out = polymarket_event_url(
        "mlb-tor-bal-2026-05-28-total-9pt5",
        captured_url=captured,
    )
    assert out == captured


def test_polymarket_event_url_returns_none_for_blank_slug_without_capture() -> None:
    assert polymarket_event_url(None) is None
    assert polymarket_event_url("") is None


# ---------------------------------------------------------------------------
# market_url_for — public switchboard used by routes / alerts / dashboard
# ---------------------------------------------------------------------------


def test_market_url_for_polymarket_strips_line() -> None:
    url = market_url_for("mlb-tor-bal-2026-05-28-total-9pt5", "polymarket")
    assert url == "https://polymarket.com/event/mlb-tor-bal-2026-05-28"


def test_market_url_for_kalshi_keeps_full_slug() -> None:
    """Kalshi URLs are per-market, not per-event, so the full slug is
    correct there. Don't accidentally strip it just because we're
    sharing one helper across platforms."""
    url = market_url_for("KXMLBGAME-26MAY28TORBAL-OVER9.5", "kalshi")
    assert url == "https://kalshi.com/markets/KXMLBGAME-26MAY28TORBAL-OVER9.5"


def test_market_url_for_blank_slug_returns_none() -> None:
    assert market_url_for("", "polymarket") is None
    assert market_url_for(None, "polymarket") is None


# ---------------------------------------------------------------------------
# internal_market_key — line-specific identifier for joins
# ---------------------------------------------------------------------------


def test_internal_market_key_for_total_includes_line() -> None:
    """The spec example. Internal key uses ``:`` separator so a search
    for ``polymarket.com/event/`` never catches an internal key."""
    parsed = parse_market_slug("mlb-tor-bal-2026-05-28-total-9pt5")
    key = internal_market_key(parsed, side="over")
    assert key == "mlb:tor-bal:2026-05-28:game_total:over:9.5"


def test_internal_market_key_for_spread_includes_side_and_line() -> None:
    parsed = parse_market_slug("mlb-tor-bal-2026-05-28-spread-home-1pt5")
    key = internal_market_key(parsed)
    assert key == "mlb:tor-bal:2026-05-28:game_spread:home:1.5"


def test_internal_market_key_for_moneyline_no_line_segment() -> None:
    parsed = parse_market_slug("mlb-tor-bal-2026-05-28-moneyline")
    key = internal_market_key(parsed, side="home")
    assert key == "mlb:tor-bal:2026-05-28:game_moneyline:home"


def test_internal_market_key_returns_none_for_unparseable_slug() -> None:
    """If parse_market_slug couldn't extract teams/date, we have nothing
    to build a key from — better to return None than fabricate an
    obviously-wrong key."""
    assert internal_market_key(None) is None
    assert internal_market_key(parse_market_slug("will-it-rain-2025")) is None


def test_internal_market_key_uses_colon_separator() -> None:
    """The colon separator is load-bearing — guarantees that an internal
    key can't be mistaken for a URL segment if it shows up in logs.
    """
    parsed = parse_market_slug("mlb-tor-bal-2026-05-28-total-9pt5")
    key = internal_market_key(parsed, side="over") or ""
    assert "polymarket.com" not in key
    assert "/" not in key
    assert "-" in key  # team pair retains hyphens
    assert ":" in key
