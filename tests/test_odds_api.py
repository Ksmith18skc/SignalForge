from __future__ import annotations

import httpx
import pytest

from app.config import Settings
from app.providers.odds_api import OddsApiError, OddsApiProvider, best_prices, normalize_odds_lines


def _sample_odds_payload():
    return {
        "id": 123,
        "home": "Knicks",
        "away": "Cavs",
        "date": "2026-05-25T23:00:00Z",
        "bookmakers": {
            "DraftKings": [
                {
                    "name": "ML",
                    "updatedAt": "2026-05-25T18:00:00Z",
                    "odds": [{"home": "2.10", "away": "1.80", "homeDirectLink": "https://dk/home"}],
                },
                {
                    "name": "Totals",
                    "updatedAt": "2026-05-25T18:01:00Z",
                    "odds": [{"hdp": 216.5, "over": "1.91", "under": "1.95"}],
                },
            ],
            "FanDuel": [
                {
                    "name": "Totals",
                    "updatedAt": "2026-05-25T18:02:00Z",
                    "odds": [{"hdp": 216.5, "over": "1.94", "under": "1.90"}],
                }
            ],
        },
    }


def test_normalize_odds_lines_flattens_markets_and_outcomes():
    rows = normalize_odds_lines(_sample_odds_payload())

    assert len(rows) == 3
    totals = [row for row in rows if row["market"] == "Totals"]
    assert totals[0]["line"] == 216.5
    assert totals[0]["outcomes"] == {"over": 1.91, "under": 1.95}


def test_best_prices_returns_best_by_outcome():
    rows = normalize_odds_lines(_sample_odds_payload())
    best = best_prices(rows)

    assert best["over"]["bookmaker"] == "FanDuel"
    assert best["over"]["price"] == 1.94
    assert best["under"]["bookmaker"] == "DraftKings"
    assert best["under"]["price"] == 1.95


def test_best_prices_can_filter_side():
    rows = normalize_odds_lines(_sample_odds_payload())
    best = best_prices(rows, "over")

    assert list(best.keys()) == ["over"]
    assert best["over"]["bookmaker"] == "FanDuel"


def test_odds_provider_requires_key_for_auth_endpoints():
    provider = OddsApiProvider(None, "https://api.odds-api.io/v3", "DraftKings")

    with pytest.raises(OddsApiError):
        provider._auth_params({"sport": "basketball"})


def test_odds_bookmakers_env_alias(monkeypatch):
    monkeypatch.setenv("ODDS_API_BOOKMAKERS", "DraftKings,FanDuel")
    assert Settings().odds_bookmakers == "DraftKings,FanDuel"


def test_odds_provider_preview_normalizes_mlb_to_baseball():
    provider = OddsApiProvider("key", "https://api.odds-api.io/v3", "DraftKings")

    preview = provider.preview_events_request(
        "mlb",
        date_from="2026-05-25T07:00:00Z",
        date_to="2026-05-26T06:59:59Z",
        include_auth=False,
    )

    assert "sport=baseball" in preview["url"]
    assert "apiKey" not in preview["url"]


@pytest.mark.asyncio
async def test_odds_provider_compare_lines(monkeypatch):
    calls = []

    class _Response:
        status_code = 200
        headers: dict[str, str] = {}

        def raise_for_status(self):
            return None

        def json(self):
            return _sample_odds_payload()

    class _Client:
        def __init__(self, timeout, **kwargs):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, params=None):
            calls.append((url, params))
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    provider = OddsApiProvider("key", "https://api.odds-api.io/v3", "DraftKings,FanDuel")

    result = await provider.compare_lines(123, market="Totals", side="over", line=216.5)

    assert calls == [
        (
            "https://api.odds-api.io/v3/odds",
            {"eventId": 123, "bookmakers": "DraftKings,FanDuel", "apiKey": "key"},
        )
    ]
    assert result["best_by_outcome"]["over"]["bookmaker"] == "FanDuel"
    assert len(result["rows"]) == 2


@pytest.mark.asyncio
async def test_odds_provider_events_omits_empty_league(monkeypatch):
    calls = []

    class _Response:
        status_code = 200
        headers: dict[str, str] = {}

        def raise_for_status(self):
            return None

        def json(self):
            return []

    class _Client:
        def __init__(self, timeout, **kwargs):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, params=None):
            calls.append((url, params))
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    provider = OddsApiProvider("key", "https://api.odds-api.io/v3", "DraftKings")

    await provider.events(
        "mlb",
        league=None,
        date_from="2026-05-25T07:00:00Z",
        date_to="2026-05-26T06:59:59Z",
    )

    assert calls == [
        (
            "https://api.odds-api.io/v3/events",
            {
                "sport": "baseball",
                "from": "2026-05-25T07:00:00Z",
                "to": "2026-05-26T06:59:59Z",
                "apiKey": "key",
            },
        )
    ]


@pytest.mark.asyncio
async def test_odds_provider_retries_forbidden_bookmaker_request_with_allowed_books(monkeypatch):
    calls = []

    class _Response403:
        status_code = 403
        headers: dict[str, str] = {}

        def __init__(self):
            self.text = "Allowed: DraftKings, FanDuel"

        def raise_for_status(self):
            return None

        def json(self):
            return {"error": "Allowed: DraftKings, FanDuel"}

    class _Response200:
        status_code = 200
        headers: dict[str, str] = {}

        def raise_for_status(self):
            return None

        def json(self):
            return _sample_odds_payload()

    class _Client:
        def __init__(self, timeout, **kwargs):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, params=None):
            calls.append((url, params))
            return _Response403() if len(calls) == 1 else _Response200()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    provider = OddsApiProvider("key", "https://api.odds-api.io/v3", "DraftKings,FanDuel,BetMGM,Caesars")

    payload = await provider.odds(123)

    assert payload["id"] == 123
    assert calls == [
        (
            "https://api.odds-api.io/v3/odds",
            {"eventId": 123, "bookmakers": "DraftKings,FanDuel,BetMGM,Caesars", "apiKey": "key"},
        ),
        (
            "https://api.odds-api.io/v3/odds",
            {"eventId": 123, "bookmakers": "DraftKings,FanDuel", "apiKey": "key"},
        ),
    ]
