from __future__ import annotations

import pytest

from app.providers.weather_api import WeatherApiProvider, summarize_baseball_weather


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeAsyncClient:
    calls = []

    def __init__(self, timeout):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def get(self, url, params):
        self.calls.append((url, params))
        return FakeResponse(
            {
                "location": {"name": "Chicago"},
                "current": {
                    "temp_f": 62.0,
                    "humidity": 71,
                    "wind_mph": 18.4,
                    "wind_degree": 215,
                    "wind_dir": "SW",
                    "precip_in": 0.02,
                    "condition": {"text": "Light rain"},
                    "last_updated": "2026-05-25 18:00",
                },
            }
        )


def test_summarize_baseball_weather_extracts_run_total_fields():
    summary = summarize_baseball_weather(
        {
            "temp_f": 83,
            "humidity": 64,
            "wind_mph": 14,
            "wind_degree": 270,
            "wind_dir": "W",
            "precip_in": 0.1,
            "condition": {"text": "Partly cloudy"},
        }
    )

    assert summary["temp_f"] == 83
    assert summary["wind_mph"] == 14
    assert summary["wind_dir"] == "W"
    assert summary["precip_in"] == 0.1


@pytest.mark.asyncio
async def test_weather_provider_calls_current_with_key(monkeypatch):
    FakeAsyncClient.calls = []
    monkeypatch.setattr("app.providers.weather_api.httpx.AsyncClient", FakeAsyncClient)
    provider = WeatherApiProvider("secret", "https://api.weatherapi.com/v1")

    payload = await provider.baseball_weather("Wrigley Field")

    assert payload["weather"]["wind_mph"] == 18.4
    url, params = FakeAsyncClient.calls[0]
    assert url == "https://api.weatherapi.com/v1/current.json"
    assert params["q"] == "Wrigley Field"
    assert params["key"] == "secret"
