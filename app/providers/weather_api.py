"""WeatherAPI.com provider for MLB weather context."""

from __future__ import annotations

from typing import Any

import httpx


class WeatherApiError(RuntimeError):
    """Raised for WeatherAPI configuration or upstream failures."""


class WeatherApiProvider:
    """Small async client for https://www.weatherapi.com/docs/."""

    def __init__(
        self,
        api_key: str | None,
        base_url: str = "https://api.weatherapi.com/v1",
        timeout: float = 15.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _params(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self._api_key:
            raise WeatherApiError("SIGNALFORGE_WEATHER_API_KEY is not configured")
        out = _clean_params(params)
        out["key"] = self._api_key
        return out

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(f"{self._base_url}{path}", params=self._params(params))
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and "error" in payload:
                err = payload["error"]
                if isinstance(err, dict):
                    raise WeatherApiError(str(err.get("message") or err))
                raise WeatherApiError(str(err))
            return payload

    async def current(self, q: str, *, aqi: str = "no") -> dict[str, Any]:
        return await self._get("/current.json", {"q": q, "aqi": aqi})

    async def forecast(
        self,
        q: str,
        *,
        days: int = 1,
        dt: str | None = None,
        hour: int | None = None,
        alerts: str = "no",
        aqi: str = "no",
    ) -> dict[str, Any]:
        return await self._get(
            "/forecast.json",
            {"q": q, "days": days, "dt": dt, "hour": hour, "alerts": alerts, "aqi": aqi},
        )

    async def history(
        self,
        q: str,
        *,
        dt: str,
        hour: int | None = None,
        end_dt: str | None = None,
    ) -> dict[str, Any]:
        return await self._get("/history.json", {"q": q, "dt": dt, "hour": hour, "end_dt": end_dt})

    async def baseball_weather(
        self,
        q: str,
        *,
        game_date: str | None = None,
        hour: int | None = None,
    ) -> dict[str, Any]:
        payload = (
            await self.forecast(q, days=1, dt=game_date, hour=hour)
            if game_date or hour is not None
            else await self.current(q)
        )
        observation = _select_observation(payload, hour)
        return {
            "location": payload.get("location"),
            "weather": summarize_baseball_weather(observation),
            "raw": payload,
        }


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if value not in (None, "", [])}


def _select_observation(payload: dict[str, Any], hour: int | None = None) -> dict[str, Any]:
    current = payload.get("current")
    if isinstance(current, dict):
        return current

    forecast = payload.get("forecast") or {}
    days = forecast.get("forecastday") if isinstance(forecast, dict) else None
    if not isinstance(days, list) or not days:
        return {}

    hours = days[0].get("hour") if isinstance(days[0], dict) else None
    if not isinstance(hours, list) or not hours:
        day = days[0].get("day") if isinstance(days[0], dict) else {}
        return day if isinstance(day, dict) else {}

    if hour is not None:
        for row in hours:
            if isinstance(row, dict) and str(row.get("time", "")).endswith(f" {hour:02d}:00"):
                return row
    return hours[0] if isinstance(hours[0], dict) else {}


def summarize_baseball_weather(observation: dict[str, Any]) -> dict[str, Any]:
    condition = observation.get("condition") if isinstance(observation.get("condition"), dict) else {}
    return {
        "temp_f": observation.get("temp_f") or observation.get("avgtemp_f"),
        "temp_c": observation.get("temp_c") or observation.get("avgtemp_c"),
        "humidity": observation.get("humidity") or observation.get("avghumidity"),
        "wind_mph": observation.get("wind_mph") or observation.get("maxwind_mph"),
        "wind_kph": observation.get("wind_kph") or observation.get("maxwind_kph"),
        "wind_degree": observation.get("wind_degree"),
        "wind_dir": observation.get("wind_dir"),
        "precip_in": observation.get("precip_in") or observation.get("totalprecip_in"),
        "precip_mm": observation.get("precip_mm") or observation.get("totalprecip_mm"),
        "condition": condition.get("text") if isinstance(condition, dict) else None,
        "time": observation.get("time") or observation.get("last_updated"),
    }
