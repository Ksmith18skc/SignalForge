"""MLB environment scoring for run totals and strikeout props."""

from __future__ import annotations

from typing import Any


def score_environment(weather: dict[str, Any] | None, *, park_factor: float | None = None) -> dict[str, Any]:
    warnings: list[str] = []
    weather = weather or {}
    temp_f = _num(weather.get("temp_f"))
    humidity = _num(weather.get("humidity"))
    wind_mph = _num(weather.get("wind_mph"))
    wind_dir = str(weather.get("wind_dir") or "").upper()
    precip_in = _num(weather.get("precip_in")) or 0.0

    if not weather:
        warnings.append("Weather missing; confidence downgraded")
    if temp_f is None:
        warnings.append("Temperature missing")
    if wind_mph is None:
        warnings.append("Wind speed missing")
    if not wind_dir:
        warnings.append("Wind direction missing")

    temperature_score = _clamp(50 + ((temp_f or 70) - 70) * 1.5)
    humidity_score = _clamp(50 + ((humidity or 50) - 50) * 0.25)
    wind_score = _wind_score(wind_mph or 0.0, wind_dir)
    precipitation_risk = _clamp(precip_in * 250)
    if wind_mph and wind_mph >= 20:
        warnings.append("Extreme wind may materially affect run environment")
    if precip_in >= 0.1:
        warnings.append("Precipitation risk may affect lineups or game continuity")

    park = park_factor if park_factor is not None else 50.0
    run_environment_score = _clamp(
        0.35 * temperature_score
        + 0.35 * wind_score
        + 0.15 * humidity_score
        + 0.15 * park
        - precipitation_risk * 0.25
    )
    under_environment_score = _clamp(100 - run_environment_score + precipitation_risk * 0.15)
    k_environment_score = _clamp(0.6 * under_environment_score + 0.4 * (100 - max(0.0, (temp_f or 70) - 70)))

    return {
        "temperature_score": round(temperature_score, 2),
        "wind_score": round(wind_score, 2),
        "humidity_score": round(humidity_score, 2),
        "precipitation_risk": round(precipitation_risk, 2),
        "park_factor": park,
        "run_environment_score": round(run_environment_score, 2),
        "under_environment_score": round(under_environment_score, 2),
        "k_environment_score": round(k_environment_score, 2),
        "warnings": warnings,
    }


def _wind_score(wind_mph: float, wind_dir: str) -> float:
    # Without park orientation, use coarse baseball heuristics. Out/in keywords
    # from manual/future stadium mapping get priority; cardinal directions stay neutral-ish.
    if "OUT" in wind_dir:
        return _clamp(50 + wind_mph * 2.2)
    if "IN" in wind_dir:
        return _clamp(50 - wind_mph * 2.2)
    if wind_dir in {"S", "SW", "SE"}:
        return _clamp(50 + wind_mph * 0.7)
    if wind_dir in {"N", "NW", "NE"}:
        return _clamp(50 - wind_mph * 0.7)
    return _clamp(50 + wind_mph * 0.1)


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))
