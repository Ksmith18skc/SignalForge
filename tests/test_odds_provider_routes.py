from __future__ import annotations

import asyncio

from app.api.routes import odds_providers_health, odds_providers_self_test
from app.models import ProviderHealthState


def test_odds_provider_health_endpoint_reports_persisted_state(db_session) -> None:
    db_session.add_all(
        [
            ProviderHealthState(
                provider="Odds-API",
                enabled=True,
                last_success_at=None,
                last_error_at=None,
                cooldown_until=None,
                recent_failures=1,
                last_status_code=400,
                last_error="bad request",
                last_successful_strategy="plan_limited:DraftKings,FanDuel",
                last_refresh_event_count=0,
                refresh_errors=1,
            ),
            ProviderHealthState(
                provider="SportsGameOdds",
                enabled=True,
                last_success_at=None,
                last_error_at=None,
                cooldown_until=None,
                recent_failures=0,
                last_status_code=200,
                last_error=None,
                last_successful_strategy="mlb_events_params",
                last_refresh_event_count=8,
                refresh_errors=0,
            ),
        ]
    )
    db_session.commit()

    payload = odds_providers_health(db_session)

    assert payload["providers"]
    assert payload["primary"]["provider"] == "Odds-API"
    assert payload["primary"]["refresh_errors"] == 1
    assert payload["primary"]["plan_limit_warning"] == "Odds-API plan limited to DraftKings/FanDuel."
    assert payload["backup"]["provider"] == "SportsGameOdds"
    assert payload["backup"]["last_refresh_event_count"] == 8


def test_odds_provider_self_test_endpoint_returns_sanitized_results(monkeypatch, db_session) -> None:
    async def _ok_events(self, sport, **kwargs):  # noqa: ANN001
        return []

    async def _ok_get(self, path, params=None, max_retries=1):  # noqa: ANN001
        return {"success": True, "data": []}

    monkeypatch.setattr("app.providers.odds_api.OddsApiProvider.events", _ok_events)
    monkeypatch.setattr("app.providers.sportsgameodds.SportsGameOddsProvider._get", _ok_get)

    payload = asyncio.run(odds_providers_self_test(db_session))

    assert payload["results"]
    assert all(result["ok"] for result in payload["results"])
    assert all("apiKey" not in result["sanitized_url"] for result in payload["results"])
