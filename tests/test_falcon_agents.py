"""Tests for the Falcon agent registry + env override layer."""

from __future__ import annotations

import importlib
import os
from unittest.mock import patch

from app.providers import falcon_agents


def test_registry_exposes_expected_agent_ids():
    expected = {
        "POLYMARKET_MARKETS": 574,
        "POLYMARKET_TRADES": 556,
        "POLYMARKET_CANDLES": 568,
        "POLYMARKET_ORDERBOOK": 572,
        "POLYMARKET_PNL": 569,
        "POLYMARKET_LEADERBOARD": 579,
        "SCORE_LEADERBOARD": 584,
        "WALLET_360": 581,
        "MARKET_INSIGHTS": 575,
        "KALSHI_MARKETS": 565,
        "KALSHI_TRADES": 573,
        "SOCIAL_PULSE": 585,
    }
    # Re-import to defeat any test-order memoisation of the registry.
    importlib.reload(falcon_agents)
    for name, value in expected.items():
        assert falcon_agents.AGENT_IDS[name] == value
        assert getattr(falcon_agents.Agent, name) == value


def test_env_override_takes_precedence():
    with patch.dict(os.environ, {"FALCON_AGENT_WALLET_360": "9001"}, clear=False):
        falcon_agents.reload_registry()
        assert falcon_agents.AGENT_IDS["WALLET_360"] == 9001
        assert falcon_agents.Agent.WALLET_360 == 9001
    # Cleanup: reload without the override.
    falcon_agents.reload_registry()
    assert falcon_agents.Agent.WALLET_360 == 581


def test_invalid_env_override_falls_back_to_default():
    with patch.dict(os.environ, {"FALCON_AGENT_KALSHI_TRADES": "not-an-int"}, clear=False):
        falcon_agents.reload_registry()
        assert falcon_agents.AGENT_IDS["KALSHI_TRADES"] == 573
    falcon_agents.reload_registry()


def test_all_specs_includes_human_labels():
    specs = falcon_agents.all_specs()
    by_name = {s.name: s for s in specs}
    assert by_name["WALLET_360"].label == "Wallet 360"
    assert by_name["SOCIAL_PULSE"].label == "Social Pulse"
    assert by_name["POLYMARKET_LEADERBOARD"].id == 579
