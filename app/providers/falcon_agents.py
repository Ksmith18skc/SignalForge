"""Falcon agent registry.

All Falcon agent IDs live here so the rest of the codebase never hardcodes a
numeric ID. Defaults match the working IDs the operator supplied; each one can
be overridden by an env var of the form ``FALCON_AGENT_<NAME>=<int>`` so a
single deployment can be repointed at a different agent without a code change.

The registry exposes:

* ``AGENT_<NAME>`` module constants for direct import.
* ``AGENT_IDS`` dict for the diagnostics endpoint / health panel.
* ``AGENT_LABELS`` for human-readable agent names.
* ``Agent`` enum-style class so callers can use ``Agent.WALLET_360`` rather
  than a magic number.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Defaults — verified by the operator. Override per-deploy with env vars.
_DEFAULTS: dict[str, int] = {
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

_LABELS: dict[str, str] = {
    "POLYMARKET_MARKETS": "Polymarket Markets",
    "POLYMARKET_TRADES": "Polymarket Trades",
    "POLYMARKET_CANDLES": "Polymarket Candlesticks",
    "POLYMARKET_ORDERBOOK": "Polymarket Orderbook",
    "POLYMARKET_PNL": "Polymarket PnL",
    "POLYMARKET_LEADERBOARD": "Polymarket Leaderboard",
    "SCORE_LEADERBOARD": "Falcon Score Leaderboard",
    "WALLET_360": "Wallet 360",
    "MARKET_INSIGHTS": "Market Insights",
    "KALSHI_MARKETS": "Kalshi Markets",
    "KALSHI_TRADES": "Kalshi Trades",
    "SOCIAL_PULSE": "Social Pulse",
}


def _env_override(name: str, default: int) -> int:
    raw = os.environ.get(f"FALCON_AGENT_{name}")
    if not raw:
        return default
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        # Bad override → fall back to default rather than crash boot.
        return default


def _build_registry() -> dict[str, int]:
    return {name: _env_override(name, default) for name, default in _DEFAULTS.items()}


# Live registry — re-resolved at import time. Tests that mutate env should
# call ``reload_registry()`` to pick up the change.
AGENT_IDS: dict[str, int] = _build_registry()
AGENT_LABELS: dict[str, str] = dict(_LABELS)


def reload_registry() -> dict[str, int]:
    """Re-read env overrides and refresh the module-level registry."""
    global AGENT_IDS
    AGENT_IDS = _build_registry()
    # Also refresh the Agent class constants so existing references stay in sync.
    for name, value in AGENT_IDS.items():
        setattr(Agent, name, value)
    return AGENT_IDS


@dataclass(frozen=True)
class _AgentSpec:
    """Static descriptor for an agent — used by the diagnostics endpoint."""

    name: str
    label: str
    id: int


def agent_spec(name: str) -> _AgentSpec:
    return _AgentSpec(name=name, label=AGENT_LABELS.get(name, name), id=AGENT_IDS[name])


def all_specs() -> list[_AgentSpec]:
    return [agent_spec(name) for name in AGENT_IDS]


# ---- ergonomic constants -------------------------------------------------


class Agent:
    """Namespace for resolved agent IDs.

    Prefer ``Agent.WALLET_360`` over the raw int — it survives env overrides
    and keeps grep-ability ('agent_id=Agent.X' shows every call site).
    """

    POLYMARKET_MARKETS = AGENT_IDS["POLYMARKET_MARKETS"]
    POLYMARKET_TRADES = AGENT_IDS["POLYMARKET_TRADES"]
    POLYMARKET_CANDLES = AGENT_IDS["POLYMARKET_CANDLES"]
    POLYMARKET_ORDERBOOK = AGENT_IDS["POLYMARKET_ORDERBOOK"]
    POLYMARKET_PNL = AGENT_IDS["POLYMARKET_PNL"]
    POLYMARKET_LEADERBOARD = AGENT_IDS["POLYMARKET_LEADERBOARD"]
    SCORE_LEADERBOARD = AGENT_IDS["SCORE_LEADERBOARD"]
    WALLET_360 = AGENT_IDS["WALLET_360"]
    MARKET_INSIGHTS = AGENT_IDS["MARKET_INSIGHTS"]
    KALSHI_MARKETS = AGENT_IDS["KALSHI_MARKETS"]
    KALSHI_TRADES = AGENT_IDS["KALSHI_TRADES"]
    SOCIAL_PULSE = AGENT_IDS["SOCIAL_PULSE"]


__all__ = [
    "AGENT_IDS",
    "AGENT_LABELS",
    "Agent",
    "all_specs",
    "agent_spec",
    "reload_registry",
]
