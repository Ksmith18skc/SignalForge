"""Provider abstraction.

Every concrete provider returns plain dicts (not ORM objects) so the ingestion
layer owns persistence. Async methods let us swap in real HTTP clients later
without rewriting callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class ProviderSource(str, Enum):
    FALCON = "Falcon"
    POLYMARKET_ANALYTICS = "PolymarketAnalytics"
    POLYCOPY = "Polycopy"
    KALSHI = "Kalshi"
    MOCK = "Mock"


class BaseProvider(ABC):
    """Common interface for all data providers.

    Concrete providers should return dicts with stable, snake_case keys so
    the ingestion service can normalize them into ORM rows without caring
    about which upstream produced them.
    """

    source: ProviderSource

    @abstractmethod
    async def get_trader_stats(self, wallet: str) -> dict[str, Any]:
        """Return profile metrics for a single wallet."""

    @abstractmethod
    async def get_trader_trades(self, wallet: str, limit: int = 50) -> list[dict[str, Any]]:
        """Return the wallet's most recent trades, newest first."""

    @abstractmethod
    async def get_market_data(self, market_slug: str) -> dict[str, Any]:
        """Return prices + liquidity for a single market."""

    @abstractmethod
    async def get_orderbook(self, market_slug: str) -> dict[str, Any]:
        """Return top-of-book for a single market (bids/asks)."""

    @abstractmethod
    async def get_cross_market_comparison(self, topic: str) -> list[dict[str, Any]]:
        """Find equivalent markets across venues for arbitrage detection."""

    @abstractmethod
    async def get_sentiment_signals(self, market_slug: str) -> dict[str, Any]:
        """Return sentiment / flow indicators for a single market."""

    async def list_active_markets(self, limit: int = 25) -> list[dict[str, Any]]:
        """Optional: enumerate active markets. Default no-op for providers that
        don't support market discovery."""
        return []
