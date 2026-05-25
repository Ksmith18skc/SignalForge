"""KalshiProvider — placeholder for Kalshi market data.

Same shape as PolymarketProvider: when no API key is configured, all calls
go through MockProvider so SignalForge stays runnable end-to-end.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.providers.base import BaseProvider, ProviderSource
from app.providers.mock import MockProvider

logger = logging.getLogger(__name__)


class KalshiProvider(BaseProvider):
    source = ProviderSource.KALSHI

    def __init__(self, api_key: str | None, base_url: str, timeout: float = 10.0) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._fallback = MockProvider()

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": "SignalForge/0.1"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, params=params, headers=self._headers())
            resp.raise_for_status()
            return resp.json()

    async def get_trader_stats(self, wallet: str) -> dict[str, Any]:
        logger.debug("KalshiProvider.get_trader_stats falling back to mock for %s", wallet)
        data = await self._fallback.get_trader_stats(wallet)
        data["source"] = self.source.value
        return data

    async def get_trader_trades(self, wallet: str, limit: int = 50) -> list[dict[str, Any]]:
        trades = await self._fallback.get_trader_trades(wallet, limit)
        for t in trades:
            t["source"] = self.source.value
        return trades

    async def get_market_data(self, market_slug: str) -> dict[str, Any]:
        data = await self._fallback.get_market_data(market_slug)
        data["source"] = self.source.value
        data["platform"] = "kalshi"
        return data

    async def get_orderbook(self, market_slug: str) -> dict[str, Any]:
        data = await self._fallback.get_orderbook(market_slug)
        data["source"] = self.source.value
        return data

    async def get_cross_market_comparison(self, topic: str) -> list[dict[str, Any]]:
        return await self._fallback.get_cross_market_comparison(topic)

    async def get_sentiment_signals(self, market_slug: str) -> dict[str, Any]:
        data = await self._fallback.get_sentiment_signals(market_slug)
        data["source"] = self.source.value
        return data

    async def list_active_markets(self, limit: int = 25) -> list[dict[str, Any]]:
        markets = await self._fallback.list_active_markets(limit)
        for m in markets:
            m["platform"] = "kalshi"
        return markets
