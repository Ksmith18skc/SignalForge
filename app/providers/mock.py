"""MockProvider — deterministic synthetic data so the MVP runs without keys.

This is what every other provider falls back to when its credentials are
missing. The shape of returned dicts mirrors what FalconProvider will return
once wired up, so the ingestion layer doesn't need a separate code path.
"""

from __future__ import annotations

import hashlib
import random
from datetime import datetime, timedelta
from typing import Any

from app.providers.base import BaseProvider, ProviderSource

_SAMPLE_MARKETS = [
    {
        "slug": "us-recession-2026",
        "title": "Will the US enter a recession in 2026?",
        "category": "macro",
    },
    {
        "slug": "fed-cuts-june-2026",
        "title": "Will the Fed cut rates at the June 2026 meeting?",
        "category": "macro",
    },
    {
        "slug": "btc-150k-eoy-2026",
        "title": "Will BTC close above $150k on 2026-12-31?",
        "category": "crypto",
    },
    {
        "slug": "nba-finals-2026-celtics",
        "title": "Will the Celtics win the 2026 NBA Finals?",
        "category": "sports",
    },
    {
        "slug": "openai-ipo-2026",
        "title": "Will OpenAI file for IPO in 2026?",
        "category": "business",
    },
    {
        "slug": "mlb-mvp-2026-judge",
        "title": "Will Aaron Judge win the 2026 AL MVP?",
        "category": "sports",
    },
]


def _stable_seed(s: str) -> int:
    """Same string -> same number. Lets the mock return consistent values."""
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**31)


class MockProvider(BaseProvider):
    source = ProviderSource.MOCK

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    # ---- trader -----------------------------------------------------------

    async def get_trader_stats(self, wallet: str) -> dict[str, Any]:
        rng = random.Random(_stable_seed(wallet))
        return {
            "wallet": wallet,
            "trader_rank": rng.randint(1, 5000),
            "total_pnl": round(rng.uniform(-50_000, 500_000), 2),
            "net_worth": round(rng.uniform(1_000, 1_000_000), 2),
            "seven_day_return": round(rng.uniform(-0.2, 0.5), 4),
            "win_rate": round(rng.uniform(0.45, 0.75), 4),
            "total_positions": rng.randint(5, 500),
            "category_strengths": {
                "sports": round(rng.uniform(0, 1), 2),
                "politics": round(rng.uniform(0, 1), 2),
                "crypto": round(rng.uniform(0, 1), 2),
                "macro": round(rng.uniform(0, 1), 2),
            },
            "polycopy_rank": rng.randint(1, 1000),
            "polycopy_pnl": round(rng.uniform(-10_000, 200_000), 2),
            "polycopy_win_rate": round(rng.uniform(0.45, 0.8), 4),
            "polycopy_trade_count": rng.randint(10, 800),
            "source": self.source.value,
        }

    async def get_trader_trades(self, wallet: str, limit: int = 50) -> list[dict[str, Any]]:
        rng = random.Random(_stable_seed(wallet) + 1)
        trades = []
        for i in range(min(limit, 10)):
            market = rng.choice(_SAMPLE_MARKETS)
            trades.append(
                {
                    "wallet": wallet,
                    "market_slug": market["slug"],
                    "market_title": market["title"],
                    "category": market["category"],
                    "side": rng.choice(["YES", "NO"]),
                    "price": round(rng.uniform(0.05, 0.95), 3),
                    "size_usd": round(rng.uniform(100, 25_000), 2),
                    "timestamp": (datetime.utcnow() - timedelta(minutes=i * 7)).isoformat(),
                    "external_id": f"mock-{wallet[:6]}-{i}",
                    "source": self.source.value,
                }
            )
        return trades

    # ---- market -----------------------------------------------------------

    async def get_market_data(self, market_slug: str) -> dict[str, Any]:
        rng = random.Random(_stable_seed(market_slug))
        yes = round(rng.uniform(0.05, 0.95), 3)
        title = next(
            (m["title"] for m in _SAMPLE_MARKETS if m["slug"] == market_slug),
            market_slug.replace("-", " ").title(),
        )
        category = next(
            (m["category"] for m in _SAMPLE_MARKETS if m["slug"] == market_slug),
            "general",
        )
        return {
            "slug": market_slug,
            "title": title,
            "category": category,
            "platform": "polymarket",
            "yes_price": yes,
            "no_price": round(1 - yes, 3),
            "liquidity_usd": round(rng.uniform(5_000, 500_000), 2),
            "volume_24h_usd": round(rng.uniform(1_000, 1_000_000), 2),
            "end_date": (datetime.utcnow() + timedelta(days=rng.randint(7, 180))).isoformat(),
            "is_active": True,
            "source": self.source.value,
        }

    async def get_orderbook(self, market_slug: str) -> dict[str, Any]:
        market = await self.get_market_data(market_slug)
        yes = market["yes_price"] or 0.5
        spread = 0.01
        return {
            "market_slug": market_slug,
            "bids": [{"price": round(yes - spread * i, 4), "size": 1000 * i} for i in range(1, 6)],
            "asks": [{"price": round(yes + spread * i, 4), "size": 1000 * i} for i in range(1, 6)],
            "source": self.source.value,
        }

    async def get_cross_market_comparison(self, topic: str) -> list[dict[str, Any]]:
        rng = random.Random(_stable_seed(topic))
        base = round(rng.uniform(0.2, 0.8), 3)
        return [
            {
                "platform": "polymarket",
                "slug": f"{topic}-poly",
                "yes_price": base,
                "liquidity_usd": round(rng.uniform(10_000, 300_000), 2),
            },
            {
                "platform": "kalshi",
                "slug": f"{topic}-kal",
                "yes_price": round(base + rng.uniform(-0.08, 0.08), 3),
                "liquidity_usd": round(rng.uniform(10_000, 300_000), 2),
            },
        ]

    async def get_sentiment_signals(self, market_slug: str) -> dict[str, Any]:
        rng = random.Random(_stable_seed(market_slug) + 2)
        return {
            "market_slug": market_slug,
            "smart_money_flow_usd_24h": round(rng.uniform(-100_000, 200_000), 2),
            "retail_flow_usd_24h": round(rng.uniform(-50_000, 50_000), 2),
            "net_sentiment": round(rng.uniform(-1, 1), 3),
            "source": self.source.value,
        }

    async def list_active_markets(self, limit: int = 25) -> list[dict[str, Any]]:
        out = []
        for m in _SAMPLE_MARKETS[:limit]:
            out.append(await self.get_market_data(m["slug"]))
        return out
