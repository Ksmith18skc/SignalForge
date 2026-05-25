"""Ingestion: pull provider data, normalize, persist.

Order of precedence per the spec:
  1. Falcon first (primary)
  2. Polymarket Analytics enrichment on top
  3. Polycopy enrichment on top
  4. Fall back to MockProvider for anything missing
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Market, MarketSnapshot, Trade, Trader
from app.providers.base import BaseProvider, ProviderSource
from app.providers.falcon import FalconProvider
from app.providers.kalshi import KalshiProvider
from app.providers.mock import MockProvider
from app.providers.polymarket import PolymarketProvider

logger = logging.getLogger(__name__)


def build_providers() -> dict[str, BaseProvider]:
    """Construct the active provider set. Missing credentials -> MockProvider."""
    s = get_settings()
    providers: dict[str, BaseProvider] = {}

    if s.has_falcon_credentials():
        providers["primary"] = FalconProvider(s.falcon_api_key, s.falcon_base_url)
        logger.info("Falcon provider active")
    else:
        providers["primary"] = MockProvider()
        logger.warning("FALCON_API_KEY missing — using MockProvider as primary")

    providers["polymarket"] = PolymarketProvider(s.polymarket_api_key, s.polymarket_base_url)
    providers["kalshi"] = KalshiProvider(s.kalshi_api_key, s.kalshi_base_url)
    providers["mock"] = MockProvider()
    return providers


# --------------------------------------------------------------------------
# Trader enrichment
# --------------------------------------------------------------------------


_TRADER_FIELDS_FROM_STATS = {
    "trust_score",
    "trader_rank",
    "total_pnl",
    "net_worth",
    "seven_day_return",
    "win_rate",
    "total_positions",
    "category_strengths",
    "polycopy_rank",
    "polycopy_pnl",
    "polycopy_win_rate",
    "polycopy_trade_count",
}


def _apply_stats(trader: Trader, stats: dict[str, Any]) -> None:
    for field in _TRADER_FIELDS_FROM_STATS:
        if field in stats and stats[field] is not None:
            setattr(trader, field, stats[field])


async def enrich_trader(
    db: Session,
    trader: Trader,
    providers: dict[str, BaseProvider],
) -> Trader:
    """Pull stats from each provider in priority order and merge in."""
    lookup_key = trader.wallet_address or trader.nickname

    # 1. Primary (Falcon, with mock fallback).
    primary = providers["primary"]
    try:
        stats = await primary.get_trader_stats(lookup_key)
        _apply_stats(trader, stats)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Primary provider failed for %s: %s", lookup_key, exc)

    # 2. Polymarket Analytics enrichment (only overwrites if value present).
    try:
        poly_stats = await providers["polymarket"].get_trader_stats(lookup_key)
        _apply_stats(trader, poly_stats)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Polymarket enrichment failed for %s: %s", lookup_key, exc)

    trader.updated_at = datetime.utcnow()
    db.add(trader)
    db.flush()
    return trader


async def fetch_recent_trades(
    db: Session,
    trader: Trader,
    providers: dict[str, BaseProvider],
    limit: int = 20,
) -> list[Trade]:
    """Pull recent trades for a trader and persist any new ones."""
    lookup_key = trader.wallet_address or trader.nickname
    primary = providers["primary"]
    raw = await primary.get_trader_trades(lookup_key, limit=limit)

    new_trades: list[Trade] = []
    for entry in raw:
        market = upsert_market_from_dict(db, entry)
        external_id = entry.get("external_id")
        if external_id:
            existing = db.scalar(
                select(Trade).where(
                    Trade.external_id == external_id,
                    Trade.trader_id == trader.id,
                )
            )
            if existing:
                if entry.get("outcome") and not existing.outcome:
                    existing.outcome = entry.get("outcome")
                    db.add(existing)
                continue

        ts = entry.get("timestamp")
        ts_parsed = (
            datetime.fromisoformat(ts) if isinstance(ts, str) else (ts or datetime.utcnow())
        )

        trade = Trade(
            trader_id=trader.id,
            market_id=market.id,
            side=entry.get("side", "YES"),
            outcome=entry.get("outcome"),
            price=float(entry.get("price", 0.5)),
            size_usd=float(entry.get("size_usd", 0.0)),
            source=entry.get("source", ProviderSource.MOCK.value),
            external_id=external_id,
            timestamp=ts_parsed,
        )
        db.add(trade)
        new_trades.append(trade)

    db.flush()
    return new_trades


# --------------------------------------------------------------------------
# Market upsert
# --------------------------------------------------------------------------


def upsert_market_from_dict(db: Session, data: dict[str, Any]) -> Market:
    slug = data.get("market_slug") or data.get("slug")
    if not slug:
        raise ValueError("market dict missing slug")

    market = db.scalar(select(Market).where(Market.slug == slug))
    title = data.get("market_title") or data.get("title") or slug

    if market is None:
        market = Market(
            slug=slug,
            title=title,
            platform=data.get("platform", "polymarket"),
            category=data.get("category"),
        )
        db.add(market)

    for f in ("yes_price", "no_price", "liquidity_usd", "volume_24h_usd"):
        if data.get(f) is not None:
            setattr(market, f, data[f])

    end_date = data.get("end_date")
    if isinstance(end_date, str):
        try:
            market.end_date = datetime.fromisoformat(end_date)
        except ValueError:
            pass

    market.updated_at = datetime.utcnow()
    db.flush()
    return market


async def refresh_market(
    db: Session,
    market: Market,
    providers: dict[str, BaseProvider],
) -> MarketSnapshot:
    primary = providers["primary"]
    data = await primary.get_market_data(market.slug)
    upsert_market_from_dict(db, {**data, "slug": market.slug})

    snap = MarketSnapshot(
        market_id=market.id,
        yes_price=data.get("yes_price"),
        no_price=data.get("no_price"),
        liquidity_usd=data.get("liquidity_usd"),
        volume_24h_usd=data.get("volume_24h_usd"),
        captured_at=datetime.utcnow(),
    )
    db.add(snap)
    db.flush()
    return snap


async def discover_markets(
    db: Session,
    providers: dict[str, BaseProvider],
    limit: int = 25,
) -> list[Market]:
    primary = providers["primary"]
    raw = await primary.list_active_markets(limit=limit)
    markets = [upsert_market_from_dict(db, m) for m in raw]
    return markets
