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
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Market, MarketSnapshot, Trade, Trader
from app.providers.base import BaseProvider, ProviderSource
from app.providers.falcon import FalconProvider
from app.providers.kalshi import KalshiProvider
from app.providers.mock import MockProvider
from app.providers.polymarket import PolymarketProvider
from app.services import ingestion_health

logger = logging.getLogger(__name__)

# Hard cap so a runaway provider can't push megabytes of "external_id" into the
# trades index. TEXT has no DB-level limit; this is a sanity check, not a
# storage constraint.
_EXTERNAL_ID_MAX_LEN = 2048
_EXTERNAL_ID_WARN_LEN = 1000


def _sanitize_external_id(raw: Any) -> str | None:
    """Coerce, validate, and (only as last resort) truncate an external_id.

    Returns None if the value is missing/empty. Logs a warning above
    _EXTERNAL_ID_WARN_LEN and truncates only above _EXTERNAL_ID_MAX_LEN so we
    preserve the full ID whenever it's "merely long" instead of "absurd".
    """
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    length = len(value)
    if length > _EXTERNAL_ID_WARN_LEN:
        logger.warning(
            "external_id is %d chars (warn>%d, cap=%d) — preview=%r",
            length, _EXTERNAL_ID_WARN_LEN, _EXTERNAL_ID_MAX_LEN, value[:60],
        )
    if length > _EXTERNAL_ID_MAX_LEN:
        ingestion_health.record_trade_skipped_oversized()
        logger.error(
            "external_id length %d exceeds cap %d — truncating",
            length, _EXTERNAL_ID_MAX_LEN,
        )
        return value[:_EXTERNAL_ID_MAX_LEN]
    return value


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
    """Pull recent trades for a trader and persist any new ones.

    One bad row (oversized external_id, FK error, etc.) is isolated with a
    savepoint and must not poison the outer Session — the scanner needs to
    keep going.
    """
    lookup_key = trader.wallet_address or trader.nickname
    primary = providers["primary"]
    raw = await primary.get_trader_trades(lookup_key, limit=limit)

    new_trades: list[Trade] = []
    for entry in raw:
        try:
            trade = _ingest_one_trade(db, trader, entry)
        except SQLAlchemyError as exc:
            # The savepoint context auto-rolled back this trade's writes; the
            # outer Session is still usable. Just record + move on.
            ingestion_health.record_failure(f"{type(exc).__name__}: {exc}")
            ingestion_health.record_rollback()
            logger.warning(
                "trade insert failed for trader=%s external_id=%r: %s",
                trader.nickname,
                str(entry.get("external_id"))[:80],
                exc,
            )
            continue
        except Exception as exc:  # noqa: BLE001
            ingestion_health.record_failure(f"{type(exc).__name__}: {exc}")
            ingestion_health.record_rollback()
            logger.exception(
                "unexpected error ingesting trade for trader=%s: %s",
                trader.nickname, exc,
            )
            continue
        if trade is not None:
            new_trades.append(trade)
            ingestion_health.record_trade_inserted()

    return new_trades


def _ingest_one_trade(
    db: Session,
    trader: Trader,
    entry: dict[str, Any],
) -> Trade | None:
    """Persist a single trade inside its own savepoint.

    Returns the new Trade row, or None if the entry was an idempotent skip
    (duplicate external_id). Raises on actual DB errors so the caller can
    record + rollback.

    Wrapping everything (market upsert, dedupe check, insert) in one
    begin_nested() means a failure on ANY step rolls back just this trade's
    work — the outer scanner transaction keeps its earlier writes intact.
    """
    # SAVEPOINT so a failed insert here can be rolled back without dragging
    # down the outer scanner transaction. begin_nested() works on both
    # SQLite (emulated savepoints) and Postgres.
    with db.begin_nested():
        external_id = _sanitize_external_id(entry.get("external_id"))

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
                    db.flush()
                return None

        market = upsert_market_from_dict(db, entry)

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
        db.flush()
    return trade


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
