"""Pydantic schemas for API I/O."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CopyMode = Literal["disabled", "alert_only", "paper", "live"]
SignalSource = Literal["Falcon", "PolymarketAnalytics", "Polycopy", "Mock"]


# ----------------------------- Trader ---------------------------------------


class TraderBase(BaseModel):
    nickname: str
    wallet_address: str | None = None
    platform: str = "polymarket"
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None
    trust_score: float = 50.0

    trader_rank: int | None = None
    total_pnl: float | None = None
    net_worth: float | None = None
    seven_day_return: float | None = None
    win_rate: float | None = None
    category_strengths: dict[str, float] = Field(default_factory=dict)
    total_positions: int | None = None

    polycopy_rank: int | None = None
    polycopy_pnl: float | None = None
    polycopy_win_rate: float | None = None
    polycopy_trade_count: int | None = None
    copy_enabled: bool = False
    copy_mode: CopyMode = "alert_only"


class TraderCreate(TraderBase):
    pass


class TraderOut(TraderBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    updated_at: datetime


# ----------------------------- Market ---------------------------------------


class MarketOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    slug: str
    platform: str
    title: str
    category: str | None
    yes_price: float | None
    no_price: float | None
    liquidity_usd: float | None
    volume_24h_usd: float | None
    end_date: datetime | None
    is_active: bool


# ----------------------------- Signal ---------------------------------------


class SignalOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    market_id: int
    trader_id: int | None
    signal_type: str
    side: str | None
    outcome: str | None = None
    entry_price: float | None
    size_usd: float | None
    score: float
    confidence: float
    reason: str
    source: SignalSource
    score_breakdown: dict[str, Any] | None
    generated_for_date: str | None = None
    created_at: datetime

    # Enriched fields for the dashboard. Populated by the route, not the ORM.
    wallet: str | None = None
    trader_nickname: str | None = None
    market_title: str | None = None
    market_slug: str | None = None
    market_platform: str | None = None
    market_created_at: datetime | None = None
    market_updated_at: datetime | None = None
    market_end_date: datetime | None = None
    market_url: str | None = None
    trader_url: str | None = None


# ----------------------------- Alert ----------------------------------------


class AlertOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    signal_id: int | None
    channel: str
    status: str
    message: str
    error: str | None
    generated_for_date: str | None = None
    created_at: datetime


# ----------------------------- Dashboard ------------------------------------


class WatchlistHealth(BaseModel):
    total_traders: int
    enriched_traders: int
    avg_trust_score: float
    avg_win_rate: float | None
    enabled_for_copy: int


class DashboardSummary(BaseModel):
    active_signals: list[SignalOut]
    top_traders: list[TraderOut]
    highest_conviction_markets: list[MarketOut]
    recent_alerts: list[AlertOut]
    simulated_pnl_usd: float
    watchlist_health: WatchlistHealth


# ----------------------------- Scan responses -------------------------------


class ScanResult(BaseModel):
    generated_for_date: str | None = None
    markets_seen: int = 0
    markets_for_card_date: int = 0
    stale_markets_skipped: int = 0
    positions_written: int = 0
    alerts_written: int = 0
    preserved_prior_date_rows: int = 0
    reason: str = "ok"
    scanned_markets: int
    scanned_traders: int
    new_signals: int
    new_alerts: int
    duration_seconds: float
