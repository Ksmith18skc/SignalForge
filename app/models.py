"""SQLAlchemy ORM models.

These mirror the entities described in the SignalForge spec: traders, markets,
trades, positions, signals, alerts, and market_snapshots. The trader model
carries enrichment fields from Polymarket Analytics + Polycopy so we can score
on a combined profile.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base
from app.services.card_date import arizona_today


def _utcnow() -> datetime:
    return datetime.utcnow()


class Trader(Base):
    __tablename__ = "traders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    wallet_address: Mapped[str | None] = mapped_column(String(128), unique=True, index=True, nullable=True)
    nickname: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    platform: Mapped[str] = mapped_column(String(32), default="polymarket")  # polymarket | kalshi
    tags: Mapped[list[str] | None] = mapped_column(JSON, default=list)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # operator-assigned trust (0-100)
    trust_score: Mapped[float] = mapped_column(Float, default=50.0)

    # --- Polymarket Analytics enrichment ---
    trader_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_worth: Mapped[float | None] = mapped_column(Float, nullable=True)
    seven_day_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    win_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    category_strengths: Mapped[dict[str, float] | None] = mapped_column(JSON, default=dict)
    total_positions: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # --- Polycopy enrichment ---
    polycopy_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    polycopy_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    polycopy_win_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    polycopy_trade_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    copy_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    # disabled | alert_only | paper | live   (MVP forces alert_only by default)
    copy_mode: Mapped[str] = mapped_column(String(16), default="alert_only")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)

    trades: Mapped[list["Trade"]] = relationship(back_populates="trader", cascade="all, delete-orphan")


class Market(Base):
    __tablename__ = "markets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    slug: Mapped[str] = mapped_column(String(256), unique=True, index=True)
    platform: Mapped[str] = mapped_column(String(32), default="polymarket")
    title: Mapped[str] = mapped_column(String(512))
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    yes_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    no_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    liquidity_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume_24h_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    end_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)

    trades: Mapped[list["Trade"]] = relationship(back_populates="market", cascade="all, delete-orphan")
    snapshots: Mapped[list["MarketSnapshot"]] = relationship(
        back_populates="market", cascade="all, delete-orphan"
    )


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trader_id: Mapped[int] = mapped_column(ForeignKey("traders.id"), index=True)
    market_id: Mapped[int] = mapped_column(ForeignKey("markets.id"), index=True)
    side: Mapped[str] = mapped_column(String(8))  # YES | NO | BUY | SELL
    outcome: Mapped[str | None] = mapped_column(String(64), nullable=True)  # Over | Under | team | YES | NO
    price: Mapped[float] = mapped_column(Float)
    size_usd: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32), default="mock")
    # Falcon trade IDs can be long composite strings (wallet+market+timestamp).
    # Use TEXT so we never truncate; the index is created with the column.
    external_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)

    trader: Mapped[Trader] = relationship(back_populates="trades")
    market: Mapped[Market] = relationship(back_populates="trades")


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[int] = mapped_column(ForeignKey("markets.id"), index=True)
    trader_id: Mapped[int | None] = mapped_column(ForeignKey("traders.id"), nullable=True, index=True)
    signal_type: Mapped[str] = mapped_column(String(64))
    side: Mapped[str | None] = mapped_column(String(8), nullable=True)
    outcome: Mapped[str | None] = mapped_column(String(64), nullable=True)
    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    size_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(32), default="Mock")  # Falcon | PolymarketAnalytics | Polycopy | Mock
    score_breakdown: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=dict)
    generated_for_date: Mapped[str | None] = mapped_column(
        String(10), nullable=True, index=True, default=arizona_today
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)

    market: Mapped[Market] = relationship()
    trader: Mapped[Trader | None] = relationship()
    alerts: Mapped[list["Alert"]] = relationship(back_populates="signal", cascade="all, delete-orphan")


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id: Mapped[int | None] = mapped_column(ForeignKey("signals.id"), nullable=True, index=True)
    channel: Mapped[str] = mapped_column(String(32))  # console | discord | telegram | email
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending | sent | failed
    message: Mapped[str] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    generated_for_date: Mapped[str | None] = mapped_column(
        String(10), nullable=True, index=True, default=arizona_today
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)

    signal: Mapped[Signal | None] = relationship(back_populates="alerts")


class MarketSnapshot(Base):
    """Point-in-time pricing/liquidity snapshot for a market.

    The scanner writes these on each pass so the signal engine can detect
    price moves after a smart wallet entry.
    """

    __tablename__ = "market_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[int] = mapped_column(ForeignKey("markets.id"), index=True)
    yes_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    no_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    liquidity_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume_24h_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)

    market: Mapped[Market] = relationship(back_populates="snapshots")

