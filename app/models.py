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
    positions: Mapped[list["Position"]] = relationship(back_populates="trader", cascade="all, delete-orphan")


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
    external_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)

    trader: Mapped[Trader] = relationship(back_populates="trades")
    market: Mapped[Market] = relationship(back_populates="trades")


class Position(Base):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trader_id: Mapped[int] = mapped_column(ForeignKey("traders.id"), index=True)
    market_id: Mapped[int] = mapped_column(ForeignKey("markets.id"), index=True)
    side: Mapped[str] = mapped_column(String(8))
    avg_price: Mapped[float] = mapped_column(Float)
    size_usd: Mapped[float] = mapped_column(Float)
    unrealized_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)

    trader: Mapped[Trader] = relationship(back_populates="positions")
    market: Mapped[Market] = relationship()


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


class PitcherStatcastSummary(Base):
    __tablename__ = "pitcher_statcast_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, index=True)
    player_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    season: Mapped[int] = mapped_column(Integer, index=True)
    last_n_days: Mapped[int] = mapped_column(Integer, index=True)
    games: Mapped[int | None] = mapped_column(Integer, nullable=True)
    innings_pitched: Mapped[float | None] = mapped_column(Float, nullable=True)
    strikeouts: Mapped[int | None] = mapped_column(Integer, nullable=True)
    walks: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pitch_count_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    strikeouts_per_start: Mapped[float | None] = mapped_column(Float, nullable=True)
    whiff_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    chase_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    k_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)
    source: Mapped[str] = mapped_column(String(32), default="pybaseball")


class BatterStatcastSummary(Base):
    __tablename__ = "batter_statcast_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, index=True)
    player_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    season: Mapped[int] = mapped_column(Integer, index=True)
    last_n_days: Mapped[int] = mapped_column(Integer, index=True)
    games: Mapped[int | None] = mapped_column(Integer, nullable=True)
    innings_pitched: Mapped[float | None] = mapped_column(Float, nullable=True)
    strikeouts: Mapped[int | None] = mapped_column(Integer, nullable=True)
    walks: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pitch_count_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    strikeouts_per_start: Mapped[float | None] = mapped_column(Float, nullable=True)
    whiff_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    chase_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    k_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)
    source: Mapped[str] = mapped_column(String(32), default="pybaseball")


class MlbGame(Base):
    __tablename__ = "mlb_games"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_pk: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    game_date: Mapped[str] = mapped_column(String(10), index=True)
    home_team: Mapped[str] = mapped_column(String(128))
    away_team: Mapped[str] = mapped_column(String(128))
    venue: Mapped[str | None] = mapped_column(String(128), nullable=True)
    probable_home_pitcher: Mapped[str | None] = mapped_column(String(128), nullable=True)
    probable_home_pitcher_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    probable_away_pitcher: Mapped[str | None] = mapped_column(String(128), nullable=True)
    probable_away_pitcher_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    game_status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    start_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    weather_location_query: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)


class MlbGameEnvironmentSnapshot(Base):
    __tablename__ = "mlb_game_environment_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_pk: Mapped[int] = mapped_column(Integer, index=True)
    temperature_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    humidity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    precipitation_risk: Mapped[float | None] = mapped_column(Float, nullable=True)
    park_factor: Mapped[float | None] = mapped_column(Float, nullable=True)
    run_environment_score: Mapped[float] = mapped_column(Float, default=50.0)
    under_environment_score: Mapped[float] = mapped_column(Float, default=50.0)
    k_environment_score: Mapped[float] = mapped_column(Float, default=50.0)
    warnings: Mapped[list[str] | None] = mapped_column(JSON, default=list)
    raw_weather: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=dict)
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    source: Mapped[str] = mapped_column(String(32), default="weatherapi")


class MlbOddsSnapshot(Base):
    __tablename__ = "mlb_odds_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_pk: Mapped[int] = mapped_column(Integer, index=True)
    market: Mapped[str] = mapped_column(String(64), default="game_total")
    sportsbook_event_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    consensus_total_line: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_over_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_over_book: Mapped[str | None] = mapped_column(String(64), nullable=True)
    best_under_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_under_book: Mapped[str | None] = mapped_column(String(64), nullable=True)
    consensus_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    line_disagreement: Mapped[float] = mapped_column(Float, default=0.0)
    book_count: Mapped[int] = mapped_column(Integer, default=0)
    stale_book_candidates: Mapped[list[str] | None] = mapped_column(JSON, default=list)
    movement_direction: Mapped[str | None] = mapped_column(String(16), nullable=True)
    steam_velocity: Mapped[float | None] = mapped_column(Float, nullable=True)
    rows: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, default=list)
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    source: Mapped[str] = mapped_column(String(32), default="odds_api")


class MlbPitcherPropSnapshot(Base):
    __tablename__ = "mlb_pitcher_prop_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_pk: Mapped[int] = mapped_column(Integer, index=True)
    pitcher_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    pitcher_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prop: Mapped[str] = mapped_column(String(64), default="strikeouts")
    line: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_over_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_over_book: Mapped[str | None] = mapped_column(String(64), nullable=True)
    best_under_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_under_book: Mapped[str | None] = mapped_column(String(64), nullable=True)
    consensus_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    line_disagreement: Mapped[float] = mapped_column(Float, default=0.0)
    book_count: Mapped[int] = mapped_column(Integer, default=0)
    movement_direction: Mapped[str | None] = mapped_column(String(16), nullable=True)
    steam_velocity: Mapped[float | None] = mapped_column(Float, nullable=True)
    rows: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, default=list)
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    source: Mapped[str] = mapped_column(String(32), default="odds_api")


class PitcherPropOddsSnapshot(Base):
    __tablename__ = "pitcher_prop_odds_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_pk: Mapped[int] = mapped_column(Integer, index=True)
    sportsbook_event_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    player_name: Mapped[str] = mapped_column(String(128), index=True)
    matched_pitcher_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    line: Mapped[float] = mapped_column(Float)
    over_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    under_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    sportsbook: Mapped[str] = mapped_column(String(64), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    raw: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=dict)
    source: Mapped[str] = mapped_column(String(32), default="odds_api")


class MlbEdge(Base):
    __tablename__ = "mlb_edges"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_pk: Mapped[int] = mapped_column(Integer, index=True)
    edge_type: Mapped[str] = mapped_column(String(32), index=True)  # game_total | pitcher_strikeouts
    market: Mapped[str] = mapped_column(String(256))
    side: Mapped[str] = mapped_column(String(16))
    line: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_book: Mapped[str | None] = mapped_column(String(64), nullable=True)
    best_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    consensus_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    score: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    confidence: Mapped[str] = mapped_column(String(16), default="low")
    action: Mapped[str] = mapped_column(String(64), default="Pass")
    chase_risk: Mapped[str] = mapped_column(String(16), default="medium")
    reasons: Mapped[list[str] | None] = mapped_column(JSON, default=list)
    warnings: Mapped[list[str] | None] = mapped_column(JSON, default=list)
    data_sources_used: Mapped[list[str] | None] = mapped_column(JSON, default=list)
    factors: Mapped[dict[str, float] | None] = mapped_column(JSON, default=dict)
    generated_for_date: Mapped[str] = mapped_column(String(10), index=True)
    opening_line: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_line: Mapped[float | None] = mapped_column(Float, nullable=True)
    recommended_line: Mapped[float | None] = mapped_column(Float, nullable=True)
    closing_line: Mapped[float | None] = mapped_column(Float, nullable=True)
    closing_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    result: Mapped[str | None] = mapped_column(String(64), nullable=True)
    win_loss_push: Mapped[str | None] = mapped_column(String(8), nullable=True)
    implied_probability_at_entry: Mapped[float | None] = mapped_column(Float, nullable=True)
    implied_probability_at_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    clv_points: Mapped[float | None] = mapped_column(Float, nullable=True)
    clv_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    roi_units: Mapped[float | None] = mapped_column(Float, nullable=True)
    graded_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)


class MlbEdgeFactor(Base):
    __tablename__ = "mlb_edge_factors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    edge_id: Mapped[int] = mapped_column(ForeignKey("mlb_edges.id"), index=True)
    name: Mapped[str] = mapped_column(String(64))
    value: Mapped[float] = mapped_column(Float)
    weight: Mapped[float] = mapped_column(Float)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)


class MlbDailyCard(Base):
    __tablename__ = "mlb_daily_cards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    card_date: Mapped[str] = mapped_column(String(10), unique=True, index=True)
    top_game_totals: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, default=list)
    top_pitcher_strikeouts: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, default=list)
    near_misses: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, default=list)
    pass_list: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, default=list)
    data_quality_summary: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)
