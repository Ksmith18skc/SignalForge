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
    # Falcon trade IDs can be long composite strings (wallet+market+timestamp).
    # Use TEXT so we never truncate; the index is created with the column.
    external_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
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
    normalized_market_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    market_scope: Mapped[str | None] = mapped_column(String(32), nullable=True)
    is_valid: Mapped[bool] = mapped_column(Boolean, default=True)
    validation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
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
    # Tracked-wallet consensus + contributors joined to this edge at scan time.
    # Nullable: pitcher-K edges and totals with no wallet market stay None.
    wallet_context: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    # Additive per-factor point contributions (sum ≈ score − 50). Powers the
    # card's "+12 / −6" score decomposition. Nullable for pre-migration rows.
    score_contributions: Mapped[dict[str, float] | None] = mapped_column(JSON, nullable=True)
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


# ---------------------------------------------------------------------------
# Falcon adaptive-learning subsystem
#
# Persistent, append-or-upsert tables that feed the adaptive scoring,
# calibration, tiering, and explainability layers. None of these tables
# replace existing signal/trade tables — they sit alongside as a learning
# substrate.
# ---------------------------------------------------------------------------


class WalletLearningStats(Base):
    """Per-wallet rolling performance stats.

    One row per wallet. Updated by ``falcon_learning`` on backfill (from
    Wallet 360 + Polymarket PnL) and by ``falcon_retraining`` after every
    signal grading cycle. Bayesian-smoothed metrics ensure single graded
    signals can't flip a wallet's tier.
    """

    __tablename__ = "wallet_learning_stats"

    wallet_address: Mapped[str] = mapped_column(String(128), primary_key=True)
    wallet_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    total_signals: Mapped[int] = mapped_column(Integer, default=0)
    wins: Mapped[int] = mapped_column(Integer, default=0)
    losses: Mapped[int] = mapped_column(Integer, default=0)
    pushes: Mapped[int] = mapped_column(Integer, default=0)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    roi: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_clv: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Direction-specific accuracy (BUY / SELL aka fade).
    sharp_side_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    fade_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Brier-style calibration error (lower is better; None until sample exists).
    calibration_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    sample_size: Mapped[int] = mapped_column(Integer, default=0)
    # Bayesian-smoothed confidence-weight in [0, 1].
    confidence_weight: Mapped[float] = mapped_column(Float, default=0.5)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_updated: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow, index=True,
    )


class WalletMarketSpecialization(Base):
    """Per-(wallet, sport, market_type) performance row.

    Lets the engine reason about "elite in MLB totals, weak in NBA spreads"
    rather than collapsing a wallet to a single tier.
    """

    __tablename__ = "wallet_market_specialization"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    wallet_address: Mapped[str] = mapped_column(String(128), index=True)
    sport: Mapped[str] = mapped_column(String(32), index=True)
    market_type: Mapped[str] = mapped_column(String(64), index=True)
    signals: Mapped[int] = mapped_column(Integer, default=0)
    roi: Mapped[float | None] = mapped_column(Float, nullable=True)
    win_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_clv: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Fraction of signals on the favored side (0 = always underdog, 1 = always favored).
    side_bias: Mapped[float | None] = mapped_column(Float, nullable=True)
    volatility_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 0..1 specialisation; combines sample weight + ROI dominance vs other markets.
    specialization_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_updated: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow, index=True,
    )


class WalletTierHistory(Base):
    """Snapshot of a wallet's tier at a moment in time.

    Append-only — each retraining cycle inserts a new row so tier movement
    is auditable. ``tier`` is one of: elite, trusted, neutral, weak, fade.
    """

    __tablename__ = "wallet_tier_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    wallet_address: Mapped[str] = mapped_column(String(128), index=True)
    tier: Mapped[str] = mapped_column(String(16), index=True)
    sport: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    market_type: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    rolling_roi: Mapped[float | None] = mapped_column(Float, nullable=True)
    rolling_clv: Mapped[float | None] = mapped_column(Float, nullable=True)
    sample_size: Mapped[int] = mapped_column(Integer, default=0)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)


class WalletBehaviorProfile(Base):
    """Behavioural archetype tagging.

    Populated by the deterministic SignalForge clusterer (Wallet 360 + PnL
    features). One row per wallet × archetype label so a wallet can carry
    multiple tags ("sharp_steam", "contrarian_sniper", etc).
    """

    __tablename__ = "wallet_behavior_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    wallet_address: Mapped[str] = mapped_column(String(128), index=True)
    archetype: Mapped[str] = mapped_column(String(64), index=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    features: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=dict)
    source: Mapped[str] = mapped_column(String(32), default="signalforge")
    last_updated: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow, index=True,
    )


class SignalFactorAttribution(Base):
    """One row per (signal, factor) at signal-emit time.

    Captures both the raw factor value (0..1) and the adaptive weight the
    factor carried for this signal's context. After grading, ``realized_pnl``
    / ``win_loss_push`` / ``clv_points`` are backfilled and the
    ``falcon_retraining`` job re-derives factor effectiveness.
    """

    __tablename__ = "signal_factor_attribution"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id: Mapped[int] = mapped_column(ForeignKey("signals.id"), index=True)
    factor_name: Mapped[str] = mapped_column(String(64), index=True)
    factor_value: Mapped[float] = mapped_column(Float)
    factor_weight: Mapped[float] = mapped_column(Float)
    sport: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    market_type: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    win_loss_push: Mapped[str | None] = mapped_column(String(8), nullable=True)
    realized_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    clv_points: Mapped[float | None] = mapped_column(Float, nullable=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    graded_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class SignalWalletContribution(Base):
    """Which wallets contributed to a signal and at what weight."""

    __tablename__ = "signal_wallet_contributions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id: Mapped[int] = mapped_column(ForeignKey("signals.id"), index=True)
    wallet_address: Mapped[str] = mapped_column(String(128), index=True)
    contribution_weight: Mapped[float] = mapped_column(Float, default=0.0)
    side: Mapped[str | None] = mapped_column(String(8), nullable=True)
    size_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)


class SignalLearningSnapshot(Base):
    """Per-signal context snapshot: factors, regime, conflict, calibrated prob.

    Stored as a JSON blob so the explainer panel can replay the exact context
    the engine saw when it emitted the signal, even after factor weights
    have shifted.
    """

    __tablename__ = "signal_learning_snapshots"

    signal_id: Mapped[int] = mapped_column(
        ForeignKey("signals.id"), primary_key=True,
    )
    sport: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    market_type: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    raw_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    calibrated_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    factor_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=dict)
    regime_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=dict)
    conflict_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=dict)
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)


class AdaptiveFactorWeight(Base):
    """Adaptive weight for ``(factor_name, sport, market_type)`` triple.

    ``current_weight`` is what the scorer multiplies the factor value by.
    Updated in place by ``falcon_retraining``. ``sample_size`` < ``min_sample``
    means the scorer should fall back to the static ScoringWeights default.
    """

    __tablename__ = "adaptive_factor_weights"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    factor_name: Mapped[str] = mapped_column(String(64), index=True)
    sport: Mapped[str] = mapped_column(String(32), default="*", index=True)
    market_type: Mapped[str] = mapped_column(String(64), default="*", index=True)
    rolling_roi: Mapped[float | None] = mapped_column(Float, nullable=True)
    rolling_clv: Mapped[float | None] = mapped_column(Float, nullable=True)
    predictive_power: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    sample_size: Mapped[int] = mapped_column(Integer, default=0)
    current_weight: Mapped[float] = mapped_column(Float, default=1.0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow, index=True,
    )


class ConfidenceBandLearning(Base):
    """Score-band → calibrated probability mapping.

    Bins raw signal score (0..100) into bands and tracks the realised win
    rate so the dashboard can show a calibrated probability next to the raw
    score. Score bands are inclusive of ``score_min`` and exclusive of
    ``score_max`` (e.g. [70, 75)).
    """

    __tablename__ = "confidence_band_learning"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sport: Mapped[str] = mapped_column(String(32), default="*", index=True)
    market_type: Mapped[str] = mapped_column(String(64), default="*", index=True)
    score_min: Mapped[float] = mapped_column(Float)
    score_max: Mapped[float] = mapped_column(Float)
    signals: Mapped[int] = mapped_column(Integer, default=0)
    win_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    roi: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_clv: Mapped[float | None] = mapped_column(Float, nullable=True)
    calibrated_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow, index=True,
    )


class SignalRegimeSnapshot(Base):
    """Immutable per-signal market-regime snapshot.

    One row per ``signal_id`` (primary key). Once a snapshot is written it
    is **never updated** — callers that try to re-persist see the existing
    row returned unchanged. This is what makes the regime context an
    auditable historical record: a follow-up retraining pass can re-derive
    bucket statistics from the same raw inputs every time.

    Captured asynchronously by ``falcon_regime_capture`` immediately after a
    signal is emitted. Partial data (some agents unavailable) is allowed —
    ``components`` records what landed and ``enrichment_status`` summarises
    the fan-out result.
    """

    __tablename__ = "signal_regime_snapshots"

    signal_id: Mapped[int] = mapped_column(
        ForeignKey("signals.id"), primary_key=True,
    )
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)

    # Spec'd top-level fields.
    market_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    line_velocity: Mapped[float | None] = mapped_column(Float, nullable=True)
    line_acceleration: Mapped[float | None] = mapped_column(Float, nullable=True)
    volatility_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    liquidity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    orderflow_state: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    steam_state: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    sentiment_state: Mapped[str | None] = mapped_column(String(32), nullable=True)
    orderbook_imbalance: Mapped[float | None] = mapped_column(Float, nullable=True)
    consensus_concentration: Mapped[float | None] = mapped_column(Float, nullable=True)
    elite_disagreement_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    whale_activity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    candlestick_state: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Conflict flags carried over from the contrarian regime detector.
    conflict_flags: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=dict)

    # Final bucket label the engine assigned. Used for per-regime learning.
    regime_classification: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    # What the fan-out actually produced.
    components: Mapped[dict[str, bool] | None] = mapped_column(JSON, default=dict)
    enrichment_status: Mapped[str] = mapped_column(String(16), default="pending")  # pending | partial | complete | failed
    errors: Mapped[list[str] | None] = mapped_column(JSON, default=list)

    # Full agent payloads preserved verbatim so retraining never needs to
    # re-issue the original calls.
    raw_payload_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=dict)


class RegimeLearningStats(Base):
    """Per-regime-classification realised performance.

    Refreshed by ``falcon_retraining.recompute_regime_learning_stats`` after
    every grading pass. Lets the explainer answer "historical ROI of similar
    regimes" without scanning the snapshot table at request time.
    """

    __tablename__ = "regime_learning_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    regime_classification: Mapped[str] = mapped_column(String(64), index=True)
    sport: Mapped[str] = mapped_column(String(32), default="*", index=True)
    market_type: Mapped[str] = mapped_column(String(64), default="*", index=True)
    signals: Mapped[int] = mapped_column(Integer, default=0)
    wins: Mapped[int] = mapped_column(Integer, default=0)
    losses: Mapped[int] = mapped_column(Integer, default=0)
    pushes: Mapped[int] = mapped_column(Integer, default=0)
    avg_roi: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_clv: Mapped[float | None] = mapped_column(Float, nullable=True)
    positive_clv_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    win_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow, index=True,
    )


class SignalRegimeFeatures(Base):
    """Per-signal market-regime snapshot from Falcon agents.

    Derived from Polymarket Candles (568) + Orderbook (572) + Trades (556)
    + Social Pulse (585) + Market Insights (575). Stored as both summary
    columns (cheap to query/filter) and a raw JSON payload (full audit).
    """

    __tablename__ = "signal_regime_features"

    signal_id: Mapped[int] = mapped_column(
        ForeignKey("signals.id"), primary_key=True,
    )
    spread_size: Mapped[float | None] = mapped_column(Float, nullable=True)
    underdog_flag: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    favorite_flag: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    line_movement_velocity: Mapped[float | None] = mapped_column(Float, nullable=True)
    steam_timing_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_volatility: Mapped[float | None] = mapped_column(Float, nullable=True)
    wallet_disagreement_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    late_movement_flag: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    consensus_concentration: Mapped[float | None] = mapped_column(Float, nullable=True)
    liquidity_regime: Mapped[str | None] = mapped_column(String(32), nullable=True)
    public_sharp_divergence: Mapped[float | None] = mapped_column(Float, nullable=True)
    sentiment_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=dict)
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)


class MlbFinalScore(Base):
    """Persisted final-score table.

    Grading reads from here first so a redeploy that drops the live API cache
    (or transient StatsAPI outages) cannot silently strand ungraded edges.
    One row per ``game_pk``; ``generated_for_date`` is the Arizona card date
    so the row aligns with the matching ``MlbEdge.generated_for_date``.
    """

    __tablename__ = "mlb_final_scores"

    game_pk: Mapped[int] = mapped_column(Integer, primary_key=True)
    generated_for_date: Mapped[str] = mapped_column(String(10), index=True)
    home_team: Mapped[str] = mapped_column(String(128))
    away_team: Mapped[str] = mapped_column(String(128))
    home_score: Mapped[int] = mapped_column(Integer)
    away_score: Mapped[int] = mapped_column(Integer)
    total_runs: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(64), default="Final")
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow, index=True)


class OddsSnapshot(Base):
    """Single source of truth for raw Odds-API payloads.

    One row per (sport, event_id, market_type[, sportsbook]). For the
    centralized cache:
      * market_type="events_list" → list payload of all events for a sport+date
        (event_id is a synthetic key like "_events_2026-05-25")
      * market_type="event_odds"  → full odds payload for one event_id
        (contains every bookmaker × market the upstream returned)

    Splitting per-sportsbook rows is supported (sportsbook column is the
    distinguisher) but the MLB pipeline cache stores the whole payload as one
    row because Odds-API returns all books in a single response.
    """

    __tablename__ = "odds_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sport: Mapped[str] = mapped_column(String(32), index=True)
    event_id: Mapped[str] = mapped_column(String(128), index=True)
    market_type: Mapped[str] = mapped_column(String(64), index=True)
    sportsbook: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    payload: Mapped[dict[str, Any] | list[Any] | None] = mapped_column(JSON, default=dict)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)


class ProviderHealthState(Base):
    """Persisted provider status used for cooldowns and operator diagnostics."""

    __tablename__ = "provider_health_state"

    provider: Mapped[str] = mapped_column(String(64), primary_key=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_error_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    cooldown_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    recent_failures: Mapped[int] = mapped_column(Integer, default=0)
    last_status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_successful_strategy: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_refresh_event_count: Mapped[int] = mapped_column(Integer, default=0)
    refresh_errors: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow, index=True)


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
