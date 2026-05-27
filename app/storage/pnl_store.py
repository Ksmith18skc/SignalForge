"""Personal P&L tracker schema + CRUD helpers.

These tables are owned by *you* the operator — they describe your own
Kalshi/Polymarket wallets, the trades and positions inside them, and how
those line up against SignalForge recommendations. They are intentionally
kept separate from the smart-money `traders`/`trades`/`positions` tables
in `app.models` so the wallet-flow analytics and the personal-PnL view
can never blur into each other.

Schema (one section per table):

    MyWallet               — a wallet/account you own (one row per address).
    MyTrade                — an individual fill from one of those wallets.
    MyPosition             — aggregated current position derived from fills.
    WalletSnapshot         — point-in-time cash + open-position valuation.
    RecommendationSnapshot — frozen copy of a SignalForge callout at the
                              moment it became actionable; the source of
                              truth for attribution + CLV math.
    SignalAttribution      — link rows that say "this MyTrade trailed this
                              RecommendationSnapshot" with grades + labels.
    PnlAlert               — operator-facing alerts emitted by the
                              `pnl_alerts` service (e.g. negative edge,
                              strong CLV, overexposure, missed callout).
    ClosedTradeOutcome     — settlement record once a position resolves.

Everything is plain SQLAlchemy 2.x mapped on the same `Base` as
`app.models`, so the existing `init_db()` `create_all` picks them up
automatically when this module is imported at startup.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    select,
)
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship

from app.db import Base


def _utcnow() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


class MyWallet(Base):
    """A single Kalshi/Polymarket account that belongs to you.

    `address` is the Polymarket on-chain wallet address (0x...) or the
    Kalshi account identifier ("kalshi:<api_key_id>"). It's unique so
    `(platform, address)` cannot duplicate.
    """

    __tablename__ = "pnl_my_wallets"
    __table_args__ = (UniqueConstraint("platform", "address", name="uq_my_wallet"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    platform: Mapped[str] = mapped_column(String(32), index=True)  # polymarket | kalshi
    address: Mapped[str] = mapped_column(String(128), index=True)
    label: Mapped[str | None] = mapped_column(String(128), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_sync_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    trades: Mapped[list["MyTrade"]] = relationship(
        back_populates="wallet", cascade="all, delete-orphan"
    )
    positions: Mapped[list["MyPosition"]] = relationship(
        back_populates="wallet", cascade="all, delete-orphan"
    )
    snapshots: Mapped[list["WalletSnapshot"]] = relationship(
        back_populates="wallet", cascade="all, delete-orphan"
    )


class MyTrade(Base):
    """A single fill from one of your wallets.

    We persist *fills*, not orders — Polymarket and Kalshi both expose
    realised fills, and fills are what P&L math needs. `external_id`
    de-dupes across syncs; `(wallet_id, external_id)` is unique.

    Assumptions about the upstream payloads:
      - `price` is always 0.0–1.0 (prediction-market probability), never
        American odds. Both venues use that convention.
      - `size_shares` is the share count; `size_usd = size_shares * price`
        unless the venue reports an explicit notional, in which case we
        store that.
      - `side` is normalized to BUY/SELL. `outcome` carries YES/NO or the
        team/over-under string the venue uses.
    """

    __tablename__ = "pnl_my_trades"
    __table_args__ = (
        UniqueConstraint("wallet_id", "external_id", name="uq_my_trade_external"),
        Index("ix_my_trades_market_slug", "market_slug"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    wallet_id: Mapped[int] = mapped_column(ForeignKey("pnl_my_wallets.id"), index=True)
    external_id: Mapped[str] = mapped_column(String(256), index=True)
    platform: Mapped[str] = mapped_column(String(32))  # polymarket | kalshi
    market_slug: Mapped[str] = mapped_column(String(256))
    market_title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    side: Mapped[str] = mapped_column(String(8))  # BUY | SELL
    outcome: Mapped[str | None] = mapped_column(String(64), nullable=True)
    price: Mapped[float] = mapped_column(Float)
    size_shares: Mapped[float] = mapped_column(Float, default=0.0)
    size_usd: Mapped[float] = mapped_column(Float, default=0.0)
    fees_usd: Mapped[float] = mapped_column(Float, default=0.0)
    sport: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    event_date: Mapped[str | None] = mapped_column(String(10), nullable=True, index=True)
    source: Mapped[str] = mapped_column(String(32), default="manual")  # polymarket | kalshi | manual | mock
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    raw: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=dict)

    wallet: Mapped[MyWallet] = relationship(back_populates="trades")


class MyPosition(Base):
    """Aggregated open/closed position derived from `MyTrade` fills.

    Recomputed by `pnl_tracker.rebuild_positions_for_wallet(...)` after
    every sync. We keep it as a stored table so the dashboard doesn't have
    to fold thousands of fills on every render.
    """

    __tablename__ = "pnl_my_positions"
    __table_args__ = (
        UniqueConstraint(
            "wallet_id",
            "market_slug",
            "outcome",
            name="uq_my_position_slot",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    wallet_id: Mapped[int] = mapped_column(ForeignKey("pnl_my_wallets.id"), index=True)
    platform: Mapped[str] = mapped_column(String(32))
    market_slug: Mapped[str] = mapped_column(String(256), index=True)
    market_title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    outcome: Mapped[str | None] = mapped_column(String(64), nullable=True)
    side: Mapped[str | None] = mapped_column(String(8), nullable=True)
    avg_entry_price: Mapped[float] = mapped_column(Float, default=0.0)
    shares: Mapped[float] = mapped_column(Float, default=0.0)
    cost_basis_usd: Mapped[float] = mapped_column(Float, default=0.0)
    current_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_value_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    unrealized_pnl_usd: Mapped[float] = mapped_column(Float, default=0.0)
    realized_pnl_usd: Mapped[float] = mapped_column(Float, default=0.0)
    fair_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    edge_at_entry: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_edge: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_tier: Mapped[str | None] = mapped_column(String(32), nullable=True)
    signal_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    sport: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    event_date: Mapped[str | None] = mapped_column(String(10), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)  # open | closed
    is_stale_price: Mapped[bool] = mapped_column(Boolean, default=False)
    opened_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)

    wallet: Mapped[MyWallet] = relationship(back_populates="positions")


class WalletSnapshot(Base):
    """Periodic snapshot of total wallet value. Used to compute daily P&L
    (today_total - yesterday_total) without re-pricing the entire history."""

    __tablename__ = "pnl_wallet_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    wallet_id: Mapped[int] = mapped_column(ForeignKey("pnl_my_wallets.id"), index=True)
    cash_balance_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    open_position_value_usd: Mapped[float] = mapped_column(Float, default=0.0)
    realized_pnl_usd: Mapped[float] = mapped_column(Float, default=0.0)
    unrealized_pnl_usd: Mapped[float] = mapped_column(Float, default=0.0)
    total_value_usd: Mapped[float] = mapped_column(Float, default=0.0)
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    is_estimated: Mapped[bool] = mapped_column(Boolean, default=False)

    wallet: Mapped[MyWallet] = relationship(back_populates="snapshots")


class RecommendationSnapshot(Base):
    """Frozen view of a SignalForge callout at the moment it became
    actionable. Attribution + CLV both depend on having this snapshot;
    re-deriving the numbers from the live `MlbEdge` / `Signal` row would
    silently rewrite history every time the market moves.
    """

    __tablename__ = "pnl_recommendation_snapshots"
    __table_args__ = (
        Index("ix_pnl_reco_source_market", "source", "market_slug"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32))  # mlb_edge | signal
    source_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    market_slug: Mapped[str] = mapped_column(String(256))
    market_title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    side: Mapped[str | None] = mapped_column(String(16), nullable=True)
    outcome: Mapped[str | None] = mapped_column(String(64), nullable=True)
    sport: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    event_date: Mapped[str | None] = mapped_column(String(10), nullable=True, index=True)
    fair_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    implied_edge: Mapped[float | None] = mapped_column(Float, nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_tier: Mapped[str | None] = mapped_column(String(32), nullable=True)
    threshold_status: Mapped[str] = mapped_column(String(32), default="below")
    action: Mapped[str | None] = mapped_column(String(64), nullable=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    raw: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=dict)


class SignalAttribution(Base):
    """Link row between one of your trades and a recommendation snapshot.

    Created/refreshed by `position_matcher.match_trades_to_recommendations`.
    `label` captures the high-level relationship ("trailed_signalforge",
    "missed_signalforge", "before_threshold", etc.) so the dashboard can
    badge each row without re-running the matching rules.
    """

    __tablename__ = "pnl_signal_attributions"
    __table_args__ = (
        UniqueConstraint(
            "my_trade_id", "recommendation_id", name="uq_attribution_trade_reco"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    my_trade_id: Mapped[int | None] = mapped_column(
        ForeignKey("pnl_my_trades.id"), nullable=True, index=True
    )
    my_position_id: Mapped[int | None] = mapped_column(
        ForeignKey("pnl_my_positions.id"), nullable=True, index=True
    )
    recommendation_id: Mapped[int | None] = mapped_column(
        ForeignKey("pnl_recommendation_snapshots.id"), nullable=True, index=True
    )
    label: Mapped[str] = mapped_column(String(48), default="trailed_signalforge")
    grade: Mapped[str | None] = mapped_column(String(2), nullable=True)  # A | B | C | D | F
    entry_price_user: Mapped[float | None] = mapped_column(Float, nullable=True)
    entry_price_signal: Mapped[float | None] = mapped_column(Float, nullable=True)
    edge_at_entry: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_edge: Mapped[float | None] = mapped_column(Float, nullable=True)
    clv_points: Mapped[float | None] = mapped_column(Float, nullable=True)
    clv_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    entered_before_threshold: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    minutes_after_signal: Mapped[float | None] = mapped_column(Float, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)


class PnlAlert(Base):
    """Operator-facing alert (e.g. 'edge has flipped negative'). Status
    transitions: pending -> sent / skipped / failed (Discord), plus
    open -> acknowledged (operator)."""

    __tablename__ = "pnl_alerts"
    __table_args__ = (
        Index("ix_pnl_alert_key_status", "dedupe_key", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alert_type: Mapped[str] = mapped_column(String(48), index=True)
    severity: Mapped[str] = mapped_column(String(16), default="info")  # info | warn | crit
    title: Mapped[str] = mapped_column(String(256))
    body: Mapped[str] = mapped_column(Text, default="")
    wallet_id: Mapped[int | None] = mapped_column(ForeignKey("pnl_my_wallets.id"), nullable=True)
    market_slug: Mapped[str | None] = mapped_column(String(256), nullable=True)
    my_position_id: Mapped[int | None] = mapped_column(
        ForeignKey("pnl_my_positions.id"), nullable=True
    )
    recommendation_id: Mapped[int | None] = mapped_column(
        ForeignKey("pnl_recommendation_snapshots.id"), nullable=True
    )
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=dict)
    dedupe_key: Mapped[str] = mapped_column(String(256), index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending | sent | failed | skipped
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    delivery_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ClosedTradeOutcome(Base):
    """Settlement record for a position that has resolved. The dashboard
    uses this for realized-P&L history and grading."""

    __tablename__ = "pnl_closed_trade_outcomes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    my_position_id: Mapped[int] = mapped_column(ForeignKey("pnl_my_positions.id"), index=True)
    market_slug: Mapped[str] = mapped_column(String(256), index=True)
    final_outcome: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payout_usd: Mapped[float] = mapped_column(Float, default=0.0)
    realized_pnl_usd: Mapped[float] = mapped_column(Float, default=0.0)
    settled_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    source: Mapped[str] = mapped_column(String(32), default="manual")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def upsert_wallet(
    db: Session,
    *,
    platform: str,
    address: str,
    label: str | None = None,
) -> MyWallet:
    """Find-or-create a wallet by (platform, address). Updates label if
    the caller passed a fresh one; never overwrites with None."""
    address_norm = (address or "").strip()
    if not address_norm:
        raise ValueError("wallet address is required")
    wallet = db.scalar(
        select(MyWallet).where(
            MyWallet.platform == platform, MyWallet.address == address_norm
        )
    )
    if wallet is None:
        wallet = MyWallet(platform=platform, address=address_norm, label=label)
        db.add(wallet)
        db.flush()
    elif label and wallet.label != label:
        wallet.label = label
    return wallet


def list_wallets(db: Session, *, active_only: bool = True) -> list[MyWallet]:
    stmt = select(MyWallet).order_by(MyWallet.platform.asc(), MyWallet.id.asc())
    if active_only:
        stmt = stmt.where(MyWallet.is_active.is_(True))
    return list(db.scalars(stmt))


def list_positions(
    db: Session,
    *,
    wallet_id: int | None = None,
    status: str | None = "open",
) -> list[MyPosition]:
    stmt = select(MyPosition)
    if wallet_id is not None:
        stmt = stmt.where(MyPosition.wallet_id == wallet_id)
    if status is not None:
        stmt = stmt.where(MyPosition.status == status)
    return list(db.scalars(stmt.order_by(MyPosition.last_updated.desc())))


def list_recent_trades(
    db: Session,
    *,
    wallet_id: int | None = None,
    limit: int = 200,
) -> list[MyTrade]:
    stmt = select(MyTrade).order_by(MyTrade.timestamp.desc()).limit(limit)
    if wallet_id is not None:
        stmt = stmt.where(MyTrade.wallet_id == wallet_id)
    return list(db.scalars(stmt))


def insert_trades(db: Session, trades: Iterable[MyTrade]) -> int:
    """Insert trades while skipping duplicates by (wallet_id, external_id).

    Returns the number of *new* rows inserted. Callers should `db.flush()`
    or `db.commit()` themselves so a multi-step sync stays atomic.
    """
    count = 0
    for trade in trades:
        existing = db.scalar(
            select(MyTrade.id).where(
                MyTrade.wallet_id == trade.wallet_id,
                MyTrade.external_id == trade.external_id,
            )
        )
        if existing is not None:
            continue
        db.add(trade)
        count += 1
    return count


def write_snapshot(db: Session, snapshot: WalletSnapshot) -> WalletSnapshot:
    db.add(snapshot)
    db.flush()
    return snapshot


def latest_snapshot(db: Session, wallet_id: int) -> WalletSnapshot | None:
    return db.scalar(
        select(WalletSnapshot)
        .where(WalletSnapshot.wallet_id == wallet_id)
        .order_by(WalletSnapshot.captured_at.desc())
        .limit(1)
    )
