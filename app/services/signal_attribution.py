"""SignalForge attribution summaries for the personal P&L tracker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.storage.pnl_store import (
    MyPosition,
    MyTrade,
    RecommendationSnapshot,
    SignalAttribution,
)


@dataclass(frozen=True)
class AttributionSummary:
    trailed_pnl_usd: float
    non_signal_pnl_usd: float
    trailed_trade_count: int
    non_signal_trade_count: int
    trailed_win_rate: float | None
    average_clv_points: float | None
    best_signal: dict[str, Any] | None
    worst_signal: dict[str, Any] | None


def summarize_signal_attribution(db: Session) -> AttributionSummary:
    """Aggregate P&L into SignalForge-trailed vs non-dashboard buckets."""
    attrs = list(db.scalars(select(SignalAttribution)))
    attr_trade_ids = {a.my_trade_id for a in attrs if a.my_trade_id is not None}
    trades = list(db.scalars(select(MyTrade)))
    positions = list(db.scalars(select(MyPosition)))
    pos_by_slot = {
        (p.wallet_id, p.market_slug, p.outcome or None): p for p in positions
    }

    trailed_pnl = 0.0
    trailed_wins = 0
    trailed_losses = 0
    non_signal_pnl = 0.0
    best: dict[str, Any] | None = None
    worst: dict[str, Any] | None = None
    clv_values: list[float] = []

    rec_ids = {a.recommendation_id for a in attrs if a.recommendation_id is not None}
    recs = {
        r.id: r
        for r in db.scalars(
            select(RecommendationSnapshot).where(RecommendationSnapshot.id.in_(rec_ids))
        )
    } if rec_ids else {}

    for attr in attrs:
        trade = db.get(MyTrade, attr.my_trade_id) if attr.my_trade_id else None
        if trade is None:
            continue
        pos = pos_by_slot.get((trade.wallet_id, trade.market_slug, trade.outcome or None))
        pnl = ((pos.realized_pnl_usd or 0.0) + (pos.unrealized_pnl_usd or 0.0)) if pos else 0.0
        trailed_pnl += pnl
        if pnl > 0:
            trailed_wins += 1
        elif pnl < 0:
            trailed_losses += 1
        if attr.clv_points is not None:
            clv_values.append(attr.clv_points)
        rec = recs.get(attr.recommendation_id)
        row = {
            "market": (rec.market_title if rec else None) or trade.market_title or trade.market_slug,
            "platform": trade.platform,
            "pnl_usd": round(pnl, 2),
            "clv_points": attr.clv_points,
            "label": attr.label,
            "grade": attr.grade,
        }
        if best is None or row["pnl_usd"] > best["pnl_usd"]:
            best = row
        if worst is None or row["pnl_usd"] < worst["pnl_usd"]:
            worst = row

    for trade in trades:
        if trade.id in attr_trade_ids:
            continue
        pos = pos_by_slot.get((trade.wallet_id, trade.market_slug, trade.outcome or None))
        if pos:
            non_signal_pnl += (pos.realized_pnl_usd or 0.0) + (pos.unrealized_pnl_usd or 0.0)

    decided = trailed_wins + trailed_losses
    return AttributionSummary(
        trailed_pnl_usd=round(trailed_pnl, 2),
        non_signal_pnl_usd=round(non_signal_pnl, 2),
        trailed_trade_count=len(attr_trade_ids),
        non_signal_trade_count=len([t for t in trades if t.id not in attr_trade_ids]),
        trailed_win_rate=round(trailed_wins / decided, 4) if decided else None,
        average_clv_points=round(sum(clv_values) / len(clv_values), 4) if clv_values else None,
        best_signal=best,
        worst_signal=worst,
    )
