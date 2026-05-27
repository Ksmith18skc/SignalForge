"""Match wallet trades against SignalForge recommendations.

Pipeline (in order):

  1. `snapshot_actionable_recommendations(db)` walks the *current* live
     callouts (MLB edges + smart-money signals) and freezes one row in
     `RecommendationSnapshot` per "actionable" callout. This is what
     attribution and CLV are joined against — never the live edge row,
     which keeps moving with the market.

  2. `match_trades_to_recommendations(db)` walks every `MyTrade` and
     joins to the freshest snapshot whose `(market_slug, outcome)` lines
     up, within a ±48h window. For each match it upserts a
     `SignalAttribution` row carrying labels, grade, CLV, and edge math.

  3. `find_missed_opportunities(db)` returns snapshots that *had no
     matching trade* — i.e. dashboard told you "BUY Under 7.5" and you
     never opened it. Surfaced in the "Missed edges" table.

We try hard to keep matching rules in one file so all the threshold
labels stay consistent across the dashboard, the alerts engine, and the
attribution view.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from app.models import MlbEdge, Signal
from app.services import pnl_tracker
from app.storage.pnl_store import (
    MyPosition,
    MyTrade,
    RecommendationSnapshot,
    SignalAttribution,
)

logger = logging.getLogger(__name__)

# Maximum time delta between a user trade and a recommendation snapshot
# for matching to be considered valid. 48h covers same-day-edge + late
# next-morning entries; anything older is almost certainly coincidence.
MATCH_WINDOW = timedelta(hours=48)

# A user entry within this window *before* the rec is treated as
# "entered before threshold" (you front-ran the call). Outside that, an
# earlier entry is just an old position that happens to be the same.
PRE_THRESHOLD_WINDOW = timedelta(hours=6)

ACTIONABLE_MLB_SCORE = 80.0          # see SCORE_HIGH_CONV_MIN in dashboard_format
ACTIONABLE_SIGNAL_SCORE = 80.0

ACTIONABLE_LABELS = {
    "trailed_signalforge",
    "entered_before_threshold",
    "entered_after_signal",
    "missed_best_price",
    "no_matching_recommendation",
}


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchResult:
    trade_id: int
    recommendation_id: int | None
    label: str
    minutes_after_signal: float | None
    entered_before_threshold: bool | None
    edge_at_entry: float | None
    clv_points: float | None


def _mlb_edge_to_snapshot(edge: MlbEdge) -> RecommendationSnapshot:
    """Freeze the parts of an MlbEdge that drive attribution math."""
    fair = None
    factors = edge.factors or {}
    if isinstance(factors, dict):
        # mlb_edge_engine stores model probabilities under varied keys;
        # we accept the first one we find rather than fight the schema.
        for key in ("model_probability", "fair_probability", "model_prob", "p_fair"):
            value = factors.get(key)
            if value is None:
                continue
            try:
                fair = float(value)
                break
            except (TypeError, ValueError):
                continue
    return RecommendationSnapshot(
        source="mlb_edge",
        source_id=edge.id,
        market_slug=(edge.normalized_market_name or edge.market or "").strip(),
        market_title=edge.market,
        side=edge.side,
        outcome=edge.side,  # MLB edges encode the outcome in `side`.
        sport="mlb",
        event_date=edge.generated_for_date,
        fair_probability=fair,
        market_price=edge.best_price,
        implied_edge=pnl_tracker.compute_edge(fair, edge.best_price),
        score=edge.score,
        confidence_tier=edge.confidence,
        threshold_status=_threshold_status_for_score(edge.score, ACTIONABLE_MLB_SCORE),
        action=edge.action,
        raw={"reasons": edge.reasons or [], "warnings": edge.warnings or []},
    )


def _signal_to_snapshot(signal: Signal) -> RecommendationSnapshot:
    return RecommendationSnapshot(
        source="signal",
        source_id=signal.id,
        market_slug=(signal.market.slug if signal.market else "") or "",
        market_title=signal.market.title if signal.market else None,
        side=signal.side,
        outcome=signal.outcome,
        sport=(signal.market.category if signal.market else None),
        event_date=None,
        fair_probability=None,
        market_price=signal.entry_price,
        implied_edge=None,
        score=signal.score,
        confidence_tier=None,
        threshold_status=_threshold_status_for_score(signal.score, ACTIONABLE_SIGNAL_SCORE),
        action=None,
        raw={"reason": signal.reason},
    )


def _threshold_status_for_score(score: float | None, threshold: float) -> str:
    if score is None:
        return "unknown"
    if score >= threshold:
        return "actionable"
    if score >= threshold - 5.0:
        return "near"
    return "below"


def snapshot_actionable_recommendations(db: Session, *, lookback_hours: int = 36) -> list[RecommendationSnapshot]:
    """Persist a snapshot row for every callout that is currently
    actionable but not yet snapshotted.

    Idempotent: we use `(source, source_id)` as the dedupe key. Calling
    this on every wallet sync keeps the attribution layer fresh without
    flooding the DB on every page refresh.
    """
    since = datetime.utcnow() - timedelta(hours=lookback_hours)
    edges = list(
        db.scalars(
            select(MlbEdge).where(
                MlbEdge.score >= ACTIONABLE_MLB_SCORE,
                MlbEdge.created_at >= since,
            )
        )
    )
    signals = list(
        db.scalars(
            select(Signal).where(
                Signal.score >= ACTIONABLE_SIGNAL_SCORE,
                Signal.created_at >= since,
            )
        )
    )
    written: list[RecommendationSnapshot] = []
    for edge in edges:
        existing = db.scalar(
            select(RecommendationSnapshot.id).where(
                RecommendationSnapshot.source == "mlb_edge",
                RecommendationSnapshot.source_id == edge.id,
            )
        )
        if existing:
            continue
        snap = _mlb_edge_to_snapshot(edge)
        if not snap.market_slug:
            continue
        db.add(snap)
        written.append(snap)
    for signal in signals:
        existing = db.scalar(
            select(RecommendationSnapshot.id).where(
                RecommendationSnapshot.source == "signal",
                RecommendationSnapshot.source_id == signal.id,
            )
        )
        if existing:
            continue
        snap = _signal_to_snapshot(signal)
        if not snap.market_slug:
            continue
        db.add(snap)
        written.append(snap)
    db.flush()
    return written


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _normalize_slot(slug: str | None, outcome: str | None) -> tuple[str, str]:
    return (slug or "").strip().lower(), (outcome or "").strip().lower()


def _outcomes_align(rec: RecommendationSnapshot, trade: MyTrade) -> bool:
    """Loose match: prefer outcome match, fall back to side+slug."""
    rec_outcome = (rec.outcome or "").strip().lower()
    trade_outcome = (trade.outcome or "").strip().lower()
    if rec_outcome and trade_outcome and rec_outcome == trade_outcome:
        return True
    if not rec_outcome and not trade_outcome:
        return True
    rec_side = (rec.side or "").strip().lower()
    trade_side = (trade.side or "").strip().lower()
    if rec_side == trade_side and trade_side in {"buy", "sell"}:
        return True
    return False


def _pick_best_snapshot(
    snapshots: Iterable[RecommendationSnapshot],
    *,
    trade_time: datetime,
) -> RecommendationSnapshot | None:
    """Choose the snapshot whose captured_at is closest to the trade
    timestamp, but inside MATCH_WINDOW. None means no acceptable match."""
    best: tuple[float, RecommendationSnapshot] | None = None
    for snap in snapshots:
        delta = abs((trade_time - snap.captured_at).total_seconds())
        if delta > MATCH_WINDOW.total_seconds():
            continue
        if best is None or delta < best[0]:
            best = (delta, snap)
    return best[1] if best else None


def _label_for_match(snap: RecommendationSnapshot, trade: MyTrade) -> tuple[str, bool | None, float | None]:
    """Return (label, entered_before_threshold, minutes_after_signal)."""
    delta = (trade.timestamp - snap.captured_at).total_seconds() / 60.0
    if delta < 0:
        # Trade pre-dates the rec.
        if abs(delta) <= PRE_THRESHOLD_WINDOW.total_seconds() / 60.0:
            return "entered_before_threshold", True, delta
        return "trailed_signalforge", True, delta
    if snap.market_price is not None and trade.price is not None:
        side = (snap.side or trade.side or "").lower()
        # For a BUY, getting filled materially *above* the callout price
        # is a worse entry. 4 cents is the operator-visible threshold.
        if side == "buy" and trade.price - snap.market_price >= 0.04:
            return "missed_best_price", False, delta
        if side == "sell" and snap.market_price - trade.price >= 0.04:
            return "missed_best_price", False, delta
    return "entered_after_signal", False, delta


def match_trades_to_recommendations(
    db: Session,
    *,
    wallet_ids: Iterable[int] | None = None,
    since: datetime | None = None,
) -> list[MatchResult]:
    """Build/refresh `SignalAttribution` rows. Returns one MatchResult per
    *processed* trade (matched or not) so callers can summarise what
    just happened."""
    trade_stmt = select(MyTrade)
    if wallet_ids is not None:
        trade_stmt = trade_stmt.where(MyTrade.wallet_id.in_(list(wallet_ids)))
    if since is not None:
        trade_stmt = trade_stmt.where(MyTrade.timestamp >= since)
    trades = list(db.scalars(trade_stmt.order_by(MyTrade.timestamp.asc())))
    if not trades:
        return []

    # Group snapshots by normalized slot so the inner loop is O(trades).
    slug_set = {(_normalize_slot(t.market_slug, t.outcome))[0] for t in trades}
    snaps_by_slug: dict[str, list[RecommendationSnapshot]] = {s: [] for s in slug_set}
    snaps = list(
        db.scalars(
            select(RecommendationSnapshot).where(
                RecommendationSnapshot.market_slug.in_(
                    list({t.market_slug for t in trades if t.market_slug})
                )
            )
        )
    )
    for s in snaps:
        snaps_by_slug.setdefault((s.market_slug or "").strip().lower(), []).append(s)

    results: list[MatchResult] = []
    for trade in trades:
        slug_key = (trade.market_slug or "").strip().lower()
        candidates = [
            s for s in snaps_by_slug.get(slug_key, []) if _outcomes_align(s, trade)
        ]
        snap = _pick_best_snapshot(candidates, trade_time=trade.timestamp)
        if snap is None:
            results.append(
                MatchResult(
                    trade_id=trade.id,
                    recommendation_id=None,
                    label="no_matching_recommendation",
                    minutes_after_signal=None,
                    entered_before_threshold=None,
                    edge_at_entry=None,
                    clv_points=None,
                )
            )
            continue
        label, before, minutes_after = _label_for_match(snap, trade)
        edge_at_entry = pnl_tracker.compute_edge(snap.fair_probability, trade.price)
        clv_points, clv_pct = pnl_tracker.compute_clv(snap.market_price, trade.price)
        # CLV sign-flip for SELL — we want a positive number to mean
        # "you got a better price than the close".
        if (trade.side or "").upper() == "SELL" and clv_points is not None:
            clv_points = -clv_points
            clv_pct = -clv_pct if clv_pct is not None else None

        attr = db.scalar(
            select(SignalAttribution).where(
                SignalAttribution.my_trade_id == trade.id,
                SignalAttribution.recommendation_id == snap.id,
            )
        )
        if attr is None:
            attr = SignalAttribution(
                my_trade_id=trade.id,
                recommendation_id=snap.id,
            )
            db.add(attr)
        attr.label = label
        attr.entry_price_user = trade.price
        attr.entry_price_signal = snap.market_price
        attr.edge_at_entry = edge_at_entry
        attr.clv_points = clv_points
        attr.clv_percent = clv_pct
        attr.entered_before_threshold = before
        attr.minutes_after_signal = minutes_after
        pos = db.scalar(
            select(MyPosition).where(
                MyPosition.wallet_id == trade.wallet_id,
                MyPosition.market_slug == trade.market_slug,
                MyPosition.outcome == trade.outcome,
            )
        )
        if pos is not None:
            attr.my_position_id = pos.id
            attr.current_edge = pos.current_edge
        attr.grade = pnl_tracker.grade_trade(
            pnl_usd=trade.size_usd * (snap.market_price or trade.price or 0.0)
            - trade.size_usd,
            edge_at_entry=edge_at_entry,
            clv_points=clv_points,
            is_win=None,
        )
        results.append(
            MatchResult(
                trade_id=trade.id,
                recommendation_id=snap.id,
                label=label,
                minutes_after_signal=minutes_after,
                entered_before_threshold=before,
                edge_at_entry=edge_at_entry,
                clv_points=clv_points,
            )
        )
    db.flush()
    return results


# ---------------------------------------------------------------------------
# Missed opportunities
# ---------------------------------------------------------------------------


def find_missed_opportunities(
    db: Session,
    *,
    wallet_ids: Iterable[int] | None = None,
    lookback_hours: int = 36,
) -> list[RecommendationSnapshot]:
    """Snapshots that had no matching trade from any of `wallet_ids`.

    Caller is expected to render these as "Missed edges" — actionable
    recommendations the user did not take. We do not include snapshots
    older than `lookback_hours` because anything older is past the trade
    decision window (event most likely already started).
    """
    since = datetime.utcnow() - timedelta(hours=lookback_hours)
    snaps = list(
        db.scalars(
            select(RecommendationSnapshot)
            .where(
                RecommendationSnapshot.captured_at >= since,
                RecommendationSnapshot.threshold_status == "actionable",
            )
            .order_by(RecommendationSnapshot.score.desc())
        )
    )
    if not snaps:
        return []

    trade_filter = [MyTrade.market_slug.in_([s.market_slug for s in snaps])]
    if wallet_ids is not None:
        trade_filter.append(MyTrade.wallet_id.in_(list(wallet_ids)))
    user_trades = list(db.scalars(select(MyTrade).where(and_(*trade_filter))))
    matched_slugs: set[tuple[str, str]] = set()
    for trade in user_trades:
        matched_slugs.add(_normalize_slot(trade.market_slug, trade.outcome))
    return [
        s
        for s in snaps
        if _normalize_slot(s.market_slug, s.outcome) not in matched_slugs
    ]


def latest_recommendation_for_position(
    db: Session,
    *,
    market_slug: str,
    outcome: str | None,
) -> RecommendationSnapshot | None:
    """Most recent rec snapshot for a market/outcome — feeds the open
    position table's "current edge" and "signal status" columns."""
    stmt = (
        select(RecommendationSnapshot)
        .where(RecommendationSnapshot.market_slug == market_slug)
        .order_by(RecommendationSnapshot.captured_at.desc())
    )
    if outcome:
        stmt = stmt.where(
            or_(
                RecommendationSnapshot.outcome == outcome,
                RecommendationSnapshot.outcome.is_(None),
            )
        )
    return db.scalar(stmt.limit(1))
