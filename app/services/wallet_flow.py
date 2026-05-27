"""Wallet-flow enrichment for MLB edges.

Joins a sportsbook edge to tracked-wallet activity on the matching
Polymarket/Kalshi market(s) and returns a ``wallet_context`` payload describing
consensus, contributing traders, and a bounded ``confidence_adjustment`` the
scoring layer folds into the edge score.

Design notes:
  * Exposure is aggregated from ``trades`` (the ``positions`` table is empty in
    this deployment). A SELL flips the effective side via the resolver.
  * Only the edge's ``generated_for_date`` markets are considered, so prior-date
    wallet activity never bleeds onto today's card.
  * No wallet data never zeroes an edge — the adjustment is simply 0.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Market, Trade, Trader, WalletLearningStats, WalletTierHistory
from app.services import wallet_market_resolver as wmr
from app.services.falcon_retraining import _tier_from_stats

logger = logging.getLogger(__name__)

ELITE_TIERS = {"elite"}
TRUSTED_TIERS = {"elite", "trusted"}

# Edge types that can have a wallet-flow market. Pitcher Ks have no Polymarket
# equivalent, so they always resolve to NO WALLET DATA.
_JOINABLE_EDGE_TYPES = {"game_total", "game_spread", "game_moneyline"}


@dataclass
class _WalletAgg:
    """Per-trader rollup before we pick the dominant stance."""

    trader: Trader
    aligned_size: float = 0.0
    opposing_size: float = 0.0
    aligned_notional_price: float = 0.0  # Σ price*size for size-weighted avg entry
    opposing_notional_price: float = 0.0
    slug: str | None = None
    platform: str | None = None
    markets: set[str] = field(default_factory=set)


def _empty_context(reason: str, debug: dict[str, Any]) -> dict[str, Any]:
    debug = {**debug, "no_match_reason": reason}
    return {
        "consensus_pct": None,
        "tracked_wallet_count": 0,
        "aligned_wallets": [],
        "opposing_wallets": [],
        "aligned_exposure_usd": 0.0,
        "opposing_exposure_usd": 0.0,
        "largest_aligned_entry": 0.0,
        "largest_opposing_entry": 0.0,
        "elite_wallet_agreement": 0,
        "elite_wallet_disagreement": 0,
        "confidence_adjustment": 0.0,
        "tags": ["NO WALLET DATA"],
        "debug": debug,
    }


def _resolve_tier(db: Session, wallet_address: str | None, trust_score: float | None) -> str:
    """Latest global WalletTierHistory -> learned stats -> trust_score fallback."""
    if wallet_address:
        hist = db.scalar(
            select(WalletTierHistory)
            .where(
                WalletTierHistory.wallet_address == wallet_address,
                WalletTierHistory.market_type.is_(None),
            )
            .order_by(WalletTierHistory.captured_at.desc())
            .limit(1)
        )
        if hist and hist.tier:
            return hist.tier
        stats = db.get(WalletLearningStats, wallet_address)
        if stats is not None and (stats.sample_size or 0) > 0:
            tier, _reason = _tier_from_stats(
                roi=stats.roi,
                confidence_weight=float(stats.confidence_weight or 0.5),
                sample_size=int(stats.sample_size or 0),
                avg_clv=stats.avg_clv,
            )
            return tier
    ts = trust_score if trust_score is not None else 50.0
    if ts >= 80:
        return "elite"
    if ts >= 65:
        return "trusted"
    return "neutral"


def _line_tol_for(market_type: str, settings: Any) -> float:
    if market_type == "spread":
        return float(settings.wallet_flow_spread_line_tolerance)
    return float(settings.wallet_flow_total_line_tolerance)


def build_wallet_context(
    db: Session,
    *,
    edge: dict[str, Any],
    home_team: str | None,
    away_team: str | None,
    card_date: str,
    line_tol: float | None = None,
) -> dict[str, Any]:
    """Return the ``wallet_context`` payload for one MLB edge."""
    settings = get_settings()
    key = wmr.normalize_edge(edge, home_team=home_team, away_team=away_team)
    tol = line_tol if line_tol is not None else _line_tol_for(key.market_type, settings)

    debug: dict[str, Any] = {
        "normalized_key": {
            "league": key.league,
            "event_date": key.event_date,
            "away_abbr": key.away_abbr,
            "home_abbr": key.home_abbr,
            "market_type": key.market_type,
            "line": key.line,
            "side": key.side,
            "outcome": key.outcome,
        },
        "sportsbook_event_id": edge.get("sportsbook_event_id"),
        "line_tolerance": tol,
        "candidate_markets_considered": 0,
        "matched_slugs": [],
    }

    edge_type = str(edge.get("edge_type") or "")
    if edge_type not in _JOINABLE_EDGE_TYPES:
        return _empty_context(f"edge type '{edge_type}' has no wallet market", debug)
    if not key.team_pair() or len(key.team_pair()) != 2:
        return _empty_context("could not map both teams to slug codes", debug)
    if not key.outcome:
        return _empty_context("edge side does not map to a tradeable outcome", debug)

    # Candidate markets: same league/teams/date/market_type, line within tolerance.
    # Narrow with a cheap SQL LIKE, then apply the precise resolver match.
    date_iso = key.event_date or card_date
    like = f"{key.league}-%-{date_iso}-{key.market_type}%"
    candidates = list(db.scalars(select(Market).where(Market.slug.like(like))))
    debug["candidate_markets_considered"] = len(candidates)

    matched: list[Market] = []
    for market in candidates:
        parsed = wmr.parse_market_slug(market.slug)
        if parsed is None:
            continue
        if wmr.keys_match(key, parsed, line_tol=tol):
            matched.append(market)
    debug["matched_slugs"] = [m.slug for m in matched]

    if not matched:
        return _empty_context("no wallet market matched the edge key", debug)

    market_ids = [m.id for m in matched]
    market_by_id = {m.id: m for m in matched}
    trades = list(
        db.scalars(
            select(Trade).where(Trade.market_id.in_(market_ids))
        )
    )
    if not trades:
        return _empty_context("matched market(s) have no tracked-wallet trades", debug)

    # Aggregate per trader across matched markets.
    aggs: dict[int, _WalletAgg] = {}
    trader_ids = {t.trader_id for t in trades}
    traders = {
        t.id: t
        for t in db.scalars(select(Trader).where(Trader.id.in_(trader_ids)))
    }
    for trade in trades:
        trader = traders.get(trade.trader_id)
        if trader is None:
            continue
        classification = wmr.outcomes_align(
            key, trade_outcome=trade.outcome, trade_side=trade.side
        )
        if classification == "unrelated":
            continue
        agg = aggs.get(trade.trader_id)
        if agg is None:
            mkt = market_by_id.get(trade.market_id)
            agg = _WalletAgg(trader=trader, slug=mkt.slug if mkt else None,
                             platform=mkt.platform if mkt else None)
            aggs[trade.trader_id] = agg
        size = float(trade.size_usd or 0.0)
        price = float(trade.price or 0.0)
        if trade.market_id in market_by_id:
            agg.markets.add(market_by_id[trade.market_id].slug)
        if classification == "aligned":
            agg.aligned_size += size
            agg.aligned_notional_price += price * size
        else:
            agg.opposing_size += size
            agg.opposing_notional_price += price * size

    aligned_wallets: list[dict[str, Any]] = []
    opposing_wallets: list[dict[str, Any]] = []
    for agg in aggs.values():
        stance_aligned = agg.aligned_size >= agg.opposing_size
        size = agg.aligned_size if stance_aligned else agg.opposing_size
        if size <= 0:
            continue
        notional = agg.aligned_notional_price if stance_aligned else agg.opposing_notional_price
        avg_entry = round(notional / size, 4) if size > 0 else None
        tier = _resolve_tier(db, agg.trader.wallet_address, agg.trader.trust_score)
        stats = (
            db.get(WalletLearningStats, agg.trader.wallet_address)
            if agg.trader.wallet_address
            else None
        )
        slug = next(iter(agg.markets), agg.slug)
        contributor = {
            "trader_name": agg.trader.nickname,
            "wallet_address": agg.trader.wallet_address,
            "tier": tier,
            "avg_entry": avg_entry,
            "size_usd": round(size, 2),
            "source": agg.platform or agg.trader.platform,
            "profile_url": wmr.trader_profile_url(agg.trader.wallet_address, agg.platform),
            "market_url": wmr.market_url_for(slug, agg.platform),
            "position_url": wmr.market_url_for(slug, agg.platform),
            "historical_roi": round(stats.roi, 4) if stats and stats.roi is not None else None,
            "avg_clv": round(stats.avg_clv, 4) if stats and stats.avg_clv is not None else None,
        }
        if stance_aligned:
            aligned_wallets.append(contributor)
        else:
            opposing_wallets.append(contributor)

    if not aligned_wallets and not opposing_wallets:
        return _empty_context("trades present but none align with the edge side", debug)

    aligned_wallets.sort(key=lambda w: w["size_usd"], reverse=True)
    opposing_wallets.sort(key=lambda w: w["size_usd"], reverse=True)

    aligned_exposure = round(sum(w["size_usd"] for w in aligned_wallets), 2)
    opposing_exposure = round(sum(w["size_usd"] for w in opposing_wallets), 2)
    total_exposure = aligned_exposure + opposing_exposure
    consensus_pct = round(aligned_exposure / total_exposure * 100, 1) if total_exposure > 0 else None
    elite_agreement = sum(1 for w in aligned_wallets if w["tier"] in ELITE_TIERS)
    elite_disagreement = sum(1 for w in opposing_wallets if w["tier"] in ELITE_TIERS)
    tracked_count = len(aligned_wallets) + len(opposing_wallets)

    adjustment, tags = _score_adjustment(
        settings,
        aligned_wallets=aligned_wallets,
        opposing_wallets=opposing_wallets,
        elite_agreement=elite_agreement,
        elite_disagreement=elite_disagreement,
        consensus_pct=consensus_pct,
    )

    return {
        "consensus_pct": consensus_pct,
        "tracked_wallet_count": tracked_count,
        "aligned_wallets": aligned_wallets,
        "opposing_wallets": opposing_wallets,
        "aligned_exposure_usd": aligned_exposure,
        "opposing_exposure_usd": opposing_exposure,
        "largest_aligned_entry": max((w["size_usd"] for w in aligned_wallets), default=0.0),
        "largest_opposing_entry": max((w["size_usd"] for w in opposing_wallets), default=0.0),
        "elite_wallet_agreement": elite_agreement,
        "elite_wallet_disagreement": elite_disagreement,
        "confidence_adjustment": adjustment,
        "tags": tags,
        "debug": debug,
    }


def _score_adjustment(
    settings: Any,
    *,
    aligned_wallets: list[dict[str, Any]],
    opposing_wallets: list[dict[str, Any]],
    elite_agreement: int,
    elite_disagreement: int,
    consensus_pct: float | None,
) -> tuple[float, list[str]]:
    """Bounded score nudge + UI tags. Wallet flow influences, never dominates."""
    adjustment = 0.0
    tags: list[str] = []

    if elite_agreement > 0:
        adjustment += settings.wallet_flow_elite_score_bonus
        tags.append("ELITE AGREEMENT")
    if elite_disagreement > 0:
        adjustment -= settings.wallet_flow_opposing_penalty
        tags.append("ELITE DISAGREEMENT")

    aligned_count = len(aligned_wallets)
    opposing_count = len(opposing_wallets)
    # Crowded low-quality consensus: many aligned wallets, none elite/trusted.
    crowded = aligned_count >= 4 and not any(
        w["tier"] in TRUSTED_TIERS for w in aligned_wallets
    )
    if crowded:
        adjustment -= settings.wallet_flow_crowded_penalty
        tags.append("CROWDED SIDE")

    if aligned_count and aligned_count >= opposing_count and (consensus_pct or 0) >= 60:
        tags.append("WALLET CONFIRMED")

    max_adj = float(settings.wallet_flow_max_adjustment)
    adjustment = round(max(-max_adj, min(max_adj, adjustment)), 2)
    return adjustment, tags
