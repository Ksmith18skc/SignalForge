"""Per-recommendation Falcon enrichment.

Fans out across multiple Falcon agents to build a market-regime + conflict
payload for a single signal. Designed to be best-effort: every individual
agent call is allowed to fail and the overall result still ships with a
``components`` map flagging which enrichments succeeded.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import SignalRegimeFeatures, WalletBehaviorProfile
from app.providers.falcon import FalconProvider, FalconResult

logger = logging.getLogger(__name__)


# ---- regime extraction ---------------------------------------------------


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _candle_volatility(candle_rows: list[dict[str, Any]]) -> float | None:
    """Standard deviation of log returns. ``None`` if <3 candles or no closes."""
    closes: list[float] = []
    for row in candle_rows:
        c = _to_float(row.get("close") or row.get("c") or row.get("price"))
        if c is not None and c > 0:
            closes.append(c)
    if len(closes) < 3:
        return None
    returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return round(math.sqrt(var), 6)


def _line_velocity(candle_rows: list[dict[str, Any]]) -> float | None:
    """Absolute price change between the first and last candle, normalised by
    elapsed candles. Captures how fast the line is moving regardless of
    direction."""
    closes = [
        _to_float(r.get("close") or r.get("c") or r.get("price")) for r in candle_rows
    ]
    closes = [c for c in closes if c is not None]
    if len(closes) < 2:
        return None
    return round(abs(closes[-1] - closes[0]) / len(closes), 6)


def _orderbook_imbalance(orderbook_rows: list[dict[str, Any]]) -> float | None:
    """Signed imbalance in [-1, 1]. Positive = bid-heavy."""
    bid_size = 0.0
    ask_size = 0.0
    for row in orderbook_rows:
        side = str(row.get("side") or "").lower()
        size = _to_float(row.get("size") or row.get("amount"))
        if size is None:
            continue
        if side in ("bid", "buy", "yes"):
            bid_size += size
        elif side in ("ask", "sell", "no"):
            ask_size += size
    total = bid_size + ask_size
    if total <= 0:
        return None
    return round((bid_size - ask_size) / total, 4)


def _trade_consensus(trade_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fraction of recent traded notional on each side."""
    buy = sell = 0.0
    for row in trade_rows:
        side = str(row.get("side") or "").upper()
        notional = _to_float(row.get("size_usd") or row.get("notional") or row.get("size"))
        if notional is None:
            continue
        if side == "BUY":
            buy += notional
        elif side == "SELL":
            sell += notional
    total = buy + sell
    if total <= 0:
        return {"concentration": None, "dominant_side": None}
    dominant = "BUY" if buy >= sell else "SELL"
    concentration = round(max(buy, sell) / total, 4)
    return {"concentration": concentration, "dominant_side": dominant}


def _sentiment_score(rows: list[dict[str, Any]], summary: dict[str, Any] | None) -> float | None:
    """Pick out a sentiment value in [-1, 1] from various possible field names."""
    for source in (summary or {},) if summary else ():
        for key in ("sentiment_score", "pulse_score", "score", "net_sentiment"):
            val = _to_float(source.get(key))
            if val is not None:
                return max(-1.0, min(1.0, val))
    if not rows:
        return None
    vals = [_to_float(r.get("sentiment") or r.get("score")) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    avg = sum(vals) / len(vals)
    return round(max(-1.0, min(1.0, avg)), 4)


# ---- payload type --------------------------------------------------------


@dataclass
class RegimePayload:
    """Compact, typed regime snapshot stored alongside a signal.

    ``components`` records which Falcon agents actually contributed. The
    explainer panel uses this to be transparent about partial intelligence
    rather than pretending every field is fresh.
    """

    line_movement_velocity: float | None = None
    market_volatility: float | None = None
    orderbook_imbalance: float | None = None
    consensus_concentration: float | None = None
    dominant_trade_side: str | None = None
    sentiment_score: float | None = None
    insights_summary: dict[str, Any] | None = None
    components: dict[str, bool] = field(default_factory=dict)
    captured_at: datetime = field(default_factory=datetime.utcnow)

    def as_dict(self) -> dict[str, Any]:
        return {
            "line_movement_velocity": self.line_movement_velocity,
            "market_volatility": self.market_volatility,
            "orderbook_imbalance": self.orderbook_imbalance,
            "consensus_concentration": self.consensus_concentration,
            "dominant_trade_side": self.dominant_trade_side,
            "sentiment_score": self.sentiment_score,
            "insights_summary": self.insights_summary,
            "components": dict(self.components),
            "captured_at": self.captured_at.isoformat(),
        }


# ---- fan-out -------------------------------------------------------------


async def build_regime_payload(
    falcon: FalconProvider,
    *,
    market_slug: str,
    candle_interval: str = "1h",
    candle_limit: int = 24,
) -> RegimePayload:
    """Best-effort regime payload for one market.

    All four Falcon calls run; any individual failure is tolerated and the
    ``components`` map reflects what succeeded. The caller can persist the
    payload via ``persist_regime_features`` once a signal_id exists.
    """
    payload = RegimePayload()

    candles: FalconResult = await falcon.fetch_polymarket_candles(
        market_slug=market_slug, interval=candle_interval, limit=candle_limit,
    )
    payload.components["candles"] = candles.available
    if candles.available:
        payload.market_volatility = _candle_volatility(candles.rows)
        payload.line_movement_velocity = _line_velocity(candles.rows)

    book: FalconResult = await falcon.fetch_polymarket_orderbook(market_slug=market_slug)
    payload.components["orderbook"] = book.available
    if book.available:
        payload.orderbook_imbalance = _orderbook_imbalance(book.rows)

    trades: FalconResult = await falcon.fetch_polymarket_trades(market_slug=market_slug, limit=50)
    payload.components["trades"] = trades.available
    if trades.available:
        consensus = _trade_consensus(trades.rows)
        payload.consensus_concentration = consensus.get("concentration")
        payload.dominant_trade_side = consensus.get("dominant_side")

    social: FalconResult = await falcon.fetch_social_pulse(market_slug=market_slug)
    payload.components["social_pulse"] = social.available
    if social.available:
        payload.sentiment_score = _sentiment_score(social.rows, social.summary)

    insights: FalconResult = await falcon.fetch_market_insights(market_slug=market_slug)
    payload.components["market_insights"] = insights.available
    if insights.available and insights.summary:
        payload.insights_summary = {
            k: v for k, v in insights.summary.items()
            if k in (
                "headline", "summary", "category", "key_factors", "narrative",
                "regime", "classification",
            )
        }
    return payload


def persist_regime_features(
    db: Session,
    *,
    signal_id: int,
    payload: RegimePayload,
) -> SignalRegimeFeatures:
    """Idempotent upsert of a ``SignalRegimeFeatures`` row."""
    existing = db.get(SignalRegimeFeatures, signal_id)
    if existing is None:
        existing = SignalRegimeFeatures(signal_id=signal_id)
        db.add(existing)
    existing.line_movement_velocity = payload.line_movement_velocity
    existing.market_volatility = payload.market_volatility
    existing.consensus_concentration = payload.consensus_concentration
    existing.public_sharp_divergence = payload.orderbook_imbalance
    existing.sentiment_score = payload.sentiment_score
    existing.raw_payload = payload.as_dict()
    existing.captured_at = datetime.utcnow()
    return existing


# ---- conflict / contrarian regime ---------------------------------------


def detect_conflict(
    *,
    same_side_wallets: int,
    total_watched: int,
    elite_disagreement_count: int = 0,
    sentiment_score: float | None = None,
    consensus_concentration: float | None = None,
    orderbook_imbalance: float | None = None,
) -> dict[str, Any]:
    """Classify the contrarian / conflict regime around a candidate signal.

    Returns a dict of binary flags plus a single ``conviction_penalty`` in
    [0, 0.5] the scoring layer should subtract from raw conviction.
    """
    crowded_side = False
    elite_disagreement = False
    trap_signal = False
    sentiment_against = False
    asymmetric_conviction = False
    penalty = 0.0

    crowded_threshold = max(3, int(total_watched * 0.5))
    if same_side_wallets >= crowded_threshold:
        crowded_side = True
        penalty += 0.05
    if elite_disagreement_count >= 1:
        elite_disagreement = True
        penalty += 0.15
    if (consensus_concentration or 0.0) >= 0.85 and (orderbook_imbalance or 0.0) < -0.2:
        # Heavy retail buying with bid/ask leaning the other way.
        trap_signal = True
        penalty += 0.15
    if sentiment_score is not None and sentiment_score <= -0.3 and same_side_wallets >= 2:
        sentiment_against = True
        penalty += 0.05
    if (
        same_side_wallets >= crowded_threshold
        and elite_disagreement_count >= 1
    ):
        asymmetric_conviction = True
        penalty += 0.05

    penalty = round(min(0.5, penalty), 4)
    return {
        "crowded_side": crowded_side,
        "trap_signal": trap_signal,
        "elite_disagreement": elite_disagreement,
        "weak_consensus_quality": crowded_side and elite_disagreement,
        "volatility_risk": trap_signal or asymmetric_conviction,
        "sentiment_against": sentiment_against,
        "asymmetric_conviction": asymmetric_conviction,
        "conviction_penalty": penalty,
    }


# ---- wallet behaviour clustering ----------------------------------------


_ARCHETYPES = (
    "sharp_steam",
    "late_momentum_chaser",
    "high_volume_low_edge",
    "contrarian_sniper",
    "market_maker_follower",
)


def derive_behavior_profile(
    *,
    wallet_360_summary: dict[str, Any],
    pnl_summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Deterministic clusterer: Wallet 360 features → archetype scores.

    Each returned dict carries ``archetype`` and ``score`` in [0, 1]. We
    score every archetype rather than picking one so the dashboard can
    show "70% sharp_steam / 20% market_maker_follower" instead of a hard
    label that's wrong on the edge of the cluster.
    """
    f = wallet_360_summary or {}
    pnl = pnl_summary or {}

    profit_factor = _to_float(f.get("profit_factor")) or 1.0
    win_rate = _to_float(f.get("win_rate_last_30day") or f.get("win_rate")) or 0.5
    roi = _to_float(f.get("roi")) or 0.0
    total_trades = _to_float(f.get("total_trades")) or 0.0
    avg_size = _to_float(f.get("avg_position_size_usd") or pnl.get("avg_position_size_usd")) or 0.0
    timing_score = _to_float(f.get("entry_timing_score")) or 0.5
    concentration = _to_float(f.get("market_concentration_ratio")) or 0.5

    def _clip(x: float) -> float:
        return max(0.0, min(1.0, x))

    scores = {
        "sharp_steam": _clip(0.5 * win_rate + 0.3 * (profit_factor / 2.0) + 0.2 * timing_score),
        "late_momentum_chaser": _clip(
            (1.0 - timing_score) * 0.7 + (1.0 - win_rate) * 0.3
        ),
        "high_volume_low_edge": _clip(
            _clip(total_trades / 1000.0) * 0.6 + (1.0 - max(win_rate, 0.4)) * 0.4
        ),
        "contrarian_sniper": _clip(
            (1.0 - concentration) * 0.4 + win_rate * 0.3 + _clip(roi / 25.0) * 0.3
        ),
        "market_maker_follower": _clip(concentration * 0.7 + (1.0 - timing_score) * 0.3),
    }
    return [{"archetype": k, "score": round(v, 4)} for k, v in scores.items()]


def upsert_behavior_profile(
    db: Session,
    *,
    wallet_address: str,
    profiles: list[dict[str, Any]],
    features: dict[str, Any] | None = None,
) -> int:
    """Upsert one row per (wallet, archetype). Returns number of rows written."""
    written = 0
    for profile in profiles:
        archetype = profile.get("archetype")
        score = _to_float(profile.get("score"))
        if not archetype or score is None:
            continue
        existing = db.scalar(
            select(WalletBehaviorProfile).where(
                WalletBehaviorProfile.wallet_address == wallet_address,
                WalletBehaviorProfile.archetype == archetype,
            )
        )
        if existing is None:
            db.add(
                WalletBehaviorProfile(
                    wallet_address=wallet_address,
                    archetype=archetype,
                    score=score,
                    features=features or {},
                )
            )
        else:
            existing.score = score
            if features is not None:
                existing.features = features
        written += 1
    return written
