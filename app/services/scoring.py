"""Signal scoring.

Weighted combination of five components, each normalized to [0, 1]:

  wallet_quality          35%   — operator trust + win rate + rank
  multi_wallet_consensus  25%   — how many watched wallets are on the same side
  liquidity               15%   — market depth (USD)
  entry_timing            15%   — early entry vs current price drift
  price_inefficiency      10%   — cross-venue / cross-market gap

Final score is rounded to a 0-100 integer-friendly float so it's easy to
display and threshold against `settings.signal_score_threshold`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.config import ScoringWeights, get_settings
from app.models import Market, Trader


@dataclass
class ScoreInputs:
    trader: Trader | None = None
    same_side_wallets: int = 1
    total_watched_wallets: int = 1
    market: Market | None = None
    entry_price: float | None = None
    current_price: float | None = None
    price_gap: float = 0.0  # absolute price delta across venues (0-1 scale)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreBreakdown:
    wallet_quality: float
    multi_wallet_consensus: float
    liquidity: float
    entry_timing: float
    price_inefficiency: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return {
            "wallet_quality": round(self.wallet_quality, 4),
            "multi_wallet_consensus": round(self.multi_wallet_consensus, 4),
            "liquidity": round(self.liquidity, 4),
            "entry_timing": round(self.entry_timing, 4),
            "price_inefficiency": round(self.price_inefficiency, 4),
            "total": round(self.total, 2),
        }


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _wallet_quality(trader: Trader | None) -> float:
    """Combine operator trust, win_rate, and rank into one [0,1] number.

    Trust score is operator-assigned and weighted heavily. Win rate is the
    market-proven signal. Rank is inverted (lower rank = better)."""
    if trader is None:
        return 0.4  # unknown wallet — neutral-low

    trust = _clip((trader.trust_score or 50.0) / 100.0)
    win_rate = _clip(trader.win_rate or 0.5)
    if trader.trader_rank and trader.trader_rank > 0:
        # rank 1 -> ~1.0; rank 5000 -> ~0.0
        rank = _clip(1 - math.log10(trader.trader_rank) / math.log10(5000))
    else:
        rank = 0.5

    return _clip(0.5 * trust + 0.3 * win_rate + 0.2 * rank)


def _multi_wallet_consensus(same_side: int, total_watched: int) -> float:
    if same_side <= 0:
        return 0.0
    # Saturating curve: 1 wallet = 0.2, 2 = 0.5, 3 = 0.7, 5+ = 1.0
    pts = {0: 0.0, 1: 0.2, 2: 0.5, 3: 0.7, 4: 0.85}
    raw = pts.get(same_side, 1.0)
    # mild boost if it's a large fraction of the watchlist
    if total_watched > 0:
        raw = _clip(raw + 0.1 * (same_side / total_watched))
    return _clip(raw)


def _liquidity(market: Market | None) -> float:
    if market is None or market.liquidity_usd is None:
        return 0.2
    liq = max(market.liquidity_usd, 1.0)
    # 1k -> 0.0, 100k -> ~0.66, 1M+ -> ~1.0 (log scale)
    return _clip(math.log10(liq) / 6.0)


def _entry_timing(entry: float | None, current: float | None) -> float:
    if entry is None or current is None:
        return 0.5
    # The further the current price has moved AWAY from entry in the smart-money
    # direction, the more validated the entry. We treat any move > 5pp as strong.
    drift = abs(current - entry)
    return _clip(drift / 0.10)


def _price_inefficiency(price_gap: float) -> float:
    # 0 gap = 0.0; 5pp gap = 0.5; 10pp+ gap = 1.0
    return _clip(abs(price_gap) / 0.10)


def score_signal(
    inputs: ScoreInputs,
    weights: ScoringWeights | None = None,
    *,
    db: Session | None = None,
    sport: str | None = None,
    market_type: str | None = None,
    conviction_penalty: float = 0.0,
) -> ScoreBreakdown:
    """Compute the weighted score. Returns a breakdown and total on 0-100.

    ``db``/``sport``/``market_type`` are accepted for call-site compatibility
    but no longer alter the weights — static priors are always used.

    ``conviction_penalty`` (0..0.5) is subtracted from the final score after
    weighting — used by the contrarian-conflict regime to mark crowded /
    elite-disagreement signals as lower conviction without changing the
    underlying factor values.
    """
    w = weights or get_settings().scoring
    components = {
        "wallet_quality": _wallet_quality(inputs.trader),
        "multi_wallet_consensus": _multi_wallet_consensus(
            inputs.same_side_wallets, inputs.total_watched_wallets
        ),
        "liquidity": _liquidity(inputs.market),
        "entry_timing": _entry_timing(inputs.entry_price, inputs.current_price),
        "price_inefficiency": _price_inefficiency(inputs.price_gap),
    }
    static_weights = {
        "wallet_quality": w.wallet_quality,
        "multi_wallet_consensus": w.multi_wallet_consensus,
        "liquidity": w.liquidity,
        "entry_timing": w.entry_timing,
        "price_inefficiency": w.price_inefficiency,
    }
    # Static priors only — the adaptive per-scope weight learning layer was
    # removed when the project refocused on wallet consensus.
    effective_weights = static_weights

    total = sum(components[name] * effective_weights[name] for name in components) * 100.0
    if conviction_penalty:
        total -= conviction_penalty * 100.0
    breakdown = ScoreBreakdown(total=_clip(total, 0, 100), **components)
    # Stash effective weights on the breakdown so callers can persist them
    # at signal-emit time without re-deriving the lookup.
    breakdown.__dict__["effective_weights"] = effective_weights
    return breakdown
