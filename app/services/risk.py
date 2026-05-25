"""Risk guardrails.

MVP is alert-only — these checks return a `RiskDecision` describing what would
have been blocked if real trading were on, plus a sized recommendation if it
passes. No order placement happens here. Ever.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from app.config import RiskLimits, get_settings
from app.models import Trader

Mode = Literal["disabled", "alert_only", "paper", "live"]


@dataclass
class TradeRequest:
    market_id: int
    side: str
    price: float
    desired_size_usd: float
    trader: Trader | None = None
    todays_exposure_usd: float = 0.0
    per_market_exposure_usd: float = 0.0


@dataclass
class RiskDecision:
    allowed: bool
    mode: Mode
    recommended_size_usd: float
    reasons: list[str] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)

    @property
    def is_paper_or_alert(self) -> bool:
        return self.mode in ("alert_only", "paper")


def evaluate(
    request: TradeRequest,
    limits: RiskLimits | None = None,
    mode: Mode | None = None,
) -> RiskDecision:
    """Decide whether (and how big) to ack a trade request.

    Live mode is force-disabled by config in the MVP. Even if a caller asks for
    `live`, we downgrade to `alert_only` unless `enable_auto_trading=True`.
    """
    settings = get_settings()
    limits = limits or settings.risk
    effective_mode: Mode = mode or settings.default_copy_mode

    # Hard guard: MVP does not place real trades.
    if effective_mode == "live" and not settings.enable_auto_trading:
        effective_mode = "alert_only"

    bankroll = max(limits.bankroll_usd, 1.0)
    max_position = bankroll * limits.max_position_size_pct
    max_daily = bankroll * limits.max_daily_exposure_pct
    max_per_market = bankroll * limits.max_per_market_exposure_pct

    reasons: list[str] = []
    blocked: list[str] = []

    requested = max(request.desired_size_usd, 0.0)
    sized = min(requested, max_position)
    if sized < requested:
        reasons.append(
            f"Position capped from ${requested:,.0f} to ${sized:,.0f} "
            f"(max_position_size_pct={limits.max_position_size_pct:.0%})"
        )

    daily_room = max(max_daily - request.todays_exposure_usd, 0.0)
    if sized > daily_room:
        before = sized
        sized = daily_room
        reasons.append(
            f"Position trimmed to ${sized:,.0f} (was ${before:,.0f}) "
            f"by daily exposure cap of ${max_daily:,.0f}"
        )

    market_room = max(max_per_market - request.per_market_exposure_usd, 0.0)
    if sized > market_room:
        before = sized
        sized = market_room
        reasons.append(
            f"Position trimmed to ${sized:,.0f} (was ${before:,.0f}) "
            f"by per-market cap of ${max_per_market:,.0f}"
        )

    if sized <= 0:
        blocked.append("No risk budget left after applying caps")

    if request.trader is not None:
        if not request.trader.copy_enabled:
            blocked.append(f"Trader '{request.trader.nickname}' has copy_enabled=False")
        if request.trader.copy_mode == "disabled":
            blocked.append(f"Trader '{request.trader.nickname}' copy_mode=disabled")

    allowed = not blocked and effective_mode != "disabled"

    return RiskDecision(
        allowed=allowed,
        mode=effective_mode,
        recommended_size_usd=round(sized, 2),
        reasons=reasons,
        blocked_by=blocked,
    )
