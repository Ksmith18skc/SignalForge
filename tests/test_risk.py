"""Tests for the risk service."""

from __future__ import annotations

from app.config import RiskLimits, get_settings
from app.models import Trader
from app.services.risk import TradeRequest, evaluate


def test_position_capped_to_max_pct():
    limits = RiskLimits(bankroll_usd=10_000, max_position_size_pct=0.05)
    decision = evaluate(
        TradeRequest(market_id=1, side="YES", price=0.5, desired_size_usd=5_000),
        limits=limits,
        mode="paper",
    )
    # Should be capped to 5% of 10k = 500
    assert decision.recommended_size_usd <= 500.0
    assert any("capped" in r for r in decision.reasons)


def test_daily_exposure_trims_size():
    limits = RiskLimits(
        bankroll_usd=10_000,
        max_position_size_pct=0.5,
        max_daily_exposure_pct=0.2,
    )
    decision = evaluate(
        TradeRequest(
            market_id=1,
            side="YES",
            price=0.5,
            desired_size_usd=5_000,
            todays_exposure_usd=1_500,
        ),
        limits=limits,
        mode="paper",
    )
    # daily cap 2k - already used 1.5k -> 500 left
    assert decision.recommended_size_usd == 500.0


def test_per_market_cap_trims_size():
    limits = RiskLimits(
        bankroll_usd=10_000,
        max_position_size_pct=0.5,
        max_per_market_exposure_pct=0.1,
    )
    decision = evaluate(
        TradeRequest(
            market_id=42,
            side="YES",
            price=0.5,
            desired_size_usd=5_000,
            per_market_exposure_usd=800,
        ),
        limits=limits,
        mode="paper",
    )
    # per-market cap 1k - already used 800 -> 200 left
    assert decision.recommended_size_usd == 200.0


def test_live_mode_force_downgrades_in_mvp(monkeypatch):
    # MVP default: auto trading disabled, so even live mode -> alert_only
    settings = get_settings()
    monkeypatch.setattr(settings, "enable_auto_trading", False)
    decision = evaluate(
        TradeRequest(market_id=1, side="YES", price=0.5, desired_size_usd=100),
        mode="live",
    )
    assert decision.mode == "alert_only"


def test_disabled_copy_blocks_trade():
    trader = Trader(nickname="x", copy_enabled=False, copy_mode="disabled")
    decision = evaluate(
        TradeRequest(market_id=1, side="YES", price=0.5, desired_size_usd=100, trader=trader),
        mode="paper",
    )
    assert decision.allowed is False
    assert any("copy_enabled" in b for b in decision.blocked_by)


def test_alert_only_does_not_block():
    """alert_only is allowed — it just means no real money moves."""
    trader = Trader(nickname="x", copy_enabled=True, copy_mode="alert_only")
    decision = evaluate(
        TradeRequest(market_id=1, side="YES", price=0.5, desired_size_usd=100, trader=trader),
        mode="alert_only",
    )
    assert decision.mode == "alert_only"
    assert decision.allowed is True
    assert decision.is_paper_or_alert
