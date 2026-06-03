"""Unit tests for the display helpers the Streamlit dashboard + alerts depend on.

These helpers are pure (no Streamlit, no DB), so they're exercised directly.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.utils.dashboard_format import (
    DASH,
    american_to_implied_probability,
    compact_time_ago,
    confidence_label,
    factor_label,
    format_money_short,
    score_tier,
    score_tier_kind,
    short_addr,
)


def test_american_to_implied_probability_handles_american_and_decimal() -> None:
    assert round(american_to_implied_probability(-110), 4) == 0.5238
    assert round(american_to_implied_probability("+100"), 4) == 0.5
    assert round(american_to_implied_probability(1.91), 4) == 0.5236
    # Ambiguous / junk inputs are rejected.
    assert american_to_implied_probability(None) is None
    assert american_to_implied_probability(50) is None  # |x| < 100, not decimal


def test_score_tier_bands() -> None:
    assert score_tier(90) == "HIGH CONV"
    assert score_tier(78) == "STRONG"
    assert score_tier(66) == "LEAN"
    assert score_tier(56) == "WATCH"
    assert score_tier(10) == "PASS"
    assert score_tier(None) == "PASS"


def test_score_tier_kind_maps_to_css_classes() -> None:
    assert score_tier_kind(90) == "gold"
    assert score_tier_kind(10) == "muted"


def test_confidence_label_priority() -> None:
    assert confidence_label(90)[0] == "HIGH CONV"
    assert confidence_label(70)[0] == "ACTIONABLE WATCH"
    assert confidence_label(70, action="pass")[0] == "PASS"
    assert confidence_label(40, action="watch setup")[0] == "WATCH SETUP"


def test_factor_label_known_and_fallback() -> None:
    assert factor_label("multi_wallet_consensus") == "Multi-wallet consensus"
    assert factor_label("some_unknown_factor") == "Some Unknown Factor"
    assert factor_label("") == DASH


def test_format_money_short() -> None:
    assert format_money_short(1_500_000) == "$1.5M"
    assert format_money_short(95_000) == "$95k"
    assert format_money_short(412) == "$412"
    assert format_money_short(None) == DASH


def test_short_addr() -> None:
    assert short_addr("0x1234567890abcdef") == "0x1234…cdef"
    assert short_addr(None) == DASH


def test_compact_time_ago() -> None:
    now = datetime(2026, 6, 2, 12, 0, 0, tzinfo=timezone.utc)
    assert compact_time_ago(now - timedelta(seconds=8), now=now) == "8s"
    assert compact_time_ago(now - timedelta(minutes=4), now=now) == "4m"
    assert compact_time_ago(now - timedelta(hours=2), now=now) == "2h"
    assert compact_time_ago(now - timedelta(days=3), now=now) == "3d"
    assert compact_time_ago(None, now=now) == DASH
