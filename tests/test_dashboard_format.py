"""Unit tests for `app.utils.dashboard_format` — the formatting layer the
Streamlit dashboard depends on. The dashboard imports these helpers, so any
regression here surfaces as a broken card render."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.utils.dashboard_format import (
    DASH,
    SCORE_BUCKETS,
    american_from_price,
    american_to_implied_probability,
    compact_time_ago,
    confidence_label,
    confidence_word,
    factor_label,
    format_edge_delta,
    format_hit_rate,
    format_price_with_implied_prob,
    odds_provider_label,
    score_bucket_label,
    score_distribution,
    score_tier,
    score_tier_kind,
)


class TestImpliedProbability:
    def test_plus_american(self):
        # +100 should be exactly 50%; the tier border case.
        assert american_to_implied_probability(100) == pytest.approx(0.5, rel=1e-6)

    def test_plus_138_matches_spec(self):
        # +138 -> 100 / 238 ≈ 0.4202
        prob = american_to_implied_probability(138)
        assert prob is not None
        assert round(prob * 100, 1) == 42.0

    def test_minus_120_matches_spec(self):
        # -120 -> 120 / 220 ≈ 0.5454
        prob = american_to_implied_probability(-120)
        assert prob is not None
        assert round(prob * 100, 1) == 54.5

    def test_decimal_price_treated_as_decimal(self):
        # 1.91 is decimal odds (≈ -110 American). 1/1.91 ≈ 0.5236.
        prob = american_to_implied_probability(1.91)
        assert prob is not None
        assert round(prob, 3) == 0.524

    def test_string_inputs(self):
        assert american_to_implied_probability("+138") == pytest.approx(
            american_to_implied_probability(138)
        )
        assert american_to_implied_probability("-120") == pytest.approx(
            american_to_implied_probability(-120)
        )

    def test_invalid_returns_none(self):
        assert american_to_implied_probability(None) is None
        assert american_to_implied_probability("") is None
        assert american_to_implied_probability("abc") is None
        assert american_to_implied_probability(0) is None

    def test_decimal_at_unity_returns_none(self):
        # Decimal-odds at exactly 1.0 imply certainty and produce no value.
        assert american_to_implied_probability(1.0) is None


class TestPriceFormatting:
    def test_format_price_with_implied_prob(self):
        assert format_price_with_implied_prob(138) == "+138 (42.0%)"
        assert format_price_with_implied_prob(-120) == "-120 (54.5%)"

    def test_format_price_dashes_on_missing(self):
        assert format_price_with_implied_prob(None) == DASH
        assert format_price_with_implied_prob("") == DASH

    def test_american_from_decimal(self):
        # 1.91 ≈ -110
        assert american_from_price(1.91) == "-110"
        # 2.38 ≈ +138
        assert american_from_price(2.38) == "+138"

    def test_american_passthrough(self):
        assert american_from_price(-110) == "-110"
        assert american_from_price(138) == "+138"


class TestEdgeDelta:
    def test_positive_delta(self):
        assert format_edge_delta(8.0, 6.5, unit="Ks") == "+1.5 Ks"

    def test_negative_delta(self):
        assert format_edge_delta(6.0, 6.5, unit="Ks") == "-0.5 Ks"

    def test_missing_returns_dash(self):
        assert format_edge_delta(None, 6.5) == DASH
        assert format_edge_delta(6.5, None) == DASH


class TestHitRate:
    def test_renders_fraction(self):
        assert format_hit_rate(4, 5) == "4/5"

    def test_missing_returns_phrase(self):
        assert format_hit_rate(None, 5) == "insufficient history"
        assert format_hit_rate(4, None) == "insufficient history"
        assert format_hit_rate(4, 0) == "insufficient history"


class TestScoreTier:
    def test_high_conv(self):
        assert score_tier(90) == "HIGH CONV"
        assert score_tier_kind(90) == "gold"

    def test_strong(self):
        assert score_tier(78) == "STRONG"
        assert score_tier_kind(78) == "green"

    def test_lean(self):
        assert score_tier(67) == "LEAN"
        assert score_tier_kind(67) == "purple"

    def test_watch(self):
        assert score_tier(58) == "WATCH"
        assert score_tier_kind(58) == "cyan"

    def test_pass(self):
        assert score_tier(20) == "PASS"
        assert score_tier_kind(20) == "muted"

    def test_none(self):
        assert score_tier(None) == "PASS"


class TestConfidenceLabel:
    def test_high_conv_priority_overrides_action(self):
        # Score >= 85 always wins, regardless of action.
        label, kind = confidence_label(91, action="watch")
        assert label == "HIGH CONV"
        assert kind == "gold"

    def test_actionable_watch_when_score_in_lean_band(self):
        label, kind = confidence_label(72, action="bettable only at price")
        assert label == "ACTIONABLE WATCH"
        assert kind == "green"

    def test_pass_label_when_action_pass(self):
        label, kind = confidence_label(40, action="Pass")
        assert label == "PASS"
        assert kind == "muted"

    def test_watch_setup_when_action_watch_and_below_threshold(self):
        label, kind = confidence_label(60, action="watch")
        assert label == "WATCH SETUP"
        assert kind == "purple"

    def test_actionable_watch_overrides_watch_action(self):
        # 70 is at/above actionable threshold AND action is watch -> the
        # spec prefers ACTIONABLE WATCH (the more decision-forward label).
        label, _ = confidence_label(70, action="watch")
        assert label == "ACTIONABLE WATCH"


class TestConfidenceWord:
    def test_low_med_high_vh(self):
        assert confidence_word("low") == ("WATCH", "cyan")
        assert confidence_word("medium") == ("LEAN", "purple")
        assert confidence_word("high") == ("STRONG", "green")
        assert confidence_word("very_high") == ("HIGH CONV", "gold")

    def test_unknown_returns_placeholder(self):
        assert confidence_word(None) == ("CONF ?", "muted")
        assert confidence_word("bogus") == ("CONF ?", "muted")


class TestFactorLabel:
    def test_renamed_factors(self):
        assert factor_label("odds_edge") == "Sportsbook price edge"
        assert factor_label("environment") == "Run environment rating"
        assert factor_label("pitcher_recent_form") == "Recent form rating"
        assert factor_label("matchup_k_profile") == "Opponent K matchup"
        assert factor_label("smart_money") == "Wallet flow signal"
        assert factor_label("line_movement") == "Line movement"

    def test_fallback_titlecases_unknown_keys(self):
        assert factor_label("brand_new_factor") == "Brand New Factor"


class TestScoreBuckets:
    def test_buckets_are_contiguous(self):
        # Every bucket label should appear in score_distribution output.
        scores = [10, 52, 57, 62, 67, 71, 85]
        dist = score_distribution(scores)
        # Each bucket should be a key.
        for _, _, label in SCORE_BUCKETS:
            assert label in dist
        assert sum(dist.values()) == len(scores)

    def test_score_bucket_label(self):
        assert score_bucket_label(67) == "65–70"
        assert score_bucket_label(80) == "70+"
        assert score_bucket_label(40) == "<50"
        assert score_bucket_label(None) is None


class TestProviderLabel:
    def test_sgo_is_fallback(self):
        assert odds_provider_label("sportsgameodds") == ("SportsGameOdds", True)
        assert odds_provider_label("SportsGameOdds") == ("SportsGameOdds", True)

    def test_primary_default(self):
        assert odds_provider_label(None) == ("Odds-API.io", False)
        assert odds_provider_label("odds_api") == ("Odds-API.io", False)


class TestCompactTimeAgo:
    def test_seconds(self):
        now = datetime(2026, 5, 26, 12, 0, 0, tzinfo=timezone.utc)
        past = now - timedelta(seconds=8)
        assert compact_time_ago(past, now=now) == "8s"

    def test_minutes(self):
        now = datetime(2026, 5, 26, 12, 0, 0, tzinfo=timezone.utc)
        past = now - timedelta(minutes=12)
        assert compact_time_ago(past, now=now) == "12m"

    def test_hours(self):
        now = datetime(2026, 5, 26, 12, 0, 0, tzinfo=timezone.utc)
        past = now - timedelta(hours=3)
        assert compact_time_ago(past, now=now) == "3h"

    def test_days(self):
        now = datetime(2026, 5, 26, 12, 0, 0, tzinfo=timezone.utc)
        past = now - timedelta(days=2)
        assert compact_time_ago(past, now=now) == "2d"

    def test_missing_returns_dash(self):
        assert compact_time_ago(None) == DASH
        assert compact_time_ago("") == DASH
