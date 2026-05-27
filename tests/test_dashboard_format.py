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
    best_executable_edge,
    build_consensus_wallets,
    compact_time_ago,
    consensus_wallets_chips_html,
    conviction_tier,
    edge_risk_flags,
    edge_source_stack,
    executable_edge_rows,
    format_score_contributions,
    confidence_label,
    confidence_word,
    edge_vs_market,
    factor_label,
    format_card_title,
    format_cents,
    format_edge_delta,
    format_hit_rate,
    format_money_short,
    format_price_with_implied_prob,
    format_probability,
    odds_provider_label,
    polished_missing,
    score_bucket_label,
    score_distribution,
    score_tier,
    score_tier_kind,
    team_short,
    wallet_alignment_percent,
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


class TestCardTitleFormat:
    def test_pitcher_strikeouts_clean_title(self):
        edge = {
            "edge_type": "pitcher_strikeouts",
            "market": "Joe Ryan Strikeouts - Over 6.5",
            "side": "over",
            "line": 6.5,
        }
        assert format_card_title(edge) == "Joe Ryan — Over 6.5 Ks"

    def test_pitcher_drops_dangling_hyphen_when_line_missing(self):
        edge = {
            "edge_type": "pitcher_strikeouts",
            "market": "Joe Ryan Strikeouts -",
            "side": "over",
            "line": None,
        }
        # Should not end with a hyphen; should still describe the bet.
        title = format_card_title(edge)
        assert not title.rstrip().endswith("-")
        assert "Joe Ryan" in title
        assert "Over" in title

    def test_game_total_with_team_names_uses_abbreviations(self):
        edge = {
            "edge_type": "game_total",
            "market": "Full Game Total - Under 8.5",
            "side": "under",
            "line": 8.5,
            "home_team": "Kansas City Royals",
            "away_team": "New York Yankees",
        }
        assert format_card_title(edge) == "NYY @ KC — Under 8.5"

    def test_game_total_without_team_names(self):
        edge = {
            "edge_type": "game_total",
            "market": "Full Game Total - Over 9.0",
            "side": "over",
            "line": 9.0,
        }
        title = format_card_title(edge)
        # Should still render a sensible 'Over 9' headline even with no
        # team names — never the raw 'Full Game Total -' with hyphen.
        assert "Over 9" in title
        assert not title.endswith("-")

    def test_unknown_edge_type_strips_trailing_hyphen(self):
        edge = {
            "edge_type": "weird_type",
            "market": "Some Market -",
            "side": None,
            "line": None,
        }
        assert format_card_title(edge) == "Some Market"

    def test_integer_line_renders_without_decimal(self):
        edge = {
            "edge_type": "pitcher_strikeouts",
            "market": "Joe Ryan Strikeouts - Over 7",
            "side": "over",
            "line": 7,
        }
        assert format_card_title(edge) == "Joe Ryan — Over 7 Ks"


class TestTeamShort:
    def test_known_team(self):
        assert team_short("New York Yankees") == "NYY"
        assert team_short("Kansas City Royals") == "KC"
        assert team_short("MIAMI MARLINS") == "MIA"

    def test_unknown_team_passthrough(self):
        assert team_short("Some New Franchise") == "Some New Franchise"

    def test_empty_returns_empty(self):
        assert team_short(None) == ""
        assert team_short("") == ""


class TestPolishedMissing:
    def test_known_kinds_return_premium_phrasing(self):
        assert polished_missing("projection") == "Model projection not yet calibrated"
        assert polished_missing("history") == "Limited recent sample"
        assert polished_missing("clv_pending") == "CLV pending"
        assert polished_missing("closing") == "Awaiting closing line"
        # Never the ugly raw 'pending' / 'insufficient history'.
        for ugly in ("pending", "insufficient history", "n/a", "null", "None"):
            for kind in ("projection", "history", "clv_pending", "closing", "movement"):
                assert polished_missing(kind).lower() != ugly.lower()

    def test_unknown_kind_falls_back(self):
        assert polished_missing("not-a-kind") == "Data unavailable"


class TestFormatProbability:
    def test_fractional_input(self):
        assert format_probability(0.42) == "42.0%"

    def test_percent_input(self):
        assert format_probability(42) == "42.0%"

    def test_missing(self):
        assert format_probability(None) == DASH
        assert format_probability("") == DASH


class TestFormatCents:
    def test_fraction(self):
        assert format_cents(0.34) == "34¢"

    def test_int(self):
        assert format_cents(34) == "34¢"

    def test_missing(self):
        assert format_cents(None) == DASH


class TestEdgeVsMarket:
    def test_signed_delta_in_percent(self):
        assert edge_vs_market(0.5, 0.42) == "+8.0%"
        assert edge_vs_market(0.4, 0.50) == "-10.0%"

    def test_returns_none_when_missing(self):
        assert edge_vs_market(None, 0.42) is None
        assert edge_vs_market(0.5, None) is None


class TestMoneyShort:
    def test_thousands_and_millions(self):
        assert format_money_short(95_000) == "$95k"
        assert format_money_short(1_200_000) == "$1.2M"
        assert format_money_short(412_000) == "$412k"

    def test_small_value(self):
        assert format_money_short(123) == "$123"

    def test_missing(self):
        assert format_money_short(None) == DASH


class TestWalletAlignment:
    def test_full_alignment(self):
        assert wallet_alignment_percent(4, 4) == 100.0

    def test_partial(self):
        assert wallet_alignment_percent(3, 5) == 60.0

    def test_missing_or_zero(self):
        assert wallet_alignment_percent(None, 4) is None
        assert wallet_alignment_percent(2, 0) is None


class TestBuildConsensusWallets:
    def _fill(self, **kw):
        base = {
            "trader_nickname": "LaBradfordSmith22",
            "wallet": "0xabc",
            "trader_id": 1,
            "market_slug": "mlb-stl-mil-2026-05-26",
            "outcome": "St. Louis Cardinals",
            "side": "BUY",
            "size_usd": 100.0,
            "entry_price": 0.50,
            "created_at": "2026-05-26T10:00:00",
        }
        base.update(kw)
        return base

    def test_same_trader_many_fills_appears_once(self):
        fills = [self._fill(size_usd=10.0) for _ in range(10)]
        rows = build_consensus_wallets(fills)
        assert len(rows) == 1
        assert rows[0]["fill_count"] == 10
        assert rows[0]["total_size_usd"] == 100.0

    def test_avg_entry_is_size_weighted(self):
        fills = [
            self._fill(size_usd=100.0, entry_price=0.40),
            self._fill(size_usd=300.0, entry_price=0.60),
        ]
        rows = build_consensus_wallets(fills)
        # (0.40*100 + 0.60*300) / 400 = 0.55
        assert rows[0]["avg_entry"] == 0.55

    def test_buy_sell_netting(self):
        fills = [
            self._fill(side="BUY", size_usd=100.0),
            self._fill(side="SELL", size_usd=40.0),
        ]
        rows = build_consensus_wallets(fills)
        assert rows[0]["total_size_usd"] == 140.0   # gross
        assert rows[0]["net_size_usd"] == 60.0      # 100 - 40
        assert rows[0]["net_side"] == "BUY"
        # Net flips to SELL when sells dominate.
        sells = [self._fill(side="BUY", size_usd=10.0), self._fill(side="SELL", size_usd=90.0)]
        assert build_consensus_wallets(sells)[0]["net_side"] == "SELL"

    def test_duplicate_nickname_same_wallet_merges(self):
        # Same wallet, two different trader_id rows, mixed case address.
        fills = [
            self._fill(trader_id=1, wallet="0xABC"),
            self._fill(trader_id=2, wallet="0xabc"),
        ]
        rows = build_consensus_wallets(fills)
        assert len(rows) == 1
        assert rows[0]["fill_count"] == 2

    def test_missing_wallet_falls_back_to_trader_id(self):
        fills = [
            self._fill(wallet=None, trader_id=7),
            self._fill(wallet=None, trader_id=7),
            self._fill(wallet=None, trader_id=9),
        ]
        rows = build_consensus_wallets(fills)
        assert len(rows) == 2
        assert sorted(r["fill_count"] for r in rows) == [1, 2]

    def test_sorted_by_total_size_desc(self):
        fills = [
            self._fill(wallet="0xsmall", size_usd=50.0),
            self._fill(wallet="0xbig", size_usd=500.0),
        ]
        rows = build_consensus_wallets(fills)
        assert rows[0]["wallet_address"] == "0xbig"


class TestEdgeSourceStack:
    def test_sportsbook_and_wallet_and_risk_tags(self):
        edge = {
            "factors": {"odds_edge": 95, "movement": 75},
            "odds_stale": True,
            "wallet_context": {
                "tracked_wallet_count": 3,
                "elite_wallet_disagreement": 1,
                "tags": ["WALLET CONFIRMED", "CROWDED SIDE"],
                "execution": {"implied_prob": 0.56},
            },
        }
        labels = [lbl for lbl, _ in edge_source_stack(edge)]
        assert "SPORTSBOOK EDGE" in labels
        assert "WALLET CONFIRMED" in labels
        assert "ELITE DISAGREEMENT" in labels
        assert "STEAM MOVE" in labels
        assert "PREDICTION-MARKET EDGE" in labels
        assert "CROWDED CONSENSUS" in labels
        assert "STALE ODDS" in labels

    def test_model_only_when_no_corroboration(self):
        edge = {"factors": {"odds_edge": 50, "movement": 50}, "wallet_context": {"tracked_wallet_count": 0}}
        labels = [lbl for lbl, _ in edge_source_stack(edge)]
        assert labels == ["MODEL ONLY"]


class TestEdgeRiskFlags:
    def test_flags_present(self):
        edge = {
            "odds_stale": True,
            "chase_risk": "high",
            "wallet_context": {
                "tracked_wallet_count": 0,
                "elite_wallet_disagreement": 2,
                "tags": ["CROWDED SIDE"],
                "execution": {"liquidity_usd": 100.0},
            },
        }
        labels = [lbl for lbl, _ in edge_risk_flags(edge)]
        assert "Stale odds" in labels
        assert "No wallet confirmation" in labels
        assert "Crowded side" in labels
        assert "Sharp disagreement" in labels
        assert "Low liquidity" in labels
        assert "Late adverse movement" in labels

    def test_no_flags_clean_edge(self):
        edge = {"odds_stale": False, "chase_risk": "low",
                "wallet_context": {"tracked_wallet_count": 3, "tags": ["WALLET CONFIRMED"]}}
        assert edge_risk_flags(edge) == []


class TestConvictionTier:
    def test_high_conv(self):
        assert conviction_tier(90) == ("HIGH CONV", "gold")

    def test_pass(self):
        assert conviction_tier(40) == ("PASS", "muted")


class TestScoreContributions:
    def test_sorted_signed_and_wallet_line(self):
        contribs = {"odds_edge": 13.5, "data_quality": -1.0, "movement": 0.0}
        rows = format_score_contributions(contribs, wallet_adjustment=-5.0)
        # Largest magnitude first; wallet line included.
        assert rows[0][0] == "Sportsbook price edge" and rows[0][1] == 13.5 and rows[0][2] == "green"
        labels = [r[0] for r in rows]
        assert "Wallet flow" in labels
        wallet_row = next(r for r in rows if r[0] == "Wallet flow")
        assert wallet_row[1] == -5.0 and wallet_row[2] == "red"

    def test_zero_wallet_adjustment_omitted(self):
        rows = format_score_contributions({"odds_edge": 5.0}, wallet_adjustment=0.0)
        assert all(r[0] != "Wallet flow" for r in rows)


class TestExecutableEdge:
    def _edge(self):
        return {
            "best_price": 2.40, "best_book": "FanDuel", "source_url": "http://sb",
            "calibrated_probability": 0.62,
            "wallet_context": {"execution": {"platform": "polymarket", "side_price": 0.50,
                                             "implied_prob": 0.50, "market_url": "http://pm"}},
        }

    def test_rows_include_pm_and_sportsbook(self):
        rows = executable_edge_rows(self._edge())
        venues = [r["venue"] for r in rows]
        assert "Polymarket" in venues
        assert any("Sportsbook" in v for v in venues)

    def test_best_edge_picks_largest_positive(self):
        best = best_executable_edge(self._edge())
        # fair 0.62; sportsbook implied 1/2.4=0.4167 → +20.3; PM 0.50 → +12.0.
        assert best["venue"].startswith("Sportsbook")
        assert best["edge_pct"] == 20.3

    def test_no_fair_prob_returns_none(self):
        edge = {"best_price": 2.4, "wallet_context": {}}
        assert best_executable_edge(edge) is None


class TestConsensusWalletsChipsHtml:
    def test_no_raw_html_from_trader_name(self):
        rows = build_consensus_wallets([
            {"trader_nickname": "<script>x</script>", "wallet": "0xa",
             "size_usd": 100.0, "side": "BUY", "market_slug": "m", "outcome": "Over"},
        ])
        out = consensus_wallets_chips_html(rows)
        assert "<script>" not in out          # name markup never leaks
        assert "&lt;script&gt;" in out         # it is escaped instead
        assert out.count("sf-chips") == 1      # one real wrapper, rendered once

    def test_limit_caps_chip_count(self):
        rows = build_consensus_wallets([
            {"wallet": f"0x{i}", "size_usd": float(100 - i), "side": "BUY",
             "market_slug": "m", "outcome": "Over"}
            for i in range(6)
        ])
        assert consensus_wallets_chips_html(rows, limit=3).count("<span") == 3

    def test_empty_returns_empty_string(self):
        assert consensus_wallets_chips_html([]) == ""
