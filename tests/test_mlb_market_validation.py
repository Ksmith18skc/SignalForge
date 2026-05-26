from __future__ import annotations

from app.services.mlb_market_validation import (
    MarketSubtype,
    classify_market_subtype,
    is_valid_line,
    is_valid_price,
    normalized_total_name,
)
from app.services.mlb_odds_analysis import analyze_game_totals, odds_edge_score, summarize_markets


def test_market_classifier_does_not_treat_pitcher_total_strikeouts_as_game_total():
    row = {"market": "Pitcher Total Strikeouts", "label": "Gerrit Cole Over 6.5"}

    assert classify_market_subtype(row) == MarketSubtype.PLAYER_PROP


def test_market_classifier_identifies_team_and_first_five_totals():
    assert classify_market_subtype({"market": "Team Total"}) == MarketSubtype.TEAM_TOTAL
    assert classify_market_subtype({"market": "1st 5 Innings Total"}) == MarketSubtype.FIRST_5_TOTAL


def test_strict_line_ranges():
    assert is_valid_line(MarketSubtype.FULL_GAME_TOTAL, 8.5)
    assert not is_valid_line(MarketSubtype.FULL_GAME_TOTAL, 3.7)
    assert is_valid_line(MarketSubtype.FIRST_5_TOTAL, 4.5)
    assert not is_valid_line(MarketSubtype.FIRST_5_TOTAL, 12.0)
    assert is_valid_line(MarketSubtype.PLAYER_PROP, 6.5)
    assert not is_valid_line(MarketSubtype.PLAYER_PROP, 20.0)


def test_malformed_prices_are_rejected():
    assert is_valid_price(1.91)
    assert not is_valid_price(None)
    assert not is_valid_price(0)
    assert not is_valid_price(-110)


def test_pitcher_total_strikeouts_payload_does_not_create_game_total():
    payload = {
        "id": "ev_pitcher_props",
        "bookmakers": {
            "DraftKings": [
                {
                    "name": "Pitcher Total Strikeouts",
                    "odds": [{"hdp": 6.5, "over": 1.91, "under": 1.91, "label": "Gerrit Cole"}],
                }
            ]
        },
    }

    analysis = analyze_game_totals(payload)

    assert analysis["is_valid"] is False
    assert analysis["book_count"] == 0


def test_debug_market_summary_does_not_flag_pitcher_props_as_totals():
    payload = {
        "id": "ev_pitcher_props",
        "bookmakers": {
            "DraftKings": [
                {
                    "name": "Pitcher Total Strikeouts",
                    "odds": [{"hdp": 6.5, "over": 1.91, "under": 1.91, "label": "Gerrit Cole"}],
                }
            ]
        },
    }

    summary = summarize_markets(payload)

    assert summary["has_totals"] is False
    assert summary["has_pitcher_ks"] is True


def test_normalized_total_display_names_are_explicit():
    assert normalized_total_name(
        scope=MarketSubtype.FULL_GAME_TOTAL,
        side="under",
        line=8.5,
        home="New York Yankees",
        away="Kansas City Royals",
    ) == "Full Game Total - Under 8.5"
    assert normalized_total_name(
        scope=MarketSubtype.FIRST_5_TOTAL,
        side="over",
        line=4.5,
        home="New York Yankees",
        away="Kansas City Royals",
    ) == "First 5 Innings Total - Over 4.5"


def test_odds_score_caps_malformed_or_extreme_inputs():
    malformed = {"book_count": 4, "line_disagreement": 1.0, "best_over_price": -110, "consensus_price": 1.9}
    assert odds_edge_score(malformed, "over") == 40.0

    extreme = {"book_count": 10, "line_disagreement": 10, "best_over_price": 10.0, "consensus_price": 1.01}
    assert odds_edge_score(extreme, "over") == 95.0
