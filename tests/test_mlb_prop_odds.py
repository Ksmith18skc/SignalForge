from __future__ import annotations

from app.services.mlb_prop_odds import (
    consensus_for_pitcher,
    names_match,
    normalize_name,
    normalize_pitcher_strikeout_props,
)


def _payload():
    return {
        "id": 123,
        "bookmakers": {
            "DraftKings": [
                {
                    "name": "Pitcher Strikeouts",
                    "updatedAt": "2026-05-25T12:00:00Z",
                    "odds": [
                        {
                            "playerName": "Tarik Skubal",
                            "hdp": 6.5,
                            "over": 1.91,
                            "under": 1.83,
                        }
                    ],
                }
            ],
            "FanDuel": [
                {
                    "name": "Player Strikeouts",
                    "updatedAt": "2026-05-25T12:01:00Z",
                    "odds": [
                        {
                            "label": "Tarik Skubal Over 7.5 Strikeouts",
                            "hdp": 7.5,
                            "over": 2.1,
                            "under": 1.72,
                        }
                    ],
                }
            ],
            "RandomBook": [
                {
                    "name": "Pitcher Strikeouts",
                    "odds": [{"playerName": "Tarik Skubal", "hdp": 6.5, "over": 1.9}],
                }
            ],
        },
    }


def test_normalize_pitcher_strikeout_props_supported_books_only():
    rows = normalize_pitcher_strikeout_props(_payload())

    assert len(rows) == 2
    assert rows[0].player_name == "Tarik Skubal"
    assert rows[0].line == 6.5
    assert rows[0].sportsbook == "DraftKings"
    assert rows[1].sportsbook == "FanDuel"


def test_name_normalization_and_fuzzy_matching():
    assert normalize_name("José Berríos Jr.") == "jose berrios"
    assert names_match("Jose Berrios", "José Berríos Jr.")
    assert names_match("T. Skubal", "Tarik Skubal")


def test_consensus_calculates_best_lines_prices_and_disagreement():
    rows = normalize_pitcher_strikeout_props(_payload())

    consensus = consensus_for_pitcher(rows, "Tarik Skubal")

    assert consensus["book_count"] == 2
    assert consensus["line"] == 7.0
    assert consensus["best_over_line"] == 7.5
    assert consensus["best_over_price"] == 2.1
    assert consensus["best_over_book"] == "FanDuel"
    assert consensus["best_under_book"] == "DraftKings"
    assert consensus["line_disagreement"] == 1.0
    assert consensus["average_implied_probability"] is not None


def test_consensus_handles_missing_books():
    rows = normalize_pitcher_strikeout_props(
        {"bookmakers": {"DraftKings": _payload()["bookmakers"]["DraftKings"]}}
    )

    consensus = consensus_for_pitcher(rows, "Tarik Skubal")

    assert consensus["book_count"] == 1
    assert consensus["warnings"] == ["Fewer than 2 books available"]


def test_consensus_missing_pitcher_returns_empty_warning():
    consensus = consensus_for_pitcher(normalize_pitcher_strikeout_props(_payload()), "Logan Webb")

    assert consensus["line"] is None
    assert "No pitcher strikeout props found" in consensus["warnings"][0]
