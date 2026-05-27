"""Additive score-contribution decomposition (#7)."""

from __future__ import annotations

from app.services.mlb_edge_scoring import (
    TOTAL_WEIGHTS,
    additive_contributions,
    weighted_score,
)
from app.services.mlb_totals_model import total_edges


def test_additive_contributions_sum_to_score_minus_baseline():
    factors = {
        "odds_edge": 95, "movement": 50, "environment": 63,
        "pitcher_matchup": 50, "smart_money": 50, "data_quality": 100,
    }
    contribs = additive_contributions(factors, TOTAL_WEIGHTS)
    # Per-term 2dp rounding can drift by a cent from the exact score.
    assert abs(sum(contribs.values()) - (weighted_score(factors, TOTAL_WEIGHTS) - 50)) <= 0.05


def test_total_edges_emits_score_contributions():
    game = {"game_pk": 1, "home_team": "Kansas City Royals", "away_team": "New York Yankees"}
    odds = {
        "consensus_total_line": 8.5, "best_over_price": 2.0, "best_under_price": 1.9,
        "best_over_book": "FD", "best_under_book": "DK", "consensus_price": 1.95,
        "book_count": 3, "rows": [{"bookmaker": "FD"}], "line_disagreement": 0.0,
        "is_valid": True,
    }
    env = {"run_environment_score": 63, "under_environment_score": 60, "warnings": []}
    for edge in total_edges(game=game, odds_analysis=odds, environment=env):
        contribs = edge["score_contributions"]
        assert set(contribs) == set(TOTAL_WEIGHTS)
        assert abs(sum(contribs.values()) - (edge["score"] - 50)) <= 0.05
