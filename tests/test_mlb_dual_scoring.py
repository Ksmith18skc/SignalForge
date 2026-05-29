from __future__ import annotations

from app.services.mlb_edge_scoring import watchlist_sort_key
from app.services.mlb_totals_model import total_edges


def _game() -> dict:
    return {
        "game_pk": 1,
        "home_team": "Kansas City Royals",
        "away_team": "New York Yankees",
    }


def _environment() -> dict:
    return {
        "run_environment_score": 50,
        "under_environment_score": 50,
        "warnings": [],
    }


def _odds(*, over_price: float, under_price: float = 1.9) -> dict:
    return {
        "consensus_total_line": 8.5,
        "best_over_price": over_price,
        "best_under_price": under_price,
        "best_over_book": "FD",
        "best_under_book": "DK",
        "consensus_price": 1.9,
        "book_count": 5,
        "rows": [{"bookmaker": f"Book {idx}"} for idx in range(5)],
        "line_disagreement": 0.0,
        "is_valid": True,
    }


def _over_edge(over_price: float) -> dict:
    edges = total_edges(
        game=_game(),
        odds_analysis=_odds(over_price=over_price),
        environment=_environment(),
        pitcher_matchup_score=50,
        smart_money_score=50,
    )
    return next(edge for edge in edges if edge["side"] == "over")


def test_sportsbook_edge_does_not_affect_prediction_score() -> None:
    cheap_price = _over_edge(2.8)
    bad_price = _over_edge(1.5)

    assert cheap_price["prediction_score"] == bad_price["prediction_score"]
    assert cheap_price["legacy_score"] > bad_price["legacy_score"]


def test_execution_score_includes_sportsbook_pricing_edge() -> None:
    cheap_price = _over_edge(2.8)
    bad_price = _over_edge(1.5)

    assert cheap_price["execution_score"] > bad_price["execution_score"]
    assert "sportsbook_price_edge" in cheap_price["execution_breakdown"]


def test_cheap_price_trap_badge_condition() -> None:
    edge = _over_edge(2.8)

    assert edge["execution_score"] >= 70
    assert edge["prediction_score"] < 65
    assert edge["cheap_price_trap"] is True


def test_cheap_price_only_signal_loses_ranking_to_model_wallet_signal() -> None:
    cheap_price_only = {
        "prediction_score": 60,
        "execution_score": 92,
        "factors": {"wallet_alignment": 50},
    }
    model_wallet_aligned = {
        "prediction_score": 66,
        "execution_score": 55,
        "factors": {"wallet_alignment": 82},
    }

    ranked = sorted(
        [cheap_price_only, model_wallet_aligned],
        key=watchlist_sort_key,
        reverse=True,
    )

    assert ranked[0] is model_wallet_aligned


def test_watchlist_sorts_by_prediction_score_first() -> None:
    higher_prediction = {
        "prediction_score": 70,
        "execution_score": 40,
        "factors": {"wallet_alignment": 30},
    }
    lower_prediction_better_everything_else = {
        "prediction_score": 69,
        "execution_score": 99,
        "factors": {"wallet_alignment": 99},
    }

    ranked = sorted(
        [lower_prediction_better_everything_else, higher_prediction],
        key=watchlist_sort_key,
        reverse=True,
    )

    assert ranked[0] is higher_prediction
