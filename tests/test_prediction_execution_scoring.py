"""Prediction / Execution split — invariants for the dual-score refactor.

These tests pin the four properties the scoring split exists to enforce:

1. ``prediction_score`` does NOT include sportsbook price edge — only
   model conviction (projection / wallets / matchup / environment /
   model confidence).
2. A cheap price-only signal loses watchlist ranking to a
   model+wallet-aligned signal even when the cheap candidate's legacy
   score is higher.
3. The watchlist sort key uses prediction_score first, then
   wallet_alignment, then execution_score.
4. ``execution_score`` still includes sportsbook price edge (50% weight).
"""

from __future__ import annotations

from app.services.mlb_edge_scoring import (
    EXECUTION_WEIGHTS,
    PREDICTION_WEIGHTS,
    compute_execution_score,
    compute_prediction_score,
    is_cheap_price_trap,
    watchlist_sort_key,
)
from app.services.mlb_totals_model import total_edges


def _base_odds(**overrides):
    """Tightly-priced, well-behaved odds payload used as the default in
    most tests. Overrides let each test perturb one field at a time."""
    payload = {
        "consensus_total_line": 8.5,
        "best_over_price": 2.0,
        "best_under_price": 1.9,
        "best_over_book": "FD",
        "best_under_book": "DK",
        "consensus_price": 1.95,
        "book_count": 3,
        "rows": [{"bookmaker": "FD"}],
        "line_disagreement": 0.0,
        "is_valid": True,
    }
    payload.update(overrides)
    return payload


def _base_env(**overrides):
    payload = {"run_environment_score": 50, "under_environment_score": 50, "warnings": []}
    payload.update(overrides)
    return payload


def _base_game():
    return {"game_pk": 1, "home_team": "Kansas City Royals", "away_team": "New York Yankees"}


# ---------------------------------------------------------------------------
# 1. Sportsbook price edge has zero weight in prediction_score.
# ---------------------------------------------------------------------------

def test_sportsbook_price_edge_not_in_prediction_weights():
    """The PREDICTION_WEIGHTS dict must not contain a sportsbook-price-edge
    slot under any of its naming aliases."""
    forbidden = {
        "sportsbook_price_edge", "price_edge", "odds_edge",
    }
    assert not (forbidden & set(PREDICTION_WEIGHTS))


def test_doubling_sportsbook_price_edge_leaves_prediction_unchanged():
    """Holding the prediction inputs fixed, doubling the sportsbook price
    edge must NOT move prediction_score. Two edges that differ only in
    book prices must produce identical prediction_scores."""
    cheap = total_edges(
        game=_base_game(),
        odds_analysis=_base_odds(
            # A juicy price gap on the over side — pure execution edge.
            best_over_price=2.40, consensus_price=1.95,
        ),
        environment=_base_env(),
    )
    expensive = total_edges(
        game=_base_game(),
        odds_analysis=_base_odds(
            best_over_price=1.85, consensus_price=1.95,
        ),
        environment=_base_env(),
    )
    cheap_over = next(e for e in cheap if e["side"] == "over")
    expensive_over = next(e for e in expensive if e["side"] == "over")
    assert cheap_over["prediction_score"] == expensive_over["prediction_score"]
    # And the execution side should reflect the price difference.
    assert cheap_over["execution_score"] > expensive_over["execution_score"]


# ---------------------------------------------------------------------------
# 2. Cheap-price-only candidate loses ranking to model+wallet-aligned one.
# ---------------------------------------------------------------------------

def test_cheap_price_signal_loses_to_model_aligned_signal_in_sort():
    """A 'cheap price, weak model' candidate must sort BELOW a
    'fair price, strong model + wallet consensus' candidate even though
    the cheap one's execution_score is higher."""
    # Cheap-price-only candidate: legacy odds with no model conviction.
    cheap_price = {
        "prediction_score": 55.0,  # below the trap ceiling
        "execution_score": 90.0,   # great price
        "factors": {"wallet_alignment": 50.0},  # no wallet info
    }
    # Model-aligned candidate: real model conviction + wallet consensus.
    model_aligned = {
        "prediction_score": 78.0,
        "execution_score": 60.0,   # fair, not juicy
        "factors": {"wallet_alignment": 82.0},
    }
    ranked = sorted(
        [cheap_price, model_aligned], key=watchlist_sort_key, reverse=True,
    )
    assert ranked[0] is model_aligned
    assert ranked[1] is cheap_price


def test_cheap_price_trap_flag_fires_on_high_execution_low_prediction():
    """The badge fires when execution ≥70 and prediction <65 — exactly
    the regime where the legacy single score would have ranked a
    no-conviction candidate above a model-aligned one."""
    assert is_cheap_price_trap(prediction_score=55.0, execution_score=82.0)
    assert not is_cheap_price_trap(prediction_score=80.0, execution_score=82.0)
    # Execution below the floor: no trap even with weak prediction.
    assert not is_cheap_price_trap(prediction_score=40.0, execution_score=65.0)


# ---------------------------------------------------------------------------
# 3. Watchlist sort key is (prediction, wallet_alignment, execution).
# ---------------------------------------------------------------------------

def test_watchlist_sort_breaks_ties_on_wallet_then_execution():
    """When prediction_score ties, the next key must be wallet_alignment;
    when wallet_alignment also ties, execution_score breaks the tie."""
    base = {"prediction_score": 70.0, "execution_score": 60.0}
    a = {**base, "factors": {"wallet_alignment": 80.0}}
    b = {**base, "factors": {"wallet_alignment": 50.0}}
    c = {**base, "factors": {"wallet_alignment": 80.0}, "execution_score": 75.0}

    ranked = sorted([b, a, c], key=watchlist_sort_key, reverse=True)
    # c and a tie on prediction + wallet, c wins on higher execution.
    # b loses on the wallet axis even though execution is the same as a.
    assert ranked[0] is c
    assert ranked[1] is a
    assert ranked[2] is b


def test_high_prediction_outranks_high_execution_when_prediction_differs():
    """The primary sort key is prediction_score — it must dominate
    even very large differences in execution_score."""
    high_prediction = {
        "prediction_score": 80.0,
        "execution_score": 40.0,
        "factors": {"wallet_alignment": 50.0},
    }
    high_execution = {
        "prediction_score": 60.0,
        "execution_score": 95.0,
        "factors": {"wallet_alignment": 95.0},
    }
    ranked = sorted(
        [high_execution, high_prediction], key=watchlist_sort_key, reverse=True,
    )
    assert ranked[0] is high_prediction


# ---------------------------------------------------------------------------
# 4. Execution score still uses sportsbook price edge.
# ---------------------------------------------------------------------------

def test_sportsbook_price_edge_is_largest_weight_in_execution():
    """``sportsbook_price_edge`` must remain the dominant input in
    EXECUTION_WEIGHTS (≥0.5)."""
    assert "sportsbook_price_edge" in EXECUTION_WEIGHTS
    assert EXECUTION_WEIGHTS["sportsbook_price_edge"] >= 0.5
    # And it must be the single largest weight in the execution formula.
    top = max(EXECUTION_WEIGHTS.items(), key=lambda kv: kv[1])
    assert top[0] == "sportsbook_price_edge"


def test_execution_score_moves_with_sportsbook_price_edge():
    """Raising sportsbook_price_edge from 50 → 90 with all other
    execution inputs at neutral must lift execution_score by the
    expected weight-scaled amount."""
    neutral = {
        "sportsbook_price_edge": 50.0,
        "line_movement": 50.0,
        "clv_signal": 50.0,
        "market_quality": 50.0,
    }
    juicy = {**neutral, "sportsbook_price_edge": 90.0}
    score_neutral, _ = compute_execution_score(neutral)
    score_juicy, _ = compute_execution_score(juicy)
    # The +40-pt move on price edge should propagate through the 0.5
    # weight as a +20-pt move on the final score.
    assert round(score_juicy - score_neutral, 2) == 20.0


def test_prediction_score_responds_to_prediction_inputs():
    """A sanity check on the other axis: changing a prediction-only input
    (projection_edge) must shift prediction_score while leaving
    execution_score untouched if it isn't recomputed."""
    base_prediction = {
        "projection_edge": 50.0,
        "wallet_alignment": 50.0,
        "pitcher_matchup": 50.0,
        "environment": 50.0,
        "model_confidence": 50.0,
    }
    strong = {**base_prediction, "projection_edge": 90.0}
    score_base, _ = compute_prediction_score(base_prediction)
    score_strong, _ = compute_prediction_score(strong)
    # 0.35 weight × +40 = +14 pts.
    assert round(score_strong - score_base, 2) == 14.0
