from __future__ import annotations

from app.models import MlbEdge
from app.services.mlb_edge_engine import _build_daily_card
from app.services.mlb_edge_scoring import classify_edge, weighted_score
from app.services.mlb_environment import score_environment
from app.services.mlb_odds_analysis import analyze_game_totals


def test_environment_scoring_hot_wind_out_boosts_runs():
    env = score_environment({"temp_f": 88, "humidity": 62, "wind_mph": 15, "wind_dir": "OUT"})

    assert env["run_environment_score"] > 60
    assert env["under_environment_score"] < 50


def test_environment_scoring_cold_wind_in_boosts_under_and_k():
    env = score_environment({"temp_f": 48, "humidity": 45, "wind_mph": 16, "wind_dir": "IN"})

    assert env["under_environment_score"] > 60
    assert env["k_environment_score"] > 55


def test_game_total_odds_normalization_compares_books():
    payload = {
        "bookmakers": {
            "DraftKings": [
                {
                    "name": "Totals",
                    "updatedAt": "2026-05-25T12:00:00Z",
                    "odds": [{"hdp": 8.5, "over": 1.91, "under": 1.95}],
                }
            ],
            "FanDuel": [
                {
                    "name": "Totals",
                    "updatedAt": "2026-05-25T12:01:00Z",
                    "odds": [{"hdp": 9.0, "over": 1.98, "under": 1.88}],
                }
            ],
        }
    }

    analysis = analyze_game_totals(payload)

    assert analysis["book_count"] == 2
    assert analysis["consensus_total_line"] == 8.75
    assert analysis["best_over_book"] == "FanDuel"
    assert analysis["line_disagreement"] == 0.5


def test_edge_score_classification_rules():
    assert classify_edge(64)["action"] == "Pass"
    assert classify_edge(70)["action"] == "Watch"
    assert classify_edge(80)["action"] == "Bettable only at price"
    assert classify_edge(87)["action"] == "Strong candidate"
    assert weighted_score({"a": 100, "b": 0}, {"a": 0.5, "b": 0.5}) == 50


def test_daily_card_ranking_and_missing_data_downgrade(db_session):
    db_session.add_all(
        [
            MlbEdge(
                game_pk=1,
                edge_type="game_total",
                market="A at B Over 8.5",
                side="over",
                score=82,
                confidence="high",
                action="Bettable only at price",
                chase_risk="low",
                reasons=["good"],
                warnings=[],
                generated_for_date="2026-05-25",
            ),
            MlbEdge(
                game_pk=2,
                edge_type="pitcher_strikeouts",
                market="Pitcher Over 6.5 Ks",
                side="over",
                score=63,
                confidence="low",
                action="Pass",
                chase_risk="medium",
                reasons=["near"],
                warnings=["Pitcher prop odds missing"],
                generated_for_date="2026-05-25",
            ),
        ]
    )
    db_session.flush()

    card = _build_daily_card(db_session, "2026-05-25")

    assert card.top_game_totals[0]["score"] == 82
    assert card.near_misses[0]["market"] == "Pitcher Over 6.5 Ks"
    assert classify_edge(82, ["Weather missing", "Odds missing", "Statcast missing"])["confidence"] == "low"
