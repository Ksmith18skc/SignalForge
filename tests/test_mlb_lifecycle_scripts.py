from __future__ import annotations

from datetime import datetime, timedelta

from app.models import MlbEdge, MlbGame
from scripts.grade_mlb_results import _grade_pitcher_k, _grade_total
from scripts.send_mlb_performance_report import build_report
from scripts.update_mlb_closing_lines import _candidate_edges


def test_grade_total_over_win():
    edge = MlbEdge(
        game_pk=1,
        edge_type="game_total",
        market="A at B Over 8.5",
        side="over",
        line=8.5,
        recommended_line=8.5,
        score=80,
        confidence="high",
        action="Bettable only at price",
        chase_risk="low",
        generated_for_date="2026-05-25",
    )
    linescore = {"teams": {"away": {"runs": 4}, "home": {"runs": 5}}}

    assert _grade_total(edge, linescore) == ("Final total 9 runs", "win")


def test_grade_pitcher_k_under_loss():
    edge = MlbEdge(
        game_pk=1,
        edge_type="pitcher_strikeouts",
        market="Tarik Skubal Under 6.5 Ks",
        side="under",
        line=6.5,
        recommended_line=6.5,
        score=80,
        confidence="high",
        action="Bettable only at price",
        chase_risk="low",
        generated_for_date="2026-05-25",
    )
    boxscore = {
        "teams": {
            "away": {
                "players": {
                    "ID1": {
                        "person": {"fullName": "Tarik Skubal"},
                        "stats": {"pitching": {"strikeOuts": 8}},
                    }
                }
            }
        }
    }

    assert _grade_pitcher_k(edge, boxscore) == ("Tarik Skubal 8 strikeouts", "loss")


def test_candidate_edges_are_ungraded_and_near_start(db_session):
    game = MlbGame(
        game_pk=1,
        game_date="2026-05-25",
        home_team="Home",
        away_team="Away",
        start_time=datetime.utcnow() + timedelta(minutes=10),
    )
    edge = MlbEdge(
        game_pk=1,
        edge_type="game_total",
        market="Away at Home Over 8.5",
        side="over",
        line=8.5,
        score=80,
        confidence="high",
        action="Bettable only at price",
        chase_risk="low",
        generated_for_date="2026-05-25",
    )
    db_session.add_all([game, edge])
    db_session.commit()

    rows = _candidate_edges(db_session, 30)

    assert len(rows) == 1


def test_performance_report_builds_text(db_session):
    edge = MlbEdge(
        game_pk=1,
        edge_type="game_total",
        market="Away at Home Over 8.5",
        side="over",
        line=8.5,
        best_price=2.0,
        score=86,
        confidence="high",
        action="Strong candidate",
        chase_risk="low",
        generated_for_date="2026-05-25",
        win_loss_push="win",
        roi_units=1.0,
        clv_percent=0.02,
        factors={"odds_edge": 90},
    )
    db_session.add(edge)
    db_session.commit()

    report = build_report(db_session)

    assert "MLB Performance Report" in report
    assert "Graded edges: 1" in report
