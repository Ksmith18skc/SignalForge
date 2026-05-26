from __future__ import annotations

from app.models import MlbEdge
from app.services.mlb_edge_scoring import edge_to_dict
from app.services.mlb_performance import (
    clv_report,
    grade_edge,
    performance_by_market,
    performance_by_score_band,
    performance_summary,
)


def _edge(**kwargs):
    defaults = {
        "game_pk": 1,
        "edge_type": "game_total",
        "market": "A at B Over 8.5",
        "side": "over",
        "line": 8.5,
        "best_book": "DraftKings",
        "best_price": 1.91,
        "score": 82,
        "confidence": "high",
        "action": "Bettable only at price",
        "chase_risk": "low",
        "generated_for_date": "2026-05-25",
        "factors": {"odds_edge": 80, "environment": 70},
    }
    defaults.update(kwargs)
    return MlbEdge(**defaults)


def test_grade_edge_calculates_clv_and_roi():
    edge = _edge(side="over", line=8.5, best_price=2.0, recommended_line=8.5)

    grade_edge(edge, closing_line=9.0, closing_price=1.8, win_loss_push="win", result="Final 6-4")

    assert edge.implied_probability_at_entry == 0.5
    assert edge.implied_probability_at_close == 0.5556
    assert edge.clv_points == 0.5
    assert edge.clv_percent == 0.1112
    assert edge.roi_units == 1.0
    assert edge.result == "Final 6-4"


def test_performance_reports_group_by_market_and_score_band(db_session):
    e1 = _edge(score=86, edge_type="game_total", best_price=2.0, recommended_line=8.5)
    e2 = _edge(
        game_pk=2,
        edge_type="pitcher_strikeouts",
        market="Pitcher Over 6.5 Ks",
        side="over",
        score=72,
        best_price=1.8,
        recommended_line=6.5,
    )
    e3 = _edge(
        game_pk=3,
        score=90,
        is_valid=False,
        validation_reason="malformed line",
        best_price=2.0,
        recommended_line=3.7,
    )
    grade_edge(e1, closing_line=9.0, closing_price=1.8, win_loss_push="win")
    grade_edge(e2, closing_line=6.0, closing_price=1.95, win_loss_push="loss")
    grade_edge(e3, closing_line=4.0, closing_price=1.8, win_loss_push="win")
    db_session.add_all([e1, e2, e3])
    db_session.commit()

    summary = performance_summary(db_session)
    by_market = performance_by_market(db_session)
    by_band = performance_by_score_band(db_session)
    clv = clv_report(db_session)

    assert summary["graded_edges"] == 2
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert any(row["edge_type"] == "game_total" for row in by_market)
    assert any(row["score_band"] == "85+" for row in by_band)
    assert clv["edges_with_clv"] == 2


def test_edge_serialization_includes_measurement_fields():
    edge = _edge()
    grade_edge(edge, closing_line=8.0, closing_price=2.05, win_loss_push="push")
    payload = edge_to_dict(edge)

    assert "closing_line" in payload
    assert "clv_points" in payload
    assert "roi_units" in payload
    assert payload["win_loss_push"] == "push"
