from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

from app.models import MlbEdge, MlbGame
from app.services.mlb_edge_scoring import edge_to_dict
from app.services.mlb_performance import (
    arizona_today,
    arizona_window,
    arizona_yesterday,
    clv_report,
    grade_edge,
    performance_by_market,
    performance_by_score_band,
    performance_diagnostics,
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


def test_performance_summary_filters_by_date_window(db_session):
    today_edge = _edge(generated_for_date="2026-05-27", best_price=2.0, recommended_line=8.5)
    yesterday_edge = _edge(
        game_pk=2,
        generated_for_date="2026-05-26",
        best_price=2.0,
        recommended_line=8.5,
    )
    last_week_edge = _edge(
        game_pk=3,
        generated_for_date="2026-05-15",
        best_price=2.0,
        recommended_line=8.5,
    )
    grade_edge(today_edge, closing_line=9.0, closing_price=1.8, win_loss_push="win")
    grade_edge(yesterday_edge, closing_line=9.0, closing_price=1.8, win_loss_push="loss")
    grade_edge(last_week_edge, closing_line=9.0, closing_price=1.8, win_loss_push="win")
    db_session.add_all([today_edge, yesterday_edge, last_week_edge])
    db_session.commit()

    # Single-day filter: only yesterday is included.
    single = performance_summary(db_session, start_date="2026-05-26", end_date="2026-05-26")
    assert single["graded_edges"] == 1
    assert single["losses"] == 1

    # 7-day rolling window ending 2026-05-27 picks up only the two recent edges.
    rolling = performance_summary(db_session, start_date="2026-05-21", end_date="2026-05-27")
    assert rolling["graded_edges"] == 2

    # No filter returns everything.
    all_time = performance_summary(db_session)
    assert all_time["graded_edges"] == 3


def test_performance_diagnostics_reports_missing_snapshots(db_session):
    diagnostics = performance_diagnostics(
        db_session, start_date="2026-05-26", end_date="2026-05-26",
    )

    assert diagnostics["snapshot_count"] == 0
    assert diagnostics["graded_edge_count"] == 0
    assert diagnostics["reason"] and "No saved edge snapshots" in diagnostics["reason"]


def test_performance_diagnostics_reports_missing_finals_then_ungraded(db_session):
    edge = _edge(generated_for_date="2026-05-26", best_price=2.0, recommended_line=8.5)
    game = MlbGame(
        game_pk=edge.game_pk,
        game_date="2026-05-26",
        home_team="Home",
        away_team="Away",
        game_status="In Progress",
    )
    db_session.add_all([edge, game])
    db_session.commit()

    # Snapshot exists but no final scores yet.
    no_finals = performance_diagnostics(
        db_session, start_date="2026-05-26", end_date="2026-05-26",
    )
    assert no_finals["snapshot_count"] == 1
    assert no_finals["final_score_count"] == 0
    assert no_finals["reason"] and "no final game scores" in no_finals["reason"]

    # Mark game as final → reason flips to "no edges graded yet".
    game.game_status = "Final"
    db_session.commit()
    waiting = performance_diagnostics(
        db_session, start_date="2026-05-26", end_date="2026-05-26",
    )
    assert waiting["final_score_count"] == 1
    assert waiting["graded_edge_count"] == 0
    assert waiting["reason"] and "no edges have been graded" in waiting["reason"]

    # Grade the edge → reason clears.
    grade_edge(edge, closing_line=9.0, closing_price=1.8, win_loss_push="win")
    db_session.commit()
    after = performance_diagnostics(
        db_session, start_date="2026-05-26", end_date="2026-05-26",
    )
    assert after["graded_edge_count"] == 1
    assert after["reason"] is None


def test_arizona_window_and_yesterday_rollover():
    # Freeze "now" to just after UTC midnight on the 27th — in Arizona it is
    # still the 26th, so yesterday() should return the 25th, not the 26th.
    just_after_utc_midnight = datetime(2026, 5, 27, 1, 30)

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: D401 - test shim
            base = just_after_utc_midnight
            if tz is not None:
                from datetime import timezone
                return base.replace(tzinfo=timezone.utc).astimezone(tz)
            return base

    with patch("app.services.mlb_performance.datetime", _FrozenDatetime):
        assert arizona_today() == "2026-05-26"
        assert arizona_yesterday() == "2026-05-25"
        start, end = arizona_window(7)
        assert end == "2026-05-26"
        assert start == "2026-05-20"
