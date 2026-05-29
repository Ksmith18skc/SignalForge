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
    factor_attribution,
    grade_edge,
    lookup_edge_score_band,
    performance_by_market,
    performance_by_projection_bucket,
    performance_by_score_axis,
    performance_by_score_band,
    performance_by_side,
    performance_by_timing,
    performance_diagnostics,
    performance_summary,
    projection_bucket,
    projection_calibration,
    research_health,
    sample_size_label,
    score_band,
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
        "prediction_score": 78,
        "execution_score": 60,
        "confidence": "high",
        "action": "Bettable only at price",
        "chase_risk": "low",
        "generated_for_date": "2026-05-25",
        "factors": {"odds_edge": 80, "environment": 70},
    }
    defaults.update(kwargs)
    defaults["legacy_score"] = kwargs.get("legacy_score", defaults["score"])
    return MlbEdge(**defaults)


def test_grade_edge_calculates_clv_and_roi():
    edge = _edge(side="over", line=8.5, best_price=2.0, recommended_line=8.5)

    grade_edge(edge, closing_line=9.0, closing_price=1.8, win_loss_push="win", result="Final 6-4")

    assert edge.implied_probability_at_entry == 0.5
    assert edge.implied_probability_at_close == 0.5556
    assert edge.clv_points == 0.5
    # clv_percent is now line-based: clv_points / entry_total = 0.5 / 8.5.
    assert edge.clv_percent == round(0.5 / 8.5, 4)
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
    assert summary["average_prediction_score"] is not None
    assert summary["average_execution_score"] is not None


def test_performance_reports_prediction_and_execution_score_axes(db_session):
    e1 = _edge(
        game_pk=10,
        score=90,
        legacy_score=90,
        prediction_score=86,
        execution_score=62,
        best_price=2.0,
        recommended_line=8.5,
    )
    e2 = _edge(
        game_pk=11,
        score=70,
        legacy_score=70,
        prediction_score=58,
        execution_score=88,
        best_price=2.0,
        recommended_line=8.5,
    )
    grade_edge(e1, closing_line=9.0, closing_price=1.8, win_loss_push="win")
    grade_edge(e2, closing_line=8.0, closing_price=1.8, win_loss_push="loss")
    db_session.add_all([e1, e2])
    db_session.commit()

    by_prediction = performance_by_score_axis(db_session, axis="prediction")
    by_execution = performance_by_score_axis(db_session, axis="execution")

    pred_85 = next(r for r in by_prediction if r["score_band"] == "85+")
    exec_85 = next(r for r in by_execution if r["score_band"] == "85+")
    assert pred_85["graded_edges"] == 1
    assert pred_85["wins"] == 1
    assert exec_85["graded_edges"] == 1
    assert exec_85["losses"] == 1


def test_lookup_edge_score_band_aggregates_similar_graded_edges(db_session):
    # Three graded game_total edges in the 85+ band: 2 wins, 1 loss.
    edges = [
        _edge(game_pk=1, score=86, best_price=2.0, recommended_line=8.5),
        _edge(game_pk=2, score=88, best_price=2.0, recommended_line=8.5),
        _edge(game_pk=3, score=90, best_price=2.0, recommended_line=8.5),
    ]
    grade_edge(edges[0], closing_line=9.0, closing_price=1.8, win_loss_push="win")
    grade_edge(edges[1], closing_line=9.0, closing_price=1.8, win_loss_push="win")
    grade_edge(edges[2], closing_line=4.0, closing_price=1.8, win_loss_push="loss")
    db_session.add_all(edges)
    db_session.commit()

    band = lookup_edge_score_band(db_session, edge_type="game_total", score=87)
    assert band["score_band"] == "85+"
    assert band["sample_size"] == 3
    assert band["wins"] == 2 and band["losses"] == 1
    assert band["win_rate"] == round(2 / 3, 4)
    # Laplace-smoothed: (2+1)/(3+2) = 0.6.
    assert band["calibrated_probability"] == 0.6


def test_lookup_edge_score_band_respects_edge_type_and_empty(db_session):
    e = _edge(game_pk=1, score=86, best_price=2.0, recommended_line=8.5)
    grade_edge(e, closing_line=9.0, closing_price=1.8, win_loss_push="win")
    db_session.add(e)
    db_session.commit()

    # Different edge type → no matching history.
    other = lookup_edge_score_band(db_session, edge_type="pitcher_strikeouts", score=86)
    assert other["sample_size"] == 0
    assert other["calibrated_probability"] is None
    # Empty band (no graded edges < 65).
    empty = lookup_edge_score_band(db_session, edge_type="game_total", score=50)
    assert empty["sample_size"] == 0


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


# ---------------------------------------------------------------------------
# Research-upgrade analytics
# ---------------------------------------------------------------------------


def test_score_band_five_tier_segmentation():
    assert score_band(40) == "<55"
    assert score_band(60) == "55-64"
    assert score_band(70) == "65-74"
    assert score_band(80) == "75-84"
    assert score_band(90) == "85+"
    # Bad input lands in the weakest band rather than raising.
    assert score_band(None) == "<55"


def test_projection_bucket_boundaries():
    assert projection_bucket(7.0) == "<7.5"
    assert projection_bucket(8.0) == "7.5-8.5"
    assert projection_bucket(9.0) == "8.5-9.5"
    assert projection_bucket(10.0) == "9.5-10.5"
    assert projection_bucket(11.0) == "10.5+"
    assert projection_bucket(None) is None


def test_sample_size_label_tiers():
    assert sample_size_label(5)["tier"] == "exploratory"
    assert sample_size_label(100)["tier"] == "early"
    assert sample_size_label(300)["tier"] == "moderate"
    assert sample_size_label(1000)["tier"] == "strong"


def test_performance_by_side_emits_directional_bias_warning(db_session):
    # 8 overs vs 2 unders → over_share = 0.8 → over-bias warning fires.
    edges = []
    for i in range(8):
        edges.append(
            _edge(game_pk=100 + i, side="over", best_price=2.0, recommended_line=8.5, score=70)
        )
    for i in range(2):
        edges.append(
            _edge(game_pk=200 + i, side="under", best_price=2.0, recommended_line=8.5, score=70)
        )
    for edge in edges:
        grade_edge(edge, closing_line=9.0, closing_price=1.8, win_loss_push="win")
    db_session.add_all(edges)
    db_session.commit()

    side_report = performance_by_side(db_session)
    assert side_report["over"]["count"] == 8
    assert side_report["under"]["count"] == 2
    assert side_report["directional_bias_warning"] == "Model may be directionally biased toward over."


def test_projection_calibration_inflated_close_warning(db_session):
    edges = []
    for i in range(3):
        e = _edge(
            game_pk=300 + i,
            side="over",
            best_price=2.0,
            recommended_line=9.5,
            score=80,
            model_projected_total=11.0,
        )
        # Closing line stays at 9.0 → model proj. (11.0) - close (9.0) = 2.0 > 0.75.
        grade_edge(e, closing_line=9.0, closing_price=1.9, win_loss_push="win")
        e.actual_total = 8.0  # projection error = +3.0 → over-projecting warning.
        edges.append(e)
    db_session.add_all(edges)
    db_session.commit()

    cal = projection_calibration(db_session)
    assert cal["graded_game_totals"] == 3
    assert cal["rows_with_projection"] == 3
    assert any("inflated" in w for w in cal["warnings"])
    assert any("over-projecting" in w for w in cal["warnings"])


def test_performance_by_score_band_returns_all_five_bands_with_stable_flag(db_session):
    e = _edge(game_pk=400, score=90, best_price=2.0, recommended_line=8.5)
    grade_edge(e, closing_line=9.0, closing_price=1.8, win_loss_push="win")
    db_session.add(e)
    db_session.commit()

    rows = performance_by_score_band(db_session)
    bands = [row["score_band"] for row in rows]
    assert bands == ["<55", "55-64", "65-74", "75-84", "85+"]
    # Single graded edge → unstable flag for that band.
    band_row = next(r for r in rows if r["score_band"] == "85+")
    assert band_row["graded_edges"] == 1
    assert band_row["stable"] is False
    assert band_row["over_count"] == 1


def test_research_health_clv_first_ordering(db_session):
    e = _edge(game_pk=500, score=80, best_price=2.0, recommended_line=8.5)
    grade_edge(e, closing_line=9.0, closing_price=1.8, win_loss_push="win")
    db_session.add(e)
    db_session.commit()

    health = research_health(db_session)
    # Headline payload must include CLV-first metrics, ROI/win-rate, and the
    # sample-size tier so the panel can render the warning bar.
    assert "positive_clv_rate" in health
    assert "average_clv_points" in health
    assert "roi_units" in health
    assert "win_rate" in health
    assert health["graded_sample_size"] == 1
    assert health["sample_size_tier"] == "exploratory"


def test_factor_attribution_flags_unstable_factors(db_session):
    e = _edge(
        game_pk=600,
        score=80,
        best_price=2.0,
        recommended_line=8.5,
        factors={"environment": 80, "odds_edge": 90},
    )
    grade_edge(e, closing_line=9.0, closing_price=1.8, win_loss_push="win")
    db_session.add(e)
    db_session.commit()

    rows = factor_attribution(db_session)
    assert all(row["unstable"] for row in rows)  # sample size < 50


def test_performance_by_projection_bucket_groups_by_model_total(db_session):
    e_low = _edge(
        game_pk=701, score=80, best_price=2.0, recommended_line=7.0,
        model_projected_total=7.0,
    )
    e_high = _edge(
        game_pk=702, score=80, best_price=2.0, recommended_line=11.0,
        model_projected_total=11.0,
    )
    grade_edge(e_low, closing_line=7.5, closing_price=1.8, win_loss_push="win")
    grade_edge(e_high, closing_line=10.5, closing_price=1.8, win_loss_push="loss")
    db_session.add_all([e_low, e_high])
    db_session.commit()

    rows = performance_by_projection_bucket(db_session)
    by_bucket = {r["projection_bucket"]: r for r in rows}
    assert by_bucket["<7.5"]["graded_edges"] == 1
    assert by_bucket["10.5+"]["graded_edges"] == 1


def test_performance_by_timing_buckets_by_hours_before_game(db_session):
    from datetime import datetime as _dt

    edge = _edge(game_pk=800, score=80, best_price=2.0, recommended_line=8.5)
    grade_edge(edge, closing_line=9.0, closing_price=1.8, win_loss_push="win")
    edge.created_at = _dt(2026, 5, 26, 18, 0)
    game = MlbGame(
        game_pk=800, game_date="2026-05-26", home_team="H", away_team="A",
        game_status="Final",
        start_time=_dt(2026, 5, 27, 12, 0),  # 18h after creation → ">12h" bucket
    )
    db_session.add_all([edge, game])
    db_session.commit()

    rows = performance_by_timing(db_session)
    by_bucket = {r["timing_bucket"]: r for r in rows}
    assert by_bucket[">12h"]["graded_edges"] == 1
