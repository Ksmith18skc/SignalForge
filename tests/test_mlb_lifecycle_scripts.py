from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any

from app.models import MlbEdge, MlbGame
from scripts.grade_mlb_results import _grade_pitcher_k, _grade_total
from scripts.grade_mlb_results import run_async as grade_run_async
from scripts.send_mlb_performance_report import build_report
from scripts.update_mlb_closing_lines import _candidate_edges
from scripts.update_mlb_closing_lines import run_async as closing_run_async


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


class _FakeMlb:
    def __init__(self, linescores: dict[int, dict[str, Any]]):
        self._linescores = linescores

    async def linescore(self, game_pk: int) -> dict[str, Any]:
        return self._linescores.get(game_pk, {})

    async def boxscore(self, game_pk: int) -> dict[str, Any]:  # pragma: no cover
        return {}


def test_grade_run_async_returns_zero_reason_when_no_snapshots(db_session):
    result = asyncio.run(grade_run_async(date="2026-05-26", db=db_session, mlb=_FakeMlb({})))

    assert result["ok"] is True
    assert result["candidates"] == 0
    assert result["graded"] == 0
    assert result["reason"] and "No ungraded edge snapshots" in result["reason"]


def test_grade_run_async_grades_final_game_total(db_session):
    game = MlbGame(
        game_pk=42,
        game_date="2026-05-26",
        home_team="Home",
        away_team="Away",
        game_status="Final",
        start_time=datetime.utcnow() - timedelta(hours=3),
    )
    edge = MlbEdge(
        game_pk=42,
        edge_type="game_total",
        market="Away at Home Over 8.5",
        side="over",
        line=8.5,
        recommended_line=8.5,
        best_price=2.0,
        score=80,
        confidence="high",
        action="Bettable only at price",
        chase_risk="low",
        generated_for_date="2026-05-26",
    )
    db_session.add_all([game, edge])
    db_session.commit()

    linescore = {"teams": {"home": {"runs": 6}, "away": {"runs": 4}}}
    result = asyncio.run(
        grade_run_async(date="2026-05-26", db=db_session, mlb=_FakeMlb({42: linescore})),
    )

    assert result["graded"] == 1
    assert result["finals_found"] == 1
    db_session.refresh(edge)
    assert edge.win_loss_push == "win"


def test_grade_run_async_skips_when_no_finals(db_session):
    game = MlbGame(
        game_pk=43,
        game_date="2026-05-26",
        home_team="Home",
        away_team="Away",
        game_status="In Progress",
        start_time=datetime.utcnow() - timedelta(hours=1),
    )
    edge = MlbEdge(
        game_pk=43,
        edge_type="game_total",
        market="Away at Home Over 8.5",
        side="over",
        line=8.5,
        recommended_line=8.5,
        score=80,
        confidence="high",
        action="Bettable only at price",
        chase_risk="low",
        generated_for_date="2026-05-26",
    )
    db_session.add_all([game, edge])
    db_session.commit()

    result = asyncio.run(
        grade_run_async(date="2026-05-26", db=db_session, mlb=_FakeMlb({43: {}})),
    )

    assert result["graded"] == 0
    assert result["skipped_not_final"] == 1
    assert result["reason"] and "no games are final" in result["reason"]


def test_closing_lines_date_filter_skips_other_dates(db_session):
    target = MlbGame(
        game_pk=10,
        game_date="2026-05-26",
        home_team="Home",
        away_team="Away",
        start_time=datetime.utcnow() + timedelta(minutes=5),
    )
    other = MlbGame(
        game_pk=11,
        game_date="2026-05-25",
        home_team="X",
        away_team="Y",
        start_time=datetime.utcnow() + timedelta(minutes=5),
    )
    db_session.add_all([
        target,
        other,
        MlbEdge(
            game_pk=10, edge_type="game_total", market="Away at Home Over 8.5",
            side="over", line=8.5, score=80, confidence="high",
            action="Bettable only at price", chase_risk="low",
            generated_for_date="2026-05-26",
        ),
        MlbEdge(
            game_pk=11, edge_type="game_total", market="Y at X Over 7.5",
            side="over", line=7.5, score=80, confidence="high",
            action="Bettable only at price", chase_risk="low",
            generated_for_date="2026-05-25",
        ),
    ])
    db_session.commit()

    rows = _candidate_edges(db_session, 30, date="2026-05-26")

    assert {row[1].game_pk for row in rows} == {10}


def test_closing_run_async_returns_no_candidate_reason(db_session):
    result = asyncio.run(
        closing_run_async(
            window_minutes=30, date="2026-05-26", db=db_session, odds=SimpleNamespace(),
        ),
    )

    assert result["ok"] is True
    assert result["candidates"] == 0
    assert result["closing_lines_updated"] == 0
    assert result["reason"] and "No ungraded edge snapshots" in result["reason"]


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
