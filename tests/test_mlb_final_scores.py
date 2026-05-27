"""Tests for the persisted MLB final-score table and the grader fallback."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

import pytest

from app.models import MlbEdge, MlbFinalScore, MlbGame
from app.services.mlb_final_scores import (
    get_final_score,
    ingest_final_scores_for_date,
    persisted_final_score_count,
    upsert_final_score,
)
from app.services.mlb_performance import performance_diagnostics
from scripts.grade_mlb_results import run_async as grade_run_async


# --- helpers ---------------------------------------------------------------


def _schedule_payload(games: list[dict[str, Any]]) -> dict[str, Any]:
    return {"dates": [{"date": "2026-05-26", "games": games}]}


def _final_game(
    *, game_pk: int, home: str, away: str, home_score: int, away_score: int,
    status_detail: str = "Final", abstract: str = "Final",
) -> dict[str, Any]:
    return {
        "gamePk": game_pk,
        "status": {
            "abstractGameState": abstract,
            "detailedState": status_detail,
            "codedGameState": "F" if abstract == "Final" else "I",
        },
        "teams": {
            "home": {"team": {"name": home}, "score": home_score},
            "away": {"team": {"name": away}, "score": away_score},
        },
    }


def _edge(**kwargs):
    defaults = {
        "game_pk": 1,
        "edge_type": "game_total",
        "market": "Away at Home Over 8.5",
        "side": "over",
        "line": 8.5,
        "recommended_line": 8.5,
        "best_price": 2.0,
        "score": 80,
        "confidence": "high",
        "action": "Bettable only at price",
        "chase_risk": "low",
        "generated_for_date": "2026-05-26",
    }
    defaults.update(kwargs)
    return MlbEdge(**defaults)


def _game(**kwargs):
    defaults = {
        "game_pk": 1,
        "game_date": "2026-05-26",
        "home_team": "Home",
        "away_team": "Away",
        "start_time": datetime.utcnow() - timedelta(hours=3),
        "game_status": "Final",
    }
    defaults.update(kwargs)
    return MlbGame(**defaults)


class _StaticMlb:
    """Test double for MlbStatsApiProvider."""

    def __init__(
        self,
        *,
        schedule: dict[str, Any] | None = None,
        linescores: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        self._schedule = schedule or {"dates": []}
        self._linescores = linescores or {}
        self.calls: dict[str, int] = {"schedule": 0, "linescore": 0, "boxscore": 0}

    async def schedule(self, *, game_date: str, hydrate: str | None = None) -> dict[str, Any]:
        self.calls["schedule"] += 1
        return self._schedule

    async def linescore(self, game_pk: int) -> dict[str, Any]:
        self.calls["linescore"] += 1
        return self._linescores.get(game_pk, {})

    async def boxscore(self, game_pk: int) -> dict[str, Any]:  # pragma: no cover
        self.calls["boxscore"] += 1
        return {}


class _ExplodingMlb:
    """Test double that fails on every live network call. Used to prove the
    redeploy-safe path: with a persisted final score, grading must succeed
    without touching the live API."""

    async def schedule(self, *, game_date: str, hydrate: str | None = None) -> dict[str, Any]:
        raise RuntimeError("StatsAPI unreachable")

    async def linescore(self, game_pk: int) -> dict[str, Any]:
        raise RuntimeError("StatsAPI unreachable")

    async def boxscore(self, game_pk: int) -> dict[str, Any]:
        raise RuntimeError("StatsAPI unreachable")


# --- upsert ---------------------------------------------------------------


def test_upsert_final_score_inserts_then_updates(db_session):
    row = upsert_final_score(
        db_session,
        game_pk=42,
        generated_for_date="2026-05-26",
        home_team="Home",
        away_team="Away",
        home_score=4,
        away_score=3,
    )
    db_session.commit()

    assert row.total_runs == 7
    assert get_final_score(db_session, 42) is row
    assert persisted_final_score_count(db_session) == 1

    # Re-upsert with a different score → updates in place, no duplicate row.
    upsert_final_score(
        db_session,
        game_pk=42,
        generated_for_date="2026-05-26",
        home_team="Home",
        away_team="Away",
        home_score=6,
        away_score=2,
    )
    db_session.commit()

    refreshed = get_final_score(db_session, 42)
    assert refreshed.home_score == 6
    assert refreshed.total_runs == 8
    assert persisted_final_score_count(db_session) == 1  # still one row


def test_ingest_final_scores_skips_unfinished_games(db_session):
    schedule = _schedule_payload([
        _final_game(game_pk=1, home="H1", away="A1", home_score=5, away_score=4),
        # In-progress game must NOT be persisted.
        {
            "gamePk": 2,
            "status": {"abstractGameState": "Live", "detailedState": "In Progress"},
            "teams": {
                "home": {"team": {"name": "H2"}, "score": 1},
                "away": {"team": {"name": "A2"}, "score": 0},
            },
        },
    ])
    mlb = _StaticMlb(schedule=schedule)

    result = asyncio.run(ingest_final_scores_for_date(db_session, mlb, date="2026-05-26"))

    assert result["games_seen"] == 2
    assert result["finals_found"] == 1
    assert result["upserted"] == 1
    assert persisted_final_score_count(db_session) == 1
    assert get_final_score(db_session, 2) is None


# --- grading from persisted -----------------------------------------------


def test_grader_uses_persisted_score_and_skips_live_call(db_session):
    db_session.add_all([
        _game(game_pk=10),
        _edge(game_pk=10, side="over", recommended_line=8.5),
    ])
    upsert_final_score(
        db_session,
        game_pk=10,
        generated_for_date="2026-05-26",
        home_team="Home",
        away_team="Away",
        home_score=6,
        away_score=4,
    )
    db_session.commit()

    mlb = _StaticMlb(schedule={"dates": []})

    result = asyncio.run(
        grade_run_async(date="2026-05-26", db=db_session, mlb=mlb),
    )

    assert result["graded"] == 1
    assert result["graded_from_persisted"] == 1
    assert result["graded_from_live"] == 0
    # The persisted-first path must not hit the live linescore endpoint.
    assert mlb.calls["linescore"] == 0

    db_session.expire_all()
    edge = db_session.get(MlbEdge, 1)
    assert edge.win_loss_push == "win"


def test_grader_falls_back_to_live_when_no_persisted_row(db_session):
    db_session.add_all([
        _game(game_pk=20),
        _edge(game_pk=20, side="over", recommended_line=8.5),
    ])
    db_session.commit()

    # Empty schedule (so ingestion finds nothing to persist) but the live
    # linescore endpoint still returns a final result.
    mlb = _StaticMlb(
        schedule={"dates": []},
        linescores={20: {"teams": {"home": {"runs": 5}, "away": {"runs": 5}}}},
    )

    result = asyncio.run(
        grade_run_async(date="2026-05-26", db=db_session, mlb=mlb),
    )

    assert result["graded"] == 1
    assert result["graded_from_persisted"] == 0
    assert result["graded_from_live"] == 1
    assert mlb.calls["linescore"] == 1


def test_grader_runs_offline_when_persisted_row_exists(db_session):
    """Redeploy-safe path: every live API call raises, but with a persisted
    row the grader must still grade the edge."""

    db_session.add_all([
        _game(game_pk=30),
        _edge(game_pk=30, side="over", recommended_line=7.5),
    ])
    upsert_final_score(
        db_session,
        game_pk=30,
        generated_for_date="2026-05-26",
        home_team="Home",
        away_team="Away",
        home_score=3,
        away_score=2,
    )
    db_session.commit()

    mlb = _ExplodingMlb()

    # Skip the pre-grade ingestion (which would fail and is non-fatal) so the
    # test isolates the grading-from-persisted path itself.
    result = asyncio.run(
        grade_run_async(date="2026-05-26", db=db_session, mlb=mlb, skip_ingestion=True),
    )

    assert result["graded"] == 1
    assert result["graded_from_persisted"] == 1
    assert result["graded_from_live"] == 0


def test_grade_run_async_records_ingestion_summary(db_session):
    db_session.add_all([
        _game(game_pk=50, game_status="Final"),
        _edge(game_pk=50, side="over", recommended_line=8.5),
    ])
    db_session.commit()

    schedule = _schedule_payload([
        _final_game(game_pk=50, home="Home", away="Away", home_score=5, away_score=4),
    ])
    mlb = _StaticMlb(
        schedule=schedule,
        # If grading falls back to live, this would return a different total
        # than the persisted 9 — but persisted wins, so this should be unused.
        linescores={50: {"teams": {"home": {"runs": 99}, "away": {"runs": 99}}}},
    )

    result = asyncio.run(grade_run_async(date="2026-05-26", db=db_session, mlb=mlb))

    assert result["ingestion"]["upserted"] == 1
    assert result["graded"] == 1
    assert result["graded_from_persisted"] == 1
    db_session.expire_all()
    edge = db_session.get(MlbEdge, 1)
    assert edge.win_loss_push == "win"  # total=9 > line=8.5
    # The live linescore endpoint must not have been touched.
    assert mlb.calls["linescore"] == 0


# --- diagnostics ----------------------------------------------------------


def test_performance_diagnostics_separates_persisted_and_live_counts(db_session):
    db_session.add_all([
        _game(game_pk=60, game_status="Final"),
        _game(game_pk=61, game_status="Final"),
        _edge(game_pk=60),
    ])
    upsert_final_score(
        db_session,
        game_pk=60,
        generated_for_date="2026-05-26",
        home_team="Home",
        away_team="Away",
        home_score=4,
        away_score=3,
    )
    db_session.commit()

    diagnostics = performance_diagnostics(
        db_session, start_date="2026-05-26", end_date="2026-05-26",
    )

    assert diagnostics["persisted_final_score_count"] == 1
    assert diagnostics["live_final_count"] == 2
    assert diagnostics["final_score_count"] == 2  # max of the two sources
