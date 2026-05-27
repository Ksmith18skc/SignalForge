"""Regression tests for MLB 'today' defaulting and daily-card staleness.

Calls route functions directly so we avoid the TestClient startup race
against ``_background_bootstrap`` in ``app.main``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.api.routes import mlb_daily_card, mlb_edges_today
from app.models import MlbDailyCard, MlbEdge


def _make_card(db_session, *, card_date: str) -> MlbDailyCard:
    card = MlbDailyCard(
        card_date=card_date,
        top_game_totals=[{"market": "A at B Over 8.5", "score": 90}],
        top_pitcher_strikeouts=[],
        near_misses=[{"market": "C at D Over 8.0", "score": 60}],
        pass_list=[],
        data_quality_summary={},
    )
    db_session.add(card)
    db_session.commit()
    return card


def _make_edge(db_session, *, card_date: str) -> MlbEdge:
    edge = MlbEdge(
        game_pk=1,
        edge_type="game_total",
        market="A at B Over 8.5",
        side="over",
        line=8.5,
        score=90,
        confidence="high",
        action="Strong candidate",
        chase_risk="low",
        generated_for_date=card_date,
        is_valid=True,
    )
    db_session.add(edge)
    db_session.commit()
    return edge


def _freeze_arizona(year: int, month: int, day: int, hour_utc: int):
    """Context manager: pin datetime.now() to a fixed UTC moment so
    arizona_today() returns a known string."""
    frozen = datetime(year, month, day, hour_utc, 0, tzinfo=timezone.utc)

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: D401
            if tz is None:
                return frozen.replace(tzinfo=None)
            return frozen.astimezone(tz)

    return patch("app.services.mlb_performance.datetime", _FrozenDatetime)


def test_edges_today_uses_arizona_date_not_utc(db_session):
    """At 01:30 UTC on 2026-05-27, Arizona is still on 2026-05-26. The
    endpoint must use Arizona today, so an edge stamped 2026-05-26 surfaces."""
    _make_edge(db_session, card_date="2026-05-26")
    with _freeze_arizona(2026, 5, 27, hour_utc=1):
        rows = mlb_edges_today(game_date=None, limit=100, db=db_session)
    assert len(rows) == 1
    assert rows[0]["market"] == "A at B Over 8.5"


def test_daily_card_marks_stale_when_today_missing(db_session):
    """If only yesterday's card exists, /mlb/daily-card returns it with
    is_stale=True so the dashboard banner can fire."""
    _make_card(db_session, card_date="2026-05-26")
    # 18:00 UTC = 11:00 Arizona → arizona_today = 2026-05-27.
    with _freeze_arizona(2026, 5, 27, hour_utc=18):
        body = mlb_daily_card(game_date=None, db=db_session)
    assert body["card_date"] == "2026-05-26"
    assert body["requested_date"] == "2026-05-27"
    assert body["is_stale"] is True
    assert "Run MLB edge scan" in body["stale_reason"]


def test_daily_card_marks_fresh_when_today_present(db_session):
    _make_card(db_session, card_date="2026-05-27")
    with _freeze_arizona(2026, 5, 27, hour_utc=18):
        body = mlb_daily_card(game_date=None, db=db_session)
    assert body["card_date"] == "2026-05-27"
    assert body["is_stale"] is False
    assert body["requested_date"] == "2026-05-27"


def test_daily_card_raises_404_when_db_empty(db_session):
    with _freeze_arizona(2026, 5, 27, hour_utc=18):
        with pytest.raises(HTTPException) as exc_info:
            mlb_daily_card(game_date=None, db=db_session)
    assert exc_info.value.status_code == 404


def test_daily_card_explicit_date_argument_is_respected(db_session):
    """Passing ``game_date=YYYY-MM-DD`` should look up that exact card and
    return it fresh — staleness logic only applies to the auto-today path."""
    _make_card(db_session, card_date="2026-05-25")
    body = mlb_daily_card(game_date="2026-05-25", db=db_session)
    assert body["card_date"] == "2026-05-25"
    assert body["is_stale"] is False
    assert body["requested_date"] == "2026-05-25"
