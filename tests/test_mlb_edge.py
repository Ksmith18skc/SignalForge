from __future__ import annotations

from app.services.mlb_edge import statcast_context


def test_mlb_edge_downgrades_confidence_when_cache_missing(db_session):
    context = statcast_context(db_session, player_id=123, player_type="pitcher", season=2026)

    assert context["summary"] is None
    assert context["confidence_multiplier"] == 0.85
    assert context["warnings"]
