from __future__ import annotations

from scripts.update_statcast_cache import summarize_statcast


def test_summarize_statcast_builds_memory_safe_pitcher_summary():
    rows = [
        {
            "game_pk": 1,
            "player_name": "Sample Pitcher",
            "events": "strikeout",
            "description": "swinging_strike",
            "zone": 14,
        },
        {
            "game_pk": 1,
            "events": "walk",
            "description": "ball",
            "zone": 11,
        },
        {
            "game_pk": 2,
            "events": "field_out",
            "description": "hit_into_play",
            "zone": 5,
        },
    ]

    summary = summarize_statcast(rows, player_id=123, season=2026, last_n_days=14)

    assert summary["player_id"] == 123
    assert summary["player_name"] == "Sample Pitcher"
    assert summary["games"] == 2
    assert summary["strikeouts"] == 1
    assert summary["walks"] == 1
    assert summary["strikeouts_per_start"] == 0.5
    assert summary["source"] == "pybaseball_worker"
