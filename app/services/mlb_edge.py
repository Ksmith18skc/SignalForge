"""MLB edge helpers that only read cached provider data.

Daily-card generation should call this module instead of pybaseball directly.
If Statcast cache rows are missing, confidence is downgraded with an explicit
warning so the web process never performs heavy live pulls.
"""

from __future__ import annotations

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.models import BatterStatcastSummary, PitcherStatcastSummary


def statcast_context(
    db: Session,
    *,
    player_id: int,
    player_type: str,
    season: int | None = None,
    last_n_days: int | None = None,
) -> dict[str, object]:
    model = PitcherStatcastSummary if player_type == "pitcher" else BatterStatcastSummary
    query = select(model).where(model.player_id == player_id)
    if season is not None:
        query = query.where(model.season == season)
    if last_n_days is not None:
        query = query.where(model.last_n_days == last_n_days)
    summary = db.scalar(query.order_by(desc(model.updated_at)).limit(1))
    if summary is None:
        return {
            "summary": None,
            "confidence_multiplier": 0.85,
            "warnings": [
                "Missing cached Statcast summary; confidence downgraded. Run python -m scripts.update_statcast_cache."
            ],
        }
    return {
        "summary": {
            "player_id": summary.player_id,
            "player_name": summary.player_name,
            "season": summary.season,
            "last_n_days": summary.last_n_days,
            "games": summary.games,
            "innings_pitched": summary.innings_pitched,
            "strikeouts": summary.strikeouts,
            "walks": summary.walks,
            "pitch_count_avg": summary.pitch_count_avg,
            "strikeouts_per_start": summary.strikeouts_per_start,
            "whiff_rate": summary.whiff_rate,
            "chase_rate": summary.chase_rate,
            "k_rate": summary.k_rate,
            "updated_at": summary.updated_at.isoformat() if summary.updated_at else None,
            "source": summary.source,
        },
        "confidence_multiplier": 1.0,
        "warnings": [],
    }
