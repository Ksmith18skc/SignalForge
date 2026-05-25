from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Trader
from scripts.seed import SEED_TRADERS, seed_watchlist


def test_seed_watchlist_creates_and_updates_real_wallets(db_session: Session) -> None:
    created, updated = seed_watchlist(db_session)
    db_session.commit()

    assert created == len(SEED_TRADERS)
    assert updated == 0

    traders = list(db_session.scalars(select(Trader)))
    assert len(traders) == len(SEED_TRADERS)
    assert all(trader.wallet_address for trader in traders)

    surf = db_session.scalar(select(Trader).where(Trader.nickname == "surfandturf"))
    assert surf is not None
    surf.trust_score = 1
    db_session.commit()

    created, updated = seed_watchlist(db_session)
    db_session.commit()
    db_session.refresh(surf)

    assert created == 0
    assert updated == len(SEED_TRADERS)
    assert surf.trust_score == 82.66
