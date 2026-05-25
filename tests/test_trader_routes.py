from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.api.routes import create_trader, delete_trader
from app.models import Alert, Market, Signal, Trade
from app.schemas import TraderCreate


def test_create_trader_with_wallet(db_session: Session) -> None:
    trader = create_trader(
        TraderCreate(
            nickname="sharp-wallet",
            wallet_address="0x1111111111111111111111111111111111111111",
            trust_score=65,
            tags=["sports"],
            notes="manual add",
        ),
        db_session,
    )

    assert trader.nickname == "sharp-wallet"
    assert trader.wallet_address == "0x1111111111111111111111111111111111111111"
    assert trader.copy_mode == "alert_only"


def test_create_trader_rejects_duplicate_wallet(db_session: Session) -> None:
    create_trader(
        TraderCreate(
            nickname="first",
            wallet_address="0x2222222222222222222222222222222222222222",
        ),
        db_session,
    )

    with pytest.raises(HTTPException) as exc:
        create_trader(
            TraderCreate(
                nickname="second",
                wallet_address="0x2222222222222222222222222222222222222222",
            ),
            db_session,
        )

    assert exc.value.status_code == 409
    assert "wallet" in exc.value.detail


def test_delete_trader_removes_related_data(db_session: Session) -> None:
    trader = create_trader(
        TraderCreate(
            nickname="delete-me",
            wallet_address="0x3333333333333333333333333333333333333333",
        ),
        db_session,
    )
    market = Market(slug="delete-test", title="Delete test")
    db_session.add(market)
    db_session.flush()
    trade = Trade(
        trader_id=trader.id,
        market_id=market.id,
        side="BUY",
        price=0.5,
        size_usd=100,
        source="Falcon",
    )
    signal = Signal(
        trader_id=trader.id,
        market_id=market.id,
        signal_type="trusted_wallet_entry",
        side="BUY",
        entry_price=0.5,
        size_usd=100,
        source="Falcon",
    )
    db_session.add_all([trade, signal])
    db_session.flush()
    db_session.add(Alert(signal_id=signal.id, channel="console", message="test"))
    db_session.commit()

    delete_trader(trader.id, db_session)

    assert db_session.get(type(trader), trader.id) is None
    assert db_session.query(Trade).filter_by(trader_id=trader.id).count() == 0
    assert db_session.query(Signal).filter_by(trader_id=trader.id).count() == 0
    assert db_session.query(Alert).count() == 0
