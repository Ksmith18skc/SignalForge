from __future__ import annotations

from datetime import datetime, timedelta

from app.services import pnl_tracker
from app.services.position_matcher import match_trades_to_recommendations
from app.storage.pnl_store import (
    MyTrade,
    RecommendationSnapshot,
    SignalAttribution,
    insert_trades,
    upsert_wallet,
)


def test_pnl_math_and_clv_helpers() -> None:
    assert pnl_tracker.compute_position_value(100, 0.62) == 62.0
    assert pnl_tracker.compute_unrealized_pnl(100, 0.45, 0.62) == 17.0
    assert pnl_tracker.compute_edge(0.61, 0.55) == 0.05999999999999994
    points, pct = pnl_tracker.compute_clv(0.50, 0.58)
    assert round(points, 4) == 0.08
    assert round(pct, 2) == 16.0


def test_rebuild_positions_realizes_partial_sell(db_session) -> None:
    wallet = upsert_wallet(db_session, platform="mock", address="mock:test")
    insert_trades(
        db_session,
        [
            MyTrade(
                wallet_id=wallet.id,
                external_id="buy-1",
                platform="mock",
                market_slug="mlb-total",
                side="BUY",
                outcome="under",
                price=0.40,
                size_shares=100,
                size_usd=40,
                timestamp=datetime.utcnow() - timedelta(hours=2),
            ),
            MyTrade(
                wallet_id=wallet.id,
                external_id="sell-1",
                platform="mock",
                market_slug="mlb-total",
                side="SELL",
                outcome="under",
                price=0.55,
                size_shares=25,
                size_usd=13.75,
                timestamp=datetime.utcnow() - timedelta(hours=1),
            ),
        ],
    )
    db_session.flush()
    positions = pnl_tracker.rebuild_positions_for_wallet(db_session, wallet)
    pos = positions[0]
    assert pos.shares == 75
    assert pos.cost_basis_usd == 30
    assert pos.realized_pnl_usd == 3.75


def test_recommendation_match_labels_late_bad_entry(db_session) -> None:
    wallet = upsert_wallet(db_session, platform="mock", address="mock:test")
    snap = RecommendationSnapshot(
        source="mlb_edge",
        source_id=1,
        market_slug="mlb-total",
        market_title="MLB Total",
        side="buy",
        outcome="under",
        fair_probability=0.63,
        market_price=0.50,
        score=88,
        confidence_tier="high",
        threshold_status="actionable",
        captured_at=datetime.utcnow() - timedelta(minutes=30),
    )
    db_session.add(snap)
    db_session.flush()
    trade = MyTrade(
        wallet_id=wallet.id,
        external_id="late-fill",
        platform="mock",
        market_slug="mlb-total",
        market_title="MLB Total",
        side="BUY",
        outcome="under",
        price=0.56,
        size_shares=100,
        size_usd=56,
        timestamp=datetime.utcnow(),
    )
    insert_trades(db_session, [trade])
    db_session.flush()
    results = match_trades_to_recommendations(db_session)
    assert results[0].label == "missed_best_price"
    attr = db_session.query(SignalAttribution).one()
    assert round(attr.edge_at_entry, 2) == 0.07


def test_threshold_detection_entered_before_signal(db_session) -> None:
    wallet = upsert_wallet(db_session, platform="mock", address="mock:test")
    snap = RecommendationSnapshot(
        source="mlb_edge",
        source_id=2,
        market_slug="mlb-k",
        side="buy",
        outcome="over",
        fair_probability=0.60,
        market_price=0.48,
        score=90,
        threshold_status="actionable",
        captured_at=datetime.utcnow(),
    )
    db_session.add(snap)
    db_session.flush()
    insert_trades(
        db_session,
        [
            MyTrade(
                wallet_id=wallet.id,
                external_id="early-fill",
                platform="mock",
                market_slug="mlb-k",
                side="BUY",
                outcome="over",
                price=0.47,
                size_shares=10,
                size_usd=4.7,
                timestamp=datetime.utcnow() - timedelta(minutes=20),
            )
        ],
    )
    db_session.flush()
    result = match_trades_to_recommendations(db_session)[0]
    assert result.label == "entered_before_threshold"
    assert result.entered_before_threshold is True
