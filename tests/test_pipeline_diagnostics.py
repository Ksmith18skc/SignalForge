"""Tests for the end-to-end pipeline funnel diagnostics + the two pipeline
fixes it was built to expose (moneyline wallet-flow join, non-null
generated_for_date)."""

from __future__ import annotations

from datetime import datetime

from app.models import Market, Signal, Trade, Trader
from app.services import wallet_flow
from app.services.pipeline_diagnostics import pipeline_funnel
from app.services.signal_engine import SignalCandidate

CARD_DATE = "2026-05-27"


def _market(db, slug):
    m = Market(slug=slug, title=slug, platform="polymarket")
    db.add(m)
    db.flush()
    return m


def _trader(db, nickname, wallet):
    t = Trader(nickname=nickname, wallet_address=wallet, platform="polymarket")
    db.add(t)
    db.flush()
    return t


def _trade(db, trader, market, outcome, *, side="BUY", size=100.0):
    db.add(Trade(trader_id=trader.id, market_id=market.id, side=side,
                 outcome=outcome, price=0.5, size_usd=size, source="polymarket",
                 timestamp=datetime.utcnow()))


# ---------------------------------------------------------------------------
# Fix 1 — generated_for_date never persists NULL
# ---------------------------------------------------------------------------


def test_candidate_without_card_date_gets_today_not_null():
    cand = SignalCandidate(
        market_id=1, trader_id=1, signal_type="trusted_wallet_entry",
        side="BUY", outcome="Over", entry_price=0.5, size_usd=10.0,
        reason="r", generated_for_date=None,
    )
    sig = cand.to_model()
    # An explicit None used to override the column default and produce a
    # date-less signal invisible to every card-date query.
    assert sig.generated_for_date is not None
    assert len(sig.generated_for_date) == 10  # YYYY-MM-DD


# ---------------------------------------------------------------------------
# Fix 2 — moneyline edges find their bare-matchup wallet market
# ---------------------------------------------------------------------------


def test_moneyline_edge_matches_base_slug(db_session):
    market = _market(db_session, "mlb-nyy-kc-2026-05-27")  # NO -moneyline suffix
    t1 = _trader(db_session, "surf", "0xa")
    t2 = _trader(db_session, "bana", "0xb")
    _trade(db_session, t1, market, "New York Yankees")
    _trade(db_session, t2, market, "New York Yankees")
    db_session.flush()

    edge = {"edge_type": "game_moneyline", "side": "away", "line": None,
            "generated_for_date": CARD_DATE, "game_pk": 1}
    ctx = wallet_flow.build_wallet_context(
        db_session, edge=edge, home_team="Kansas City Royals",
        away_team="New York Yankees", card_date=CARD_DATE,
    )
    assert ctx["debug"]["candidate_markets_considered"] >= 1
    assert "mlb-nyy-kc-2026-05-27" in ctx["debug"]["matched_slugs"]
    assert ctx["tracked_wallet_count"] == 2


# ---------------------------------------------------------------------------
# Funnel — pinpoints the drop stage
# ---------------------------------------------------------------------------


def test_funnel_flags_card_date_mismatch(db_session):
    # A trade on a market dated a DIFFERENT day than the card date.
    market = _market(db_session, "mlb-nyy-kc-2026-05-20-total-9pt5")
    t = _trader(db_session, "surf", "0xa")
    _trade(db_session, t, market, "Over")
    db_session.flush()

    out = pipeline_funnel(db_session, card_date=CARD_DATE)
    assert out["ingestion"]["mlb_trades_total"] == 1
    assert out["ingestion"]["mlb_trades_for_card_date"] == 0
    assert "card_date_mismatch" in out["drop_stage"]


def test_funnel_reports_aligned_cards(db_session):
    market = _market(db_session, "mlb-nyy-kc-2026-05-27-total-9pt5")
    t1 = _trader(db_session, "surf", "0xa")
    t2 = _trader(db_session, "bana", "0xb")
    _trade(db_session, t1, market, "Over")
    _trade(db_session, t2, market, "Over")
    # Two distinct wallets, same market/side/outcome -> one aligned card.
    for trader in (t1, t2):
        db_session.add(Signal(
            market_id=market.id, trader_id=trader.id,
            signal_type="trusted_wallet_entry", side="BUY", outcome="Over",
            entry_price=0.5, size_usd=100.0, score=70.0, confidence=0.7,
            reason="r", source="polymarket", generated_for_date=CARD_DATE,
        ))
    db_session.flush()

    out = pipeline_funnel(db_session, card_date=CARD_DATE)
    assert out["alignment"]["aligned_cards"] == 1
    assert out["funnel"]["signals_generated"] == 2
