"""Tracked-wallet live-positions invariants.

These pin the contract introduced to fix the "No current-card wallet
flow found" bug — namely:

  1. A tracked wallet with a current active position appears in
     ``live_positions`` regardless of whether a Signal row was emitted.
  2. ``mlb-det-cws-2026-05-29-total-9pt5`` parses as 2026-05-29.
  3. A same-day wallet position is NOT rejected as date mismatch.
  4. Wallet-only signal appears when no sportsbook edge exists.
  5. Wallet-confirmed badge fires when wallet position matches an edge.
  6. Non-MLB current wallet positions appear in wallet flow but NOT
     in the MLB edge-card join.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from app.models import Market, Trade, Trader
from app.services.tracked_wallet_positions import (
    classify_wallet_against_edges,
    edges_indexed_by_key,
    live_position_debug,
    live_positions,
)
from app.services.wallet_normalize import normalize_market_key


CARD_DATE = "2026-05-29"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trader(db, nickname: str = "VeryLucky888", address: str = "0xabc"):
    trader = Trader(
        nickname=nickname, wallet_address=address, platform="polymarket",
        trust_score=70.0, tags=[],
    )
    db.add(trader)
    db.flush()
    return trader


def _market(db, slug: str, title: str, *, platform: str = "polymarket"):
    market = Market(slug=slug, platform=platform, title=title)
    db.add(market)
    db.flush()
    return market


def _trade(db, trader, market, *, hours_ago: float = 2.0, size: float = 250.0):
    ts = datetime.utcnow() - timedelta(hours=hours_ago)
    trade = Trade(
        trader_id=trader.id, market_id=market.id,
        side="BUY", outcome="Over", price=0.55, size_usd=size, source="falcon",
        timestamp=ts, external_id=f"{trader.id}-{market.id}-{ts.isoformat()}",
    )
    db.add(trade)
    db.flush()
    return trade


# ---------------------------------------------------------------------------
# 2. Slug parsing
# ---------------------------------------------------------------------------

def test_mlb_total_slug_parses_event_date():
    key = normalize_market_key("mlb-det-cws-2026-05-29-total-9pt5")
    assert key is not None
    assert key.event_date == "2026-05-29"
    assert key.sport == "mlb"
    assert key.market_subtype == "game_total"
    assert key.line == 9.5
    # Canonical form is the human-debuggable join key the matcher uses.
    assert key.canonical == "mlb:det-cws:2026-05-29:game_total:9.5"


def test_atp_slug_parses_event_date_without_subtype():
    key = normalize_market_key("atp-khachan-jong-2026-05-29")
    assert key is not None
    assert key.sport == "atp"
    assert key.event_date == "2026-05-29"
    # Subtype not recognized for ATP — but the row still has a key so
    # the dashboard never silently drops it.
    assert key.market_subtype is None
    assert key.canonical.startswith("atp:")
    assert "2026-05-29" in key.canonical


# ---------------------------------------------------------------------------
# 1 + 3. Active tracked position appears today and is not rejected for
#         a date mismatch.
# ---------------------------------------------------------------------------

def test_active_tracked_position_appears_in_live_positions(db_session):
    trader = _trader(db_session)
    market = _market(
        db_session,
        slug="mlb-det-cws-2026-05-29-total-9pt5",
        title="DET @ CWS — Over 9.5",
    )
    _trade(db_session, trader, market)
    db_session.commit()

    rows = live_positions(db_session, card_date=CARD_DATE)
    assert any(
        r["wallet_nickname"] == "VeryLucky888"
        and r["market_slug"] == "mlb-det-cws-2026-05-29-total-9pt5"
        for r in rows
    )


def test_same_day_position_is_not_rejected_for_date_mismatch(db_session):
    """The rejection-debug report must classify a today's-date trade as
    accepted, NOT bucketed into ``market_date_mismatch`` / similar."""
    trader = _trader(db_session)
    market = _market(
        db_session,
        slug="mlb-det-cws-2026-05-29-total-9pt5",
        title="DET @ CWS — Over 9.5",
    )
    _trade(db_session, trader, market)
    db_session.commit()

    debug = live_position_debug(db_session, card_date=CARD_DATE)
    assert debug["accepted_for_card_date"] == 1
    # Critically: no rejection reasons recorded for this row.
    assert debug["rejection_reasons"] == {}
    assert debug["rejected"] == 0


# ---------------------------------------------------------------------------
# 4 + 5. Wallet-only vs wallet-confirmed classification.
# ---------------------------------------------------------------------------

def test_wallet_only_signal_when_no_matching_edge(db_session):
    trader = _trader(db_session)
    market = _market(
        db_session,
        slug="mlb-det-cws-2026-05-29-total-9pt5",
        title="DET @ CWS — Over 9.5",
    )
    _trade(db_session, trader, market)
    db_session.commit()

    rows = live_positions(db_session, card_date=CARD_DATE)
    assert rows
    # Empty edge list → every wallet position lands in wallet_only.
    classified = [classify_wallet_against_edges(r, edges_indexed_by_key([])) for r in rows]
    assert all(c["edge_match_kind"] == "wallet_only" for c in classified)
    assert all(c["matched_edge"] is None for c in classified)


def test_wallet_confirmed_badge_when_position_matches_edge(db_session):
    trader = _trader(db_session)
    market = _market(
        db_session,
        slug="mlb-det-cws-2026-05-29-total-9pt5",
        title="DET @ CWS — Over 9.5",
    )
    _trade(db_session, trader, market)
    db_session.commit()

    # Build an edge whose home/away + game_date match the wallet's slug.
    edges = [{
        "id": 99, "edge_type": "game_total", "side": "over",
        "line": 9.5, "generated_for_date": CARD_DATE,
        "home_team": "CWS", "away_team": "DET",
        "market": "DET @ CWS — Over 9.5",
        "prediction_score": 78.0, "execution_score": 60.0,
    }]
    index = edges_indexed_by_key(edges)
    [row] = live_positions(db_session, card_date=CARD_DATE)
    classified = classify_wallet_against_edges(row, index)
    assert classified["edge_match_kind"] in {"exact_line", "matchup_date"}
    assert classified["matched_edge"] is not None
    assert classified["matched_edge"]["id"] == 99


# ---------------------------------------------------------------------------
# 6. Non-MLB positions appear in wallet flow but never inside the MLB
#     edge-card join.
# ---------------------------------------------------------------------------

def test_non_mlb_position_appears_in_flow_but_not_as_mlb_edge_card(db_session):
    trader = _trader(db_session, nickname="HomeRunHazard", address="0xdef")
    market = _market(
        db_session,
        slug="atp-khachan-jong-2026-05-29",
        title="Khachanov vs Jong",
    )
    _trade(db_session, trader, market)
    db_session.commit()

    rows = live_positions(db_session, card_date=CARD_DATE)
    assert len(rows) == 1
    [row] = rows
    assert row["sport"] == "atp"
    # The MLB-edge index would have only MLB keys; ATP rows must never
    # join through one of them, so the badge falls to wallet_only.
    edges = [{
        "id": 1, "edge_type": "game_total", "side": "over",
        "line": 9.5, "generated_for_date": CARD_DATE,
        "home_team": "CWS", "away_team": "DET",
        "market": "DET @ CWS — Over 9.5",
    }]
    classified = classify_wallet_against_edges(row, edges_indexed_by_key(edges))
    assert classified["edge_match_kind"] == "wallet_only"


# ---------------------------------------------------------------------------
# Bonus: a wrong-date wallet trade DOES get rejected and surfaces a
# worked example in the debug report (no silent aggregation).
# ---------------------------------------------------------------------------

def test_wrong_date_trade_surfaces_in_debug_with_example(db_session):
    trader = _trader(db_session)
    market = _market(
        db_session,
        slug="mlb-det-cws-2026-05-28-total-9pt5",
        title="DET @ CWS — Over 9.5 (yesterday)",
    )
    _trade(db_session, trader, market, hours_ago=4.0)
    db_session.commit()

    debug = live_position_debug(db_session, card_date=CARD_DATE)
    # Trade is fresh (4h ago) so it appears in the candidate pool, but
    # the slug carries yesterday's date so it's rejected.
    assert debug["raw_recent_trades"] == 1
    assert debug["rejected"] == 1
    assert debug["top_rejected_examples"]
    example = debug["top_rejected_examples"][0]
    assert example["wallet_nickname"] == "VeryLucky888"
    assert example["parsed_event_date"] == "2026-05-28"
    assert example["dashboard_card_date"] == CARD_DATE
    # Either rejection path (market_date vs slug_date) is valid — the
    # contract is "rejection is named with the parsed date." Operators
    # need to see the date, not which lookup produced it.
    assert "date_mismatch" in example["rejection_reason"]
    assert "2026-05-28" in example["rejection_reason"]
    assert example["normalized_market_key"].startswith("mlb:")
