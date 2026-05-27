"""Tests for wallet-flow enrichment + the market join layer."""

from __future__ import annotations

from app.models import Market, MlbEdge, MlbGame, Trade, Trader, WalletTierHistory
from app.services import wallet_market_resolver as wmr
from app.services.wallet_flow import build_wallet_context

CARD_DATE = "2026-05-27"


# ---------------------------------------------------------------------------
# Resolver (pure functions)
# ---------------------------------------------------------------------------


def test_parse_total_slug():
    parsed = wmr.parse_market_slug("mlb-nyy-kc-2026-05-27-total-10pt5")
    assert parsed is not None
    assert parsed.league == "mlb"
    assert parsed.event_date == "2026-05-27"
    assert parsed.away_abbr == "nyy" and parsed.home_abbr == "kc"
    assert parsed.market_type == "total"
    assert parsed.line == 10.5


def test_parse_non_game_slug_returns_none():
    assert wmr.parse_market_slug("will-it-rain-tomorrow") is None
    assert wmr.parse_market_slug(None) is None


def _total_key(side: str, line: float) -> wmr.NormalizedKey:
    edge = {"edge_type": "game_total", "side": side, "line": line, "generated_for_date": CARD_DATE}
    return wmr.normalize_edge(
        edge, home_team="Kansas City Royals", away_team="New York Yankees"
    )


def test_line_tolerance_join():
    market = wmr.parse_market_slug("mlb-nyy-kc-2026-05-27-total-10pt5")
    # 10.1 sportsbook line is within +-0.5 of the 10.5 Polymarket market.
    assert wmr.keys_match(_total_key("under", 10.1), market, line_tol=0.5) is True
    # A 12.5 market is too far from 10.1.
    far = wmr.parse_market_slug("mlb-nyy-kc-2026-05-27-total-12pt5")
    assert wmr.keys_match(_total_key("under", 10.1), far, line_tol=0.5) is False


def test_opposite_side_detected():
    key = _total_key("under", 10.1)
    assert wmr.outcomes_align(key, trade_outcome="Under", trade_side="BUY") == "aligned"
    assert wmr.outcomes_align(key, trade_outcome="Over", trade_side="BUY") == "opposing"
    # Selling Over == backing Under == aligned with an Under edge.
    assert wmr.outcomes_align(key, trade_outcome="Over", trade_side="SELL") == "aligned"


# ---------------------------------------------------------------------------
# build_wallet_context (against a session)
# ---------------------------------------------------------------------------


def _seed_game(db, game_pk=777):
    db.add(MlbGame(game_pk=game_pk, game_date=CARD_DATE,
                   home_team="Kansas City Royals", away_team="New York Yankees"))


def _edge_dict(side="under", line=10.1, game_pk=777):
    return {
        "edge_type": "game_total",
        "side": side,
        "line": line,
        "generated_for_date": CARD_DATE,
        "game_pk": game_pk,
    }


def _add_trader(db, nickname, wallet, trust=50.0):
    t = Trader(nickname=nickname, wallet_address=wallet, platform="polymarket", trust_score=trust)
    db.add(t)
    db.flush()
    return t


def _add_total_market(db, slug):
    m = Market(slug=slug, title=slug, platform="polymarket")
    db.add(m)
    db.flush()
    return m


def test_no_market_returns_no_wallet_data(db_session):
    _seed_game(db_session)
    db_session.flush()
    ctx = build_wallet_context(
        db_session, edge=_edge_dict(), home_team="Kansas City Royals",
        away_team="New York Yankees", card_date=CARD_DATE,
    )
    assert ctx["tracked_wallet_count"] == 0
    assert "NO WALLET DATA" in ctx["tags"]
    assert ctx["confidence_adjustment"] == 0.0
    assert ctx["debug"]["no_match_reason"]


def test_aligned_exposure_and_profile_url(db_session):
    _seed_game(db_session)
    market = _add_total_market(db_session, "mlb-nyy-kc-2026-05-27-total-10pt5")
    with_addr = _add_trader(db_session, "surf", "0xabc")
    no_addr = _add_trader(db_session, "anon", None)
    db_session.add(Trade(trader_id=with_addr.id, market_id=market.id, side="BUY",
                         outcome="Under", price=0.55, size_usd=300.0, source="polymarket"))
    db_session.add(Trade(trader_id=no_addr.id, market_id=market.id, side="BUY",
                         outcome="Under", price=0.50, size_usd=100.0, source="polymarket"))
    db_session.flush()

    ctx = build_wallet_context(
        db_session, edge=_edge_dict(), home_team="Kansas City Royals",
        away_team="New York Yankees", card_date=CARD_DATE,
    )
    assert ctx["tracked_wallet_count"] == 2
    assert ctx["aligned_exposure_usd"] == 400.0
    assert ctx["opposing_exposure_usd"] == 0.0
    assert ctx["consensus_pct"] == 100.0
    by_name = {w["trader_name"]: w for w in ctx["aligned_wallets"]}
    assert by_name["surf"]["profile_url"] == "https://polymarketanalytics.com/traders/0xabc"
    assert by_name["anon"]["profile_url"] is None
    assert by_name["surf"]["market_url"] == "https://polymarket.com/event/mlb-nyy-kc-2026-05-27-total-10pt5"


def test_elite_disagreement_reduces_score(db_session):
    _seed_game(db_session)
    market = _add_total_market(db_session, "mlb-nyy-kc-2026-05-27-total-10pt5")
    elite = _add_trader(db_session, "sharp", "0xelite")
    db_session.add(WalletTierHistory(wallet_address="0xelite", tier="elite", market_type=None,
                                     sample_size=50))
    # Elite trader is on Over while the edge likes Under -> opposing.
    db_session.add(Trade(trader_id=elite.id, market_id=market.id, side="BUY",
                         outcome="Over", price=0.6, size_usd=500.0, source="polymarket"))
    db_session.flush()

    ctx = build_wallet_context(
        db_session, edge=_edge_dict(side="under"), home_team="Kansas City Royals",
        away_team="New York Yankees", card_date=CARD_DATE,
    )
    assert ctx["elite_wallet_disagreement"] == 1
    assert ctx["confidence_adjustment"] < 0
    assert "ELITE DISAGREEMENT" in ctx["tags"]


def test_prior_date_positions_excluded(db_session):
    _seed_game(db_session)
    # Same teams, but yesterday's market — must not join onto today's card.
    market = _add_total_market(db_session, "mlb-nyy-kc-2026-05-26-total-10pt5")
    tr = _add_trader(db_session, "yday", "0xyday")
    db_session.add(Trade(trader_id=tr.id, market_id=market.id, side="BUY",
                         outcome="Under", price=0.5, size_usd=999.0, source="polymarket"))
    db_session.flush()

    ctx = build_wallet_context(
        db_session, edge=_edge_dict(), home_team="Kansas City Royals",
        away_team="New York Yankees", card_date=CARD_DATE,
    )
    assert ctx["tracked_wallet_count"] == 0
    assert "NO WALLET DATA" in ctx["tags"]


def test_execution_block_populated_from_priced_market(db_session):
    _seed_game(db_session)
    market = _add_total_market(db_session, "mlb-nyy-kc-2026-05-27-total-10pt5")
    market.yes_price = 0.41   # Over
    market.no_price = 0.59    # Under
    market.liquidity_usd = 5000.0
    tr = _add_trader(db_session, "surf", "0xabc")
    db_session.add(Trade(trader_id=tr.id, market_id=market.id, side="BUY",
                         outcome="Under", price=0.55, size_usd=300.0, source="polymarket"))
    db_session.flush()

    ctx = build_wallet_context(
        db_session, edge=_edge_dict(side="under"), home_team="Kansas City Royals",
        away_team="New York Yankees", card_date=CARD_DATE,
    )
    ex = ctx["execution"]
    assert ex is not None
    assert ex["platform"] == "polymarket"
    assert ex["side"] == "under"
    assert ex["side_price"] == 0.59          # Under leg = no_price
    assert ex["implied_prob"] == 0.59
    assert ex["market_url"].endswith("mlb-nyy-kc-2026-05-27-total-10pt5")


def test_execution_none_when_no_market(db_session):
    _seed_game(db_session)
    db_session.flush()
    ctx = build_wallet_context(
        db_session, edge=_edge_dict(), home_team="Kansas City Royals",
        away_team="New York Yankees", card_date=CARD_DATE,
    )
    assert ctx["execution"] is None


def test_pitcher_k_edge_gets_no_wallet_data(db_session):
    _seed_game(db_session)
    db_session.flush()
    edge = {"edge_type": "pitcher_strikeouts", "side": "over", "line": 6.5,
            "generated_for_date": CARD_DATE, "game_pk": 777}
    ctx = build_wallet_context(
        db_session, edge=edge, home_team="Kansas City Royals",
        away_team="New York Yankees", card_date=CARD_DATE,
    )
    assert "NO WALLET DATA" in ctx["tags"]
    assert "pitcher_strikeouts" in ctx["debug"]["no_match_reason"]


def test_wallet_context_persists_on_edge(db_session):
    """The new MlbEdge.wallet_context column round-trips JSON."""
    edge = MlbEdge(game_pk=777, edge_type="game_total", market="Game — Under 10.1",
                   side="under", line=10.1, score=70.0, generated_for_date=CARD_DATE,
                   wallet_context={"consensus_pct": 80.0, "tags": ["WALLET CONFIRMED"]})
    db_session.add(edge)
    db_session.flush()
    db_session.refresh(edge)
    assert edge.wallet_context["consensus_pct"] == 80.0
