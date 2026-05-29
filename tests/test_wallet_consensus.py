"""Tracked-wallet consensus aggregator tests.

The Wallet Flow tab relies on ``wallet_consensus_groups`` to surface
markets where two or more tracked wallets are on the same side. These
tests pin the contract that:

* Groups are keyed by (market_id, side, outcome) — distinct sides of
  the same market don't merge.
* The wallet-count threshold counts distinct *wallets*, not signals,
  so one wallet with two trades on the same market doesn't fake a
  consensus by itself.
* The representative payload keeps a real ``market_url`` /
  ``trader_nickname`` / ``market_slug`` so ``render_wallet_card`` can
  still render it.
* Sort order is wallet-count → total size → mean score.
"""

from __future__ import annotations

from app.utils.dashboard_format import wallet_consensus_groups


def _pos(
    *,
    market_id: int,
    side: str,
    outcome: str,
    wallet: str,
    score: float = 50.0,
    size: float = 100.0,
    trader_nickname: str | None = None,
    market_url: str | None = None,
) -> dict:
    return {
        "market_id": market_id,
        "side": side,
        "outcome": outcome,
        "wallet": wallet,
        "trader_nickname": trader_nickname or f"trader_{wallet[-3:]}",
        "score": score,
        "size_usd": size,
        "market_url": market_url or f"https://polymarket.com/event/mlb-game-{market_id}",
        "market_slug": f"mlb-game-{market_id}",
    }


def test_two_wallets_same_side_form_consensus() -> None:
    positions = [
        _pos(market_id=1, side="BUY", outcome="Over", wallet="0xaaa", score=80, size=500),
        _pos(market_id=1, side="BUY", outcome="Over", wallet="0xbbb", score=65, size=300),
    ]
    groups = wallet_consensus_groups(positions, min_wallets=2)
    assert len(groups) == 1
    g = groups[0]
    assert g["consensus_wallets"] == 2
    assert g["consensus_total_size"] == 800.0
    assert g["consensus_mean_score"] == 72.5
    # Representative is the higher-scoring signal — keeps a real card
    # payload so render_wallet_card works.
    assert g["wallet"] == "0xaaa"
    assert g["market_url"].endswith("mlb-game-1")


def test_same_wallet_two_trades_does_not_form_consensus() -> None:
    """The aggregator counts distinct wallets, not raw signals — one
    wallet hammering the same market twice must not fake a consensus.
    """
    positions = [
        _pos(market_id=1, side="BUY", outcome="Over", wallet="0xaaa", score=80, size=500),
        _pos(market_id=1, side="BUY", outcome="Over", wallet="0xaaa", score=85, size=200),
    ]
    groups = wallet_consensus_groups(positions, min_wallets=2)
    assert groups == []


def test_opposite_sides_of_same_market_dont_merge() -> None:
    """Wallets on Over and Under of the same market are NOT consensus —
    they're opposing positions.
    """
    positions = [
        _pos(market_id=1, side="BUY", outcome="Over", wallet="0xaaa"),
        _pos(market_id=1, side="BUY", outcome="Under", wallet="0xbbb"),
    ]
    assert wallet_consensus_groups(positions, min_wallets=2) == []


def test_different_markets_dont_merge() -> None:
    """Two wallets on Over of different markets are not the same play."""
    positions = [
        _pos(market_id=1, side="BUY", outcome="Over", wallet="0xaaa"),
        _pos(market_id=2, side="BUY", outcome="Over", wallet="0xbbb"),
    ]
    assert wallet_consensus_groups(positions, min_wallets=2) == []


def test_positions_without_market_id_are_skipped() -> None:
    """A position without a market_id has no join key and would otherwise
    silently bucket into ``(None, ...)`` — that masks real consensus and
    fakes it on noise. Skip these explicitly.
    """
    positions = [
        _pos(market_id=1, side="BUY", outcome="Over", wallet="0xaaa"),
        {"market_id": None, "side": "BUY", "outcome": "Over", "wallet": "0xbbb"},
    ]
    assert wallet_consensus_groups(positions, min_wallets=2) == []


def test_groups_sort_by_wallets_then_size_then_score() -> None:
    """Three groups, all >=2 wallets; sort order must be the
    contractually-promised wallets → size → mean-score."""
    positions = [
        # market 1: 2 wallets, small size
        _pos(market_id=1, side="BUY", outcome="Over", wallet="0xa1", size=100, score=90),
        _pos(market_id=1, side="BUY", outcome="Over", wallet="0xa2", size=100, score=90),
        # market 2: 3 wallets, biggest priority
        _pos(market_id=2, side="BUY", outcome="Over", wallet="0xb1", size=50, score=50),
        _pos(market_id=2, side="BUY", outcome="Over", wallet="0xb2", size=50, score=50),
        _pos(market_id=2, side="BUY", outcome="Over", wallet="0xb3", size=50, score=50),
        # market 3: 2 wallets, big size — wins tiebreak vs market 1
        _pos(market_id=3, side="BUY", outcome="Over", wallet="0xc1", size=1000, score=40),
        _pos(market_id=3, side="BUY", outcome="Over", wallet="0xc2", size=1000, score=40),
    ]
    groups = wallet_consensus_groups(positions, min_wallets=2)
    assert [g["market_id"] for g in groups] == [2, 3, 1]


def test_min_wallets_threshold_is_respected() -> None:
    """Two-wallet group fails when min_wallets=3."""
    positions = [
        _pos(market_id=1, side="BUY", outcome="Over", wallet="0xaaa"),
        _pos(market_id=1, side="BUY", outcome="Over", wallet="0xbbb"),
    ]
    assert wallet_consensus_groups(positions, min_wallets=3) == []
    # And the same data with min=2 *does* form a group.
    assert len(wallet_consensus_groups(positions, min_wallets=2)) == 1


def test_consensus_preserves_member_list_for_drill_down() -> None:
    """``consensus_members`` keeps every contributing position so the
    dashboard can render the full wallet roster on click-through.
    """
    positions = [
        _pos(market_id=1, side="BUY", outcome="Over", wallet="0xaaa", score=80),
        _pos(market_id=1, side="BUY", outcome="Over", wallet="0xbbb", score=65),
        _pos(market_id=1, side="BUY", outcome="Over", wallet="0xccc", score=70),
    ]
    groups = wallet_consensus_groups(positions, min_wallets=2)
    assert len(groups) == 1
    members = groups[0]["consensus_members"]
    assert len(members) == 3
    assert {m["wallet"] for m in members} == {"0xaaa", "0xbbb", "0xccc"}
