"""Tests for the signal scoring engine."""

from __future__ import annotations

from app.config import ScoringWeights
from app.models import Market, Trader
from app.services.scoring import ScoreInputs, score_signal


def _make_trader(**overrides):
    base = dict(
        nickname="testwallet",
        trust_score=90.0,
        win_rate=0.7,
        trader_rank=50,
    )
    base.update(overrides)
    return Trader(**base)


def _make_market(**overrides):
    base = dict(
        slug="test-market",
        title="Test market",
        liquidity_usd=100_000.0,
        yes_price=0.55,
    )
    base.update(overrides)
    return Market(**base)


def test_score_is_bounded():
    inputs = ScoreInputs(
        trader=_make_trader(trust_score=100.0, win_rate=1.0, trader_rank=1),
        same_side_wallets=5,
        total_watched_wallets=5,
        market=_make_market(liquidity_usd=10_000_000.0),
        entry_price=0.30,
        current_price=0.80,
        price_gap=0.30,
    )
    out = score_signal(inputs)
    assert 0.0 <= out.total <= 100.0
    assert out.total > 80.0  # everything maxed -> very high


def test_score_handles_missing_trader_and_market():
    inputs = ScoreInputs(trader=None, market=None)
    out = score_signal(inputs)
    assert 0.0 <= out.total <= 100.0
    # neutral defaults should still produce a positive baseline
    assert out.total > 10.0


def test_high_trust_beats_low_trust():
    market = _make_market()
    high = score_signal(
        ScoreInputs(trader=_make_trader(trust_score=95.0, win_rate=0.8), market=market)
    )
    low = score_signal(
        ScoreInputs(trader=_make_trader(trust_score=10.0, win_rate=0.45), market=market)
    )
    assert high.total > low.total


def test_multi_wallet_consensus_increases_score():
    inputs_base = dict(trader=_make_trader(), market=_make_market(), total_watched_wallets=10)
    solo = score_signal(ScoreInputs(same_side_wallets=1, **inputs_base))
    pair = score_signal(ScoreInputs(same_side_wallets=2, **inputs_base))
    crowd = score_signal(ScoreInputs(same_side_wallets=5, **inputs_base))
    assert solo.total < pair.total < crowd.total


def test_weights_sum_to_one():
    w = ScoringWeights()
    total = (
        w.wallet_quality
        + w.multi_wallet_consensus
        + w.liquidity
        + w.entry_timing
        + w.price_inefficiency
    )
    assert abs(total - 1.0) < 1e-9


def test_breakdown_components_present():
    out = score_signal(ScoreInputs(trader=_make_trader(), market=_make_market()))
    d = out.as_dict()
    assert {
        "wallet_quality",
        "multi_wallet_consensus",
        "liquidity",
        "entry_timing",
        "price_inefficiency",
        "total",
    } <= d.keys()


def test_post_entry_price_move_boosts_score():
    market = _make_market()
    trader = _make_trader()
    no_move = score_signal(
        ScoreInputs(trader=trader, market=market, entry_price=0.5, current_price=0.5)
    )
    big_move = score_signal(
        ScoreInputs(trader=trader, market=market, entry_price=0.5, current_price=0.65)
    )
    assert big_move.total > no_move.total


def test_price_inefficiency_contribution():
    market = _make_market()
    trader = _make_trader()
    flat = score_signal(ScoreInputs(trader=trader, market=market, price_gap=0.0))
    arbed = score_signal(ScoreInputs(trader=trader, market=market, price_gap=0.10))
    assert arbed.total > flat.total
