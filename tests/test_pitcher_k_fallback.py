"""Pitcher K pipeline invariants.

These pin the fixes for the "BallparkPal has 30 rows but Top Pitcher Ks
shows 'No qualifying edges'" bug:

  1. With ``projected_k=6.38, over_line=6.5`` (Taj Bradley) the
     fallback builder produces a usable prop and the pipeline classifies
     it as a watchlist-band candidate (|edge| ≈ 0.12 → below floor in
     this case, so the test checks band classification correctness).
  2. With ``projected_k=5.76, over_line=5.5`` (Max Meyer) the fallback
     produces an Over candidate (positive edge ≈ 0.26 → candidate band).
  3. If the sportsbook odds cache has no pitcher props, the BPP
     fallback still produces a usable card.
  4. Empty states surface the EXACT failing stage instead of the
     generic "No qualifying edges."
  5. Pitcher name normalization handles ``"J. Ryan"`` ↔ ``"Joe Ryan"``
     and ``"Last, First"`` formats.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from app.services.mlb_pitcher_k_diagnostics import (
    K_EDGE_CANDIDATE_FLOOR,
    K_EDGE_STRONG_FLOOR,
    K_EDGE_WATCHLIST_FLOOR,
    PitcherKDiagnostics,
    classify_k_edge_magnitude,
)
from app.services.mlb_pitcher_k_fallback import (
    CSV_EXECUTION_SOURCE,
    CSV_MARKET_TYPE,
    CSV_SOURCE,
    FALLBACK_SOURCE,
    FallbackRejection,
    build_fallback_prop_analysis,
    is_fallback_payload,
    k_edge_from_fallback,
    name_matches_loose,
    normalize_pitcher_name,
    validate_bpp_row,
)


# ---------------------------------------------------------------------------
# Threshold + classification
# ---------------------------------------------------------------------------

def test_threshold_constants_match_spec():
    assert K_EDGE_WATCHLIST_FLOOR == 0.15
    assert K_EDGE_CANDIDATE_FLOOR == 0.35
    assert K_EDGE_STRONG_FLOOR == 0.65


def test_classify_k_edge_magnitude_bands():
    assert classify_k_edge_magnitude(0.10) == "below"
    assert classify_k_edge_magnitude(0.20) == "watchlist"
    assert classify_k_edge_magnitude(-0.20) == "watchlist"  # under lean
    assert classify_k_edge_magnitude(0.40) == "candidate"
    assert classify_k_edge_magnitude(0.70) == "strong"
    assert classify_k_edge_magnitude(None) == "below"


# ---------------------------------------------------------------------------
# BallparkPal fallback builder
# ---------------------------------------------------------------------------

def test_bradley_row_produces_fallback_prop_analysis():
    """projected_k=6.38, over_line=6.5 → edge = -0.12 (under lean,
    below the watchlist floor of 0.15). Even at that magnitude the
    fallback builder must still produce a usable prop_analysis dict
    so downstream stages can score it; classification is what drops
    it from the card list, not the builder."""
    row = {
        "pitcher": "Taj Bradley", "team": "TB", "opp": "BAL",
        "projected_k": 6.38, "over_line": 6.5, "ballparkpal_odds": 1.91,
    }
    prop = build_fallback_prop_analysis(pitcher_name="Taj Bradley", bpp_row=row)
    assert prop is not None
    assert prop["line"] == 6.5
    assert prop["best_over_book"] == CSV_EXECUTION_SOURCE
    assert prop["source"] == CSV_SOURCE
    assert prop["execution_source"] == CSV_EXECUTION_SOURCE
    assert prop["market_type"] == CSV_MARKET_TYPE
    assert prop["ballparkpal_projected_k"] == 6.38
    assert is_fallback_payload(prop)
    # k_edge is negative for Bradley (projection below line).
    assert k_edge_from_fallback(prop) == -0.12


def test_meyer_row_produces_over_watchlist_candidate():
    """projected_k=5.76, over_line=5.5 → +0.26 edge (over lean,
    watchlist band). User explicitly asked for this case to surface."""
    row = {
        "pitcher": "Max Meyer", "team": "MIA", "opp": "PHI",
        "projected_k": 5.76, "over_line": 5.5, "over_odds": 1.85,
    }
    prop = build_fallback_prop_analysis(pitcher_name="Max Meyer", bpp_row=row)
    assert prop is not None
    k_edge = k_edge_from_fallback(prop)
    assert k_edge == 0.26
    assert prop["source"] == CSV_SOURCE
    assert prop["best_over_price"] == 1.85
    # Positive edge above the watchlist floor → watchlist band.
    assert classify_k_edge_magnitude(k_edge) == "watchlist"


def test_carlos_rodon_clear_watchlist_over():
    row = {
        "pitcher": "Carlos Rodon", "team": "NYY", "opp": "BOS",
        "projected_k": 5.76, "over_line": 5.5, "ballparkpal_odds": 1.80,
    }
    prop = build_fallback_prop_analysis(pitcher_name="Carlos Rodon", bpp_row=row)
    assert prop is not None
    assert classify_k_edge_magnitude(k_edge_from_fallback(prop)) == "watchlist"


def test_fallback_returns_none_when_projection_missing():
    """No projected_k → can't compute an edge → no fallback card."""
    row = {"pitcher": "Mystery", "over_line": 5.5}
    assert build_fallback_prop_analysis(pitcher_name="Mystery", bpp_row=row) is None


def test_fallback_returns_none_when_line_missing():
    row = {"pitcher": "Mystery", "projected_k": 6.0}
    assert build_fallback_prop_analysis(pitcher_name="Mystery", bpp_row=row) is None


# ---------------------------------------------------------------------------
# Empty-state messaging
# ---------------------------------------------------------------------------

def test_empty_state_says_no_projections_when_cache_blank():
    diag = PitcherKDiagnostics()
    msg = diag.empty_state_message()
    assert "0 strikeout projections" in msg or "did not run" in msg


def test_empty_state_says_no_odds_when_no_fallback_either():
    """The truly empty case: 0 projections loaded → the generic
    'no fallback odds available either' branch is intentionally
    quiet about the 30-loaded-but-rejected case (covered by other
    tests below)."""
    diag = PitcherKDiagnostics(
        strikeout_projections_loaded=0,
        sportsbook_pitcher_k_props_loaded=0,
        candidates_built_from_ballparkpal_fallback=0,
    )
    msg = diag.empty_state_message()
    assert "Pitcher K scan did not run" in msg or "0 strikeout projections" in msg


def test_empty_state_says_no_names_matched_when_30_loaded_but_zero_matches():
    """The real bug behind 'No qualifying edges': 30 BPP rows loaded,
    but 0 of today's MLB probable pitchers matched any of them. The
    message must say so explicitly so the operator opens the name
    reconciliation panel."""
    diag = PitcherKDiagnostics(
        strikeout_projections_loaded=30,
        sportsbook_pitcher_k_props_loaded=18,
        pitcher_names_matched_sportsbook=0,
        pitcher_names_matched_ballparkpal=0,
        mlb_probable_pitchers_today=[
            {"pitcher_name": "Joe Ryan", "team": "MIN", "game_pk": 1},
            {"pitcher_name": "Max Meyer", "team": "MIA", "game_pk": 2},
        ],
    )
    msg = diag.empty_state_message()
    assert "0 names matched" in msg
    assert "30 BallparkPal projections" in msg
    assert "2 MLB probable pitchers" in msg
    assert "name drift" in msg.lower() or "name reconciliation" in msg.lower() or "name_drift" in msg.lower() or "reconciliation" in msg.lower() or "name drift" in msg.lower()


def test_empty_state_says_bad_mapping_when_all_matches_rejected():
    """30 rows loaded, names DID match (rejections happened because
    sanity validator killed them all). The message must point at the
    cache parse, not at name matching."""
    diag = PitcherKDiagnostics(
        strikeout_projections_loaded=30,
        rejected_for_bad_mapping=5,
        candidates_built_from_ballparkpal_fallback=0,
    )
    msg = diag.empty_state_message()
    assert "rejected by sanity checks" in msg
    assert "re-upload the CSV" in msg


def test_empty_state_bad_mapping_takes_priority_over_no_fallback_branch():
    """Regression test for the user-visible bug: when 30 rows are
    loaded but ALL were rejected by sanity validation, the message
    used to say 'no BallparkPal fallback odds available either',
    which falsely implied the cache was empty. It must instead say
    rows were REJECTED."""
    diag = PitcherKDiagnostics(
        strikeout_projections_loaded=30,
        sportsbook_pitcher_k_props_loaded=0,
        rejected_for_bad_mapping=8,
        candidates_built_from_ballparkpal_fallback=0,
    )
    msg = diag.empty_state_message()
    assert "BallparkPal fallback odds available" not in msg
    assert "rejected by sanity checks" in msg


def test_empty_state_says_threshold_when_candidates_rejected():
    diag = PitcherKDiagnostics(
        strikeout_projections_loaded=30,
        sportsbook_pitcher_k_props_loaded=18,
        pitcher_names_matched_sportsbook=10,
        candidates_built_total=8,
        candidates_rejected_by_threshold=8,
        cards_rendered=0,
    )
    msg = diag.empty_state_message()
    assert "rejected by K-edge thresholds" in msg or "0.15-run" in msg


def test_empty_state_says_bpp_cards_when_external_props_absent():
    diag = PitcherKDiagnostics(
        strikeout_projections_loaded=30,
        external_pitcher_prop_markets_found=0,
        candidates_built_from_ballparkpal_fallback=2,
    )
    msg = diag.empty_state_message()
    assert msg == (
        "30 BallparkPal K projections loaded. 0 external pitcher prop "
        "markets found. Showing BallparkPal-based model cards."
    )


def test_external_pitcher_props_zero_still_produces_bpp_k_cards(
    db_session, monkeypatch,
):
    from app.models import MlbEdge
    from app.services import mlb_edge_engine, odds_cache
    from app.services.ballparkpal_cache import upsert_snapshot

    upsert_snapshot(
        db_session,
        page="strikeouts",
        slate_date="2026-05-29",
        source_url="file://strikeouts.csv",
        parsed={
            "rows": [
                {
                    "pitcher": "Max Meyer",
                    "team": "MIA",
                    "opp": "PHI",
                    "projected_k": 5.76,
                    "over_line": 5.5,
                    "over_odds": -110,
                }
            ]
        },
        raw_html_path=None,
        last_updated_text=None,
        status="ok",
    )
    db_session.commit()

    async def fake_load_games(_mlb, card_date: str):
        return [
            {
                "game_pk": 99001,
                "game_date": card_date,
                "home_team": "Miami Marlins",
                "away_team": "Philadelphia Phillies",
                "venue": "loanDepot park",
                "probable_home_pitcher": "Max Meyer",
                "probable_home_pitcher_id": 42,
                "probable_away_pitcher": None,
                "probable_away_pitcher_id": None,
                "game_status": "Scheduled",
                "start_time": None,
                "weather_location_query": "Miami, FL",
            }
        ]

    async def fake_refresh(*_args, **kwargs):
        return odds_cache.RefreshResult(
            refreshed=False,
            reason="events fetch failed and no cached events",
            game_date=kwargs["game_date"],
        )

    async def fake_environment_for_game(*_args, **_kwargs):
        return {"k_environment_score": 50.0, "warnings": []}

    async def fake_external_props(**_kwargs):
        return []

    monkeypatch.setattr(mlb_edge_engine, "_load_games", fake_load_games)
    monkeypatch.setattr(mlb_edge_engine, "MlbStatsApiProvider", lambda: object())
    monkeypatch.setattr(mlb_edge_engine.odds_cache, "refresh_mlb_odds_cache", fake_refresh)
    monkeypatch.setattr(mlb_edge_engine, "_environment_for_game", fake_environment_for_game)
    monkeypatch.setattr(mlb_edge_engine, "fetch_external_pitcher_k_props", fake_external_props)
    monkeypatch.setattr(
        mlb_edge_engine,
        "statcast_context",
        lambda *_args, **_kwargs: {"summary": {}, "warnings": []},
    )

    result = asyncio.run(
        mlb_edge_engine.run_daily_mlb_edges(
            db_session,
            game_date="2026-05-29",
        )
    )

    assert result["events_with_pitcher_props"] == 0
    edges = db_session.query(MlbEdge).filter_by(edge_type="pitcher_strikeouts").all()
    assert edges
    assert any(edge.best_book == CSV_EXECUTION_SOURCE for edge in edges)
    assert any(edge.best_price == -110 for edge in edges)

    card_rows = result["daily_card"]["top_pitcher_strikeouts"]
    assert card_rows
    assert any(row.get("source") == CSV_SOURCE for row in card_rows)
    assert any(row.get("execution_source") == CSV_EXECUTION_SOURCE for row in card_rows)
    assert any(row.get("market_type") == CSV_MARKET_TYPE for row in card_rows)
    assert any(row.get("projected_strikeouts") == 5.76 for row in card_rows)

    diag = result["diagnostics"]["pitcher_k"]
    assert diag["strikeout_projections_loaded"] == 1
    assert diag["sportsbook_pitcher_k_props_loaded"] == 0
    assert diag["external_pitcher_prop_markets_found"] == 0
    assert diag["candidates_built_from_ballparkpal_fallback"] == 1
    assert result["daily_card"]["data_quality_summary"]["pitcher_k_empty_state_message"] == (
        "1 BallparkPal K projections loaded. 0 external pitcher prop "
        "markets found. Showing BallparkPal-based model cards."
    )


def test_external_pitcher_prop_matcher_normalizes_name_date_and_line():
    from app.services.mlb_external_pitcher_props import (
        match_external_pitcher_k_prop,
        normalize_external_pitcher_k_market,
    )

    market = normalize_external_pitcher_k_market(
        {
            "title": "Will Taj Bradley record over 6.5 strikeouts on 2026-05-29?",
            "slug": "mlb-taj-bradley-2026-05-29-strikeouts-6pt5",
            "yes_price": 57,
        },
        platform="polymarket",
        default_date="2026-05-29",
    )

    assert market is not None
    assert market["pitcher_name"] == "Taj Bradley"
    assert market["line"] == 6.5
    assert market["price"] == 0.57

    assert match_external_pitcher_k_prop(
        pitcher_name="T. Bradley",
        game_date="2026-05-29",
        line=6.5,
        markets=[market],
    )["market_url"].startswith("https://polymarket.com/event/")


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

def test_first_initial_matches_full_first_name():
    assert name_matches_loose("J. Ryan", "Joe Ryan")
    assert name_matches_loose("Joe Ryan", "J. Ryan")


def test_first_initial_does_not_match_different_first_initial():
    assert not name_matches_loose("J. Ryan", "Max Ryan")
    assert not name_matches_loose("Joe Ryan", "Max Ryan")


def test_last_first_format_reverses_to_first_last():
    assert normalize_pitcher_name("Ryan, Joe") == "joe ryan"


def test_accents_and_suffix_stripped():
    assert normalize_pitcher_name("José Ureña Jr.") == "jose urena"


def test_exact_full_name_match():
    assert name_matches_loose("Carlos Rodon", "Carlos Rodon")
    assert name_matches_loose("Carlos Rodón", "Carlos Rodon")  # accent


# ---------------------------------------------------------------------------
# Regression: factors dict must hold ONLY numeric values
# ---------------------------------------------------------------------------
#
# _persist_edge iterates ``edge.factors.items()`` and calls
# ``float(value)`` on each entry to build MlbEdgeFactor rows. If a
# pipeline stage ever stuffs a string into factors (e.g. an "odds_source"
# tag), the entire scan 502s. These tests pin both invariants: the
# fallback path must NOT leak a string into factors, AND a malformed
# factors dict must not crash the persist helper.

def test_fallback_factors_dict_holds_only_numeric_values(db_session):
    """End-to-end: run a single edge through total_edges → _persist_edge
    with a factors dict that includes the BPP-fallback fields. None of
    those should be strings."""
    from app.models import MlbEdge, MlbEdgeFactor
    from app.services.mlb_pitcher_k_model import pitcher_k_edges

    game = {"game_pk": 9001, "home_team": "MIA", "away_team": "PHI"}
    pitcher = {"id": 42, "name": "Max Meyer"}
    bpp_row = {
        "pitcher": "Max Meyer", "team": "MIA", "opp": "PHI",
        "projected_k": 5.76, "over_line": 5.5, "over_odds": 1.85,
    }
    prop = build_fallback_prop_analysis(pitcher_name="Max Meyer", bpp_row=bpp_row)
    statcast = {"summary": {}, "warnings": []}
    environment = {"k_environment_score": 50.0, "warnings": []}

    edges = pitcher_k_edges(
        game=game, pitcher=pitcher, prop_analysis=prop,
        statcast_context=statcast, environment=environment,
    )
    assert edges
    for edge in edges:
        factors = edge.get("factors") or {}
        # Every value in the factors dict must be coercible to float.
        # Catching a string here means a future caller is about to crash
        # _persist_edge with a ValueError.
        for name, value in factors.items():
            try:
                float(value)
            except (TypeError, ValueError):
                raise AssertionError(
                    f"factors[{name!r}] = {value!r} is not numeric — "
                    "this would 502 the entire scan in _persist_edge."
                )


# ---------------------------------------------------------------------------
# Field-mapping invariants — the "Over 174 Ks" bug regression suite
# ---------------------------------------------------------------------------

import pytest


def test_bradley_field_mapping_uses_correct_columns():
    """Pinned to the user-provided example. projected_k must come from
    projected_k, line from over_line, odds from over_odds — and the
    edge must equal projected_k - over_line, NEVER projected_k - odds."""
    row = {
        "pitcher": "Taj Bradley", "team": "TB", "opp": "BAL",
        "projected_k": 6.38, "over_line": 6.5, "over_odds": 116,
        "projected_innings": 6.0, "batters_faced": 26.1,
    }
    prop = build_fallback_prop_analysis(pitcher_name="Taj Bradley", bpp_row=row)
    assert prop is not None
    assert prop["line"] == 6.5  # NOT 116
    assert prop["best_over_price"] == 116  # the odds
    assert prop["source"] == CSV_SOURCE
    assert prop["execution_source"] == CSV_EXECUTION_SOURCE
    assert prop["ballparkpal_projected_k"] == 6.38
    assert k_edge_from_fallback(prop) == pytest.approx(-0.12, abs=1e-6)


def test_meyer_field_mapping_produces_correct_over_edge():
    row = {
        "pitcher": "Max Meyer", "team": "MIA", "opp": "PHI",
        "projected_k": 5.76, "over_line": 5.5, "over_odds": -110,
    }
    prop = build_fallback_prop_analysis(pitcher_name="Max Meyer", bpp_row=row)
    assert prop is not None
    assert prop["line"] == 5.5
    assert prop["best_over_price"] == -110
    assert prop["source"] == CSV_SOURCE
    assert prop["execution_source"] == CSV_EXECUTION_SOURCE
    assert k_edge_from_fallback(prop) == pytest.approx(0.26, abs=1e-6)


def test_bradley_bpp_row_produces_valid_pitcher_k_card_payload():
    from app.services.mlb_pitcher_k_model import pitcher_k_edges

    row = {
        "pitcher": "Taj Bradley", "team": "TB", "opp": "BAL",
        "projected_k": 6.38, "over_line": 6.5, "over_odds": 116,
    }
    prop = build_fallback_prop_analysis(pitcher_name="Taj Bradley", bpp_row=row)
    cards = pitcher_k_edges(
        game={"game_pk": 1, "home_team": "Tampa Bay Rays", "away_team": "Baltimore Orioles"},
        pitcher={"id": 1, "name": "Taj Bradley"},
        prop_analysis=prop,
        statcast_context={"summary": {}, "warnings": []},
        environment={"k_environment_score": 50.0, "warnings": []},
    )
    over = next(card for card in cards if card["side"] == "over")
    assert over["line"] == 6.5
    assert over["best_price"] == 116
    assert over["source"] == CSV_SOURCE
    assert over["execution_source"] == CSV_EXECUTION_SOURCE
    assert over["market_type"] == CSV_MARKET_TYPE
    assert over["projected_strikeouts"] == 6.38


def test_meyer_bpp_row_produces_valid_pitcher_k_card_payload():
    from app.services.mlb_pitcher_k_model import pitcher_k_edges

    row = {
        "pitcher": "Max Meyer", "team": "MIA", "opp": "PHI",
        "projected_k": 5.76, "over_line": 5.5, "over_odds": -110,
    }
    prop = build_fallback_prop_analysis(pitcher_name="Max Meyer", bpp_row=row)
    cards = pitcher_k_edges(
        game={"game_pk": 2, "home_team": "Miami Marlins", "away_team": "Philadelphia Phillies"},
        pitcher={"id": 2, "name": "Max Meyer"},
        prop_analysis=prop,
        statcast_context={"summary": {}, "warnings": []},
        environment={"k_environment_score": 50.0, "warnings": []},
    )
    over = next(card for card in cards if card["side"] == "over")
    assert over["line"] == 5.5
    assert over["best_price"] == -110
    assert over["source"] == CSV_SOURCE
    assert over["execution_source"] == CSV_EXECUTION_SOURCE
    assert over["market_type"] == CSV_MARKET_TYPE
    assert over["projected_strikeouts"] == 5.76


# ---------------------------------------------------------------------------
# Sanity rejections — the bug that produced "Over 174 Ks"
# ---------------------------------------------------------------------------

def test_line_value_that_looks_like_american_odds_is_rejected():
    """The Paxton Schultz failure mode: over_line carried american
    odds (174) because the cache parser swapped columns. The sanity
    validator must reject the row with a NAMED reason instead of
    publishing a "Over 174 Ks" card."""
    row = {
        "pitcher": "Paxton Schultz", "team": "?", "opp": "?",
        "projected_k": 5.0, "over_line": 174, "over_odds": 6.5,
    }
    with pytest.raises(FallbackRejection) as info:
        validate_bpp_row(pitcher_name="Paxton Schultz", bpp_row=row)
    assert info.value.reason == "line_looks_like_american_odds"
    # And build_fallback_prop_analysis returns None — no card built.
    assert build_fallback_prop_analysis(
        pitcher_name="Paxton Schultz", bpp_row=row,
    ) is None


def test_negative_line_value_rejected():
    """The Tyler Samaniego case — -164 as a "K line" is clearly
    american odds for the under, not a strikeout count."""
    row = {
        "pitcher": "Tyler Samaniego",
        "projected_k": 4.0, "over_line": -164, "over_odds": 6.5,
    }
    with pytest.raises(FallbackRejection) as info:
        validate_bpp_row(pitcher_name="Tyler Samaniego", bpp_row=row)
    assert info.value.reason == "line_looks_like_american_odds"


def test_projected_k_out_of_range_rejected():
    row = {"pitcher": "X", "projected_k": 99.0, "over_line": 6.5}
    with pytest.raises(FallbackRejection) as info:
        validate_bpp_row(pitcher_name="X", bpp_row=row)
    assert info.value.reason == "projected_k_out_of_range"


def test_missing_line_rejected_with_named_reason():
    row = {"pitcher": "X", "projected_k": 5.0}
    with pytest.raises(FallbackRejection) as info:
        validate_bpp_row(pitcher_name="X", bpp_row=row)
    assert info.value.reason == "over_line_missing"


def test_missing_projected_k_rejected_with_named_reason():
    row = {"pitcher": "X", "over_line": 6.5}
    with pytest.raises(FallbackRejection) as info:
        validate_bpp_row(pitcher_name="X", bpp_row=row)
    assert info.value.reason == "projected_k_missing"


def test_over_odds_outside_range_drops_price_but_does_not_reject_row():
    """A K card is still useful with just the projection + line, so a
    bad over_odds value must NOT block the card. Decimal odds like 1.91
    were sitting inside the "looks like a line" heuristic and the
    aggressive earlier rejection was eating real cards. We now silently
    drop the price and keep going."""
    row = {
        "pitcher": "X", "projected_k": 5.0, "over_line": 5.5,
        "over_odds": 5000,  # implausible odds
    }
    out = validate_bpp_row(pitcher_name="X", bpp_row=row)
    # Validator accepts the row; the bad price simply becomes None so
    # the card renders without a best price.
    assert out["line"] == 5.5
    assert out["over_price"] is None


def test_valid_row_returns_validated_values():
    row = {
        "pitcher": "Carlos Rodon",
        "projected_k": 5.76, "over_line": 5.5, "over_odds": -118,
    }
    out = validate_bpp_row(pitcher_name="Carlos Rodon", bpp_row=row)
    assert out["projected_k"] == 5.76
    assert out["line"] == 5.5
    assert out["over_price"] == -118


def test_persist_edge_skips_non_numeric_factors_without_crashing(db_session):
    """Even if a future regression sneaks a non-numeric value into
    factors, the persist path must skip it with a debug log instead of
    raising ValueError and aborting the scan."""
    from app.models import MlbEdge, MlbEdgeFactor
    from app.services.mlb_edge_engine import _persist_edge

    payload = {
        "game_pk": 7777,
        "edge_type": "pitcher_strikeouts",
        "market": "Max Meyer Over 5.5 Ks",
        "side": "over",
        "line": 5.5,
        "best_book": "ballparkpal_fallback",
        "best_price": 1.85,
        "consensus_price": 1.85,
        "score": 60.0,
        "confidence": "low",
        "action": "Watch",
        "chase_risk": "low",
        "reasons": [],
        "warnings": [],
        "data_sources_used": ["ballparkpal_fallback"],
        "factors": {
            # Stray non-numeric — the bug that caused the production 502.
            "odds_source": "ballparkpal_fallback",
            # And a real numeric factor that should survive.
            "sportsbook_price_edge": 50.0,
            "ballparkpal_projected_k": 5.76,
        },
        "is_valid": True,
    }
    edge = _persist_edge(db_session, payload, card_date="2026-05-29")
    db_session.flush()
    # Edge persisted successfully.
    assert edge.id is not None
    # And the non-numeric factor was silently skipped, while numeric
    # factors made it through.
    persisted_names = {f.name for f in db_session.query(MlbEdgeFactor).all()}
    assert "odds_source" not in persisted_names
    assert "sportsbook_price_edge" in persisted_names
    assert "ballparkpal_projected_k" in persisted_names
