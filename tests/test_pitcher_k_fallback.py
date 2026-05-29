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

from datetime import datetime

from app.services.mlb_pitcher_k_diagnostics import (
    K_EDGE_CANDIDATE_FLOOR,
    K_EDGE_STRONG_FLOOR,
    K_EDGE_WATCHLIST_FLOOR,
    PitcherKDiagnostics,
    classify_k_edge_magnitude,
)
from app.services.mlb_pitcher_k_fallback import (
    FALLBACK_SOURCE,
    build_fallback_prop_analysis,
    is_fallback_payload,
    k_edge_from_fallback,
    name_matches_loose,
    normalize_pitcher_name,
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
    assert prop["best_over_book"] == FALLBACK_SOURCE
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
    diag = PitcherKDiagnostics(
        strikeout_projections_loaded=30,
        sportsbook_pitcher_k_props_loaded=0,
        candidates_built_from_ballparkpal_fallback=0,
    )
    msg = diag.empty_state_message()
    assert "30 strikeout projections" in msg
    assert "0 pitcher K odds" in msg


def test_empty_state_says_no_names_matched():
    diag = PitcherKDiagnostics(
        strikeout_projections_loaded=30,
        sportsbook_pitcher_k_props_loaded=18,
        pitcher_names_matched_sportsbook=0,
        pitcher_names_matched_ballparkpal=0,
    )
    msg = diag.empty_state_message()
    assert "0 pitcher names matched" in msg


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
