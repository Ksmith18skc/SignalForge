"""Optional join layer: compare BallparkPal projections to SignalForge edges.

This module is read-only and never modifies edges in place. It returns
diagnostic payloads (BPP projected total, gap vs market, agree/disagree
verdict) that the dashboard can render alongside the existing MLB edge
cards. Edges are still produced and graded by the SignalForge pipeline —
BPP is a second opinion, not the truth.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.services.ballparkpal_cache import latest_snapshot, snapshot_payload


def game_sims_index(db: Session, *, slate_date: str | None = None) -> dict[tuple[str, str], dict[str, Any]]:
    """Build a (home, away) → row lookup of BPP game-sim projections.

    Keys are normalized to uppercase 3-letter team abbreviations so a
    case mismatch can't silently miss a match.
    """
    snap = latest_snapshot(db, page="game_sims", slate_date=slate_date)
    payload = snapshot_payload(snap)
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload.get("rows") or []:
        home = (row.get("home_team") or "").upper()
        away = (row.get("away_team") or "").upper()
        if home and away:
            index[(home, away)] = row
    return index


def k_projection_index(db: Session, *, slate_date: str | None = None) -> dict[str, dict[str, Any]]:
    """Pitcher-name → row lookup from the Strikeout Center cache.

    We normalize on lowercased name without punctuation so "St. John"
    matches "St John" matches "STJOHN".
    """
    snap = latest_snapshot(db, page="strikeouts", slate_date=slate_date)
    payload = snapshot_payload(snap)
    out: dict[str, dict[str, Any]] = {}
    for row in payload.get("rows") or []:
        name = row.get("pitcher") or ""
        key = _normalize_name(name)
        if key:
            out[key] = row
    return out


def compare_totals(
    *,
    signalforge_projected_total: float | None,
    market_total: float | None,
    ballparkpal_total: float | None,
) -> dict[str, Any]:
    """Three-way comparison between SignalForge, market, and BallparkPal.

    Returns gap deltas plus a verdict string the dashboard can render
    directly. ``None`` inputs collapse to ``None`` gaps — we never
    fabricate a number from missing data.
    """
    bpp_vs_market = _delta(ballparkpal_total, market_total)
    sf_vs_bpp = _delta(signalforge_projected_total, ballparkpal_total)
    sf_vs_market = _delta(signalforge_projected_total, market_total)
    verdict = _verdict(sf_vs_bpp)
    return {
        "ballparkpal_total": ballparkpal_total,
        "signalforge_total": signalforge_projected_total,
        "market_total": market_total,
        "ballparkpal_total_gap_vs_market": bpp_vs_market,
        "signalforge_total_gap_vs_ballparkpal": sf_vs_bpp,
        "signalforge_total_gap_vs_market": sf_vs_market,
        "verdict": verdict,
    }


def _verdict(sf_vs_bpp: float | None) -> str:
    if sf_vs_bpp is None:
        return "no_comparison"
    # A 0.4-run gap is roughly within model noise; outside that range we
    # mark agreement/disagreement directionally.
    if abs(sf_vs_bpp) <= 0.4:
        return "agree"
    if sf_vs_bpp > 0.75:
        return "signalforge_inflated_vs_bpp"
    if sf_vs_bpp < -0.75:
        return "signalforge_deflated_vs_bpp"
    return "disagree"


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    try:
        return round(float(a) - float(b), 4)
    except (TypeError, ValueError):
        return None


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())
