"""BallparkPal fallback for pitcher K cards.

When the sportsbook odds cache is empty for pitcher strikeout props —
or just doesn't contain the pitcher we care about — we synthesize a
prop-analysis payload from the BallparkPal Strikeout Center cache.
This is the difference between "we have a projected K, an over line,
and BP odds in the CSV the operator uploaded" and "we have nothing,
so we display nothing." The cache already exposes the data; we just
have to plug it into the same shape that ``consensus_for_pitcher``
returns.

The synthesized payload is clearly marked with
``source="ballparkpal_fallback"`` so downstream consumers (and the
dashboard card) can label the card "BallparkPal fallback odds" and
the operator never thinks the line came from DraftKings.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Any

from app.services.mlb_market_validation import MarketSubtype


# BallparkPal CSV columns we know how to read. The cache parser stores
# rows verbatim with lowercase keys, so an unusual export header
# (``Projected_K`` vs ``projected_k``) gets normalized at lookup time.
_PROJECTED_K_KEYS = ("projected_k", "projected_ks", "k", "ks")
_OVER_LINE_KEYS = ("over_line", "k_line", "line", "ou_line", "over")
# BPP's CSV occasionally puts its own odds in a "ballparkpal_odds" / "bp"
# column. Some operators relabel them "over_odds" — handle both.
_OVER_ODDS_KEYS = ("over_odds", "over_price", "ballparkpal_odds", "bp_odds", "bp")
_UNDER_ODDS_KEYS = ("under_odds", "under_price", "ballparkpal_under_odds")
_PITCHER_KEYS = ("pitcher", "player", "name")
_TEAM_KEYS = ("team", "tm")
_OPP_KEYS = ("opp", "opponent")


FALLBACK_SOURCE = "ballparkpal_fallback"


def normalize_pitcher_name(value: str | None) -> str:
    """Aggressive normalization for pitcher name matching.

    Strips accents, punctuation, suffixes; lowercases. Also handles
    ``"Last, First"`` by reversing the comma form so we end up with
    ``"first last"``. The MLB StatsAPI sometimes returns ``"J. Ryan"``
    while BPP returns ``"Joe Ryan"`` — the caller pairs us with
    :func:`name_matches_loose` to allow first-initial matches.
    """
    if not value:
        return ""
    text = str(value).strip()
    if "," in text:
        last, first = (part.strip() for part in text.split(",", 1))
        text = f"{first} {last}"
    decoded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", decoded).lower()
    parts = [p for p in cleaned.split() if p not in {"jr", "sr", "ii", "iii", "iv", "v"}]
    return " ".join(parts)


def name_matches_loose(
    candidate: str | None, target: str | None,
) -> bool:
    """Match pitcher names with first-initial + last-name fallback.

    Used at the BPP-fallback boundary because BPP/Polymarket/odds-API
    each disagree on whether the first name is full or initialized.
    Avoids ``SequenceMatcher`` to keep the threshold deterministic;
    callers that need fuzzy matching can layer
    :func:`mlb_prop_odds.names_match` on top.
    """
    cand = normalize_pitcher_name(candidate)
    tgt = normalize_pitcher_name(target)
    if not cand or not tgt:
        return False
    if cand == tgt:
        return True
    cand_parts = cand.split()
    tgt_parts = tgt.split()
    if not cand_parts or not tgt_parts:
        return False
    # Last names match AND either full first names match OR first
    # initial matches. Handles "J. Ryan" ↔ "Joe Ryan" without
    # accidentally matching "John Ryan".
    if cand_parts[-1] != tgt_parts[-1]:
        return False
    cand_first = cand_parts[0]
    tgt_first = tgt_parts[0]
    if cand_first == tgt_first:
        return True
    return cand_first[:1] == tgt_first[:1]


def build_fallback_prop_analysis(
    *,
    pitcher_name: str,
    bpp_row: dict[str, Any],
    timestamp: datetime | None = None,
) -> dict[str, Any] | None:
    """Synthesize a prop-analysis payload from a BallparkPal CSV row.

    Returns ``None`` only when the row is missing the bare minimum
    (projected_k AND over_line) needed to render a card. Everything
    else is filled in with sensible neutrals so the downstream
    ``pitcher_k_edges`` pipeline doesn't crash on missing fields.
    """
    projected_k = _first_float(bpp_row, _PROJECTED_K_KEYS)
    line = _first_float(bpp_row, _OVER_LINE_KEYS)
    if projected_k is None or line is None:
        return None
    over_price = _first_float(bpp_row, _OVER_ODDS_KEYS)
    under_price = _first_float(bpp_row, _UNDER_ODDS_KEYS)
    ts = (timestamp or datetime.utcnow()).isoformat()

    # Build a single synthetic "row" so the upstream odds-row consumers
    # (`prop.get("rows")` check in the edge engine) see a non-empty list.
    synthetic_row = {
        "player_name": str(bpp_row.get("pitcher") or pitcher_name),
        "line": line,
        "over_price": over_price,
        "under_price": under_price,
        "sportsbook": FALLBACK_SOURCE,
        "timestamp": ts,
        "raw": bpp_row,
    }

    return {
        "rows": [synthetic_row],
        "line": line,
        "average_line": line,
        "best_over_line": line,
        "best_over_price": over_price,
        "best_over_book": FALLBACK_SOURCE,
        "best_under_line": line,
        "best_under_price": under_price,
        "best_under_book": FALLBACK_SOURCE,
        "consensus_price": over_price,
        "average_implied_probability": None,
        "line_disagreement": 0.0,
        "line_disagreement_score": 0.0,
        "book_count": 0,
        "movement_direction": None,
        "steam_velocity": None,
        "warnings": ["BallparkPal fallback odds — no live sportsbook coverage"],
        "is_valid": True,
        "validation_reason": "",
        "market_scope": MarketSubtype.PLAYER_PROP.value,
        "normalized_market_name": None,
        # Fallback-specific fields. The edge builder reads these to
        # short-circuit the recent-form computation (we have the
        # projected_k directly from BPP) and the card renderer reads
        # them to label the card as BPP-sourced.
        "source": FALLBACK_SOURCE,
        "ballparkpal_projected_k": projected_k,
        "ballparkpal_team": bpp_row.get("team") or bpp_row.get("tm"),
        "ballparkpal_opponent": bpp_row.get("opp") or bpp_row.get("opponent"),
    }


def k_edge_from_fallback(prop: dict[str, Any]) -> float | None:
    """Compute the signed projection-vs-line K edge.

    Positive = "projected K above the line" (over lean). Negative =
    "projected K below the line" (under lean). Returns ``None`` when
    either input is missing.
    """
    projected = prop.get("ballparkpal_projected_k")
    line = prop.get("line")
    if projected is None or line is None:
        return None
    try:
        return round(float(projected) - float(line), 4)
    except (TypeError, ValueError):
        return None


def _first_float(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    return None


def is_fallback_payload(prop: dict[str, Any] | None) -> bool:
    """Convenience used by the card renderer + diagnostics."""
    return bool(prop) and str(prop.get("source") or "") == FALLBACK_SOURCE
