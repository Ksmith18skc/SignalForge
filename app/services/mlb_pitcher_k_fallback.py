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
#
# Order matters here: the FIRST matching key wins. ``over_line`` must
# come before ``over`` so a row carrying both (e.g. after a CSV-export
# rename) picks the explicit K-line column rather than an ambiguous
# shorthand that could hold american odds.
_PROJECTED_K_KEYS = (
    "projected_k", "projected_ks", "projected_strikeouts",
    "k", "ks", "proj_k",
)
_OVER_LINE_KEYS = (
    "over_line", "k_line", "strikeout_line", "line", "ou_line", "over",
)
# Over odds. ``bp`` is the cache-canonical key written by the HTML parser.
_OVER_ODDS_KEYS = (
    "over_odds", "over_price", "ballparkpal_odds", "bp_odds", "bp",
)
_UNDER_ODDS_KEYS = (
    "under_odds", "under_price", "ballparkpal_under_odds",
)
_PITCHER_KEYS = ("pitcher", "player", "name")
_TEAM_KEYS = ("team", "tm")
_OPP_KEYS = ("opp", "opponent")


FALLBACK_SOURCE = "ballparkpal_fallback"


# Sanity ranges. Anything outside these is either a parse mistake (an
# american-odds value landed in the K-line column) or a corrupted row.
# We log + reject — we never publish a Top Pitcher K card with a line of
# "174" because the upstream parser swapped columns.
PROJECTED_K_RANGE = (0.0, 15.0)
OVER_LINE_RANGE = (0.5, 15.5)
# American odds typically run -1000 .. +1000 (with the occasional
# unusual longshot outside that band). We accept that range AND the
# decimal-odds range (1.01 .. 10.0) since some scraper tools convert.
AMERICAN_ODDS_RANGE = (-1000.0, 1000.0)
DECIMAL_ODDS_RANGE = (1.01, 10.0)
# Heuristic guards so a value that obviously belongs in the OTHER column
# gets rejected with a named reason instead of producing a 174-line
# card. A K line larger than 20 is almost certainly american odds; an
# odds value with abs < 20 is almost certainly a K line (unless it sits
# inside the decimal-odds band).
LINE_LOOKS_LIKE_ODDS_THRESHOLD = 20.0
ODDS_LOOKS_LIKE_LINE_THRESHOLD = 20.0


class FallbackRejection(Exception):
    """Raised internally when a BPP row fails sanity validation.

    The engine catches this, increments the rejection counter, and
    appends a worked example to the diagnostics payload so the
    operator can see exactly which row was rejected and why.
    """

    def __init__(self, reason: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


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


def validate_bpp_row(
    *, pitcher_name: str, bpp_row: dict[str, Any],
) -> dict[str, Any]:
    """Sanity-check a BPP strikeout row and return the validated values.

    Raises :class:`FallbackRejection` with a named ``reason`` when the
    row's numeric fields fall outside the K-prop-shaped ranges. This is
    the single chokepoint that ensures a card title can never read
    ``"Over 174 Ks"``: 174 fails the K-line range and is rejected with
    ``reason="line_looks_like_american_odds"``.
    """
    projected_k = _first_float(bpp_row, _PROJECTED_K_KEYS)
    line = _first_float(bpp_row, _OVER_LINE_KEYS)
    over_price = _first_float(bpp_row, _OVER_ODDS_KEYS)
    under_price = _first_float(bpp_row, _UNDER_ODDS_KEYS)

    if projected_k is None:
        raise FallbackRejection(
            "projected_k_missing",
            details={"pitcher": pitcher_name},
        )
    if line is None:
        raise FallbackRejection(
            "over_line_missing",
            details={"pitcher": pitcher_name, "projected_k": projected_k},
        )
    if not (PROJECTED_K_RANGE[0] <= projected_k <= PROJECTED_K_RANGE[1]):
        raise FallbackRejection(
            "projected_k_out_of_range",
            details={"pitcher": pitcher_name, "projected_k": projected_k},
        )
    # If the K-line value is large enough to look like american odds,
    # the parser almost certainly swapped over_line ↔ over_odds. Reject
    # the row rather than publish a "Over 174 Ks" card.
    if abs(line) > LINE_LOOKS_LIKE_ODDS_THRESHOLD:
        raise FallbackRejection(
            "line_looks_like_american_odds",
            details={
                "pitcher": pitcher_name,
                "over_line": line,
                "over_odds": over_price,
            },
        )
    if not (OVER_LINE_RANGE[0] <= line <= OVER_LINE_RANGE[1]):
        raise FallbackRejection(
            "over_line_out_of_range",
            details={"pitcher": pitcher_name, "over_line": line},
        )
    # Odds sanity: accept either american (-1000..+1000) or decimal
    # (1.01..10). A value of None is fine — the card will render
    # without a best price.
    if over_price is not None and not _is_valid_odds(over_price):
        raise FallbackRejection(
            "over_odds_out_of_range",
            details={"pitcher": pitcher_name, "over_odds": over_price},
        )
    if under_price is not None and not _is_valid_odds(under_price):
        # Don't reject — just drop the under price. Some CSVs only
        # ship over odds and this should not block the over card.
        under_price = None

    return {
        "projected_k": projected_k,
        "line": line,
        "over_price": over_price,
        "under_price": under_price,
    }


def _is_valid_odds(value: float) -> bool:
    """True when ``value`` is plausibly american OR decimal odds."""
    return (
        AMERICAN_ODDS_RANGE[0] <= value <= AMERICAN_ODDS_RANGE[1]
        and not (
            -ODDS_LOOKS_LIKE_LINE_THRESHOLD < value < ODDS_LOOKS_LIKE_LINE_THRESHOLD
            and not (DECIMAL_ODDS_RANGE[0] <= value <= DECIMAL_ODDS_RANGE[1])
        )
    )


def build_fallback_prop_analysis(
    *,
    pitcher_name: str,
    bpp_row: dict[str, Any],
    timestamp: datetime | None = None,
) -> dict[str, Any] | None:
    """Synthesize a prop-analysis payload from a BallparkPal CSV row.

    Returns ``None`` only when the row fails the sanity validator. The
    caller can recover the named rejection reason by calling
    :func:`validate_bpp_row` directly. Everything else is filled in with
    sensible neutrals so the downstream ``pitcher_k_edges`` pipeline
    doesn't crash on missing fields.
    """
    try:
        validated = validate_bpp_row(pitcher_name=pitcher_name, bpp_row=bpp_row)
    except FallbackRejection:
        return None
    projected_k = validated["projected_k"]
    line = validated["line"]
    over_price = validated["over_price"]
    under_price = validated["under_price"]
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
