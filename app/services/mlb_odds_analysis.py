"""Odds normalization and comparison helpers for MLB edges."""

from __future__ import annotations

import logging
import re
from statistics import mean
from typing import Any

from app.providers.odds_api import best_prices, normalize_odds_lines

logger = logging.getLogger(__name__)

# Different sportsbooks call full-game totals different things on Odds-API:
#   DraftKings: "Totals"
#   FanDuel: "Total Points" / "Total"
#   BetMGM: "Game Total"
#   Others observed: "Over/Under", "OVER_UNDER", "Over Under", "Full Game Total"
# We match case-insensitively against an exact name OR an "OU" substring.
TOTAL_MARKET_NAMES = {
    "totals",
    "total",
    "game total",
    "full game total",
    "total points",
    "total runs",
    "game total runs",
    "over/under",
    "over under",
    "over_under",
    "ou",
}

# Substrings that imply "this is a totals/Over-Under market". Used as a
# fallback when the name isn't an exact hit in TOTAL_MARKET_NAMES — covers
# things like "Game Total (Runs)" or "Full-Game Total".
TOTAL_MARKET_PATTERNS = ("total", "over/under", "over under", "over_under")

K_MARKET_HINTS = {
    "strikeouts",
    "pitcher strikeouts",
    "player strikeouts",
    "pitcher strikeouts (k)",
    "total strikeouts",
    "pitcher total strikeouts",
    "strikeouts thrown",
    "ks thrown",
    "ks",
    "pitcher ks",
}


def _market_name_lower(row: dict[str, Any]) -> str:
    return str(row.get("market") or "").strip().lower()


def is_total_market(row: dict[str, Any]) -> bool:
    name = _market_name_lower(row)
    if not name:
        return False
    if name in TOTAL_MARKET_NAMES:
        return True
    # Don't mistake "Team Total" or "1st Half Total" for the full game total.
    if any(prefix in name for prefix in ("team total", "1st half", "first half", "first 5", "1st 5", "alt ")):
        return False
    return any(p in name for p in TOTAL_MARKET_PATTERNS)


def is_pitcher_k_market(row: dict[str, Any]) -> bool:
    name = _market_name_lower(row)
    label = str(row.get("label") or "").lower()
    if not name and not label:
        return False
    return any(h in name or h in label for h in K_MARKET_HINTS)


def analyze_game_totals(odds_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not odds_payload:
        return _empty("Odds missing")
    all_rows = normalize_odds_lines(odds_payload)
    rows = [row for row in all_rows if is_total_market(row)]
    if not rows:
        seen_markets = sorted({str(r.get("market") or "?") for r in all_rows})
        logger.warning(
            "No totals market found in event %s. Markets present: %s",
            odds_payload.get("id"), seen_markets[:10],
        )
        return _empty(
            f"No full-game totals found. Markets present: {seen_markets[:10]}"
        )
    lines = [float(r["line"]) for r in rows if r.get("line") is not None]
    best = best_prices(rows)
    books = sorted({str(r.get("bookmaker")) for r in rows if r.get("bookmaker")})
    stale = [
        str(r.get("bookmaker"))
        for r in rows
        if not r.get("updated_at")
    ]
    return {
        "rows": rows,
        "consensus_total_line": round(mean(lines), 2) if lines else None,
        "best_over_price": (best.get("over") or {}).get("price"),
        "best_over_book": (best.get("over") or {}).get("bookmaker"),
        "best_under_price": (best.get("under") or {}).get("price"),
        "best_under_book": (best.get("under") or {}).get("bookmaker"),
        "consensus_price": _consensus_price(rows),
        "line_disagreement": round(max(lines) - min(lines), 2) if len(lines) > 1 else 0.0,
        "book_count": len(books),
        "stale_book_candidates": stale,
        "movement_direction": None,
        "steam_velocity": None,
        "warnings": [] if len(books) >= 2 else ["Fewer than 2 books available"],
    }


def analyze_pitcher_k_props(
    odds_payload: dict[str, Any] | None,
    *,
    pitcher_name: str | None = None,
) -> dict[str, Any]:
    if not odds_payload:
        return _empty("Pitcher prop odds missing")
    all_rows = normalize_odds_lines(odds_payload)
    rows = [row for row in all_rows if _is_k_prop(row, pitcher_name)]
    if not rows:
        seen_markets = sorted({str(r.get("market") or "?") for r in all_rows})
        logger.warning(
            "No pitcher K market found in event %s for pitcher %r. Markets present: %s",
            odds_payload.get("id"), pitcher_name, seen_markets[:10],
        )
        return _empty(
            f"No pitcher strikeout props found. Markets present: {seen_markets[:10]}"
        )
    lines = [float(r["line"]) for r in rows if r.get("line") is not None]
    best = best_prices(rows)
    books = sorted({str(r.get("bookmaker")) for r in rows if r.get("bookmaker")})
    return {
        "rows": rows,
        "line": round(mean(lines), 2) if lines else None,
        "best_over_price": (best.get("over") or {}).get("price"),
        "best_over_book": (best.get("over") or {}).get("bookmaker"),
        "best_under_price": (best.get("under") or {}).get("price"),
        "best_under_book": (best.get("under") or {}).get("bookmaker"),
        "consensus_price": _consensus_price(rows),
        "line_disagreement": round(max(lines) - min(lines), 2) if len(lines) > 1 else 0.0,
        "book_count": len(books),
        "movement_direction": None,
        "steam_velocity": None,
        "warnings": [] if len(books) >= 2 else ["Fewer than 2 books available"],
    }


def odds_edge_score(analysis: dict[str, Any], side: str) -> float:
    book_count = int(analysis.get("book_count") or 0)
    line_disagreement = float(analysis.get("line_disagreement") or 0.0)
    side_price = analysis.get(f"best_{side.lower()}_price")
    consensus = analysis.get("consensus_price")
    price_bonus = 0.0
    try:
        if side_price is not None and consensus is not None:
            price_bonus = (float(side_price) - float(consensus)) * 25
    except (TypeError, ValueError):
        pass
    return _clamp(50 + min(book_count, 5) * 4 + line_disagreement * 8 + price_bonus)


def movement_score(analysis: dict[str, Any], side: str) -> float:
    direction = str(analysis.get("movement_direction") or "").lower()
    steam = float(analysis.get("steam_velocity") or 0.0)
    if not direction:
        return 50.0
    if side.lower() in direction:
        return _clamp(62 + steam)
    return _clamp(42 - steam)


def _is_k_prop(row: dict[str, Any], pitcher_name: str | None) -> bool:
    if not is_pitcher_k_market(row):
        return False
    if not pitcher_name:
        return True
    label = str(row.get("label") or "").lower()
    # Some providers stamp the player in the market name instead of the label.
    market = str(row.get("market") or "").lower()
    return pitcher_name.lower() in label or pitcher_name.lower() in market


def _consensus_price(rows: list[dict[str, Any]]) -> float | None:
    prices: list[float] = []
    for row in rows:
        outcomes = row.get("outcomes") or {}
        if isinstance(outcomes, dict):
            prices.extend(float(v) for v in outcomes.values() if _is_number(v))
    return round(mean(prices), 4) if prices else None


def _empty(warning: str) -> dict[str, Any]:
    return {
        "rows": [],
        "consensus_total_line": None,
        "line": None,
        "best_over_price": None,
        "best_over_book": None,
        "best_under_price": None,
        "best_under_book": None,
        "consensus_price": None,
        "line_disagreement": 0.0,
        "book_count": 0,
        "stale_book_candidates": [],
        "movement_direction": None,
        "steam_velocity": None,
        "warnings": [warning],
    }


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def summarize_markets(odds_payload: dict[str, Any] | None) -> dict[str, Any]:
    """Inventory the markets in a raw odds payload — powers /mlb/debug/odds/markets.

    Returns the unique market names exactly as the Odds-API returned them,
    plus boolean flags for "did we find a totals market" / "did we find any
    pitcher-K market", and a sample of the first three rows per market.
    """
    if not odds_payload:
        return {
            "event_id": None,
            "market_names": [],
            "has_totals": False,
            "has_pitcher_ks": False,
            "totals_rows": [],
            "pitcher_k_rows": [],
            "samples": [],
        }
    rows = normalize_odds_lines(odds_payload)
    market_names = sorted({str(r.get("market") or "?") for r in rows})
    totals_rows = [r for r in rows if is_total_market(r)]
    k_rows = [r for r in rows if is_pitcher_k_market(r)]
    return {
        "event_id": odds_payload.get("id"),
        "bookmakers": list((odds_payload.get("bookmakers") or {}).keys()),
        "market_names": market_names,
        "has_totals": bool(totals_rows),
        "has_pitcher_ks": bool(k_rows),
        "totals_market_names": sorted({str(r.get("market")) for r in totals_rows}),
        "pitcher_k_market_names": sorted({str(r.get("market")) for r in k_rows}),
        "totals_rows": totals_rows[:5],
        "pitcher_k_rows": k_rows[:5],
        "samples": rows[:3],
    }
