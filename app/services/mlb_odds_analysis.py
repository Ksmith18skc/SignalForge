"""Odds normalization and comparison helpers for MLB edges."""

from __future__ import annotations

from statistics import mean
from typing import Any

from app.providers.odds_api import best_prices, normalize_odds_lines


TOTAL_MARKET_NAMES = {"totals", "total", "game total", "full game total"}
K_MARKET_HINTS = {"strikeouts", "pitcher strikeouts", "player strikeouts"}


def analyze_game_totals(odds_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not odds_payload:
        return _empty("Odds missing")
    rows = [
        row for row in normalize_odds_lines(odds_payload)
        if str(row.get("market") or "").lower() in TOTAL_MARKET_NAMES
        or "total" in str(row.get("market") or "").lower()
    ]
    if not rows:
        return _empty("No full-game totals found")
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
    rows = [
        row for row in normalize_odds_lines(odds_payload)
        if _is_k_prop(row, pitcher_name)
    ]
    if not rows:
        return _empty("No pitcher strikeout props found")
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
    market = str(row.get("market") or "").lower()
    label = str(row.get("label") or "").lower()
    is_k = any(hint in market or hint in label for hint in K_MARKET_HINTS)
    if not pitcher_name:
        return is_k
    return is_k and pitcher_name.lower() in label


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
