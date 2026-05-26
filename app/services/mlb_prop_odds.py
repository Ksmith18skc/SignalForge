"""Dedicated MLB pitcher strikeout prop odds normalization."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from statistics import mean
from typing import Any

SUPPORTED_BOOKS = {"draftkings", "fanduel", "betmgm", "caesars"}
# Substring hints; we lowercase the market name/label/key/type before checking.
# Covers DK's "Pitcher Strikeouts", FD's "Player Strikeouts", BetMGM's
# "Total Strikeouts", and Caesars' "Strikeouts (Pitcher)" variants.
K_MARKET_HINTS = (
    "strikeout",
    "strikeouts",
    "pitcher strikeouts",
    "player strikeouts",
    "pitcher total strikeouts",
    "total strikeouts",
    "strikeouts thrown",
    "pitcher ks",
    "ks thrown",
)
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


@dataclass(frozen=True)
class PitcherPropLine:
    player_name: str
    line: float
    over_price: float | None
    under_price: float | None
    sportsbook: str
    timestamp: datetime
    raw: dict[str, Any]


def normalize_pitcher_strikeout_props(payload: dict[str, Any] | None) -> list[PitcherPropLine]:
    if not payload:
        return []
    out: list[PitcherPropLine] = []
    for sportsbook, markets in (payload.get("bookmakers") or {}).items():
        if _book_key(sportsbook) not in SUPPORTED_BOOKS:
            continue
        if not isinstance(markets, list):
            continue
        for market in markets:
            if not isinstance(market, dict) or not _is_k_market(market):
                continue
            for odds in market.get("odds") or []:
                line = _num(odds.get("hdp") or odds.get("line") or odds.get("point"))
                if line is None:
                    continue
                player = _player_name(odds, market)
                if not player:
                    continue
                out.append(
                    PitcherPropLine(
                        player_name=player,
                        line=line,
                        over_price=_num(odds.get("over") or odds.get("Over")),
                        under_price=_num(odds.get("under") or odds.get("Under")),
                        sportsbook=str(sportsbook),
                        timestamp=_timestamp(market.get("updatedAt") or odds.get("updatedAt")),
                        raw={"market": market, "odds": odds},
                    )
                )
    return out


def consensus_for_pitcher(
    lines: list[PitcherPropLine],
    pitcher_name: str,
) -> dict[str, Any]:
    matched = [line for line in lines if names_match(line.player_name, pitcher_name)]
    warnings: list[str] = []
    if not matched:
        return _empty(f"No pitcher strikeout props found for {pitcher_name}")
    books = sorted({line.sportsbook for line in matched})
    if len(books) < 2:
        warnings.append("Fewer than 2 books available")
    all_lines = [line.line for line in matched]
    over_candidates = [line for line in matched if line.over_price is not None]
    under_candidates = [line for line in matched if line.under_price is not None]
    best_over = max(over_candidates, key=lambda line: line.over_price or 0, default=None)
    best_under = max(under_candidates, key=lambda line: line.under_price or 0, default=None)
    implied = [
        prob
        for line in matched
        for prob in (implied_probability(line.over_price), implied_probability(line.under_price))
        if prob is not None
    ]
    return {
        "rows": [line_to_dict(line) for line in matched],
        "line": round(mean(all_lines), 2),
        "average_line": round(mean(all_lines), 2),
        "best_over_line": best_over.line if best_over else None,
        "best_over_price": best_over.over_price if best_over else None,
        "best_over_book": best_over.sportsbook if best_over else None,
        "best_under_line": best_under.line if best_under else None,
        "best_under_price": best_under.under_price if best_under else None,
        "best_under_book": best_under.sportsbook if best_under else None,
        "consensus_price": round(mean([p for p in [*(line.over_price for line in matched), *(line.under_price for line in matched)] if p is not None]), 4)
        if any(line.over_price is not None or line.under_price is not None for line in matched)
        else None,
        "average_implied_probability": round(mean(implied), 4) if implied else None,
        "line_disagreement": round(max(all_lines) - min(all_lines), 2) if len(all_lines) > 1 else 0.0,
        "line_disagreement_score": line_disagreement_score(all_lines),
        "book_count": len(books),
        "movement_direction": None,
        "steam_velocity": None,
        "warnings": warnings,
    }


def names_match(candidate: str, target: str, *, threshold: float = 0.86) -> bool:
    cand = normalize_name(candidate)
    tgt = normalize_name(target)
    if not cand or not tgt:
        return False
    if cand == tgt:
        return True
    if cand.split()[-1:] == tgt.split()[-1:] and cand[0] == tgt[0]:
        return True
    return SequenceMatcher(None, cand, tgt).ratio() >= threshold


def normalize_name(value: str | None) -> str:
    if not value:
        return ""
    cleaned = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", cleaned).lower()
    parts = [part for part in cleaned.split() if part not in SUFFIXES]
    return " ".join(parts)


def line_disagreement_score(lines: list[float]) -> float:
    if len(lines) < 2:
        return 0.0
    return round(min(100.0, (max(lines) - min(lines)) * 25), 2)


def line_to_dict(line: PitcherPropLine) -> dict[str, Any]:
    return {
        "player_name": line.player_name,
        "line": line.line,
        "over_price": line.over_price,
        "under_price": line.under_price,
        "sportsbook": line.sportsbook,
        "timestamp": line.timestamp.isoformat(),
        "raw": line.raw,
    }


def implied_probability(price: Any) -> float | None:
    val = _num(price)
    if val is None or val <= 1:
        return None
    return round(1 / val, 4)


def _is_k_market(market: dict[str, Any]) -> bool:
    text = " ".join(str(market.get(key) or "") for key in ("name", "label", "type", "key")).lower()
    return any(hint in text for hint in K_MARKET_HINTS)


def _player_name(odds: dict[str, Any], market: dict[str, Any]) -> str | None:
    for key in ("player", "playerName", "participant", "name"):
        value = odds.get(key)
        if value:
            return str(value)
    label = str(odds.get("label") or market.get("label") or "")
    label = re.sub(r"\b(over|under|strikeouts?|pitcher|player|ks?)\b", " ", label, flags=re.I)
    label = re.sub(r"\d+(\.\d+)?", " ", label)
    label = re.sub(r"[^a-zA-Z\s'.-]", " ", label)
    label = label.strip(" .-")
    cleaned = " ".join(label.split())
    return cleaned or None


def _timestamp(value: Any) -> datetime:
    if value:
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            pass
    return datetime.utcnow()


def _book_key(value: str) -> str:
    return re.sub(r"[^a-z]", "", value.lower())


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _empty(warning: str) -> dict[str, Any]:
    return {
        "rows": [],
        "line": None,
        "average_line": None,
        "best_over_line": None,
        "best_over_price": None,
        "best_over_book": None,
        "best_under_line": None,
        "best_under_price": None,
        "best_under_book": None,
        "consensus_price": None,
        "average_implied_probability": None,
        "line_disagreement": 0.0,
        "line_disagreement_score": 0.0,
        "book_count": 0,
        "movement_direction": None,
        "steam_velocity": None,
        "warnings": [warning],
    }
