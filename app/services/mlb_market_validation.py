"""MLB market validation and normalization helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)


class MarketSubtype(str, Enum):
    FULL_GAME_TOTAL = "FULL_GAME_TOTAL"
    TEAM_TOTAL = "TEAM_TOTAL"
    FIRST_5_TOTAL = "FIRST_5_TOTAL"
    PLAYER_PROP = "PLAYER_PROP"
    ALT_TOTAL = "ALT_TOTAL"
    MONEYLINE = "MONEYLINE"
    SPREAD = "SPREAD"
    NRFI_YRFI = "NRFI_YRFI"
    UNKNOWN = "UNKNOWN"


@dataclass
class ValidationSample:
    reason: str
    market: str | None = None
    line: float | None = None
    book: str | None = None
    side: str | None = None
    scope: str | None = None
    payload: dict[str, Any] | None = None


@dataclass
class ValidationTracker:
    rejected_markets: int = 0
    invalid_odds: int = 0
    malformed_lines: int = 0
    provider_mismatches: int = 0
    updated_at: datetime | None = None
    samples: dict[str, list[ValidationSample]] = field(
        default_factory=lambda: {
            "rejected_markets": [],
            "invalid_odds": [],
            "malformed_lines": [],
            "provider_mismatches": [],
        }
    )

    def _record(self, key: str, sample: ValidationSample) -> None:
        bucket = self.samples.setdefault(key, [])
        bucket.append(sample)
        if len(bucket) > 50:
            del bucket[: len(bucket) - 50]
        self.updated_at = datetime.utcnow()


_TRACKER = ValidationTracker()


def normalize_book_key(value: str | None) -> str:
    if not value:
        return ""
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def trusted_books() -> set[str]:
    settings = get_settings()
    return {normalize_book_key(b) for b in settings.odds_bookmakers.split(",") if b.strip()}


def classify_market_subtype(row: dict[str, Any]) -> MarketSubtype:
    name = str(row.get("market") or "").lower()
    label = str(row.get("label") or "").lower()
    text = f"{name} {label}".strip()
    if not text:
        return MarketSubtype.UNKNOWN
    if "strikeout" in text or "pitcher k" in text or "player prop" in text:
        return MarketSubtype.PLAYER_PROP
    if "nrfi" in text or "yrfi" in text:
        return MarketSubtype.NRFI_YRFI
    if "moneyline" in text:
        return MarketSubtype.MONEYLINE
    if "spread" in text or "run line" in text:
        return MarketSubtype.SPREAD
    if "first 5" in text or "1st 5" in text or "first five" in text:
        return MarketSubtype.FIRST_5_TOTAL
    if "team total" in text or "home total" in text or "away total" in text:
        return MarketSubtype.TEAM_TOTAL
    if "alt" in text or "alternate" in text:
        return MarketSubtype.ALT_TOTAL
    if (
        "total" in text
        or "over/under" in text
        or "over under" in text
        or text in {"ou", "o/u", "over_under"}
    ):
        return MarketSubtype.FULL_GAME_TOTAL
    return MarketSubtype.UNKNOWN


def is_valid_price(price: Any) -> bool:
    try:
        val = float(price)
    except (TypeError, ValueError):
        return False
    return 1.01 <= val <= 10.0


def is_valid_line(scope: MarketSubtype, line: float | None) -> bool:
    if line is None:
        return False
    if scope == MarketSubtype.FULL_GAME_TOTAL:
        return 4.5 <= line <= 15.0
    if scope == MarketSubtype.FIRST_5_TOTAL:
        return 2.5 <= line <= 10.0
    if scope == MarketSubtype.PLAYER_PROP:
        return 0.5 <= line <= 15.0
    return True


def normalized_total_name(
    *,
    scope: MarketSubtype,
    side: str | None,
    line: float | None,
    home: str | None,
    away: str | None,
    label: str | None = None,
) -> str:
    line_label = f"{line:.1f}" if line is not None else "?"
    suffix = f"{side.title()} {line_label}" if side and line is not None else "Total"
    if scope == MarketSubtype.FIRST_5_TOTAL:
        return f"First 5 Innings Total - {suffix}"
    if scope == MarketSubtype.TEAM_TOTAL:
        team = infer_team_from_label(label, home, away)
        name = team or "Team"
        return f"{name} Team Total - {suffix}"
    return f"Full Game Total - {suffix}"


def normalized_prop_name(*, player: str | None, side: str | None, line: float | None) -> str:
    line_label = f"{line:.1f}" if line is not None else "?"
    suffix = f"{side.title()} {line_label}" if side and line is not None else "Strikeouts"
    name = player or "Pitcher"
    return f"{name} Strikeouts - {suffix}"


def infer_team_from_label(label: str | None, home: str | None, away: str | None) -> str | None:
    if not label:
        return None
    label_norm = label.lower()
    for team in (home, away):
        if not team:
            continue
        team_norm = team.lower()
        if team_norm in label_norm:
            return team
    return None


def record_rejection(reason: str, *, row: dict[str, Any] | None = None, scope: MarketSubtype | None = None) -> None:
    _TRACKER.rejected_markets += 1
    _TRACKER._record(
        "rejected_markets",
        ValidationSample(
            reason=reason,
            market=str(row.get("market")) if row else None,
            line=row.get("line") if row else None,
            book=str(row.get("bookmaker")) if row else None,
            scope=scope.value if scope else None,
            payload=row,
        ),
    )
    logger.info(
        "market rejected: %s | market=%s line=%s book=%s scope=%s",
        reason,
        str(row.get("market")) if row else None,
        row.get("line") if row else None,
        str(row.get("bookmaker")) if row else None,
        scope.value if scope else None,
    )


def record_invalid_odds(reason: str, *, row: dict[str, Any] | None = None) -> None:
    _TRACKER.invalid_odds += 1
    _TRACKER._record(
        "invalid_odds",
        ValidationSample(
            reason=reason,
            market=str(row.get("market")) if row else None,
            line=row.get("line") if row else None,
            book=str(row.get("bookmaker")) if row else None,
            payload=row,
        ),
    )
    logger.warning(
        "invalid odds: %s | market=%s line=%s book=%s",
        reason,
        str(row.get("market")) if row else None,
        row.get("line") if row else None,
        str(row.get("bookmaker")) if row else None,
    )


def record_malformed_line(reason: str, *, row: dict[str, Any] | None = None) -> None:
    _TRACKER.malformed_lines += 1
    _TRACKER._record(
        "malformed_lines",
        ValidationSample(
            reason=reason,
            market=str(row.get("market")) if row else None,
            line=row.get("line") if row else None,
            book=str(row.get("bookmaker")) if row else None,
            payload=row,
        ),
    )
    logger.warning(
        "malformed line: %s | market=%s line=%s book=%s",
        reason,
        str(row.get("market")) if row else None,
        row.get("line") if row else None,
        str(row.get("bookmaker")) if row else None,
    )


def record_provider_mismatch(reason: str, *, details: dict[str, Any] | None = None) -> None:
    _TRACKER.provider_mismatches += 1
    _TRACKER._record(
        "provider_mismatches",
        ValidationSample(reason=reason, payload=details),
    )
    logger.warning("provider mismatch: %s | details=%s", reason, details)


def validation_report() -> dict[str, Any]:
    def _samples(key: str) -> list[dict[str, Any]]:
        return [s.__dict__ for s in _TRACKER.samples.get(key, [])]

    return {
        "rejected_markets": _TRACKER.rejected_markets,
        "invalid_odds": _TRACKER.invalid_odds,
        "malformed_lines": _TRACKER.malformed_lines,
        "provider_mismatches": _TRACKER.provider_mismatches,
        "updated_at": _TRACKER.updated_at.isoformat() if _TRACKER.updated_at else None,
        "samples": {
            "rejected_markets": _samples("rejected_markets"),
            "invalid_odds": _samples("invalid_odds"),
            "malformed_lines": _samples("malformed_lines"),
            "provider_mismatches": _samples("provider_mismatches"),
        },
    }
