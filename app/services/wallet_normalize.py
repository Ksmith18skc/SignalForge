"""Canonical-key normalization for tracked-wallet market matching.

The wallet → MLB-edge matcher used to fail silently because vendor slug
formats and our MlbEdge ``normalized_market_name`` formats were never
reconciled. A Polymarket slug like

    mlb-det-cws-2026-05-29-total-9pt5

needs to be rewritten into a sport-aware canonical form like

    mlb:det-cws:2026-05-29:game_total:9.5

so a wallet position can be matched against an MlbEdge that emits the
same key. The key is intentionally human-debuggable — the rejection
debug view shows the parsed key so the operator can spot a normalization
bug at a glance.

This module owns the small parsing primitives plus the
``normalize_market_key`` entry point. It deliberately does not import
the database — the same function runs on a slug string from any source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date as date_cls
from typing import Any

# Wallet provider slugs typically embed a date as YYYY-MM-DD anywhere in
# the slug. We extract from the LEFT-most match because some MLB props
# carry a trailing ``-total-9pt5`` segment whose numerics could otherwise
# fool a too-greedy date regex.
_SLUG_DATE_RE = re.compile(r"(20\d{2})[-_](\d{2})[-_](\d{2})")
# Lines that arrived as ``9pt5`` / ``8pt0`` in slug form get rewritten to
# ``9.5`` / ``8.0`` so they compare equal to numeric lines.
_PT_LINE_RE = re.compile(r"(?P<int>\d+)pt(?P<frac>\d+)")
# A list of league prefixes we explicitly recognize. Anything else falls
# through to the generic ``other`` bucket so unknown sports still get a
# stable key (and therefore still appear in the "Tracked Wallet Live
# Positions" panel, just without sport-specific routing).
_SPORT_PREFIXES = {
    "mlb": "mlb",
    "nba": "nba",
    "nfl": "nfl",
    "nhl": "nhl",
    "wnba": "wnba",
    "atp": "atp",  # tennis ATP
    "wta": "wta",  # tennis WTA
    "ncaaf": "ncaaf",
    "ncaab": "ncaab",
}
# MLB-specific market-scope hints. Matches the slug's TAIL segment after
# the date. Order matters — the longer hint must come before its prefix.
_MLB_MARKET_HINTS = [
    ("first-5-innings-total", "first_5_innings_total"),
    ("f5-total", "first_5_innings_total"),
    ("total", "game_total"),
    ("strikeouts", "pitcher_strikeouts"),
    ("ks", "pitcher_strikeouts"),
]


@dataclass(frozen=True)
class NormalizedMarketKey:
    """Parsed components of a wallet market key.

    ``canonical`` is the human-debuggable string (used as the join key);
    the structured fields let the matcher do partial joins (e.g.
    "same matchup + date even if line differs"). ``raw_slug`` is kept so
    the rejection debug view can show what the matcher actually saw.
    """

    canonical: str
    raw_slug: str
    sport: str | None
    event_date: str | None
    matchup: str | None
    market_subtype: str | None
    line: float | None

    @property
    def matchup_date_key(self) -> str | None:
        """The looser join — used when an MlbEdge for the same matchup
        and date exists but the line doesn't match the wallet's line."""
        if self.sport and self.event_date and self.matchup:
            return f"{self.sport}:{self.matchup}:{self.event_date}"
        return None

    @property
    def sport_date_key(self) -> str | None:
        """The loosest join — used for the wallet-only display when no
        per-event MLB edge exists yet (e.g. ATP markets)."""
        if self.sport and self.event_date:
            return f"{self.sport}:{self.event_date}"
        return None


def normalize_market_key(slug: str | None) -> NormalizedMarketKey | None:
    """Parse a Polymarket-style market slug into a sport-aware key.

    Returns ``None`` only when the input is empty/None. Unknown sports
    and missing dates still produce a key — they get the ``other`` sport
    bucket or a ``None`` event_date, but never silently drop out of the
    pipeline. (The whole point of this module is "don't hide the row
    just because we couldn't normalize it.")
    """
    if not slug:
        return None
    raw = str(slug).strip().lower()
    if not raw:
        return None

    event_date_iso = _extract_date(raw)
    sport = _extract_sport(raw)

    # Strip the sport prefix and the date so the leftover segments form a
    # tidy matchup / market-subtype pair.
    body = raw
    if sport and body.startswith(f"{sport}-"):
        body = body[len(sport) + 1:]
    if event_date_iso:
        body = body.replace(event_date_iso, "")
        body = re.sub(r"-{2,}", "-", body).strip("-")

    market_subtype, after_subtype = _extract_market_subtype(body, sport)
    line = _extract_line(after_subtype)
    matchup = _extract_matchup(after_subtype, market_subtype, line)

    parts: list[str] = [sport or "other"]
    if matchup:
        parts.append(matchup)
    if event_date_iso:
        parts.append(event_date_iso)
    if market_subtype:
        parts.append(market_subtype)
    if line is not None:
        # Format with the minimum precision needed — 9.5 not 9.50.
        line_str = f"{line:g}"
        parts.append(line_str)
    canonical = ":".join(parts) if len(parts) > 1 else parts[0]
    return NormalizedMarketKey(
        canonical=canonical,
        raw_slug=raw,
        sport=sport,
        event_date=event_date_iso,
        matchup=matchup,
        market_subtype=market_subtype,
        line=line,
    )


def _extract_date(slug: str) -> str | None:
    match = _SLUG_DATE_RE.search(slug)
    if not match:
        return None
    try:
        return date_cls(
            int(match.group(1)), int(match.group(2)), int(match.group(3)),
        ).isoformat()
    except ValueError:
        return None


def _extract_sport(slug: str) -> str | None:
    # Match the leading segment up to the first hyphen.
    head, _, _ = slug.partition("-")
    return _SPORT_PREFIXES.get(head)


def _extract_market_subtype(body: str, sport: str | None) -> tuple[str | None, str]:
    """Return ``(subtype, remaining_body)``.

    For MLB we look for the canonical hints (`-total-9pt5`, `-ks-`, etc).
    Other sports fall through to ``None``; the matchup picks up the full
    body so we still get a usable canonical key.
    """
    if sport == "mlb":
        for hint, subtype in _MLB_MARKET_HINTS:
            # The hint can appear as a trailing segment (with a possible
            # ``-9pt5`` line tail) or in the middle.
            pattern = rf"(?:^|-){re.escape(hint)}(?:-|$)"
            m = re.search(pattern, body)
            if m:
                # The remaining body is everything BEFORE the hint plus
                # everything after the line (typically empty), since the
                # hint always sits between the matchup and the line.
                remainder = body[m.end():]
                before = body[:m.start()].strip("-")
                return subtype, f"{before}-{remainder}".strip("-")
    return None, body


def _extract_line(remainder: str) -> float | None:
    """Pull a line value from a remainder like ``9pt5`` / ``9-5`` / ``9.5``."""
    if not remainder:
        return None
    match = _PT_LINE_RE.search(remainder)
    if match:
        try:
            return float(f"{match.group('int')}.{match.group('frac')}")
        except (TypeError, ValueError):
            return None
    # Fall back to a plain decimal anywhere in the remainder.
    dec_match = re.search(r"(\d+\.\d+)", remainder)
    if dec_match:
        try:
            return float(dec_match.group(1))
        except (TypeError, ValueError):
            return None
    return None


def _extract_matchup(remainder: str, subtype: str | None, line: float | None) -> str | None:
    """Best-effort matchup string — tokens that aren't the line or
    subtype hint. Returns None if nothing meaningful is left."""
    if not remainder:
        return None
    text = remainder
    if line is not None:
        text = _PT_LINE_RE.sub("", text)
        text = re.sub(r"\d+\.\d+", "", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    if not text:
        return None
    # Tighten common multi-team slugs — det-cws stays det-cws, but
    # something like "joe-ryan" should also pass through unchanged.
    return text


def looks_like_same_card_date(
    key: NormalizedMarketKey | None, card_date: str | None,
) -> bool:
    """True when the parsed event date matches the card_date, or when no
    date could be parsed (so we err on showing the row rather than
    hiding it). This is the "do not reject for date mismatch" guardrail
    the dashboard uses when filling the Tracked Wallet Live Positions
    panel — it wants false positives over silent drops.
    """
    if key is None or card_date is None:
        return True
    if key.event_date is None:
        return True
    return key.event_date == card_date
