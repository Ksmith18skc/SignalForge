"""Normalization / join layer between sportsbook MLB edges and wallet-flow markets.

Sportsbook edges (``MlbEdge``) are keyed by ``game_pk`` with full team names and a
decimal total line (e.g. ``10.1``). Tracked-wallet activity lives in Polymarket/Kalshi
``markets`` whose slugs encode the same game::

    mlb-{away}-{home}-{YYYY-MM-DD}-total-{N}pt5
    mlb-{away}-{home}-{YYYY-MM-DD}-spread-{home|away}-{N}pt5
    mlb-{away}-{home}-{YYYY-MM-DD}                       # base / moneyline

This module is pure (no DB) so the join rules can be unit-tested in isolation and reused
by the enrichment service, the alerts engine, and the dashboard.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.services.card_date import parse_slug_date
from app.utils.dashboard_format import TEAM_ABBR

# Full-team-name -> 3-letter slug code (invert the display map). Slugs are lower-case.
ABBR_BY_FULLNAME: dict[str, str] = {name: abbr.lower() for name, abbr in TEAM_ABBR.items()}

# `total-10pt5` -> 10.5 ; `spread-home-1pt5` -> 1.5
_LINE_RE = re.compile(r"^(\d+)pt(\d+)$")


def fullname_to_abbr(team: str | None) -> str | None:
    """Map an MLB StatsAPI full team name to its lower-case slug code."""
    if not team:
        return None
    return ABBR_BY_FULLNAME.get(str(team).strip().lower())


def parse_line_token(token: str | None) -> float | None:
    """`10pt5` -> 10.5, `8pt0` -> 8.0. Returns None for non-line tokens."""
    if not token:
        return None
    match = _LINE_RE.match(token)
    if not match:
        return None
    whole, frac = match.groups()
    try:
        return float(f"{whole}.{frac}")
    except ValueError:
        return None


@dataclass(frozen=True)
class ParsedMarket:
    """Structured view of a wallet-flow market slug."""

    league: str
    event_date: str | None  # ISO YYYY-MM-DD
    away_abbr: str
    home_abbr: str
    market_type: str  # total | spread | moneyline
    line: float | None
    side_hint: str | None  # spread home/away; None for total/moneyline
    slug: str

    def team_pair(self) -> frozenset[str]:
        return frozenset({self.away_abbr, self.home_abbr})


@dataclass(frozen=True)
class NormalizedKey:
    """Normalized join key derived from a sportsbook MLB edge."""

    league: str
    event_date: str | None
    away_abbr: str | None
    home_abbr: str | None
    market_type: str  # total | spread | moneyline
    line: float | None
    side: str | None  # over | under | home | away
    outcome: str | None  # Over | Under | <team> (matches Trade.outcome convention)

    def team_pair(self) -> frozenset[str]:
        return frozenset(a for a in (self.away_abbr, self.home_abbr) if a)


def parse_market_slug(slug: str | None) -> ParsedMarket | None:
    """Parse ``mlb-{away}-{home}-{date}-{type}[-...]`` into a :class:`ParsedMarket`.

    Returns ``None`` when the slug is human-readable / not a recognised game slug.
    """
    if not slug:
        return None
    parts = slug.split("-")
    if len(parts) < 4:
        return None
    league = parts[0].lower()

    # Locate the YYYY-MM-DD triple; teams are the tokens between league and date.
    date_idx: int | None = None
    for i in range(1, len(parts) - 2):
        if (
            re.fullmatch(r"20\d{2}", parts[i] or "")
            and re.fullmatch(r"\d{2}", parts[i + 1] or "")
            and re.fullmatch(r"\d{2}", parts[i + 2] or "")
        ):
            date_idx = i
            break
    if date_idx is None or date_idx < 3:
        # Need at least league + away + home before the date.
        return None

    teams = parts[1:date_idx]
    if len(teams) != 2:
        return None
    away_abbr, home_abbr = teams[0].lower(), teams[1].lower()
    event_date = parse_slug_date(slug)
    event_iso = event_date.isoformat() if event_date else None

    rest = parts[date_idx + 3:]
    market_type = "moneyline"
    line: float | None = None
    side_hint: str | None = None
    if rest[:1] == ["total"]:
        market_type = "total"
        line = parse_line_token(rest[1]) if len(rest) >= 2 else None
    elif rest[:1] == ["spread"]:
        market_type = "spread"
        if len(rest) >= 3:
            side_hint = rest[1].lower()
            line = parse_line_token(rest[2])
        elif len(rest) >= 2:
            line = parse_line_token(rest[1])
    elif rest[:1] == ["moneyline"]:
        market_type = "moneyline"

    return ParsedMarket(
        league=league,
        event_date=event_iso,
        away_abbr=away_abbr,
        home_abbr=home_abbr,
        market_type=market_type,
        line=line,
        side_hint=side_hint,
        slug=slug,
    )


_EDGE_TYPE_TO_MARKET = {
    "game_total": "total",
    "game_spread": "spread",
    "game_moneyline": "moneyline",
}


def normalize_edge(
    edge: dict[str, Any],
    *,
    home_team: str | None,
    away_team: str | None,
) -> NormalizedKey:
    """Build a :class:`NormalizedKey` from an MLB edge dict + its game's teams."""
    edge_type = str(edge.get("edge_type") or "")
    market_type = _EDGE_TYPE_TO_MARKET.get(edge_type, edge_type or "unknown")
    side = (edge.get("side") or "").strip().lower() or None

    outcome: str | None = None
    if market_type == "total" and side in {"over", "under"}:
        outcome = side.title()  # Over | Under — matches Trade.outcome
    elif market_type in {"spread", "moneyline"}:
        # The trade outcome for team-side markets is the full team name.
        if side == "home":
            outcome = home_team
        elif side == "away":
            outcome = away_team

    line = edge.get("line")
    try:
        line_val = float(line) if line is not None else None
    except (TypeError, ValueError):
        line_val = None

    return NormalizedKey(
        league="mlb",
        event_date=edge.get("generated_for_date"),
        away_abbr=fullname_to_abbr(away_team),
        home_abbr=fullname_to_abbr(home_team),
        market_type=market_type,
        line=line_val,
        side=side,
        outcome=outcome,
    )


def keys_match(key: NormalizedKey, market: ParsedMarket, *, line_tol: float = 0.5) -> bool:
    """True when a parsed market slug refers to the same game/market as the edge key."""
    if key.league != market.league:
        return False
    if key.event_date and market.event_date and key.event_date != market.event_date:
        return False
    if key.market_type != market.market_type:
        return False
    # Team pair must match (orientation-tolerant); both abbrs required on the edge side.
    if not key.team_pair() or key.team_pair() != market.team_pair():
        return False
    # Line tolerance only applies when both sides carry a line (totals/spreads).
    if key.line is not None and market.line is not None:
        if abs(key.line - market.line) > line_tol + 1e-9:
            return False
    return True


def outcomes_align(
    key: NormalizedKey,
    *,
    trade_outcome: str | None,
    trade_side: str | None,
) -> str:
    """Classify a wallet trade relative to the edge: aligned | opposing | unrelated.

    A ``SELL`` flips the effective side (selling Over == backing Under).
    """
    edge_outcome = (key.outcome or "").strip().lower()
    t_outcome = (trade_outcome or "").strip().lower()
    if not edge_outcome or not t_outcome:
        return "unrelated"

    same_outcome = edge_outcome == t_outcome
    opposite_outcome = _is_opposite(key.market_type, edge_outcome, t_outcome)
    if not (same_outcome or opposite_outcome):
        return "unrelated"

    is_sell = (trade_side or "").strip().upper() == "SELL"
    backs_outcome = same_outcome ^ is_sell  # SELL flips
    return "aligned" if backs_outcome else "opposing"


def _is_opposite(market_type: str, a: str, b: str) -> bool:
    if market_type == "total":
        return {a, b} == {"over", "under"}
    # For team markets, any two *different* recognised outcomes are opposite sides.
    return a != b


# ---------------------------------------------------------------------------
# URL builders — single source of truth shared with the alerts engine. Return
# None whenever a real URL can't be formed (never fabricate a fake link).
# ---------------------------------------------------------------------------


def market_url_for(slug: str | None, platform: str | None) -> str | None:
    if not slug:
        return None
    if (platform or "").strip().lower() == "kalshi":
        return f"https://kalshi.com/markets/{slug.upper()}"
    return f"https://polymarket.com/event/{slug}"


def trader_profile_url(wallet_address: str | None, platform: str | None = None) -> str | None:
    if not wallet_address:
        return None
    if (platform or "").strip().lower() == "kalshi":
        # Kalshi has no public per-trader profile URL scheme; omit rather than fake.
        return None
    return f"https://polymarketanalytics.com/traders/{wallet_address}"
