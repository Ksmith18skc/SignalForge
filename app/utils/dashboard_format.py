"""Display helpers shared by `dashboard.py`, `alerts.py`, and their tests.

Kept free of any Streamlit imports so unit tests can exercise the formatting
logic without pulling the front-end stack. These are presentation helpers plus
the pure wallet-consensus aggregator the dashboard renders from.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

DASH = "—"


# Team-name -> short code map. Used by wallet_market_resolver to normalize
# matchup slugs across vendors.
TEAM_ABBR: dict[str, str] = {
    "arizona diamondbacks": "ARI",
    "atlanta braves": "ATL",
    "baltimore orioles": "BAL",
    "boston red sox": "BOS",
    "chicago cubs": "CHC",
    "chicago white sox": "CHW",
    "cincinnati reds": "CIN",
    "cleveland guardians": "CLE",
    "colorado rockies": "COL",
    "detroit tigers": "DET",
    "houston astros": "HOU",
    "kansas city royals": "KC",
    "los angeles angels": "LAA",
    "los angeles dodgers": "LAD",
    "miami marlins": "MIA",
    "milwaukee brewers": "MIL",
    "minnesota twins": "MIN",
    "new york mets": "NYM",
    "new york yankees": "NYY",
    "athletics": "OAK",
    "oakland athletics": "OAK",
    "philadelphia phillies": "PHI",
    "pittsburgh pirates": "PIT",
    "san diego padres": "SD",
    "san francisco giants": "SF",
    "seattle mariners": "SEA",
    "st. louis cardinals": "STL",
    "tampa bay rays": "TB",
    "texas rangers": "TEX",
    "toronto blue jays": "TOR",
    "washington nationals": "WSH",
}


# ---------------------------------------------------------------------------
# Numeric / price helpers
# ---------------------------------------------------------------------------


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_price(value: Any) -> float | None:
    """Best-effort price parse. Accepts American (-110, "+138") or decimal
    (1.91) prices and returns the float in its original style. None if junk."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        s = value.strip().replace(",", "")
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_decimal_odds(value: float) -> bool:
    """A bare number is decimal odds when it falls strictly between 1 and 50."""
    return 1.0 < value < 50.0


def american_to_implied_probability(price: Any) -> float | None:
    """Convert a sportsbook price to its raw implied probability in [0, 1].

    Accepts American (-110, +138), decimal (1.91), or string ("+138"). Inputs in
    the ambiguous gap (|x| < 100 and not a plausible decimal > 1) are rejected.
    """
    parsed = _parse_price(price)
    if parsed is None:
        return None
    if _is_decimal_odds(parsed):
        if parsed <= 1.0:
            return None
        return 1.0 / parsed
    if abs(parsed) < 100:
        return None
    if parsed > 0:
        return 100.0 / (parsed + 100.0)
    return (-parsed) / ((-parsed) + 100.0)


# ---------------------------------------------------------------------------
# Score tiering, confidence labels, factor labels
# ---------------------------------------------------------------------------

SCORE_ACTIONABLE_MIN = 65
SCORE_STRONG_MIN = 75
SCORE_HIGH_CONV_MIN = 85


def score_tier(score: Any) -> str:
    """Plain tier label used by the badge + bar.

      HIGH CONV  >= 85
      STRONG     >= 75
      LEAN       >= 65
      WATCH      >= 55
      PASS       <  55
    """
    s = _to_float(score)
    if s is None:
        return "PASS"
    if s >= SCORE_HIGH_CONV_MIN:
        return "HIGH CONV"
    if s >= SCORE_STRONG_MIN:
        return "STRONG"
    if s >= SCORE_ACTIONABLE_MIN:
        return "LEAN"
    if s >= 55:
        return "WATCH"
    return "PASS"


def score_tier_kind(score: Any) -> str:
    """Badge color for `score_tier(score)`. Maps to CSS classes."""
    return {
        "HIGH CONV": "gold",
        "STRONG": "green",
        "LEAN": "purple",
        "WATCH": "cyan",
        "PASS": "muted",
    }[score_tier(score)]


def confidence_label(
    score: Any,
    action: Any = None,
    confidence: Any = None,
    *,
    actionable_threshold: float = SCORE_ACTIONABLE_MIN,
    high_conv_threshold: float = SCORE_HIGH_CONV_MIN,
) -> tuple[str, str]:
    """Card-level label combining score + action. Returns (label, kind)."""
    s = _to_float(score)
    act = str(action or "").strip().lower()
    if s is not None and s >= high_conv_threshold:
        return ("HIGH CONV", "gold")
    if s is not None and s >= actionable_threshold and act != "pass":
        return ("ACTIONABLE WATCH", "green")
    if act == "pass":
        return ("PASS", "muted")
    if act.startswith("watch"):
        return ("WATCH SETUP", "purple")
    return (score_tier(s), score_tier_kind(s))


# Factor-name renaming for the scoring model factors.
FACTOR_LABELS: dict[str, str] = {
    "wallet_quality": "Wallet quality",
    "multi_wallet_consensus": "Multi-wallet consensus",
    "liquidity": "Liquidity",
    "entry_timing": "Entry timing",
    "price_inefficiency": "Price inefficiency",
    "smart_money": "Wallet flow signal",
}


def factor_label(name: str) -> str:
    """Friendly label for a factor key. Falls back to title-cased name."""
    key = str(name or "").strip().lower()
    if key in FACTOR_LABELS:
        return FACTOR_LABELS[key]
    return str(name or "").replace("_", " ").strip().title() or DASH


# ---------------------------------------------------------------------------
# Time + money helpers
# ---------------------------------------------------------------------------


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def compact_time_ago(value: Any, *, now: datetime | None = None) -> str:
    """Single-token relative age: '8s', '4m', '2h', '3d'. DASH if missing."""
    dt = _parse_dt(value)
    if dt is None:
        return DASH
    base = now or datetime.now(timezone.utc)
    secs = (base - dt).total_seconds()
    if secs < 0:
        secs = 0
    if secs < 60:
        return f"{int(secs)}s"
    if secs < 3600:
        return f"{int(secs // 60)}m"
    if secs < 86400:
        return f"{int(secs // 3600)}h"
    return f"{int(secs // 86400)}d"


def format_money_short(value: Any, *, default: str = DASH) -> str:
    """'$95k' / '$1.2M' / '$412' compact money."""
    n = _to_float(value)
    if n is None:
        return default
    a = abs(n)
    sign = "-" if n < 0 else ""
    if a >= 1_000_000:
        return f"{sign}${a / 1_000_000:.1f}M"
    if a >= 1_000:
        return f"{sign}${a / 1_000:.0f}k"
    return f"{sign}${a:.0f}"


def short_addr(addr: str | None) -> str:
    if not addr:
        return DASH
    if len(addr) <= 14:
        return addr
    return f"{addr[:6]}…{addr[-4:]}"


# ---------------------------------------------------------------------------
# Wallet-to-wallet consensus aggregation
# ---------------------------------------------------------------------------


def _as_float_safe(value: Any) -> float | None:
    """Lenient float coercion used by the consensus aggregator."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _wallet_identity(pos: dict[str, Any]) -> str:
    """Stable identity for a position's wallet across payload shapes."""
    for key in ("wallet", "wallet_address", "wallet_nickname", "trader_nickname"):
        val = (pos.get(key) or "")
        if isinstance(val, str):
            val = val.strip()
        if val:
            return str(val).lower()
    tid = pos.get("trader_id")
    return f"trader:{tid}" if tid is not None else ""


def wallet_consensus_groups(
    positions: list[dict[str, Any]],
    *,
    min_wallets: int = 2,
) -> list[dict[str, Any]]:
    """Detect markets where >= ``min_wallets`` tracked wallets are on the same
    side — pure wallet-to-wallet consensus, independent of any edge or sport.

    Groups by ``(market_id, side, outcome)`` and returns one row per group with
    the participating wallets, total size, mean score, and a representative
    payload (the highest-score member, with its original fields intact so the
    dashboard can render wallet names / market links directly).

    Pure: no Streamlit dependency, no DB. Tested via tests/test_wallet_consensus.py.
    """
    groups: dict[tuple[Any, str, str], list[dict[str, Any]]] = {}
    for pos in positions:
        market_id = pos.get("market_id")
        if not market_id:
            continue
        market_key = (
            market_id,
            (pos.get("side") or "").strip().upper(),
            (pos.get("outcome") or "").strip(),
        )
        groups.setdefault(market_key, []).append(pos)

    consensus: list[dict[str, Any]] = []
    for _key, members in groups.items():
        distinct_wallets = {_wallet_identity(m) for m in members}
        distinct_wallets.discard("")
        if len(distinct_wallets) < min_wallets:
            continue
        members_sorted = sorted(
            members, key=lambda m: (m.get("score") or 0.0), reverse=True,
        )
        rep = dict(members_sorted[0])
        scores = [m.get("score") or 0.0 for m in members]
        sizes = [_as_float_safe(m.get("size_usd")) or 0.0 for m in members]
        rep["consensus_wallets"] = len(distinct_wallets)
        rep["consensus_total_size"] = sum(sizes)
        rep["consensus_mean_score"] = sum(scores) / len(scores) if scores else 0.0
        rep["consensus_members"] = members
        consensus.append(rep)
    consensus.sort(
        key=lambda r: (
            r.get("consensus_wallets") or 0,
            r.get("consensus_total_size") or 0.0,
            r.get("consensus_mean_score") or 0.0,
        ),
        reverse=True,
    )
    return consensus
