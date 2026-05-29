"""Display helpers shared by `dashboard.py` and its tests.

Kept free of any Streamlit imports so unit tests can exercise the formatting
logic without pulling the front-end stack. Backend code may also reuse these
helpers (e.g. when formatting Discord summaries), but they are presentation
only — they never mutate edge state.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Any

DASH = "—"


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
    """Best-effort price parse. Accepts:
      - American odds as int/float: -110, 138
      - American odds as str: "+138", "-120"
      - Decimal odds as float: 1.91, 2.38

    Returns the value as a *float* in its original style; callers decide
    how to interpret it. Returns None for unparseable inputs.
    """
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
    """A bare number is decimal odds when it falls strictly between 1 and 50.
    Anything outside that band is treated as American (e.g. -110, +138, 250)."""
    return 1.0 < value < 50.0


def american_to_implied_probability(price: Any) -> float | None:
    """Convert a sportsbook price to its raw (vig-inclusive) implied
    probability in the [0, 1] range. Returns None on any unparseable input.

    Accepts American (-110, +138), decimal (1.91), or string ("+138")
    representations. Inputs in the ambiguous gap (|x| < 100 and x not in
    a plausible decimal range > 1) are rejected as junk so callers don't
    surface phantom probabilities.
    """
    parsed = _parse_price(price)
    if parsed is None:
        return None
    if _is_decimal_odds(parsed):
        if parsed <= 1.0:
            return None
        return 1.0 / parsed
    # American: outside (1, 50) decimal band, accept only values with
    # |x| >= 100, the conventional American-odds floor.
    if abs(parsed) < 100:
        return None
    if parsed > 0:
        return 100.0 / (parsed + 100.0)
    return (-parsed) / ((-parsed) + 100.0)


def american_from_price(price: Any) -> str | None:
    """Render any parseable price as American odds: '+138' or '-120'.
    Returns None on unparseable input (so callers can decide on the dash).
    """
    parsed = _parse_price(price)
    if parsed is None:
        return None
    if _is_decimal_odds(parsed):
        if parsed <= 1.0:
            return None
        if parsed >= 2.0:
            return f"+{int(round((parsed - 1) * 100))}"
        return f"-{int(round(100 / (parsed - 1)))}"
    # American: |x| < 100 is junk (e.g. 1.0 sneaking past the decimal band).
    if abs(parsed) < 100:
        return None
    sign = "+" if parsed > 0 else "-"
    return f"{sign}{int(round(abs(parsed)))}"


def format_price_with_implied_prob(price: Any) -> str:
    """'+138 (42.0%)' or DASH if price is missing/unparseable."""
    american = american_from_price(price)
    if american is None:
        return DASH
    prob = american_to_implied_probability(price)
    if prob is None:
        return american
    return f"{american} ({prob * 100:.1f}%)"


def format_edge_delta(
    model_value: Any,
    market_value: Any,
    *,
    unit: str = "",
    decimals: int = 1,
) -> str:
    """Signed delta (model - market) like '+1.4 Ks' or '-0.6'. Returns DASH
    when either side is missing."""
    m = _to_float(model_value)
    k = _to_float(market_value)
    if m is None or k is None:
        return DASH
    delta = m - k
    sign = "+" if delta >= 0 else "-"
    body = f"{sign}{abs(delta):.{decimals}f}"
    return f"{body} {unit}".strip()


def format_hit_rate(hits: Any, attempts: Any) -> str:
    """'4/5' when both counts are present. 'insufficient history' otherwise."""
    h = _to_float(hits)
    a = _to_float(attempts)
    if h is None or a is None or a <= 0:
        return "insufficient history"
    return f"{int(h)}/{int(a)}"


# ---------------------------------------------------------------------------
# Score tiering, confidence labels, factor labels
# ---------------------------------------------------------------------------

# Thresholds match `app.services.mlb_edge_scoring.classify_edge`. Kept here so
# tests don't need a DB; if those move, update both.
SCORE_PASS_MAX = 65
SCORE_WATCH_MAX = 75
SCORE_ACTIONABLE_MIN = 65
SCORE_STRONG_MIN = 75
SCORE_HIGH_CONV_MIN = 85


# Score-distribution buckets used in the score-distribution panel.
SCORE_BUCKETS: tuple[tuple[int, int | None, str], ...] = (
    (0, 50, "<50"),
    (50, 55, "50–55"),
    (55, 60, "55–60"),
    (60, 65, "60–65"),
    (65, 70, "65–70"),
    (70, None, "70+"),
)


def score_tier(score: Any) -> str:
    """Plain tier label used by the badge + bar.

    Replaces the older 'Strong Candidate / Bettable / Watch / Pass' wording
    with the trust-first taxonomy the dashboard now exposes:
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
    """Badge color for `score_tier(score)`. Maps to existing CSS classes."""
    tier = score_tier(score)
    return {
        "HIGH CONV": "gold",
        "STRONG": "green",
        "LEAN": "purple",
        "WATCH": "cyan",
        "PASS": "muted",
    }[tier]


def confidence_label(
    score: Any,
    action: Any = None,
    confidence: Any = None,
    *,
    actionable_threshold: float = SCORE_ACTIONABLE_MIN,
    high_conv_threshold: float = SCORE_HIGH_CONV_MIN,
) -> tuple[str, str]:
    """Card-level label that combines score + action.

    Returns (label, kind) where kind is one of muted|green|gold|purple|cyan
    for the badge class.

    Priority (most specific wins):
      score >= high_conv_threshold  -> HIGH CONV   (gold)
      score >= actionable_threshold -> ACTIONABLE WATCH (green)
      action.lower() == 'pass'      -> PASS        (muted)
      action.lower().startswith('watch') -> WATCH SETUP (purple)
      fall through to the score band -> WATCH / LEAN / STRONG / HIGH CONV
    """
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


def confidence_word(confidence: Any) -> tuple[str, str]:
    """Map raw 'low|medium|high|very_high' to public label + badge kind.

    Per spec:
      low       -> WATCH
      medium    -> LEAN
      high      -> STRONG
      very_high -> HIGH CONV
    Anything else returns ("CONF ?", "muted").
    """
    c = str(confidence or "").strip().lower().replace("-", "_").replace(" ", "_")
    table = {
        "low": ("WATCH", "cyan"),
        "medium": ("LEAN", "purple"),
        "med": ("LEAN", "purple"),
        "high": ("STRONG", "green"),
        "very_high": ("HIGH CONV", "gold"),
        "veryhigh": ("HIGH CONV", "gold"),
    }
    return table.get(c, ("CONF ?", "muted"))


# Factor-name renaming — never imply these are probabilities.
FACTOR_LABELS: dict[str, str] = {
    "odds_edge": "Sportsbook price edge",
    "odds_edge_score": "Sportsbook price edge",
    "sportsbook_price_edge": "Sportsbook price edge",
    "movement": "Line movement",
    "line_movement": "Line movement",
    "environment": "Run environment rating",
    "run_environment": "Run environment rating",
    "environment_supports_over": "Run environment rating",
    "k_environment": "K environment rating",
    "pitcher_matchup": "Pitcher matchup",
    "pitcher_recent_form": "Recent form rating",
    "recent_form": "Recent form rating",
    "recent_form_score": "Recent form rating",
    "matchup_k_profile": "Opponent K matchup",
    "smart_money": "Wallet flow signal",
    "data_quality": "Data quality",
    "book_count": "Book count",
}


def factor_label(name: str) -> str:
    """Friendly label for a factor key. Falls back to title-cased name."""
    key = str(name or "").strip().lower()
    if key in FACTOR_LABELS:
        return FACTOR_LABELS[key]
    return str(name or "").replace("_", " ").strip().title() or DASH


def score_bucket_label(score: Any) -> str | None:
    """Return the band label (e.g. '65–70') for a score; None if invalid."""
    s = _to_float(score)
    if s is None:
        return None
    for lo, hi, label in SCORE_BUCKETS:
        if hi is None:
            if s >= lo:
                return label
        elif lo <= s < hi:
            return label
    return None


def score_distribution(scores: list[Any]) -> dict[str, int]:
    """Count how many scores fall in each named bucket."""
    counts: dict[str, int] = {label: 0 for _, _, label in SCORE_BUCKETS}
    for s in scores:
        bucket = score_bucket_label(s)
        if bucket is not None:
            counts[bucket] += 1
    return counts


# ---------------------------------------------------------------------------
# Time helpers
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
    """Single-token relative age: '8s', '4m', '2h', '3d'. DASH if missing.

    Future timestamps render as '0s' rather than negative durations so the
    label stays compact in a metrics strip.
    """
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


# ---------------------------------------------------------------------------
# Misc string helpers
# ---------------------------------------------------------------------------


def odds_provider_label(source: Any) -> tuple[str, bool]:
    """('Odds-API.io' | 'SportsGameOdds', is_fallback). Defaults to primary."""
    raw = str(source or "").strip().lower()
    if "sports" in raw and "game" in raw:
        return ("SportsGameOdds", True)
    if raw in {"sportsgameodds", "sgo"}:
        return ("SportsGameOdds", True)
    return ("Odds-API.io", False)


# ---------------------------------------------------------------------------
# Card titles — the headline on every edge card
# ---------------------------------------------------------------------------


# Team abbreviations for the common cases we see on the slate. Falls back to
# the full team name when not in the map so the title still renders cleanly.
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


def team_short(team: Any) -> str:
    """Return a 2-4 letter team code when known, else the full name."""
    if not team:
        return ""
    key = str(team).strip().lower()
    return TEAM_ABBR.get(key, str(team).strip())


def _side_label(side: Any) -> str:
    s = str(side or "").strip().lower()
    if s in {"over", "o"}:
        return "Over"
    if s in {"under", "u"}:
        return "Under"
    if s in {"home"}:
        return "Home"
    if s in {"away"}:
        return "Away"
    if s in {"yes", "y", "buy"}:
        return s.upper()
    if s in {"no", "n", "sell"}:
        return s.upper()
    return str(side or "").strip().title()


def _line_label(line: Any) -> str | None:
    try:
        f = float(line)
    except (TypeError, ValueError):
        return None
    if abs(f - round(f)) < 1e-9:
        return f"{int(round(f))}"
    return f"{f:.1f}"


def _pitcher_name_from_market(market: str | None) -> str | None:
    """Pull a pitcher name from market strings like 'Joe Ryan Strikeouts -
    Over 6.5'. Returns None when the market doesn't follow the schema."""
    if not market:
        return None
    s = str(market)
    if "Strikeouts" not in s:
        return None
    head = s.split("Strikeouts", 1)[0].strip()
    return head or None


def _resolve_market_line(edge: dict[str, Any]) -> Any:
    """Return the executable market line, never the model projection.

    Fields are checked in order of strongest guarantee that the value
    came from a sportsbook / prediction-market quote, *not* from our
    internal model:

    1. ``market_line`` — explicit field, when callers populate it
    2. ``line`` — the legacy field that *should* be the market line
       (but historically has been overloaded). When ``line`` is a
       non-standard step (e.g. 9.77) we ignore it: real sportsbook
       totals always end in ``.0`` or ``.5``, so any other value is
       almost certainly a projection that leaked into the wrong field.
    3. ``None`` — render the title without a line rather than
       displaying a projection-derived value.
    """
    explicit = edge.get("market_line")
    if explicit is not None:
        return explicit
    candidate = edge.get("line")
    try:
        f = float(candidate)
    except (TypeError, ValueError):
        return candidate
    # Sportsbook totals/spreads land on whole or half steps; a model
    # projection like 9.77 must never reach the card title.
    doubled = f * 2.0
    if abs(doubled - round(doubled)) < 1e-6:
        return candidate
    return None


def format_card_title(edge: dict[str, Any]) -> str:
    """Concise, decision-first headline. Examples:
        'Joe Ryan — Over 6.5 Ks'
        'NYY @ KC — Under 8.5'
        'SEA @ OAK — Moneyline'

    Never returns a dangling hyphen and never inserts a placeholder
    ('?', '—') into the suffix; if a field is missing, that piece is just
    omitted so the title still reads cleanly.
    """
    edge_type = str(edge.get("edge_type") or "").lower()
    side_label = _side_label(edge.get("side"))
    # Title must reflect the *executable* market line, never the model
    # projection — see _resolve_market_line for the half-step guard.
    line_label = _line_label(_resolve_market_line(edge))
    market_scope = str(edge.get("market_scope") or "").lower()
    home = team_short(edge.get("home_team"))
    away = team_short(edge.get("away_team"))
    matchup = f"{away} @ {home}" if home and away else (home or away or "")

    if edge_type == "pitcher_strikeouts":
        name = _pitcher_name_from_market(edge.get("market"))
        head = name or "Pitcher"
        if side_label and line_label:
            return f"{head} — {side_label} {line_label} Ks"
        if side_label:
            return f"{head} — {side_label} Ks"
        return f"{head} — Strikeouts"

    if edge_type == "game_total" or "total" in edge_type or "total" in market_scope:
        if not matchup:
            matchup = _matchup_from_string(edge.get("market"))
        scope_prefix = ""
        if "first_5" in market_scope or "first 5" in market_scope:
            scope_prefix = "F5 "
        elif "team_total" in market_scope or "team total" in market_scope:
            scope_prefix = "Team Total "
        if side_label and line_label:
            return f"{matchup or 'Game'} — {scope_prefix}{side_label} {line_label}".strip()
        # Side present but no usable executable line — better to render
        # "HOU @ TEX — Over" than to silently swap in a model projection
        # or fall back to a generic "Total" that drops the side.
        if side_label:
            return f"{matchup or 'Game'} — {scope_prefix}{side_label}".strip()
        return f"{matchup or 'Game'} — {scope_prefix}Total".strip()

    if "moneyline" in edge_type or "moneyline" in market_scope:
        suffix = f"Moneyline {side_label}".strip() if side_label else "Moneyline"
        return f"{matchup or 'Game'} — {suffix}"

    if "spread" in edge_type or "spread" in market_scope:
        if side_label and line_label:
            return f"{matchup or 'Game'} — Spread {side_label} {line_label}"
        return f"{matchup or 'Game'} — Spread"

    # Fallback: strip dangling hyphens off the raw market string so we
    # never render 'Joe Ryan Strikeouts -' style titles.
    raw = str(edge.get("market") or "").strip()
    raw = raw.rstrip(" -–—·")
    return raw or "Market"


def _matchup_from_string(value: Any) -> str:
    """Best-effort matchup extraction from 'NYY vs KC ...' style strings."""
    if not value:
        return ""
    s = str(value)
    for sep in [" vs ", " @ ", " at "]:
        if sep in s.lower():
            chunk = s.split("-")[0]
            return chunk.strip()
    return ""


# ---------------------------------------------------------------------------
# Polished labels for missing data (per UI spec)
# ---------------------------------------------------------------------------

MISSING_LABELS: dict[str, str] = {
    "projection": "Model projection not yet calibrated",
    "history": "Limited recent sample",
    "hit_rate": "Limited recent sample",
    "clv_pending": "CLV pending",
    "closing": "Awaiting closing line",
    "movement": "Movement data building",
    "form": "History building",
    "factors": "Composition not yet available",
}


def polished_missing(kind: str) -> str:
    """Return a polished, premium-terminal phrasing for a missing data
    section. Falls back to 'Data unavailable' for unknown kinds."""
    return MISSING_LABELS.get(kind, "Data unavailable")


# ---------------------------------------------------------------------------
# Probability formatting for prediction-market and SignalForge sections
# ---------------------------------------------------------------------------


def format_probability(value: Any, *, default: str = DASH) -> str:
    """Render a 0–1 probability as '42.0%'. Inputs already in % (e.g. 42)
    are passed through. Returns the default sentinel when missing."""
    if value is None or value == "":
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if 0 <= f <= 1:
        return f"{f * 100:.1f}%"
    if 0 <= f <= 100:
        return f"{f:.1f}%"
    return default


def format_cents(value: Any, *, default: str = DASH) -> str:
    """Render a prediction-market probability as cents ('34¢'). Used in
    Kalshi/Polymarket lines where the betting interface speaks in cents."""
    if value is None or value == "":
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if 0 < f < 1:
        return f"{int(round(f * 100))}¢"
    if 1 <= f <= 100:
        return f"{int(round(f))}¢"
    return default


def edge_vs_market(model_prob: Any, market_prob: Any) -> str | None:
    """Signed edge string like '+4.2%' (SF probability minus market
    implied). Returns None when either side is missing."""
    if model_prob is None or market_prob is None:
        return None
    try:
        mp = float(model_prob)
        kp = float(market_prob)
    except (TypeError, ValueError):
        return None
    if 0 <= mp <= 1:
        mp *= 100
    if 0 <= kp <= 1:
        kp *= 100
    delta = mp - kp
    sign = "+" if delta >= 0 else "-"
    return f"{sign}{abs(delta):.1f}%"


# ---------------------------------------------------------------------------
# Sharp-money / wallet-flow formatting
# ---------------------------------------------------------------------------


def wallet_alignment_percent(side_count: Any, total_count: Any) -> float | None:
    """Fraction of tracked wallets aligned with the alert's side, 0..100.
    Returns None when either count is missing."""
    s = _to_float(side_count)
    t = _to_float(total_count)
    if s is None or t is None or t <= 0:
        return None
    return min(100.0, max(0.0, 100.0 * s / t))


def format_money_short(value: Any, *, default: str = DASH) -> str:
    """'$95k' / '$1.2M' / '$412k' compact money for sharp-money sections."""
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


# ---------------------------------------------------------------------------
# Consensus-wallet aggregation
#
# A "fill" is one trade/signal row. Smart-money cards must show one row per
# *unique wallet* (aggregating that wallet's many fills), never one row per
# fill — otherwise the same trader appears repeatedly with many sizes.
# ---------------------------------------------------------------------------


def _short_addr(addr: str | None) -> str:
    if not addr:
        return DASH
    if len(addr) <= 14:
        return addr
    return f"{addr[:6]}…{addr[-4:]}"


def _fill_key(fill: dict[str, Any]) -> tuple[str, str, str]:
    """Identity for grouping fills: prefer wallet_address, fall back to
    trader_id, then scope by market_slug + outcome (case-insensitive)."""
    wallet = (fill.get("wallet") or fill.get("wallet_address") or "").strip().lower()
    if not wallet:
        tid = fill.get("trader_id")
        wallet = f"trader:{tid}" if tid is not None else ""
    market = (fill.get("market_slug") or fill.get("market_id") or "").__str__().strip().lower()
    outcome = (fill.get("outcome") or "").strip().lower()
    return (wallet, market, outcome)


def build_consensus_wallets(fills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse per-fill rows into one consensus row per unique wallet.

    Grouped by (wallet_address|trader_id, market_slug, outcome). Aggregates:
      * total_size_usd  – Σ size_usd (gross)
      * avg_entry       – size-weighted average entry price
      * fill_count      – number of fills merged
      * first_seen_at / last_seen_at
      * net_side        – BUY/SELL after netting signed notional
      * net_size_usd    – signed Σ (BUY +, SELL −)

    Sorted by total_size_usd descending.
    """
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str]] = []
    for fill in fills or []:
        key = _fill_key(fill)
        size = _to_float(fill.get("size_usd")) or 0.0
        price = _to_float(fill.get("entry_price") or fill.get("price"))
        side = str(fill.get("side") or "").strip().upper()
        signed = -size if side == "SELL" else size
        ts = fill.get("created_at") or fill.get("timestamp")
        ts_str = str(ts) if ts is not None else ""

        agg = grouped.get(key)
        if agg is None:
            agg = {
                "name": fill.get("trader_nickname")
                or fill.get("name")
                or _short_addr(fill.get("wallet") or fill.get("wallet_address")),
                "wallet_address": fill.get("wallet") or fill.get("wallet_address"),
                "trader_id": fill.get("trader_id"),
                "tier": fill.get("tier"),
                "profile_url": fill.get("trader_url") or fill.get("profile_url"),
                "market_url": fill.get("market_url"),
                "outcome": fill.get("outcome"),
                "total_size_usd": 0.0,
                "_num": 0.0,   # Σ price*size for the weighted average
                "_den": 0.0,
                "net_size_usd": 0.0,
                "fill_count": 0,
                "first_seen_at": ts_str or None,
                "last_seen_at": ts_str or None,
            }
            grouped[key] = agg
            order.append(key)

        agg["total_size_usd"] += size
        agg["net_size_usd"] += signed
        agg["fill_count"] += 1
        if price is not None and size > 0:
            agg["_num"] += price * size
            agg["_den"] += size
        if ts_str:
            if not agg["first_seen_at"] or ts_str < agg["first_seen_at"]:
                agg["first_seen_at"] = ts_str
            if not agg["last_seen_at"] or ts_str > agg["last_seen_at"]:
                agg["last_seen_at"] = ts_str

    rows: list[dict[str, Any]] = []
    for key in order:
        agg = grouped[key]
        den = agg.pop("_den")
        num = agg.pop("_num")
        agg["avg_entry"] = round(num / den, 4) if den else None
        agg["total_size_usd"] = round(agg["total_size_usd"], 2)
        agg["net_side"] = "BUY" if agg["net_size_usd"] >= 0 else "SELL"
        agg["net_size_usd"] = round(agg["net_size_usd"], 2)
        agg["wallet_short"] = _short_addr(agg.get("wallet_address"))
        rows.append(agg)

    rows.sort(key=lambda r: r.get("total_size_usd") or 0.0, reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Edge-card intelligence helpers
#
# Pure derivations shared by the edge card and the watchlist card so the
# "what is this / can I execute it / why trust it / what's the risk" logic
# lives in one tested place.
# ---------------------------------------------------------------------------

# Factor score (0-100) at/above which the sportsbook price edge is "real".
_SPORTSBOOK_EDGE_MIN = 70.0
# Movement factor at/above which we call it a steam move.
_STEAM_MOVE_MIN = 72.0
# Matched-market liquidity below which we flag low liquidity.
_LOW_LIQUIDITY_USD = 2_500.0


def conviction_tier(score: Any) -> tuple[str, str]:
    """(`label`, `kind`) conviction tier for a score — wraps `score_tier`."""
    return score_tier(score), score_tier_kind(score)


def edge_source_stack(edge: dict[str, Any]) -> list[tuple[str, str]]:
    """Top-level signal-source indicators: where the edge comes from.

    Returns ``(label, kind)`` pairs (kind = pill colour). Order reflects
    decision priority: confirmation first, then risk/quality flags.
    """
    factors = edge.get("factors") or {}
    wc = edge.get("wallet_context") or {}
    tags = set(wc.get("tags") or [])
    out: list[tuple[str, str]] = []

    odds_edge = _to_float(factors.get("odds_edge"))
    if odds_edge is not None and odds_edge >= _SPORTSBOOK_EDGE_MIN:
        out.append(("SPORTSBOOK EDGE", "green"))
    if "WALLET CONFIRMED" in tags:
        out.append(("WALLET CONFIRMED", "green"))
    if (wc.get("elite_wallet_disagreement") or 0) > 0:
        out.append(("ELITE DISAGREEMENT", "red"))
    movement = _to_float(factors.get("movement"))
    if movement is not None and movement >= _STEAM_MOVE_MIN:
        out.append(("STEAM MOVE", "purple"))
    execution = wc.get("execution") or {}
    if execution.get("implied_prob") is not None:
        out.append(("PREDICTION-MARKET EDGE", "cyan"))
    if "CROWDED SIDE" in tags:
        out.append(("CROWDED CONSENSUS", "purple"))
    if edge.get("odds_stale"):
        out.append(("STALE ODDS", "red"))
    # Model-only: nothing external corroborates the model's projection.
    sportsbook_ok = odds_edge is not None and odds_edge >= _SPORTSBOOK_EDGE_MIN
    wallet_present = int(wc.get("tracked_wallet_count") or 0) > 0
    if not sportsbook_ok and not wallet_present and not execution.get("implied_prob"):
        out.append(("MODEL ONLY", "muted"))
    return out


def edge_risk_flags(edge: dict[str, Any]) -> list[tuple[str, str]]:
    """Explicit risk indicators present on the edge (only the ones that apply)."""
    factors = edge.get("factors") or {}
    wc = edge.get("wallet_context") or {}
    tags = set(wc.get("tags") or [])
    execution = wc.get("execution") or {}
    out: list[tuple[str, str]] = []

    if edge.get("odds_stale"):
        out.append(("Stale odds", "red"))
    if int(wc.get("tracked_wallet_count") or 0) == 0:
        out.append(("No wallet confirmation", "muted"))
    if "CROWDED SIDE" in tags:
        out.append(("Crowded side", "purple"))
    if (wc.get("elite_wallet_disagreement") or 0) > 0:
        out.append(("Sharp disagreement", "red"))
    liq = _to_float(execution.get("liquidity_usd"))
    if liq is not None and liq < _LOW_LIQUIDITY_USD:
        out.append(("Low liquidity", "purple"))
    if str(edge.get("chase_risk") or "").lower() == "high":
        out.append(("Late adverse movement", "red"))
    return out


def format_score_contributions(
    contributions: dict[str, Any] | None,
    *,
    wallet_adjustment: Any = None,
) -> list[tuple[str, float, str]]:
    """`(label, signed_points, kind)` rows, largest magnitude first.

    ``kind`` is green for positive, red for negative, muted for ~zero. The
    wallet ``confidence_adjustment`` is appended as its own line when nonzero.
    """
    rows: list[tuple[str, float, str]] = []
    for name, value in (contributions or {}).items():
        pts = _to_float(value)
        if pts is None:
            continue
        rows.append((factor_label(name), round(pts, 2), _signed_kind(pts)))
    adj = _to_float(wallet_adjustment)
    if adj is not None and abs(adj) >= 0.01:
        rows.append(("Wallet flow", round(adj, 2), _signed_kind(adj)))
    rows.sort(key=lambda r: abs(r[1]), reverse=True)
    return rows


def _signed_kind(points: float) -> str:
    if points > 0.05:
        return "green"
    if points < -0.05:
        return "red"
    return "muted"


def executable_edge_rows(edge: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per venue with its price + implied probability, for the
    execution section. Missing venues are omitted (no fake quotes)."""
    rows: list[dict[str, Any]] = []
    wc = edge.get("wallet_context") or {}
    execution = wc.get("execution") or {}
    platform = str(execution.get("platform") or "").lower()
    if execution.get("side_price") is not None:
        label = "Kalshi" if platform == "kalshi" else "Polymarket"
        rows.append({
            "venue": label,
            "price": execution.get("side_price"),
            "implied_prob": execution.get("implied_prob"),
            "url": execution.get("market_url"),
        })
    sb_implied = american_to_implied_probability(edge.get("best_price"))
    if sb_implied is not None:
        rows.append({
            "venue": f"Sportsbook ({edge.get('best_book') or 'book'})",
            "price": edge.get("best_price"),
            "implied_prob": sb_implied,
            "url": edge.get("source_url"),
        })
    return rows


def best_executable_edge(edge: dict[str, Any]) -> dict[str, Any] | None:
    """Best edge vs the SignalForge fair probability across venues.

    edge_pct = fair_prob − venue_implied (positive = value). Returns the
    venue with the largest positive edge, or ``None`` when there's no fair
    probability to compare against (honest "uncalibrated" state).
    """
    fair = _to_float(
        edge.get("sf_fair_probability")
        or edge.get("model_probability")
        or edge.get("calibrated_probability")
    )
    if fair is None:
        return None
    best: dict[str, Any] | None = None
    for row in executable_edge_rows(edge):
        implied = _to_float(row.get("implied_prob"))
        if implied is None:
            continue
        edge_pct = round((fair - implied) * 100, 1)
        if best is None or edge_pct > best["edge_pct"]:
            best = {"venue": row["venue"], "edge_pct": edge_pct,
                    "fair_prob": round(fair, 4), "implied_prob": round(implied, 4),
                    "url": row.get("url")}
    return best


def consensus_wallets_chips_html(wallets: list[dict[str, Any]], *, limit: int | None = None) -> str:
    """Render consensus wallets as a single clean HTML chip row.

    All dynamic text is HTML-escaped so trader names can never inject markup,
    and each chip is emitted once (no nested/escaped badge strings). Returns
    an empty string when there is nothing to show.
    """
    if not wallets:
        return ""
    shown = wallets[: limit] if limit else wallets
    chips: list[str] = []
    for w in shown:
        name = html.escape(str(w.get("name") or DASH))
        size = format_money_short(w.get("total_size_usd"))
        fills = int(w.get("fill_count") or 0)
        fill_label = f" ×{fills}" if fills > 1 else ""
        chips.append(
            f"<span class='sf-badge sf-badge-muted'>{name} · {size}{fill_label}</span>"
        )
    return f"<div class='sf-chips'>{''.join(chips)}</div>"


def _as_float_safe(value: Any) -> float | None:
    """Lenient float coercion used by the consensus aggregator."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def wallet_consensus_groups(
    positions: list[dict[str, Any]],
    *,
    min_wallets: int = 2,
) -> list[dict[str, Any]]:
    """Detect markets where >= ``min_wallets`` tracked wallets are on
    the same side.

    Groups by ``(market_id, side, outcome)`` and returns one row per
    group with the participating wallets, total size, mean score, and
    a representative card payload (the highest-score signal in the
    group, with the original trader_nickname / market_url / market_slug
    intact so ``render_wallet_card`` can still render it).

    Pure: no Streamlit dependency, no DB. Tested directly via
    ``tests/test_wallet_consensus.py``.
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
        distinct_wallets = {
            (m.get("wallet") or m.get("trader_nickname") or "")
            for m in members
        }
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
