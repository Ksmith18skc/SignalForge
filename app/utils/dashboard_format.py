"""Display helpers shared by `dashboard.py` and its tests.

Kept free of any Streamlit imports so unit tests can exercise the formatting
logic without pulling the front-end stack. Backend code may also reuse these
helpers (e.g. when formatting Discord summaries), but they are presentation
only — they never mutate edge state.
"""

from __future__ import annotations

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
