"""SignalForge — Institutional sports-betting intelligence terminal.

Streamlit front-end for the SignalForge FastAPI backend. The design target
is "Bloomberg Terminal meets high-end sportsbook trading desk": dense,
decision-first, terminal-styled. No backend logic lives here — every fact
on the page is fetched from a SignalForge endpoint.

Backend URL: $SIGNALFORGE_API_URL (default http://localhost:8000).
Launch: `streamlit run dashboard.py`.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any, Iterable

import httpx
import pandas as pd
import streamlit as st

API_BASE = os.environ.get("SIGNALFORGE_API_URL", "http://localhost:8000").rstrip("/")
DEFAULT_TIMEOUT = 10.0
SCAN_TIMEOUT = 60.0
MLB_RUN_TIMEOUT = 180.0


# =============================================================================
# Streamlit page config + terminal CSS
# =============================================================================

st.set_page_config(
    page_title="SignalForge Terminal",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS tokens deliberately mirror the brand spec (black/charcoal, deep green
# for live/profit, gold for high-conviction, purple for model intelligence,
# red only for risk, cyan only for secondary system info).
TERMINAL_CSS = """
<style>
:root {
  --bg:        #05070A;
  --panel:     #0B0F14;
  --panel-2:   #111827;
  --border:    #263241;
  --text:      #E5E7EB;
  --muted:     #8A94A6;
  --green:     #00C853;
  --green-soft:rgba(0, 200, 83, 0.14);
  --gold:      #D4AF37;
  --gold-soft: rgba(212, 175, 55, 0.16);
  --purple:    #8B5CF6;
  --purple-soft:rgba(139, 92, 246, 0.16);
  --red:       #FF4D5A;
  --red-soft:  rgba(255, 77, 90, 0.16);
  --cyan:      #22D3EE;
}

/* --- Base canvas --- */
.stApp {
  background: var(--bg);
  color: var(--text);
  font-family: 'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace;
}
section[data-testid="stSidebar"] {
  background: var(--panel);
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text); }
section[data-testid="stSidebar"] .sf-meta { color: var(--muted); }

/* Tighten Streamlit's default vertical padding so the terminal feels dense. */
.main .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1500px; }
.stMarkdown p { margin-bottom: 0.3rem; }
.element-container { margin-bottom: 0.4rem; }

h1, h2, h3, h4 { color: var(--text); letter-spacing: 0.02em; }
h1 { font-weight: 700; font-size: 1.6rem; }
h2 { font-weight: 600; font-size: 1.2rem; }
h3 { font-weight: 600; font-size: 1.0rem; text-transform: uppercase; color: var(--muted); letter-spacing: 0.1em; }

/* --- Header strip --- */
.sf-header {
  background: linear-gradient(180deg, var(--panel-2) 0%, var(--panel) 100%);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 14px 18px;
  margin-bottom: 14px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.sf-brand {
  display: flex;
  align-items: baseline;
  gap: 14px;
}
.sf-brand-mark {
  color: var(--gold);
  font-size: 1.6rem;
  font-weight: 800;
  letter-spacing: 0.04em;
}
.sf-brand-name {
  font-size: 1.2rem;
  font-weight: 700;
  color: var(--text);
  letter-spacing: 0.08em;
}
.sf-brand-tagline {
  color: var(--muted);
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
}

/* --- Badges --- */
.sf-badge {
  display: inline-block;
  padding: 3px 9px;
  border-radius: 4px;
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  border: 1px solid;
  margin-right: 4px;
  margin-bottom: 2px;
  vertical-align: middle;
}
.sf-badge-muted   { background: rgba(138, 148, 166, 0.08); color: var(--muted);  border-color: var(--border); }
.sf-badge-green   { background: var(--green-soft);  color: var(--green);  border-color: var(--green);  }
.sf-badge-gold    { background: var(--gold-soft);   color: var(--gold);   border-color: var(--gold);   }
.sf-badge-purple  { background: var(--purple-soft); color: var(--purple); border-color: var(--purple); }
.sf-badge-red     { background: var(--red-soft);    color: var(--red);    border-color: var(--red);    }
.sf-badge-cyan    { background: rgba(34,211,238,0.10); color: var(--cyan); border-color: var(--cyan); }

/* --- Metric strip --- */
div[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.7rem 0.9rem;
}
div[data-testid="stMetric"] label {
  color: var(--muted) !important;
  text-transform: uppercase;
  font-size: 0.65rem !important;
  letter-spacing: 0.12em;
}
div[data-testid="stMetricValue"] {
  color: var(--text) !important;
  font-weight: 700;
  font-size: 1.4rem !important;
}
div[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* --- Cards --- */
.sf-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 14px 16px;
  margin-bottom: 10px;
}
.sf-card.gold   { border-left: 3px solid var(--gold);   }
.sf-card.green  { border-left: 3px solid var(--green);  }
.sf-card.purple { border-left: 3px solid var(--purple); }
.sf-card.red    { border-left: 3px solid var(--red);    }
.sf-card-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  margin-bottom: 6px;
}
.sf-card-title {
  font-size: 1.0rem;
  font-weight: 700;
  color: var(--text);
  letter-spacing: 0.02em;
}
.sf-card-sub {
  color: var(--muted);
  font-size: 0.78rem;
  letter-spacing: 0.04em;
}
.sf-card-row {
  color: var(--text);
  font-size: 0.88rem;
  margin-top: 4px;
}
.sf-card-row .k { color: var(--muted); margin-right: 6px; }
.sf-reasons {
  margin: 6px 0 0 0;
  padding-left: 1.1em;
  color: var(--text);
  font-size: 0.85rem;
}
.sf-reasons li { margin-bottom: 2px; }

/* --- Tables --- */
div[data-testid="stDataFrame"] {
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--panel);
}

/* --- Buttons --- */
.stButton>button {
  background: var(--panel-2);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 4px;
  font-weight: 600;
  letter-spacing: 0.04em;
  padding: 0.45rem 0.9rem;
}
.stButton>button:hover {
  border-color: var(--cyan);
  color: var(--cyan);
}
.stButton>button[kind="primary"] {
  background: var(--green);
  color: #00130a;
  border: 1px solid var(--green);
}
.stButton>button[kind="primary"]:hover {
  background: #00b248;
  color: #00130a;
  border-color: #00b248;
}

/* --- Tabs --- */
button[data-baseweb="tab"] {
  color: var(--muted) !important;
  background: transparent !important;
  font-size: 0.8rem !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  padding-top: 0.6rem !important;
  padding-bottom: 0.6rem !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--text) !important;
  border-bottom: 2px solid var(--gold) !important;
}

/* --- Misc --- */
.sf-meta { color: var(--muted); font-size: 0.78rem; }
.sf-divider {
  height: 1px;
  background: var(--border);
  margin: 12px 0;
}
hr { border-color: var(--border) !important; }
code { background: var(--panel-2); color: var(--cyan); padding: 1px 6px; border-radius: 3px; }
</style>
"""

st.markdown(TERMINAL_CSS, unsafe_allow_html=True)


# =============================================================================
# API client + structured error handling
# =============================================================================


class ApiError(Exception):
    """Backend call failed. Carries enough context to triage from the UI."""

    def __init__(
        self,
        message: str,
        *,
        method: str,
        url: str,
        status_code: int | None = None,
        body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.method = method
        self.url = url
        self.status_code = status_code
        self.body = body

    def short_body(self, limit: int = 800) -> str:
        if not self.body:
            return ""
        return self.body if len(self.body) <= limit else f"{self.body[:limit]}…"

    def suggestion(self) -> str | None:
        """Operator-facing remediation hint for common failure modes."""
        body = (self.body or "").lower()
        sc = self.status_code
        if sc == 429 or "rate" in body or "throttle" in body:
            return "Hit upstream rate limit. Wait a few minutes or use cached data."
        if sc == 500 and "/run-scan" in self.url:
            return "Wallet scanner backend error. Check /health and recent scanner logs."
        if sc == 502 or sc == 503:
            return "Backend unreachable or upstream provider down. Try again shortly."
        if sc == 404:
            return "Endpoint missing — backend may need to be redeployed."
        if sc is None:
            return "Network or timeout (Render cold start can take ~30s on first call)."
        return None


def _client() -> httpx.Client:
    return httpx.Client(base_url=API_BASE, timeout=DEFAULT_TIMEOUT)


def api_get(path: str, params: dict[str, Any] | None = None, timeout: float = DEFAULT_TIMEOUT) -> Any:
    """GET an endpoint and parse JSON, raising ApiError on any failure."""
    url = f"{API_BASE}{path}"
    try:
        with _client() as c:
            r = c.get(path, params=params, timeout=timeout)
    except httpx.HTTPError as exc:
        raise ApiError(f"{type(exc).__name__}: {exc}", method="GET", url=url) from exc
    if r.is_success:
        try:
            return r.json()
        except ValueError as exc:
            raise ApiError(
                f"Non-JSON response: {exc}", method="GET", url=url,
                status_code=r.status_code, body=r.text,
            ) from exc
    raise ApiError(
        f"HTTP {r.status_code}", method="GET", url=url,
        status_code=r.status_code, body=r.text,
    )


def api_post(path: str, json: Any = None, timeout: float = DEFAULT_TIMEOUT) -> Any:
    url = f"{API_BASE}{path}"
    try:
        with _client() as c:
            r = c.post(path, json=json, timeout=timeout)
    except httpx.HTTPError as exc:
        raise ApiError(f"{type(exc).__name__}: {exc}", method="POST", url=url) from exc
    if r.is_success:
        try:
            return r.json()
        except ValueError as exc:
            raise ApiError(
                f"Non-JSON response: {exc}", method="POST", url=url,
                status_code=r.status_code, body=r.text,
            ) from exc
    raise ApiError(
        f"HTTP {r.status_code}", method="POST", url=url,
        status_code=r.status_code, body=r.text,
    )


def render_api_error(err: ApiError, *, prefix: str = "Request failed") -> None:
    """Render a structured failure: prefix + endpoint chip + body + suggestion."""
    status = f"HTTP {err.status_code}" if err.status_code is not None else "transport error"
    st.error(f"{prefix}: {status} from `{err.method} {err.url}`")
    body = err.short_body()
    if body:
        st.code(body, language="json" if body.lstrip().startswith(("{", "[")) else "text")
    else:
        st.caption(str(err))
    tip = err.suggestion()
    if tip:
        st.caption(f"💡 {tip}")


def safe_get(path: str, *, default: Any, params: dict[str, Any] | None = None) -> Any:
    """GET that swallows errors and returns a default — for dashboard renders
    that must keep going even if one panel's endpoint is down. The error gets
    stashed in session_state so the Health tab can surface it."""
    try:
        return api_get(path, params=params)
    except ApiError as exc:
        errors = st.session_state.setdefault("_fetch_errors", {})
        errors[path] = exc
        return default


# =============================================================================
# Formatting + tier helpers
# =============================================================================

DASH = "—"


def fmt_num(value: Any, *, fmt: str = "{:.2f}", default: str = DASH) -> str:
    if value is None or value == "":
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return fmt.format(f)


def fmt_score(value: Any) -> str:
    """Score is already on a 0-100 scale; never apply % formatting."""
    if value is None:
        return DASH
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return DASH


def fmt_pct(value: Any, *, default: str = DASH) -> str:
    if value is None:
        return default
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return default


def fmt_money(value: Any, *, default: str = DASH) -> str:
    if value is None:
        return default
    try:
        return f"${float(value):,.0f}"
    except (TypeError, ValueError):
        return default


def decimal_to_american(value: Any) -> str:
    """Convert a decimal price (e.g. 1.91) to American (-110). Returns DASH if
    not a number. Decimal odds <=1 fall through as DASH."""
    if value is None:
        return DASH
    try:
        d = float(value)
    except (TypeError, ValueError):
        return DASH
    if d <= 1.0:
        return DASH
    if d >= 2.0:
        return f"+{int(round((d - 1) * 100))}"
    return f"-{int(round(100 / (d - 1)))}"


def fmt_price(value: Any) -> str:
    """Prefer American odds when input is decimal-style. Pass-through otherwise."""
    if value is None:
        return DASH
    try:
        d = float(value)
    except (TypeError, ValueError):
        return str(value)
    if 1.0 < d < 50.0:
        return decimal_to_american(d)
    return f"{d:.2f}"


def fmt_dt(value: Any, *, short: bool = True) -> str:
    if not value:
        return DASH
    try:
        dt = (
            value
            if isinstance(value, datetime)
            else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        )
    except (ValueError, TypeError):
        return str(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if short:
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def shorten_wallet(addr: str | None) -> str:
    if not addr:
        return DASH
    if len(addr) <= 14:
        return addr
    return f"{addr[:6]}…{addr[-4:]}"


def badge(text: str, kind: str = "muted") -> str:
    """Inline HTML chip. kind ∈ muted|green|gold|purple|red|cyan."""
    return f'<span class="sf-badge sf-badge-{kind}">{text}</span>'


def tier_for_score(score: Any) -> tuple[str, str]:
    """(label, badge_kind) per the frontend tier spec."""
    if score is None:
        return ("Pass", "muted")
    try:
        s = float(score)
    except (TypeError, ValueError):
        return ("Pass", "muted")
    if s >= 85:
        return ("Strong Candidate", "gold")
    if s >= 75:
        return ("Bettable", "green")
    if s >= 65:
        return ("Watch", "purple")
    return ("Pass", "muted")


def card_kind_for_tier(tier: str) -> str:
    return {
        "Strong Candidate": "gold",
        "Bettable": "green",
        "Watch": "purple",
    }.get(tier, "")


def chase_risk_badge(risk: Any) -> str:
    r = str(risk or "").lower()
    if r == "low":
        return badge("Low Chase", "green")
    if r == "medium":
        return badge("Medium Chase", "purple")
    if r == "high":
        return badge("High Chase", "red")
    return badge("Chase ?", "muted")


def confidence_badge(conf: Any) -> str:
    c = str(conf or "").lower()
    if c == "high":
        return badge("High Conf", "gold")
    if c == "medium":
        return badge("Med Conf", "purple")
    if c == "low":
        return badge("Low Conf", "muted")
    return badge("Conf ?", "muted")


def status_badge(ok: bool, *, ok_label: str, bad_label: str, neutral: bool = False) -> str:
    if neutral:
        return badge(ok_label if ok else bad_label, "cyan")
    return badge(ok_label, "green") if ok else badge(bad_label, "red")


def configured_badge(info: Any, name: str) -> str:
    """Provider status from /health.providers.<name>.

    Three states: not configured (muted), configured + healthy (green),
    configured but failing (red)."""
    if not isinstance(info, dict):
        return badge(f"{name}: ?", "muted")
    if not info.get("configured"):
        return badge(f"{name}: off", "muted")
    healthy = info.get("healthy")
    if healthy is True:
        return badge(f"{name}: live", "green")
    if healthy is False:
        return badge(f"{name}: failing", "red")
    return badge(f"{name}: configured", "cyan")


def matchup_from_market(market: str | None) -> str:
    """Best-effort 'AWAY @ HOME' extraction from a market string like
    'New York Yankees vs Boston Red Sox Over 8.5'. Falls back to first
    8 words to keep card titles short."""
    if not market:
        return DASH
    m = re.split(r"\s+(?:Over|Under|At|@|vs\.?)\s+", market, maxsplit=1)
    head = m[0].strip()
    if " vs " in head.lower():
        return head
    return " ".join(head.split()[:8])


# =============================================================================
# Card renderers
# =============================================================================


def render_edge_card(edge: dict[str, Any]) -> None:
    """Dense MLB edge card — title row, score/price strip, top reasons, action."""
    score = edge.get("score")
    tier, tier_kind = tier_for_score(score)
    card_kind = card_kind_for_tier(tier)

    market = edge.get("market") or DASH
    matchup = matchup_from_market(market)
    side = (edge.get("side") or DASH).title()
    line = edge.get("line")
    line_str = fmt_num(line, fmt="{:.1f}") if line is not None else DASH
    book = edge.get("best_book") or DASH
    price = fmt_price(edge.get("best_price"))
    action = edge.get("action") or DASH
    edge_type = edge.get("edge_type") or ""

    reasons = (edge.get("reasons") or [])[:3]
    warnings = edge.get("warnings") or []
    sources = edge.get("data_sources_used") or []

    badges = " ".join([
        badge(tier, tier_kind),
        confidence_badge(edge.get("confidence")),
        chase_risk_badge(edge.get("chase_risk")),
        badge(edge_type.replace("_", " "), "cyan") if edge_type else "",
    ])
    warn_badges = " ".join(badge(w[:34], "red") for w in warnings[:3])
    source_badges = " ".join(badge(s, "purple") for s in sources[:4])

    reasons_html = (
        "<ul class='sf-reasons'>" +
        "".join(f"<li>{r}</li>" for r in reasons) +
        "</ul>"
    ) if reasons else "<div class='sf-meta'>No supporting reasons returned.</div>"

    body = f"""
    <div class="sf-card {card_kind}">
      <div class="sf-card-head">
        <div>
          <div class="sf-card-title">{matchup}</div>
          <div class="sf-card-sub">{side} {line_str} · {market}</div>
        </div>
        <div style="text-align:right;">
          <div class="sf-card-title" style="color:var(--gold);">{fmt_score(score)}</div>
          <div class="sf-card-sub">{book} · {price}</div>
        </div>
      </div>
      <div>{badges}</div>
      {reasons_html}
      <div class="sf-card-row"><span class="k">Action:</span>{action}</div>
      <div class="sf-card-row" style="margin-top:6px;">{source_badges}{warn_badges}</div>
    </div>
    """
    st.markdown(body, unsafe_allow_html=True)


def render_wallet_card(signal: dict[str, Any]) -> None:
    score = signal.get("score")
    tier, tier_kind = tier_for_score(score)
    card_kind = card_kind_for_tier(tier)

    trader = signal.get("trader_nickname") or DASH
    wallet = shorten_wallet(signal.get("wallet"))
    market = signal.get("market_title") or signal.get("market_slug") or DASH
    side = signal.get("side") or DASH
    outcome = signal.get("outcome") or ""
    entry = fmt_num(signal.get("entry_price"), fmt="{:.3f}")
    size = fmt_money(signal.get("size_usd"))
    source = signal.get("source") or DASH
    reason = signal.get("reason") or ""

    badges_html = " ".join([
        badge(tier, tier_kind),
        badge(f"src: {source}", "purple" if source == "Falcon" else "cyan"),
        badge(side, "green" if side in {"YES", "BUY"} else ("red" if side in {"NO", "SELL"} else "muted")),
    ])

    body = f"""
    <div class="sf-card {card_kind}">
      <div class="sf-card-head">
        <div>
          <div class="sf-card-title">{trader} <span class="sf-card-sub">{wallet}</span></div>
          <div class="sf-card-sub">{market}</div>
        </div>
        <div style="text-align:right;">
          <div class="sf-card-title" style="color:var(--gold);">{fmt_score(score)}</div>
          <div class="sf-card-sub">{side} {outcome} · entry {entry} · {size}</div>
        </div>
      </div>
      <div>{badges_html}</div>
      <div class="sf-card-row sf-meta">{reason}</div>
    </div>
    """
    st.markdown(body, unsafe_allow_html=True)


def render_empty_state(title: str, body: str, *, actions: list[tuple[str, callable]] | None = None) -> None:
    """Friendly empty state — title, body, optional action buttons."""
    st.markdown(
        f"<div class='sf-card'><div class='sf-card-title'>{title}</div>"
        f"<div class='sf-card-row sf-meta'>{body}</div></div>",
        unsafe_allow_html=True,
    )
    if not actions:
        return
    cols = st.columns(len(actions))
    for col, (label, fn) in zip(cols, actions):
        with col:
            if st.button(label, key=f"empty-{label}", use_container_width=True):
                fn()


# =============================================================================
# Action handlers (sidebar + buttons)
# =============================================================================


def action_run_wallet_scan() -> None:
    with st.spinner("Running wallet scan..."):
        try:
            result = api_post("/run-scan", timeout=SCAN_TIMEOUT)
        except ApiError as exc:
            render_api_error(exc, prefix="Wallet scan failed")
            return
    st.toast(
        f"Wallet scan: {result.get('new_signals', 0)} signals, "
        f"{result.get('new_alerts', 0)} alerts, "
        f"{result.get('duration_seconds', 0):.2f}s"
    )
    st.cache_data.clear()
    st.rerun()


def action_run_mlb_edge_scan() -> None:
    with st.spinner("Running MLB edge engine (may take a moment)..."):
        try:
            result = api_post("/mlb/edges/run", timeout=MLB_RUN_TIMEOUT)
        except ApiError as exc:
            render_api_error(exc, prefix="MLB edge scan failed")
            return
    st.success(
        f"MLB scan: {result.get('edges', 0)} edges across {result.get('games', 0)} games "
        f"(odds events: {result.get('odds_events', 0)})."
    )
    st.cache_data.clear()
    st.rerun()


def action_refresh_odds_cache() -> None:
    with st.spinner("Refreshing odds cache..."):
        try:
            result = api_post("/mlb/debug/odds-cache/refresh", timeout=MLB_RUN_TIMEOUT)
        except ApiError as exc:
            render_api_error(exc, prefix="Odds cache refresh failed")
            return
    st.success(
        f"Cache refresh: events={result.get('events_fetched', 0)}, "
        f"odds_calls={result.get('odds_calls', 0)}, "
        f"rate_limited={result.get('rate_limited', 0)}"
    )
    st.cache_data.clear()
    st.rerun()


def action_test_backend() -> None:
    try:
        payload = api_get("/health")
    except ApiError as exc:
        render_api_error(exc, prefix="Backend test failed")
        return
    st.success(
        f"Backend OK · env={payload.get('environment')} · "
        f"db={payload.get('database', {}).get('backend', '?')}"
    )


# =============================================================================
# Cached fetchers
# =============================================================================


@st.cache_data(ttl=10, show_spinner=False)
def fetch_health() -> dict[str, Any] | None:
    try:
        return api_get("/health")
    except ApiError as exc:
        st.session_state.setdefault("_fetch_errors", {})["/health"] = exc
        return None


@st.cache_data(ttl=10, show_spinner=False)
def fetch_summary() -> dict[str, Any]:
    return safe_get("/dashboard-summary", default={})


@st.cache_data(ttl=10, show_spinner=False)
def fetch_traders() -> list[dict[str, Any]]:
    return safe_get("/traders", default=[])


@st.cache_data(ttl=10, show_spinner=False)
def fetch_signals(limit: int = 500) -> list[dict[str, Any]]:
    return safe_get("/signals", default=[], params={"limit": limit})


@st.cache_data(ttl=10, show_spinner=False)
def fetch_alerts(limit: int = 200) -> list[dict[str, Any]]:
    return safe_get("/alerts", default=[], params={"limit": limit})


@st.cache_data(ttl=30, show_spinner=False)
def fetch_mlb_edges(limit: int = 100) -> list[dict[str, Any]]:
    return safe_get("/mlb/edges/today", default=[], params={"limit": limit})


@st.cache_data(ttl=30, show_spinner=False)
def fetch_mlb_daily_card() -> dict[str, Any] | None:
    return safe_get("/mlb/daily-card", default=None)


@st.cache_data(ttl=30, show_spinner=False)
def fetch_mlb_sources() -> dict[str, Any]:
    return safe_get("/mlb/debug/sources", default={})


@st.cache_data(ttl=30, show_spinner=False)
def fetch_odds_cache() -> dict[str, Any]:
    return safe_get("/mlb/debug/odds-cache", default={})


@st.cache_data(ttl=30, show_spinner=False)
def fetch_odds_event_match() -> dict[str, Any]:
    return safe_get("/mlb/debug/odds/event-match", default={})


@st.cache_data(ttl=30, show_spinner=False)
def fetch_pitcher_props(limit: int = 100) -> dict[str, Any]:
    return safe_get("/mlb/debug/pitcher-props", default={"count": 0, "rows": []},
                    params={"limit": limit})


@st.cache_data(ttl=30, show_spinner=False)
def fetch_mlb_performance() -> dict[str, Any]:
    out: dict[str, Any] = {
        "summary": safe_get("/mlb/performance/summary", default={}),
        "by_market": safe_get("/mlb/performance/by-market", default=[]),
        "by_score_band": safe_get("/mlb/performance/by-score-band", default=[]),
        "clv": safe_get("/mlb/performance/clv", default={}),
    }
    return out


# =============================================================================
# Signal → position aggregation (kept from prior dashboard)
# =============================================================================


def _market_label(sig: dict[str, Any]) -> str:
    return sig.get("market_title") or f"market#{sig.get('market_id')}"


def _market_slug(sig: dict[str, Any]) -> str:
    return sig.get("market_slug") or str(sig.get("market_title") or "")


def _market_url(sig: dict[str, Any]) -> str | None:
    slug = _market_slug(sig)
    if not slug:
        return None
    platform = str(sig.get("market_platform") or sig.get("source") or "").lower()
    if "kalshi" in platform:
        return f"https://kalshi.com/markets/{slug.upper()}"
    return f"https://polymarket.com/event/{slug}"


def _trader_url(sig: dict[str, Any]) -> str | None:
    wallet = sig.get("wallet")
    if not wallet:
        return None
    return f"https://polymarketanalytics.com/traders/{wallet}"


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_market_parts(sig: dict[str, Any]) -> dict[str, str]:
    slug = _market_slug(sig)
    parts = slug.split("-") if slug else []
    date_idx = next(
        (
            i for i in range(max(len(parts) - 2, 0))
            if re.fullmatch(r"20\d{2}", parts[i] or "")
            and re.fullmatch(r"\d{2}", parts[i + 1] or "")
            and re.fullmatch(r"\d{2}", parts[i + 2] or "")
        ),
        None,
    )
    if date_idx is None or date_idx < 2:
        return {"league": "", "matchup": _market_label(sig), "contract": "", "event_date": ""}
    league = parts[0].upper()
    teams = [team.upper() for team in parts[1:date_idx]]
    matchup = " vs ".join(teams) if teams else _market_label(sig)
    event_date = "-".join(parts[date_idx:date_idx + 3])
    rest = parts[date_idx + 3:]
    contract = "Moneyline"
    if rest[:1] == ["total"]:
        contract = f"Total {rest[1].replace('pt', '.') if len(rest) > 1 else ''}".strip()
    elif rest[:1] == ["spread"]:
        contract = f"Spread {rest[2].replace('pt', '.') if len(rest) > 2 else ''}".strip()
    return {"league": league, "matchup": matchup, "contract": contract, "event_date": event_date}


def aggregate_signals_to_positions(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple, dict[str, Any]] = {}
    for s in signals:
        key = (
            s.get("source"), s.get("trader_id"), s.get("wallet"),
            s.get("market_id"), _market_label(s), s.get("side"),
            s.get("outcome"), s.get("signal_type"),
            round(_as_float(s.get("entry_price")) or 0.0, 8),
            round(_as_float(s.get("size_usd")) or 0.0, 2),
        )
        prev = latest.get(key)
        if prev is None or str(s.get("created_at") or "") > str(prev.get("created_at") or ""):
            latest[key] = s

    groups: dict[tuple, list[dict[str, Any]]] = {}
    for s in latest.values():
        groups.setdefault(
            (s.get("source"), s.get("trader_id"), s.get("market_id"),
             _market_label(s), s.get("side"), s.get("outcome")),
            [],
        ).append(s)

    positions = []
    for grouped in groups.values():
        first = grouped[0]
        total_size = sum(_as_float(s.get("size_usd")) or 0.0 for s in grouped)
        entries = [(_as_float(s.get("entry_price")), _as_float(s.get("size_usd")) or 0.0) for s in grouped]
        num = sum((e or 0.0) * sz for e, sz in entries if e is not None and sz > 0)
        den = sum(sz for e, sz in entries if e is not None and sz > 0)
        avg_entry = (num / den) if den else None
        latest_sig = max(grouped, key=lambda s: str(s.get("created_at") or ""))
        types = sorted({str(s.get("signal_type")) for s in grouped if s.get("signal_type")})
        position = dict(latest_sig)
        position["score"] = max(_as_float(s.get("score")) or 0.0 for s in grouped)
        position["confidence"] = max(_as_float(s.get("confidence")) or 0.0 for s in grouped)
        position["signal_type"] = " + ".join(types)
        position["entry_price"] = avg_entry
        position["size_usd"] = total_size
        position["signal_count"] = len(grouped)
        position["market_url"] = _market_url(first)
        position["trader_url"] = _trader_url(first)
        position.update(_parse_market_parts(first))
        positions.append(position)
    return sorted(positions, key=lambda p: (p.get("score") or 0.0), reverse=True)


# =============================================================================
# Bail early if backend offline
# =============================================================================

st.session_state.setdefault("_fetch_errors", {})
health = fetch_health()

if health is None:
    st.markdown(
        f"""
        <div class='sf-header'>
          <div class='sf-brand'>
            <span class='sf-brand-mark'>◆</span>
            <span class='sf-brand-name'>SIGNALFORGE</span>
            <span class='sf-brand-tagline'>Prediction Market + MLB Edge Terminal</span>
          </div>
          <div>{badge("BACKEND OFFLINE", "red")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    err = st.session_state["_fetch_errors"].get("/health")
    if isinstance(err, ApiError):
        render_api_error(err, prefix="Cannot reach SignalForge backend")
    st.info(
        f"Backend URL: **{API_BASE}**\n\n"
        "Start it locally with:\n```bash\nuvicorn app.main:app --reload\n```\n"
        "Or set `SIGNALFORGE_API_URL` to your deployed instance."
    )
    st.stop()


# =============================================================================
# Pre-fetch everything once per render
# =============================================================================

summary = fetch_summary() or {}
traders = fetch_traders()
signals_all = fetch_signals(limit=500)
positions_all = aggregate_signals_to_positions(signals_all)
alerts_all = fetch_alerts(limit=200)
mlb_edges_all = fetch_mlb_edges(limit=100)
mlb_daily_card = fetch_mlb_daily_card()
mlb_sources = fetch_mlb_sources()
mlb_performance = fetch_mlb_performance()
odds_cache_payload = fetch_odds_cache()
event_match_payload = fetch_odds_event_match()
pitcher_props_payload = fetch_pitcher_props()

providers_block = health.get("providers", {}) or {}
falcon_info = providers_block.get("falcon", {}) or {}
if isinstance(falcon_info, bool):
    falcon_info = {"configured": falcon_info, "healthy": False, "calls": 0, "successes": 0}
odds_block = providers_block.get("odds_api", {}) or {}
odds_cache_meta = odds_block.get("cache", {}) or {}

# Computed metrics used both in the header strip and the Command Center.
mlb_actionable = [e for e in mlb_edges_all if str(e.get("action") or "").lower() != "pass"]
high_conviction = [e for e in mlb_edges_all if (e.get("score") or 0) >= 85]
missing_odds_edges = [
    e for e in mlb_edges_all
    if any("odds" in str(w).lower() for w in (e.get("warnings") or []))
]
perf_summary = (mlb_performance.get("summary") or {})
clv_block = (mlb_performance.get("clv") or {})


# =============================================================================
# Sidebar — filters + diagnostics
# =============================================================================

with st.sidebar:
    st.markdown(
        "<div class='sf-brand-name' style='font-size:1.05rem;'>◆ SIGNALFORGE</div>"
        "<div class='sf-meta'>Filters & Diagnostics</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='sf-divider'></div>", unsafe_allow_html=True)

    st.markdown("**Signal score**")
    score_min, score_max = st.slider(
        "Score", min_value=0, max_value=100, value=(60, 100), step=5,
        label_visibility="collapsed",
    )

    trader_options = ["(all)"] + sorted({t.get("nickname", "") for t in traders if t.get("nickname")})
    selected_trader = st.selectbox("Trader", trader_options, index=0)

    league_options = ["(all)"] + sorted({p.get("league") for p in positions_all if p.get("league")})
    selected_league = st.selectbox("League", league_options, index=0)

    source_options = ["(all)", "Falcon", "PolymarketAnalytics", "Polycopy", "Mock"]
    selected_source = st.selectbox("Source", source_options, index=0)

    contract_options = ["(all)"] + sorted({p.get("contract") for p in positions_all if p.get("contract")})
    selected_contract = st.selectbox("Contract", contract_options, index=0)

    st.markdown("<div class='sf-divider'></div>", unsafe_allow_html=True)
    st.markdown("**Wallet display**")
    show_full_wallet = st.checkbox("Show full wallet addresses", value=False)
    debug_mode = st.checkbox("Show raw JSON in Debug tab", value=True)

    st.markdown("<div class='sf-divider'></div>", unsafe_allow_html=True)
    st.markdown("**Connection**")
    st.markdown(f"<div class='sf-meta'>API URL: <code>{API_BASE}</code></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sf-meta'>Backend: {status_badge(True, ok_label='ok', bad_label='down')}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='sf-meta'>Last health check: {fmt_dt(health.get('timestamp'))}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='sf-meta'>Rendered: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</div>",
        unsafe_allow_html=True,
    )


def apply_filters(positions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for p in positions:
        score = _as_float(p.get("score")) or 0.0
        if not (score_min <= score <= score_max):
            continue
        if selected_trader != "(all)" and p.get("trader_nickname") != selected_trader:
            continue
        if selected_league != "(all)" and p.get("league") != selected_league:
            continue
        if selected_source != "(all)" and p.get("source") != selected_source:
            continue
        if selected_contract != "(all)" and p.get("contract") != selected_contract:
            continue
        out.append(p)
    return out


filtered_positions = apply_filters(positions_all)


# =============================================================================
# Header — brand, status badges, action bar
# =============================================================================

odds_cache_status = "empty"
oc_metrics = odds_cache_payload.get("metrics") or {}
if odds_cache_payload.get("fresh", 0) > 0:
    odds_cache_status = "fresh"
elif odds_cache_payload.get("rows", 0) > 0:
    odds_cache_status = "stale"

odds_cache_badge = badge(
    f"Odds cache: {odds_cache_status}",
    "green" if odds_cache_status == "fresh" else ("red" if odds_cache_status == "empty" else "purple"),
)
backend_badge = badge("Backend: live", "green")
mlb_count = mlb_sources.get("row_counts", {}).get("mlb_games", 0)
mkt_badge = badge(
    f"MLB games today: {mlb_count}",
    "green" if mlb_count > 0 else "muted",
)

st.markdown(
    f"""
    <div class='sf-header'>
      <div class='sf-brand'>
        <span class='sf-brand-mark'>◆</span>
        <span class='sf-brand-name'>SIGNALFORGE</span>
        <span class='sf-brand-tagline'>Prediction Market + MLB Edge Terminal</span>
      </div>
      <div>
        {backend_badge}{mkt_badge}{odds_cache_badge}
        <span class='sf-meta' style='margin-left:8px;'>Last refresh:
          {fmt_dt(oc_metrics.get('last_refresh_at') or health.get('timestamp'))}
        </span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

action_cols = st.columns([1, 1, 1, 1, 6])
with action_cols[0]:
    if st.button("Run wallet scan", use_container_width=True, type="primary"):
        action_run_wallet_scan()
with action_cols[1]:
    if st.button("Run MLB edge scan", use_container_width=True, type="primary"):
        action_run_mlb_edge_scan()
with action_cols[2]:
    if st.button("Refresh odds cache", use_container_width=True):
        action_refresh_odds_cache()
with action_cols[3]:
    if st.button("Test backend", use_container_width=True):
        action_test_backend()


# Header metric strip — the 7 numbers an operator wants in one glance.
m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
m1.metric("Active positions", len(positions_all))
m2.metric("MLB edges today", len(mlb_edges_all))
m3.metric("High conviction", len(high_conviction))
m4.metric("Avg CLV", fmt_pct(clv_block.get("average_clv_percent")))
m5.metric("Graded edges", perf_summary.get("graded_edges") or 0)
m6.metric(
    "Odds cache",
    odds_cache_status.upper(),
    delta=f"{odds_cache_payload.get('fresh', 0)} fresh",
    delta_color="normal",
)
falcon_calls = int(falcon_info.get("last_scan_calls") or falcon_info.get("calls") or 0)
falcon_ok = int(falcon_info.get("last_scan_successes") or falcon_info.get("successes") or 0)
m7.metric(
    "Falcon",
    f"{falcon_ok}/{falcon_calls}" if falcon_calls else "—",
    delta=("healthy" if falcon_info.get("healthy") else "offline"),
    delta_color="normal" if falcon_info.get("healthy") else "inverse",
)


# =============================================================================
# Tab navigation
# =============================================================================

(
    tab_command,
    tab_mlb,
    tab_wallet,
    tab_perf,
    tab_odds,
    tab_watchlist,
    tab_alerts,
    tab_health,
) = st.tabs([
    "Command Center",
    "MLB Terminal",
    "Wallet Flow",
    "Performance / CLV",
    "Odds Cache",
    "Watchlist",
    "Alerts",
    "Health / Debug",
])


# =============================================================================
# Command Center
# =============================================================================

with tab_command:
    left, right = st.columns([3, 2], gap="medium")

    with left:
        st.markdown("### Today's Top Decisions")
        top_decisions = sorted(
            mlb_actionable, key=lambda e: e.get("score") or 0, reverse=True
        )[:5]
        if top_decisions:
            for edge in top_decisions:
                render_edge_card(edge)
        else:
            render_empty_state(
                "No actionable edges yet.",
                "Odds may be missing or stale, or model confidence is below threshold. "
                "Run an MLB scan or refresh the odds cache.",
                actions=[
                    ("Run MLB edge scan", action_run_mlb_edge_scan),
                    ("Refresh odds cache", action_refresh_odds_cache),
                ],
            )

        st.markdown("### Highest Conviction Wallet Flow")
        top_wallets = [p for p in filtered_positions if (p.get("score") or 0) >= score_min][:5]
        if top_wallets:
            for sig in top_wallets:
                render_wallet_card(sig)
        else:
            render_empty_state(
                "No high-conviction wallet flow.",
                "Lower the score filter in the sidebar, or wait for the next scan pass.",
            )

    with right:
        st.markdown("### System Health")
        provs = providers_block
        st.markdown(
            "<div class='sf-card'>"
            + "<div class='sf-card-row'>"
            + badge(f"DB: {health.get('database', {}).get('backend', '?')}", "cyan")
            + configured_badge(provs.get("falcon"), "Falcon")
            + configured_badge(provs.get("odds_api"), "Odds API")
            + (badge(
                f"Odds cache: {odds_cache_status}",
                "green" if odds_cache_status == "fresh" else ("purple" if odds_cache_status == "stale" else "red"),
              ))
            + configured_badge(provs.get("weather_api"), "Weather")
            + configured_badge(provs.get("mlb_stats_api"), "MLB Stats")
            + (badge("Pybaseball: live OFF", "muted")
               if not (provs.get("pybaseball", {}) or {}).get("allow_live_requests") else
               badge("Pybaseball: live ON", "purple"))
            + "</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown("### Market Pulse")
        rc = (mlb_sources.get("row_counts") or {})
        edges_block_total = len(mlb_edges_all)
        edges_with_warnings = sum(1 for e in mlb_edges_all if e.get("warnings"))
        st.markdown(
            "<div class='sf-card'>"
            + f"<div class='sf-card-row'><span class='k'>MLB games:</span>{rc.get('mlb_games', 0)}</div>"
            + f"<div class='sf-card-row'><span class='k'>Odds snapshots:</span>{rc.get('odds_snapshots', 0)}</div>"
            + f"<div class='sf-card-row'><span class='k'>Pitcher prop snapshots:</span>{rc.get('pitcher_prop_snapshots', 0)}</div>"
            + f"<div class='sf-card-row'><span class='k'>Active edges:</span>{edges_block_total}</div>"
            + f"<div class='sf-card-row'><span class='k'>High conviction:</span>{len(high_conviction)}</div>"
            + f"<div class='sf-card-row'><span class='k'>Missing odds:</span>{len(missing_odds_edges)}</div>"
            + f"<div class='sf-card-row'><span class='k'>With warnings:</span>{edges_with_warnings}</div>"
            + "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("### Recent Alerts")
        recent_sent = [a for a in alerts_all if a.get("status") == "sent"][:5]
        if recent_sent:
            for a in recent_sent:
                channel = a.get("channel") or "?"
                ch_badge = badge(channel, "purple" if channel == "discord" else "cyan")
                st.markdown(
                    "<div class='sf-card'>"
                    + f"<div class='sf-card-row'>{ch_badge}<span class='sf-meta'> · {fmt_dt(a.get('created_at'))}</span></div>"
                    + f"<div class='sf-card-row'>{(a.get('message') or '')[:160]}</div>"
                    + "</div>",
                    unsafe_allow_html=True,
                )
        else:
            render_empty_state(
                "No alerts dispatched yet.",
                "Alerts post automatically when a signal exceeds the discord threshold.",
            )


# =============================================================================
# MLB Terminal
# =============================================================================

with tab_mlb:
    # Daily card hero — three columns of edge cards.
    st.markdown("### Daily Card")
    if mlb_daily_card:
        hero_cols = st.columns(3, gap="medium")
        sections = [
            ("Top Game Totals", mlb_daily_card.get("top_game_totals") or []),
            ("Top Pitcher Ks", mlb_daily_card.get("top_pitcher_strikeouts") or []),
            ("Near Misses", mlb_daily_card.get("near_misses") or []),
        ]
        for col, (title, rows) in zip(hero_cols, sections):
            with col:
                st.markdown(f"<h3>{title}</h3>", unsafe_allow_html=True)
                if not rows:
                    render_empty_state(
                        "Empty",
                        "No qualifying edges in this band yet for today's slate.",
                    )
                else:
                    for row in rows[:3]:
                        render_edge_card(row)
    else:
        render_empty_state(
            "No daily card built yet.",
            "Run the MLB edge scan to materialize a card for today.",
            actions=[("Run MLB edge scan", action_run_mlb_edge_scan)],
        )

    st.markdown("### All Edges")
    if mlb_edges_all:
        df_edges = pd.DataFrame([
            {
                "score": e.get("score"),
                "tier": tier_for_score(e.get("score"))[0],
                "action": e.get("action"),
                "confidence": e.get("confidence"),
                "type": (e.get("edge_type") or "").replace("_", " "),
                "market": e.get("market"),
                "side": (e.get("side") or "").title(),
                "line": e.get("line"),
                "best_book": e.get("best_book") or DASH,
                "best_price": fmt_price(e.get("best_price")),
                "chase_risk": e.get("chase_risk"),
                "warnings": "; ".join((e.get("warnings") or [])[:3]),
            }
            for e in mlb_edges_all
        ])
        df_edges = df_edges.fillna(DASH)
        st.dataframe(
            df_edges,
            use_container_width=True,
            hide_index=True,
            height=min(420, 60 + 32 * max(len(df_edges), 1)),
            column_config={
                "score": st.column_config.NumberColumn("score", format="%.1f"),
                "line": st.column_config.NumberColumn("line", format="%.1f"),
                "best_price": st.column_config.TextColumn("best_price (US)"),
            },
        )
    else:
        render_empty_state(
            "No MLB edges available.",
            "Run the MLB edge scan; if it still shows nothing, check the Odds Cache tab.",
            actions=[("Run MLB edge scan", action_run_mlb_edge_scan)],
        )

    st.markdown("### Data Quality")
    dq = (mlb_daily_card or {}).get("data_quality_summary") or {}
    rc = (mlb_sources.get("row_counts") or {})
    pp_rows = pitcher_props_payload.get("count") if isinstance(pitcher_props_payload, dict) else 0
    statcast_rows = (
        rc.get("pitcher_prop_snapshots", 0)
        + rc.get("pitcher_prop_odds_snapshots", 0)
    )
    cards = [
        ("Edge count", dq.get("edge_count", len(mlb_edges_all))),
        ("High confidence", dq.get("high_confidence", sum(1 for e in mlb_edges_all if e.get("confidence") == "high"))),
        ("With warnings", dq.get("with_warnings", sum(1 for e in mlb_edges_all if e.get("warnings")))),
        ("Missing odds", dq.get("missing_odds", len(missing_odds_edges))),
        ("Pitcher prop rows", pp_rows),
        ("Odds snapshots", rc.get("odds_snapshots", 0)),
        ("Statcast/Prop", statcast_rows),
        ("MLB games", rc.get("mlb_games", 0)),
    ]
    dq_cols = st.columns(len(cards))
    for col, (label, value) in zip(dq_cols, cards):
        col.metric(label, value)


# =============================================================================
# Wallet Flow
# =============================================================================

with tab_wallet:
    st.markdown("### Highest Conviction Positions")
    top_positions = filtered_positions[:5]
    if top_positions:
        for pos in top_positions:
            render_wallet_card(pos)
    else:
        render_empty_state(
            "No positions match your filters.",
            "Lower the score range or clear trader/league filters in the sidebar.",
        )

    st.markdown(f"### All Positions ({len(filtered_positions)})")
    if filtered_positions:
        rows = []
        for s in filtered_positions:
            wallet_disp = s.get("wallet") if show_full_wallet else shorten_wallet(s.get("wallet"))
            tier = tier_for_score(s.get("score"))[0]
            rows.append({
                "score": s.get("score") or 0.0,
                "tier": tier,
                "trader": s.get("trader_nickname") or DASH,
                "wallet": wallet_disp or DASH,
                "league": s.get("league") or DASH,
                "matchup": s.get("matchup") or _market_label(s),
                "contract": s.get("contract") or DASH,
                "side": s.get("side") or DASH,
                "outcome": s.get("outcome") or DASH,
                "avg_entry": s.get("entry_price"),
                "size_usd": s.get("size_usd"),
                "events": s.get("signal_count") or 1,
                "source": s.get("source") or DASH,
                "market": s.get("market_url"),
            })
        df_pos = pd.DataFrame(rows).fillna(DASH)
        st.dataframe(
            df_pos,
            use_container_width=True,
            hide_index=True,
            height=min(560, 60 + 32 * len(df_pos)),
            column_config={
                "score": st.column_config.ProgressColumn(
                    "score", min_value=0, max_value=100, format="%.1f"
                ),
                "avg_entry": st.column_config.NumberColumn("avg entry", format="%.3f"),
                "size_usd": st.column_config.NumberColumn("size USD", format="$%.0f"),
                "market": st.column_config.LinkColumn("market", display_text="open"),
            },
        )
    else:
        render_empty_state(
            "No positions in the current filter set.",
            "The scanner emits one signal per pass; check back after the next scan.",
        )


# =============================================================================
# Performance / CLV
# =============================================================================

with tab_perf:
    graded = perf_summary.get("graded_edges") or 0
    if not graded:
        render_empty_state(
            "Performance tracking is empty.",
            "Tracking begins after closing lines and game results are graded. "
            "Run `scripts/grade_mlb_results.py` and `scripts/update_mlb_closing_lines.py` "
            "to seed this view.",
        )
    else:
        pc1, pc2, pc3, pc4, pc5, pc6, pc7 = st.columns(7)
        pc1.metric("Graded edges", graded)
        pc2.metric("Win rate", fmt_pct(perf_summary.get("win_rate")))
        pc3.metric("ROI units", fmt_num(perf_summary.get("roi_units"), fmt="{:+.2f}"))
        pc4.metric("Avg CLV", fmt_pct(clv_block.get("average_clv_percent")))
        pc5.metric("Positive CLV rate", fmt_pct(clv_block.get("positive_clv_rate")))
        by_market = mlb_performance.get("by_market") or []
        best = max(by_market, key=lambda m: m.get("roi_units") or -999, default=None)
        worst = min(by_market, key=lambda m: m.get("roi_units") or 999, default=None)
        pc6.metric("Best market", str((best or {}).get("edge_type") or DASH))
        pc7.metric("Worst market", str((worst or {}).get("edge_type") or DASH))

        st.markdown("### ROI by edge type")
        if by_market:
            df_market = pd.DataFrame(by_market).fillna(DASH)
            st.dataframe(df_market, use_container_width=True, hide_index=True,
                         height=min(280, 60 + 32 * len(df_market)))
            try:
                chart_df = pd.DataFrame(by_market).set_index("edge_type")[["roi_units"]]
                if not chart_df.empty:
                    st.bar_chart(chart_df, height=220)
            except Exception:
                pass
        else:
            render_empty_state("No edge-type breakdown yet.", "Need at least one graded edge per type.")

        st.markdown("### By score band")
        by_band = mlb_performance.get("by_score_band") or []
        if by_band:
            df_band = pd.DataFrame(by_band).fillna(DASH)
            st.dataframe(df_band, use_container_width=True, hide_index=True,
                         height=min(220, 60 + 32 * len(df_band)))
        else:
            render_empty_state("No score-band breakdown yet.", "Grade more edges to populate.")

        st.markdown("### CLV leaders")
        clv_cols = st.columns(2)
        with clv_cols[0]:
            st.markdown("**Top positive CLV**")
            top_pos = clv_block.get("top_positive") or []
            if top_pos:
                st.dataframe(pd.DataFrame(top_pos).fillna(DASH), use_container_width=True,
                             hide_index=True, height=min(280, 60 + 32 * len(top_pos)))
            else:
                st.caption("None")
        with clv_cols[1]:
            st.markdown("**Top negative CLV**")
            top_neg = clv_block.get("top_negative") or []
            if top_neg:
                st.dataframe(pd.DataFrame(top_neg).fillna(DASH), use_container_width=True,
                             hide_index=True, height=min(280, 60 + 32 * len(top_neg)))
            else:
                st.caption("None")

        st.markdown("### Factor attribution")
        factors = perf_summary.get("top_factors_by_performance") or []
        if factors:
            st.dataframe(pd.DataFrame(factors).fillna(DASH), use_container_width=True,
                         hide_index=True, height=min(280, 60 + 32 * len(factors)))
        else:
            st.caption("No factor attribution available.")


# =============================================================================
# Odds Cache
# =============================================================================

with tab_odds:
    if not odds_cache_payload:
        render_empty_state(
            "Odds cache empty or backend unreachable.",
            "Run a refresh to populate the cache.",
            actions=[("Refresh odds cache", action_refresh_odds_cache)],
        )
    else:
        metrics = odds_cache_payload.get("metrics") or {}
        oc1, oc2, oc3, oc4, oc5, oc6, oc7 = st.columns(7)
        oc1.metric("Cache rows", odds_cache_payload.get("rows", 0))
        oc2.metric("Fresh", odds_cache_payload.get("fresh", 0))
        oc3.metric("Stale", odds_cache_payload.get("stale", 0))
        oc4.metric("Live API calls", metrics.get("live_api_calls", 0))
        oc5.metric("Cache hits", metrics.get("cache_hits", 0))
        oc6.metric("Avoided calls", metrics.get("avoided_api_calls", 0))
        oc7.metric("Rate-limited (429)", metrics.get("rate_limited_count", 0))

        last_refresh = metrics.get("last_refresh_at")
        last_err = metrics.get("last_refresh_error")
        st.markdown(
            "<div class='sf-card'>"
            + f"<div class='sf-card-row'><span class='k'>Status:</span>"
            + (badge("FRESH", "green") if odds_cache_status == "fresh"
               else (badge("STALE", "purple") if odds_cache_status == "stale"
                     else badge("EMPTY", "red")))
            + "</div>"
            + f"<div class='sf-card-row'><span class='k'>Last refresh:</span>{fmt_dt(last_refresh)}</div>"
            + f"<div class='sf-card-row'><span class='k'>Last 429 at:</span>{fmt_dt(metrics.get('last_rate_limited_at'))}</div>"
            + f"<div class='sf-card-row'><span class='k'>Stale fallbacks served:</span>{metrics.get('stale_fallbacks', 0)}</div>"
            + (f"<div class='sf-card-row'>{badge('Last error', 'red')} <span class='sf-meta'>{last_err}</span></div>"
               if last_err else "")
            + "</div>",
            unsafe_allow_html=True,
        )

        action_cols2 = st.columns([1, 1, 1, 4])
        with action_cols2[0]:
            if st.button("Refresh now", use_container_width=True, type="primary"):
                action_refresh_odds_cache()
        with action_cols2[1]:
            if st.button("Reload events", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        with action_cols2[2]:
            if st.button("Reload matches", use_container_width=True):
                st.cache_data.clear()
                st.rerun()

        st.markdown("### By market type")
        by_type = odds_cache_payload.get("by_market_type") or {}
        if by_type:
            df_types = pd.DataFrame(
                [{"market_type": k, "rows": v} for k, v in by_type.items()]
            )
            st.dataframe(df_types, use_container_width=True, hide_index=True,
                         height=min(200, 60 + 32 * len(df_types)))
        else:
            st.caption("No cached rows yet.")

        # Event-match table — the operator's answer to "why are odds missing?"
        st.markdown("### Event Match Table")
        if event_match_payload and event_match_payload.get("matches"):
            df_match = pd.DataFrame([
                {
                    "game_pk": m.get("game_pk"),
                    "home": m.get("home_team"),
                    "away": m.get("away_team"),
                    "matched_event": m.get("matched_event_id") or DASH,
                    "strength": m.get("match_strength"),
                    "reason": m.get("reason"),
                }
                for m in event_match_payload.get("matches") or []
            ])
            st.dataframe(df_match.fillna(DASH), use_container_width=True, hide_index=True,
                         height=min(320, 60 + 32 * len(df_match)),
                         column_config={
                             "strength": st.column_config.ProgressColumn(
                                 "strength", min_value=0.0, max_value=1.0, format="%.2f"
                             ),
                         })

            unmatched_games = event_match_payload.get("unmatched_games") or []
            unmatched_events = event_match_payload.get("unmatched_events") or []
            ucols = st.columns(2)
            with ucols[0]:
                st.markdown("**Unmatched MLB games**")
                if unmatched_games:
                    st.dataframe(
                        pd.DataFrame(unmatched_games).fillna(DASH),
                        use_container_width=True, hide_index=True,
                        height=min(220, 60 + 32 * len(unmatched_games)),
                    )
                else:
                    st.caption("All MLB games are matched. ✓")
            with ucols[1]:
                st.markdown("**Unmatched odds events**")
                if unmatched_events:
                    st.dataframe(
                        pd.DataFrame(unmatched_events).fillna(DASH),
                        use_container_width=True, hide_index=True,
                        height=min(220, 60 + 32 * len(unmatched_events)),
                    )
                else:
                    st.caption("All odds events were claimed. ✓")
        else:
            render_empty_state(
                "Event match table unavailable.",
                "Refresh the cache and try again.",
            )


# =============================================================================
# Watchlist (trader CRUD)
# =============================================================================

with tab_watchlist:
    st.markdown("### Add wallet")
    with st.form("add_wallet_form", clear_on_submit=True):
        form_cols = st.columns([2, 2, 1])
        wallet_address = form_cols[0].text_input("Wallet address", placeholder="0x...").strip()
        nickname = form_cols[1].text_input("Nickname", placeholder="optional").strip()
        trust = form_cols[2].number_input("Trust", min_value=0, max_value=100, value=50, step=5)
        tags_raw = st.text_input("Tags", placeholder="sports, macro, sharp")
        notes = st.text_area("Notes", height=70, placeholder="Why this wallet is worth tracking")
        scan_after_add = st.checkbox("Run scan after adding", value=True)
        if st.form_submit_button("Add wallet", type="primary"):
            if not wallet_address.startswith("0x") or len(wallet_address) != 42:
                st.error("Enter a valid 0x wallet address.")
            else:
                tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
                payload = {
                    "nickname": nickname or wallet_address[:10],
                    "wallet_address": wallet_address,
                    "platform": "polymarket",
                    "trust_score": float(trust),
                    "tags": tags,
                    "notes": notes.strip() or None,
                    "copy_enabled": False,
                    "copy_mode": "alert_only",
                }
                try:
                    created = api_post("/traders", json=payload)
                except ApiError as exc:
                    if exc.status_code == 409:
                        st.error("That nickname or wallet is already in the watchlist.")
                    else:
                        render_api_error(exc, prefix="Could not add wallet")
                else:
                    st.success(f"Added {created.get('nickname')}.")
                    if scan_after_add:
                        action_run_wallet_scan()
                    st.cache_data.clear()
                    st.rerun()

    st.markdown("### Remove wallets")
    if not traders:
        st.caption("No wallets tracked yet.")
    else:
        trader_by_label = {
            f"{t.get('nickname','?')} · {shorten_wallet(t.get('wallet_address'))}": t
            for t in traders
        }
        labels = st.multiselect("Tracked wallets", options=list(trader_by_label),
                                placeholder="Select wallets to remove")
        confirm = st.checkbox("I confirm deletion of selected wallets and their signals")
        if st.button("Remove selected", disabled=not (labels and confirm)):
            try:
                # No api_delete helper yet; talk to the client directly. 204 is
                # the expected success code on DELETE /traders/{id}.
                with _client() as c:
                    for label in labels:
                        tid = int(trader_by_label[label]["id"])
                        r = c.delete(f"/traders/{tid}")
                        if r.status_code not in (200, 204):
                            raise ApiError(
                                f"HTTP {r.status_code}",
                                method="DELETE",
                                url=f"{API_BASE}/traders/{tid}",
                                status_code=r.status_code,
                                body=r.text,
                            )
            except ApiError as exc:
                render_api_error(exc, prefix="Could not remove wallet")
            else:
                st.success(f"Removed {len(labels)} wallet(s).")
                st.cache_data.clear()
                st.rerun()

    st.markdown("### Tracked traders")
    if traders:
        rows = []
        for t in traders:
            wallet_disp = t.get("wallet_address") if show_full_wallet else shorten_wallet(t.get("wallet_address"))
            rows.append({
                "nickname": t.get("nickname"),
                "wallet": wallet_disp or DASH,
                "platform": t.get("platform"),
                "trust": t.get("trust_score") or 0.0,
                "rank": t.get("trader_rank"),
                "win_rate": fmt_pct(t.get("win_rate")),
                "total_pnl": t.get("total_pnl"),
                "7d_return": t.get("seven_day_return"),
                "positions": t.get("total_positions"),
                "copy_mode": t.get("copy_mode"),
                "tags": ", ".join(t.get("tags") or []),
            })
        df_t = pd.DataFrame(rows).fillna(DASH)
        st.dataframe(
            df_t,
            use_container_width=True, hide_index=True,
            height=min(560, 60 + 32 * len(df_t)),
            column_config={
                "trust": st.column_config.ProgressColumn("trust", min_value=0, max_value=100, format="%.0f"),
                "total_pnl": st.column_config.NumberColumn("PnL", format="$%.0f"),
                "7d_return": st.column_config.NumberColumn("7d", format="%.2f"),
            },
        )
    else:
        render_empty_state(
            "No traders seeded yet.",
            "Run `python -m scripts.seed` once to seed the initial watchlist.",
        )


# =============================================================================
# Alerts
# =============================================================================

with tab_alerts:
    sent = [a for a in alerts_all if a.get("status") == "sent"]
    skipped = [a for a in alerts_all if a.get("status") == "skipped"]
    failed = [a for a in alerts_all if a.get("status") == "failed"]
    ac1, ac2, ac3, ac4 = st.columns(4)
    ac1.metric("Alerts total", len(alerts_all))
    ac2.metric("Sent", len(sent))
    ac3.metric("Skipped", len(skipped))
    ac4.metric("Failed", len(failed))

    if not alerts_all:
        render_empty_state(
            "No alerts dispatched yet.",
            "Alerts post when a signal exceeds the configured Discord threshold.",
        )
    else:
        rows = []
        for a in alerts_all:
            stat = (a.get("status") or "").lower()
            stat_kind = "green" if stat == "sent" else ("red" if stat == "failed" else "muted")
            rows.append({
                "created_at": fmt_dt(a.get("created_at")),
                "channel": a.get("channel"),
                "status": stat,
                "signal_id": a.get("signal_id"),
                "message": (a.get("message") or "")[:240],
                "error": (a.get("error") or "")[:200],
            })
        df_a = pd.DataFrame(rows).fillna(DASH)
        st.dataframe(
            df_a, use_container_width=True, hide_index=True,
            height=min(640, 60 + 32 * len(df_a)),
        )


# =============================================================================
# Health / Debug — raw payloads, error log
# =============================================================================

with tab_health:
    ingest = health.get("ingestion", {}) or {}
    db_block = health.get("database", {}) or {}

    hc1, hc2, hc3, hc4 = st.columns(4)
    hc1.metric("DB backend", db_block.get("backend", "?"))
    hc2.metric("Ingestion failures", ingest.get("ingestion_failures", 0))
    hc3.metric("DB rollbacks", ingest.get("db_rollbacks", 0))
    hc4.metric("Trades inserted", ingest.get("trades_inserted", 0))

    st.markdown("### Provider status")
    provs = providers_block
    badges_html = " ".join([
        configured_badge(provs.get("falcon"), "Falcon"),
        configured_badge(provs.get("polymarket"), "Polymarket"),
        configured_badge(provs.get("kalshi"), "Kalshi"),
        configured_badge(provs.get("odds_api"), "Odds API"),
        configured_badge(provs.get("weather_api"), "Weather"),
        configured_badge(provs.get("mlb_stats_api"), "MLB Stats"),
    ])
    st.markdown(f"<div class='sf-card'><div class='sf-card-row'>{badges_html}</div></div>",
                unsafe_allow_html=True)

    if ingest.get("last_ingestion_error"):
        st.markdown("### Last ingestion error")
        st.code(ingest.get("last_ingestion_error") or "", language="text")
        st.caption(f"At {fmt_dt(ingest.get('last_ingestion_error_at'))}")

    fetch_errors = st.session_state.get("_fetch_errors") or {}
    if fetch_errors:
        st.markdown("### Recent fetch errors")
        for path, err in fetch_errors.items():
            if isinstance(err, ApiError):
                with st.expander(f"{path}  ·  {err.status_code or 'transport'}", expanded=False):
                    render_api_error(err, prefix=f"Failed fetching {path}")

    if debug_mode:
        st.markdown("### Raw payloads")
        with st.expander("/health", expanded=False):
            st.json(health)
        with st.expander("/mlb/debug/sources", expanded=False):
            st.json(mlb_sources)
        with st.expander("/mlb/debug/odds-cache", expanded=False):
            st.json(odds_cache_payload)
        with st.expander("/mlb/debug/odds/event-match", expanded=False):
            st.json(event_match_payload)
        with st.expander("/dashboard-summary", expanded=False):
            st.json(summary)
    else:
        st.caption("Raw JSON hidden. Enable 'Show raw JSON' in the sidebar to expose payloads.")
