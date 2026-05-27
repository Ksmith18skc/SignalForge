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
import time
from itertools import count
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Iterable

import httpx
import pandas as pd
import streamlit as st

from app.components.pnl_dashboard import render_pnl_summary_cards, render_pnl_tracker
from app.utils.dashboard_format import (
    SCORE_ACTIONABLE_MIN,
    SCORE_BUCKETS,
    SCORE_HIGH_CONV_MIN,
    SCORE_STRONG_MIN,
    american_from_price,
    american_to_implied_probability,
    compact_time_ago,
    confidence_label as confidence_label_fn,
    confidence_word,
    edge_vs_market,
    factor_label as factor_label_fn,
    format_card_title,
    format_cents,
    format_edge_delta,
    format_hit_rate,
    format_money_short,
    format_price_with_implied_prob,
    format_probability,
    odds_provider_label,
    polished_missing,
    score_bucket_label,
    score_distribution as score_distribution_fn,
    score_tier as score_tier_fn,
    score_tier_kind,
    team_short,
    wallet_alignment_percent,
)

API_BASE = os.environ.get("SIGNALFORGE_API_URL", "http://localhost:8000").rstrip("/")
DEFAULT_TIMEOUT = 45.0
HEALTH_TIMEOUT = 60.0
SCAN_TIMEOUT = 90.0
MLB_RUN_TIMEOUT = 180.0
RETRY_PATH_PREFIXES = (
    "/health",
    "/api/status",
    "/ready",
    "/run-scan",
    "/mlb/debug/odds-cache",
    "/mlb/edges/run",
)
_EMPTY_STATE_COUNTER = count()


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
.main .block-container { padding-top: 0.6rem; padding-bottom: 1.6rem; max-width: 1500px; }
.stMarkdown p { margin-bottom: 0.15rem; line-height: 1.35; }
.element-container { margin-bottom: 0.22rem; }
.stMarkdown { line-height: 1.35; }

h1, h2, h3, h4 { color: var(--text); letter-spacing: 0.02em; margin: 0.4rem 0 0.2rem 0; }
h1 { font-weight: 700; font-size: 1.55rem; }
h2 { font-weight: 600; font-size: 1.15rem; }
h3 { font-weight: 600; font-size: 0.95rem; text-transform: uppercase; color: var(--muted); letter-spacing: 0.12em; margin-top: 0.6rem; }
h4 { font-size: 0.85rem; color: var(--muted); margin-top: 0.4rem; }

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
  padding: 8px 11px;
  margin-bottom: 6px;
}
.sf-card.gold   { border-left: 3px solid var(--gold);   box-shadow: 0 0 0 1px rgba(212,175,55,0.08); }
.sf-card.green  { border-left: 3px solid var(--green);  }
.sf-card.purple { border-left: 3px solid var(--purple); }
.sf-card.red    { border-left: 3px solid var(--red);    }
.sf-card-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 4px;
}
.sf-card-title {
  font-size: 1.02rem;
  font-weight: 700;
  color: var(--text);
  letter-spacing: 0.01em;
  line-height: 1.2;
}
.sf-card-sub {
  color: var(--muted);
  font-size: 0.74rem;
  letter-spacing: 0.04em;
  line-height: 1.3;
}
.sf-card-row {
  color: var(--text);
  font-size: 0.84rem;
  margin-top: 1px;
  line-height: 1.35;
}
.sf-card-row .k { color: var(--muted); margin-right: 6px; }
.sf-score {
    font-size: 2.0rem;
    font-weight: 800;
    letter-spacing: 0.03em;
    line-height: 1.0;
    font-variant-numeric: tabular-nums;
}
.sf-score-label {
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--muted);
}
.sf-prob {
    font-size: 1.7rem;
    font-weight: 800;
    color: var(--purple);
    letter-spacing: 0.02em;
    line-height: 1.0;
    font-variant-numeric: tabular-nums;
}
.sf-prob-row {
    display: flex;
    gap: 14px;
    align-items: flex-end;
    margin-top: 2px;
}
.sf-prob-cell .lbl {
    font-size: 0.58rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 2px;
}
.sf-prob-cell .val {
    font-size: 0.92rem;
    color: var(--text);
    font-variant-numeric: tabular-nums;
}
.sf-prob-cell .val.purple { color: var(--purple); font-weight: 700; }
.sf-prob-cell .val.green  { color: var(--green); font-weight: 700; }
.sf-prob-cell .val.gold   { color: var(--gold); font-weight: 700; }
.sf-prob-cell .val.red    { color: var(--red); font-weight: 700; }
.score-strong { color: var(--gold); }
.score-bettable { color: var(--green); }
.score-watch { color: var(--purple); }
.score-pass { color: var(--muted); }
.score-bar {
    height: 5px;
    width: 120px;
    background: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    margin-top: 6px;
    margin-left: auto;
}
.score-bar > span {
    display: block;
    height: 100%;
}
.score-bar-strong { background: var(--gold); }
.score-bar-bettable { background: var(--green); }
.score-bar-watch { background: var(--purple); }
.score-bar-pass { background: var(--muted); }
.pulse-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 8px rgba(0,200,83,0.5);
    animation: pulse 1.6s infinite;
    margin-right: 6px;
}
.pulse-odds {
    animation: pulse 1.2s infinite;
    box-shadow: 0 0 10px rgba(0,200,83,0.7);
}
@keyframes pulse {
    0% { transform: scale(1); opacity: 0.7; }
    50% { transform: scale(1.25); opacity: 1.0; }
    100% { transform: scale(1); opacity: 0.7; }
}
.sf-reasons {
  margin: 6px 0 0 0;
  padding-left: 1.1em;
  color: var(--text);
  font-size: 0.85rem;
}
.sf-reasons li { margin-bottom: 2px; }
.sf-link-buttons { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px; }
.sf-link-button {
    display: inline-block;
    padding: 4px 8px;
    border: 1px solid var(--border);
    border-radius: 4px;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text);
    background: var(--panel-2);
    text-decoration: none;
}
.sf-link-button:hover { border-color: var(--cyan); color: var(--cyan); }

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

/* --- Edge card sub-sections --- */
.sf-section {
  margin-top: 6px;
  padding-top: 5px;
  border-top: 1px dashed var(--border);
}
.sf-section-title {
  color: var(--muted);
  font-size: 0.64rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  margin-bottom: 2px;
}
.sf-kv {
  display: grid;
  grid-template-columns: 140px 1fr;
  column-gap: 10px;
  row-gap: 1px;
  font-size: 0.82rem;
}
.sf-kv .k { color: var(--muted); }
.sf-kv .v { color: var(--text); }
.sf-factor-row {
  display: grid;
  grid-template-columns: 160px 1fr 36px;
  align-items: center;
  column-gap: 8px;
  margin-bottom: 1px;
  font-size: 0.78rem;
}
.sf-factor-row .lbl { color: var(--muted); }
.sf-factor-row .val { color: var(--text); text-align: right; font-variant-numeric: tabular-nums; }
.sf-factor-bar {
  height: 5px;
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 3px;
  overflow: hidden;
}
.sf-factor-bar > span {
  display: block;
  height: 100%;
  background: var(--cyan);
}
.sf-factor-bar.hi > span { background: var(--gold); }
.sf-factor-bar.mid > span { background: var(--green); }
.sf-factor-bar.lo > span { background: var(--muted); }
.sf-trust {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 14px;
  font-size: 0.76rem;
  color: var(--text);
}
.sf-trust .ok      { color: var(--green); }
.sf-trust .warn    { color: var(--red); }
.sf-trust .neutral { color: var(--muted); }
.sf-trust .info    { color: var(--cyan); }

/* --- Live ribbon (Command Center) --- */
.sf-ribbon {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px 14px;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 7px 12px;
  margin-bottom: 10px;
  font-size: 0.78rem;
}
.sf-ribbon .seg {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  color: var(--text);
  letter-spacing: 0.04em;
}
.sf-ribbon .seg .lbl {
  color: var(--muted);
  font-size: 0.66rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
}
.sf-ribbon .seg .ok   { color: var(--green); }
.sf-ribbon .seg .warn { color: var(--red); }
.sf-ribbon .seg .info { color: var(--cyan); }
.sf-ribbon .sep { color: var(--border); }

/* --- Market Pulse chip strip --- */
.sf-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 2px;
}
.sf-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 9px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--panel);
  font-size: 0.72rem;
  color: var(--text);
  letter-spacing: 0.04em;
}
.sf-chip .lbl   { color: var(--muted); text-transform: uppercase; font-size: 0.62rem; letter-spacing: 0.14em; }
.sf-chip .val   { color: var(--text); font-weight: 700; font-variant-numeric: tabular-nums; }
.sf-chip.ok    { border-color: var(--green); }
.sf-chip.ok .val { color: var(--green); }
.sf-chip.warn  { border-color: var(--red); }
.sf-chip.warn .val { color: var(--red); }
.sf-chip.info  { border-color: var(--cyan); }
.sf-chip.info .val { color: var(--cyan); }

/* --- Sharp money block (wallet cards) --- */
.sf-sharp-pct {
  font-size: 1.6rem;
  font-weight: 800;
  color: var(--gold);
  font-variant-numeric: tabular-nums;
  line-height: 1.0;
}
.sf-sharp-meta {
  color: var(--muted);
  font-size: 0.76rem;
  margin-top: 2px;
}

/* --- Price block --- */
.sf-price-grid {
  display: grid;
  grid-template-columns: 130px 1fr;
  column-gap: 8px;
  row-gap: 1px;
  font-size: 0.82rem;
  line-height: 1.35;
}
.sf-price-grid .lbl { color: var(--muted); font-size: 0.7rem; letter-spacing: 0.12em; text-transform: uppercase; }
.sf-price-grid .val { color: var(--text); }
.sf-price-grid .val.purple { color: var(--purple); font-weight: 700; }
.sf-price-grid .val.green  { color: var(--green); font-weight: 700; }
.sf-price-grid .val.gold   { color: var(--gold); font-weight: 700; }
.sf-price-grid .edge       { color: var(--green); font-weight: 700; }
.sf-price-grid .edge.neg   { color: var(--red); }

/* --- Pill tag (compact, less shouty than the existing sf-badge) --- */
.sf-pill {
  display: inline-block;
  padding: 1px 7px;
  border-radius: 10px;
  font-size: 0.66rem;
  letter-spacing: 0.1em;
  border: 1px solid var(--border);
  color: var(--muted);
  background: rgba(138,148,166,0.06);
  text-transform: uppercase;
  margin-right: 4px;
}
.sf-pill.green  { color: var(--green); border-color: var(--green); background: var(--green-soft); }
.sf-pill.gold   { color: var(--gold); border-color: var(--gold); background: var(--gold-soft); }
.sf-pill.purple { color: var(--purple); border-color: var(--purple); background: var(--purple-soft); }
.sf-pill.red    { color: var(--red); border-color: var(--red); background: var(--red-soft); }
.sf-pill.cyan   { color: var(--cyan); border-color: var(--cyan); }
.sf-bucket-row {
  display: grid;
  grid-template-columns: 70px 1fr 50px;
  align-items: center;
  column-gap: 8px;
  font-size: 0.82rem;
  margin-bottom: 2px;
}
.sf-bucket-row .lbl { color: var(--muted); font-variant-numeric: tabular-nums; }
.sf-bucket-row .val { color: var(--text); text-align: right; font-variant-numeric: tabular-nums; }
.sf-bucket-bar {
  height: 8px;
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 3px;
  overflow: hidden;
}
.sf-bucket-bar > span {
  display: block;
  height: 100%;
  background: var(--purple);
}
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

    @property
    def is_timeout(self) -> bool:
        text = str(self).lower()
        return "timeout" in text or "timed out" in text or "readtimeout" in text

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


def _should_retry(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in RETRY_PATH_PREFIXES)


def _request_json(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json: Any = None,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int | None = None,
) -> Any:
    url = f"{API_BASE}{path}"
    retry_count = retries if retries is not None else (2 if _should_retry(path) else 0)
    attempt = 0
    last_error: ApiError | None = None
    while attempt <= retry_count:
        try:
            with _client() as c:
                if method == "GET":
                    r = c.get(path, params=params, timeout=timeout)
                else:
                    r = c.post(path, json=json, timeout=timeout)
        except httpx.HTTPError as exc:
            last_error = ApiError(f"{type(exc).__name__}: {exc}", method=method, url=url)
            if attempt >= retry_count:
                raise last_error from exc
            time.sleep(min(2 ** attempt, 8))
            attempt += 1
            continue
        if r.is_success:
            try:
                return r.json()
            except ValueError as exc:
                raise ApiError(
                    f"Non-JSON response: {exc}", method=method, url=url,
                    status_code=r.status_code, body=r.text,
                ) from exc
        error = ApiError(
            f"HTTP {r.status_code}", method=method, url=url,
            status_code=r.status_code, body=r.text,
        )
        if r.status_code in {502, 503, 504} and attempt < retry_count:
            last_error = error
            time.sleep(min(2 ** attempt, 8))
            attempt += 1
            continue
        raise error
    if last_error:
        raise last_error
    raise ApiError("request failed", method=method, url=url)


def api_get(path: str, params: dict[str, Any] | None = None, timeout: float = DEFAULT_TIMEOUT) -> Any:
    """GET an endpoint and parse JSON, raising ApiError on any failure."""
    return _request_json("GET", path, params=params, timeout=timeout)


def api_post(path: str, json: Any = None, timeout: float = DEFAULT_TIMEOUT) -> Any:
    return _request_json("POST", path, json=json, timeout=timeout)


def api_get_once(path: str, *, timeout: float = 8.0) -> Any:
    return _request_json("GET", path, timeout=timeout, retries=0)


def backend_fallback_probe() -> bool:
    """If /health timed out, check whether Render is at least serving HTTP."""
    for path in ("/api/status", "/", "/docs"):
        try:
            with httpx.Client(base_url=API_BASE, timeout=8.0) as c:
                r = c.get(path)
            if r.status_code < 500:
                return True
        except httpx.HTTPError:
            continue
    return False


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
TZ_MST = ZoneInfo("America/Phoenix")


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


def fmt_dt_mst(value: Any, *, include_tz: bool = True) -> str:
    dt = _parse_dt(value)
    if not dt:
        return DASH
    local = dt.astimezone(TZ_MST)
    label = local.strftime("%b %d, %Y %I:%M %p").replace(" 0", " ")
    return f"{label} MST" if include_tz else label


def fmt_dt_utc(value: Any) -> str:
    dt = _parse_dt(value)
    if not dt:
        return DASH
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


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


def fmt_relative(value: Any, *, now: datetime | None = None) -> str:
    dt = _parse_dt(value)
    if not dt:
        return DASH
    now = now or datetime.now(timezone.utc)
    delta = (now - dt).total_seconds()
    if delta < 0:
        delta = abs(delta)
        suffix = "from now"
    else:
        suffix = "ago"
    if delta < 60:
        return f"{int(delta)}s {suffix}"
    if delta < 3600:
        return f"{int(delta // 60)}m {suffix}"
    if delta < 86400:
        return f"{int(delta // 3600)}h {suffix}"
    return f"{int(delta // 86400)}d {suffix}"


def fmt_event_time(value: Any, *, now: datetime | None = None) -> str:
    dt = _parse_dt(value)
    if not dt:
        return DASH
    local = dt.astimezone(TZ_MST)
    return local.strftime("%b %d, %Y %I:%M %p MST").replace(" 0", " ")


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


def score_class(score: Any) -> str:
    if score is None:
        return "score-pass"
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "score-pass"
    if s >= 85:
        return "score-strong"
    if s >= 75:
        return "score-bettable"
    if s >= 65:
        return "score-watch"
    return "score-pass"


def score_bar_class(score: Any) -> str:
    if score is None:
        return "score-bar-pass"
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "score-bar-pass"
    if s >= 85:
        return "score-bar-strong"
    if s >= 75:
        return "score-bar-bettable"
    if s >= 65:
        return "score-bar-watch"
    return "score-bar-pass"


def score_percent(score: Any) -> int:
    try:
        s = float(score)
    except (TypeError, ValueError):
        return 0
    return int(min(max(s, 0.0), 100.0))


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
    """Confidence word (LOW/MED/HIGH → WATCH/LEAN/STRONG/HIGH CONV). No fake
    upgrades — this is purely a relabel of the engine's raw 'low|medium|high'
    so the card never says 'LOW CONF' next to a card under 'Top Actionable'."""
    label, kind = confidence_word(conf)
    return badge(label, kind)


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


def link_button(label: str, url: str | None) -> str:
    if not url:
        return ""
    return f'<a class="sf-link-button" href="{url}" target="_blank" rel="noopener">{label}</a>'


def render_link_buttons(links: list[tuple[str, str | None]]) -> str:
    buttons = "".join(link_button(label, url) for label, url in links if url)
    if not buttons:
        return ""
    return f"<div class='sf-link-buttons'>{buttons}</div>"


# =============================================================================
# Card renderers
# =============================================================================


def _fmt_line(value: Any, *, fmt: str = "{:.1f}") -> str:
    return fmt_num(value, fmt=fmt) if value is not None else DASH


def _factor_bar_color(value: float) -> str:
    if value >= 75:
        return "hi"
    if value >= 60:
        return "mid"
    if value >= 40:
        return ""
    return "lo"


def _pill(text: str, kind: str = "muted") -> str:
    """Compact, less-shouty alternative to the rectangular sf-badge."""
    return f"<span class='sf-pill {kind}'>{text}</span>"


def _missing_tag(kind: str) -> str:
    """Render a small neutral chip for missing data instead of a full row."""
    return _pill(polished_missing(kind), "muted")


def render_factor_bars(factors: dict[str, Any], *, limit: int = 5) -> str:
    """`Why this edge exists` — compact factor bars with renamed labels.

    Caps the visible list at `limit` so the card stays scannable; if there
    are extra factors, we tack on a small chip indicating how many more
    were compressed away."""
    if not factors:
        return f"<div class='sf-section'><div class='sf-section-title'>Why this edge exists</div>{_missing_tag('factors')}</div>"
    items: list[tuple[str, float]] = []
    for name, value in factors.items():
        if value is None:
            continue
        try:
            items.append((name, float(value)))
        except (TypeError, ValueError):
            continue
    if not items:
        return f"<div class='sf-section'><div class='sf-section-title'>Why this edge exists</div>{_missing_tag('factors')}</div>"
    # Sort by magnitude so the loudest contributor shows up first.
    items.sort(key=lambda kv: kv[1], reverse=True)
    visible = items[:limit]
    hidden = max(0, len(items) - limit)
    rows: list[str] = []
    for name, v in visible:
        label = factor_label_fn(name)
        pct = max(0.0, min(100.0, v))
        color = _factor_bar_color(pct)
        rows.append(
            "<div class='sf-factor-row'>"
            f"<span class='lbl'>{label}</span>"
            f"<span class='sf-factor-bar {color}'><span style='width:{pct:.0f}%'></span></span>"
            f"<span class='val'>{v:.0f}</span>"
            "</div>"
        )
    extra = f"<div class='sf-card-sub' style='margin-top:2px;'>+{hidden} more factor{'s' if hidden != 1 else ''}</div>" if hidden else ""
    return (
        "<div class='sf-section'>"
        "<div class='sf-section-title'>Why this edge exists</div>"
        + "".join(rows)
        + extra
        + "</div>"
    )


def render_market_price_block(edge: dict[str, Any]) -> str:
    """Reference pricing block. Sportsbook is the visible quote; the
    prediction-market row only renders when the edge actually carries
    Kalshi/Polymarket fields. SignalForge estimate is shown only when the
    backend provides one — we never invent a probability from the score."""
    edge_type = str(edge.get("edge_type") or "").lower()
    market_price = edge.get("best_price")
    best_book = edge.get("best_book") or "—"
    sb_implied = american_to_implied_probability(market_price)
    sb_value = (
        f"<span class='val'>{best_book} {american_from_price(market_price) or DASH}"
        + (f" <span class='sf-card-sub'>· {sb_implied * 100:.1f}%</span>" if sb_implied is not None else "")
        + "</span>"
    )

    pm_platform = str(edge.get("prediction_market_platform") or "").strip()
    pm_side = edge.get("prediction_market_side")
    pm_price = edge.get("prediction_market_price")
    pm_value = None
    if pm_platform and pm_price is not None:
        platform_label = pm_platform.title() if pm_platform else "Market"
        side_label = (str(pm_side or "").upper() + " ") if pm_side else ""
        pm_value = f"<span class='val purple'>{platform_label} {side_label}{format_cents(pm_price)}</span>"

    sf_prob = (
        edge.get("model_probability")
        or edge.get("signalforge_probability")
        or edge.get("estimated_probability")
    )
    if edge_type == "pitcher_strikeouts":
        sf_proj = edge.get("projected_strikeouts") or edge.get("projected_ks")
        sf_unit = "Ks"
    elif edge_type == "game_total":
        sf_proj = edge.get("projected_total") or edge.get("model_projected_total")
        sf_unit = ""
    else:
        sf_proj = None
        sf_unit = ""

    sf_value_parts: list[str] = []
    if sf_prob is not None:
        sf_value_parts.append(f"<span class='val purple'>{format_probability(sf_prob)}</span>")
    if sf_proj is not None:
        suffix = f" {sf_unit}".rstrip() if sf_unit else ""
        sf_value_parts.append(f"<span class='val'>proj {sf_proj:.1f}{suffix}</span>")
    if not sf_value_parts:
        score = edge.get("score")
        if score is not None:
            sf_value_parts.append(f"<span class='val'>Score {fmt_score(score)}</span>")
        else:
            sf_value_parts.append(_missing_tag("projection"))

    edge_value = None
    pm_implied = None
    if pm_price is not None:
        try:
            pm_implied = float(pm_price)
            if pm_implied > 1:
                pm_implied /= 100.0
        except (TypeError, ValueError):
            pm_implied = None
    if sf_prob is not None and pm_implied is not None:
        edge_str = edge_vs_market(sf_prob, pm_implied)
        if edge_str is not None:
            neg_cls = " neg" if edge_str.startswith("-") else ""
            edge_value = f"<span class='edge{neg_cls}'>{edge_str} vs prediction market</span>"
    elif sf_prob is not None and sb_implied is not None:
        edge_str = edge_vs_market(sf_prob, sb_implied)
        if edge_str is not None:
            neg_cls = " neg" if edge_str.startswith("-") else ""
            edge_value = f"<span class='edge{neg_cls}'>{edge_str} vs sportsbook</span>"

    rows: list[tuple[str, str]] = []
    if pm_value:
        rows.append(("Prediction Market", pm_value))
    rows.append((
        "Sportsbook" if pm_value else "Reference price",
        sb_value,
    ))
    rows.append(("SignalForge", " · ".join(sf_value_parts)))
    if edge_value:
        rows.append(("Market edge", edge_value))
    elif sf_prob is not None and pm_implied is None:
        rows.append(("Market edge", _missing_tag("projection")))
    body = "".join(
        f"<span class='lbl'>{k}</span>{v}" for k, v in rows
    )
    return f"<div class='sf-section'><div class='sf-section-title'>Pricing</div><div class='sf-price-grid'>{body}</div></div>"


def _render_model_vs_market(edge: dict[str, Any]) -> str:
    """Lightweight 'projection vs market line' block — only renders for
    pitcher K and game total cards, and only when a real backend
    projection field exists. Missing values collapse into a small chip
    rather than a full ugly row."""
    edge_type = str(edge.get("edge_type") or "")
    if edge_type not in {"pitcher_strikeouts", "game_total"}:
        return ""
    rows: list[tuple[str, str]] = []
    line = edge.get("line")
    if edge_type == "pitcher_strikeouts":
        proj = (
            edge.get("projected_strikeouts")
            or edge.get("model_projected_ks")
            or edge.get("projected_ks")
        )
        recent = (
            edge.get("recent_strikeouts_per_start")
            or edge.get("recent_ks_per_start")
            or (edge.get("statcast_summary") or {}).get("strikeouts_per_start")
        )
        if line is not None:
            rows.append(("Market line", f"{_fmt_line(line)} Ks"))
        if proj is not None:
            rows.append(("SF projection", f"{_fmt_line(proj)} Ks"))
            rows.append(("Edge delta", format_edge_delta(proj, line, unit="Ks")))
        if recent is not None:
            rows.append(("Recent K/start", _fmt_line(recent)))
    else:
        proj = edge.get("projected_total") or edge.get("model_projected_total")
        if line is not None:
            rows.append(("Market total", _fmt_line(line)))
        if proj is not None:
            rows.append(("SF projection", _fmt_line(proj)))
            rows.append(("Edge delta", format_edge_delta(proj, line)))

    if not rows:
        return ""
    # If we have a market line but no projection, append the polished tag
    # instead of an entire 'Projection unavailable' row.
    has_projection = any("projection" in k.lower() for k, _ in rows)
    suffix = "" if has_projection else f"<div style='margin-top:3px;'>{_missing_tag('projection')}</div>"
    body = "".join(f"<div class='k'>{k}</div><div class='v'>{v}</div>" for k, v in rows)
    return (
        "<div class='sf-section'>"
        "<div class='sf-section-title'>Model vs Market</div>"
        f"<div class='sf-kv'>{body}</div>{suffix}"
        "</div>"
    )


def _render_recent_form(edge: dict[str, Any]) -> str:
    """Recent-form panel for pitcher K cards. Real fields only — if a row
    is missing, it's omitted; a single neutral chip explains the gap."""
    if str(edge.get("edge_type") or "") != "pitcher_strikeouts":
        return ""
    line = edge.get("line")
    summary = edge.get("statcast_summary") or {}
    last3 = edge.get("last_3_starts_ks") or summary.get("last_3_starts_ks")
    last5 = edge.get("last_5_starts_ks") or summary.get("last_5_starts_ks")
    season_kps = edge.get("season_strikeouts_per_start") or summary.get(
        "season_strikeouts_per_start"
    )
    recent_kps = (
        edge.get("recent_strikeouts_per_start")
        or edge.get("recent_ks_per_start")
        or summary.get("strikeouts_per_start")
    )
    hits10 = edge.get("hits_last_10_vs_line") or summary.get("hits_last_10_vs_line")
    att10 = edge.get("attempts_last_10_vs_line") or summary.get(
        "attempts_last_10_vs_line"
    )
    hits5 = edge.get("hits_last_5_vs_line") or summary.get("hits_last_5_vs_line")
    att5 = edge.get("attempts_last_5_vs_line") or summary.get("attempts_last_5_vs_line")

    rows: list[tuple[str, str]] = []
    if last3 is not None:
        rows.append(("Last 3 K/start", _fmt_line(last3)))
    if last5 is not None:
        rows.append(("Last 5 K/start", _fmt_line(last5)))
    elif recent_kps is not None:
        rows.append(("Recent K/start", _fmt_line(recent_kps)))
    if season_kps is not None:
        rows.append(("Season K/start", _fmt_line(season_kps)))
    if line is not None:
        rate5 = format_hit_rate(hits5, att5)
        if rate5 != "insufficient history":
            rows.append((f"Hit rate vs {_fmt_line(line)} (5)", rate5))
        rate10 = format_hit_rate(hits10, att10)
        if rate10 != "insufficient history":
            rows.append((f"Hit rate vs {_fmt_line(line)} (10)", rate10))

    if not rows:
        return (
            "<div class='sf-section'>"
            "<div class='sf-section-title'>Recent Form</div>"
            f"{_missing_tag('history')}"
            "</div>"
        )
    body = "".join(f"<div class='k'>{k}</div><div class='v'>{v}</div>" for k, v in rows)
    return (
        "<div class='sf-section'>"
        "<div class='sf-section-title'>Recent Form</div>"
        f"<div class='sf-kv'>{body}</div>"
        "</div>"
    )


def _movement_tags(edge: dict[str, Any]) -> list[tuple[str, str]]:
    """Movement chips. Only emitted when a real underlying field supports
    the claim — we never invent a 'STEAM' tag."""
    tags: list[tuple[str, str]] = []
    direction = str(edge.get("movement_direction") or "").lower()
    if direction in {"steam", "steaming"}:
        tags.append(("STEAM DETECTED", "gold"))
    if direction == "drift":
        tags.append(("LINE DRIFT", "purple"))
    best_price = edge.get("best_price")
    closing_price = edge.get("closing_price")
    bp = american_to_implied_probability(best_price)
    cp = american_to_implied_probability(closing_price)
    if bp is not None and cp is not None and abs(bp - cp) >= 0.005:
        if bp < cp:
            tags.append(("PRICE WORSENING", "red"))
        else:
            tags.append(("PRICE IMPROVING", "green"))
    return tags


def _render_movement_clv(edge: dict[str, Any]) -> str:
    opening_line = edge.get("opening_line")
    current_line = edge.get("current_line")
    closing_line = edge.get("closing_line")
    best_price = edge.get("best_price")
    closing_price = edge.get("closing_price")
    clv_points = edge.get("clv_points")
    clv_percent = edge.get("clv_percent")
    graded = bool(edge.get("graded_at") or edge.get("win_loss_push"))

    rows: list[tuple[str, str]] = []
    has_line = any(v is not None for v in (opening_line, current_line, closing_line))
    if has_line:
        # Skip 'open → current' when both are identical — that's the engine
        # default and reads as visual clutter rather than information.
        start = opening_line
        cur = current_line
        end = closing_line
        parts: list[str] = []
        if start is not None:
            parts.append(_fmt_line(start))
        if cur is not None and (start is None or abs(float(cur) - float(start)) > 1e-9):
            parts.append(_fmt_line(cur))
        if end is not None:
            parts.append(_fmt_line(end))
        elif not graded:
            parts.append("<span class='sf-pill muted'>" + polished_missing("closing") + "</span>")
        if len(parts) >= 2:
            rows.append(("Line", " → ".join(parts)))

    if best_price is not None and closing_price is not None:
        sp = american_from_price(best_price) or DASH
        cp = american_from_price(closing_price) or DASH
        rows.append(("Price", f"{sp} → {cp}"))

    if clv_percent is not None or clv_points is not None:
        points = (
            f"{float(clv_points):+.1f} pts" if clv_points is not None else DASH
        )
        pct = fmt_pct(clv_percent) if clv_percent is not None else DASH
        rows.append(("CLV", f"{points} · {pct}"))

    tags = _movement_tags(edge)

    if not rows and not tags:
        # No movement data, no closing line, nothing to show. Collapse to a
        # neutral chip rather than a big empty section header.
        return (
            "<div class='sf-section'>"
            "<div class='sf-section-title'>Market Movement &amp; CLV</div>"
            f"{_missing_tag('movement')}"
            "</div>"
        )

    body = "".join(f"<div class='k'>{k}</div><div class='v'>{v}</div>" for k, v in rows)
    tag_html = " ".join(_pill(t, kind) for t, kind in tags)
    tag_block = f"<div style='margin-top:3px;'>{tag_html}</div>" if tag_html else ""
    clv_chip = (
        f"<div style='margin-top:3px;'>{_missing_tag('clv_pending')}</div>"
        if not graded and clv_percent is None and clv_points is None
        else ""
    )
    return (
        "<div class='sf-section'>"
        "<div class='sf-section-title'>Market Movement &amp; CLV</div>"
        f"<div class='sf-kv'>{body}</div>"
        f"{clv_chip}"
        f"{tag_block}"
        "</div>"
    )


def render_trust_tags(edge: dict[str, Any], *, odds_source: str, fallback: bool) -> str:
    """Single-line trust strip. Severe warnings (high chase, stale odds)
    surface explicitly; everything else stays compact and unobtrusive."""
    fresh = not bool(edge.get("odds_stale"))
    data_age = edge.get("odds_data_age_minutes")
    odds_fresh_label = "Odds fresh" if fresh else "Odds stale"
    if data_age is not None:
        odds_fresh_label += f" · {data_age}m"
    fresh_cls = "ok" if fresh else "warn"

    factors = edge.get("factors") or {}
    book_count = (
        edge.get("book_count")
        or factors.get("book_count")
        or len({(r or {}).get("bookmaker") for r in (edge.get("rows") or []) if r})
    )
    try:
        book_count_int = int(book_count) if book_count else 0
    except (TypeError, ValueError):
        book_count_int = 0

    sources = [str(s) for s in (edge.get("data_sources_used") or [])]
    statcast_ok = any("statcast" in s.lower() for s in sources)
    is_pitcher = str(edge.get("edge_type") or "") == "pitcher_strikeouts"

    warnings = edge.get("warnings") or []
    chase = str(edge.get("chase_risk") or "").lower()

    pieces: list[str] = [f"<span class='{fresh_cls}'>{'✓' if fresh else '⚠'} {odds_fresh_label}</span>"]
    if book_count_int:
        pieces.append(f"<span class='ok'>✓ {book_count_int} books</span>")
    if is_pitcher:
        pieces.append(
            f"<span class='ok'>✓ Cached Statcast</span>"
            if statcast_ok
            else "<span class='neutral'>· Statcast pending</span>"
        )
    if fallback:
        pieces.append("<span class='warn'>⚠ Fallback source</span>")
    if chase == "high":
        pieces.append("<span class='warn'>⚠ High chase</span>")
    elif chase == "medium":
        pieces.append("<span class='neutral'>· Med chase</span>")
    # Warnings only when severe — quiet by default.
    if len(warnings) >= 3:
        pieces.append(f"<span class='warn'>⚠ {len(warnings)} warnings</span>")
    return f"<div class='sf-trust'>{''.join(pieces)}</div>"


def render_time_context(edge: dict[str, Any], *, now: datetime | None = None) -> str:
    """Compact 'event · last update · data age' line. Uses tabular-num
    timestamps so the row doesn't shimmy as values tick over."""
    base_now = now or datetime.now(timezone.utc)
    game_start = edge.get("game_start_time")
    game_date = edge.get("game_date")
    event_label = fmt_event_time(game_start) if game_start else (game_date or DASH)
    odds_captured_at = edge.get("odds_snapshot_captured_at")
    best_book_at = edge.get("best_book_updated_at")
    data_age = edge.get("odds_data_age_minutes")
    age_label = f"{data_age}m" if data_age is not None else compact_time_ago(odds_captured_at, now=base_now)
    book_move = compact_time_ago(best_book_at, now=base_now)
    book_segment = f" · book moved {book_move}" if book_move != DASH else ""
    return (
        f"<div class='sf-card-sub'>{event_label} · odds {age_label} ago{book_segment}</div>"
    )


def _live_dot(edge: dict[str, Any]) -> str:
    """Pulse dot when the odds are fresh; static dim dot otherwise."""
    if edge.get("odds_stale"):
        return "<span class='pulse-dot' style='background:var(--red);box-shadow:none;animation:none;'></span>"
    return "<span class='pulse-dot'></span>"


def render_edge_card(edge: dict[str, Any]) -> None:
    """Premium edge card. Single source of truth for the visual hierarchy
    used across Command Center, MLB Terminal, Daily Card."""
    score = edge.get("score")
    label, label_kind = confidence_label_fn(
        score,
        edge.get("action"),
        edge.get("confidence"),
    )
    card_kind = {
        "gold": "gold",
        "green": "green",
        "purple": "purple",
        "red": "red",
        "cyan": "",
        "muted": "",
    }.get(label_kind, "")

    title = format_card_title(edge)
    side = (edge.get("side") or "").title()
    line = edge.get("line")
    line_str = _fmt_line(line)
    edge_type = str(edge.get("edge_type") or "")
    odds_source, fallback = odds_provider_label(edge.get("odds_snapshot_source"))
    odds_stale = bool(edge.get("odds_stale"))

    sub_bits: list[str] = []
    home = team_short(edge.get("home_team"))
    away = team_short(edge.get("away_team"))
    if home and away and edge_type == "pitcher_strikeouts":
        sub_bits.append(f"{away} @ {home}")
    if line is not None and side and edge_type != "pitcher_strikeouts":
        sub_bits.append(f"{side} {line_str}")
    market_scope = str(edge.get("market_scope") or "")
    if market_scope and market_scope.lower() != "player_prop" and market_scope.lower() != "full_game_total":
        sub_bits.append(market_scope.replace("_", " ").title())
    sub_label = " · ".join(sub_bits) if sub_bits else (edge.get("market") or "")

    # Header right-rail: score is the primary, model probability (when
    # available) is the secondary readout. The score never claims to be a
    # probability — but if the backend supplies one, we show it explicitly.
    sf_prob = (
        edge.get("model_probability")
        or edge.get("signalforge_probability")
        or edge.get("estimated_probability")
    )
    prob_row_parts: list[str] = [
        f"<div class='sf-prob-cell'><div class='lbl'>Score</div>"
        f"<div class='val'><span class='sf-score {score_class(score)}'>{fmt_score(score)}</span></div></div>"
    ]
    if sf_prob is not None:
        prob_row_parts.append(
            "<div class='sf-prob-cell'>"
            "<div class='lbl'>SF prob</div>"
            f"<div class='val purple'>{format_probability(sf_prob)}</div></div>"
        )
    market_price = edge.get("best_price")
    sb_implied = american_to_implied_probability(market_price)
    if sb_implied is not None:
        prob_row_parts.append(
            "<div class='sf-prob-cell'>"
            "<div class='lbl'>Mkt implied</div>"
            f"<div class='val'>{sb_implied * 100:.1f}%</div></div>"
        )

    # Pill chips — gold reserved for HIGH CONV, color hierarchy strict.
    pills: list[str] = [_pill(label, label_kind)]
    confidence_label_str, confidence_kind = confidence_word(edge.get("confidence"))
    if confidence_label_str != label and confidence_kind != "muted":
        pills.append(_pill(confidence_label_str, confidence_kind))
    chase = str(edge.get("chase_risk") or "").lower()
    if chase == "high":
        pills.append(_pill("AVOID CHASE", "red"))
    if odds_stale:
        pills.append(_pill("STALE", "red"))
    if fallback:
        pills.append(_pill("FALLBACK", "purple"))
    pill_html = " ".join(pills)

    reasons = (edge.get("reasons") or [])[:3]
    factors = edge.get("factors") or {}

    factor_section = render_factor_bars(factors)
    price_section = render_market_price_block(edge)
    model_section = _render_model_vs_market(edge)
    form_section = _render_recent_form(edge)
    movement_section = _render_movement_clv(edge)
    trust_section = render_trust_tags(edge, odds_source=odds_source, fallback=fallback)
    time_block = render_time_context(edge)

    reasons_html = ""
    if reasons:
        # Dedupe near-identical reasons (the engine sometimes echoes the
        # same fact in two ways). Keeps the bullet list under three.
        seen: set[str] = set()
        deduped: list[str] = []
        for r in reasons:
            key = re.sub(r"\s+", " ", str(r)).strip().lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append(r)
        if deduped:
            reasons_html = (
                "<ul class='sf-reasons'>"
                + "".join(f"<li>{r}</li>" for r in deduped[:3])
                + "</ul>"
            )

    links_html = render_link_buttons([
        ("Market", edge.get("market_url")),
        ("Source", edge.get("source_url")),
    ])

    body = f"""
    <div class="sf-card {card_kind}">
      <div class="sf-card-head">
        <div>
          <div class="sf-card-title">{_live_dot(edge)}{title}</div>
          <div class="sf-card-sub">{sub_label}</div>
          {time_block}
        </div>
        <div class="sf-prob-row" style="margin-left:auto;">
          {''.join(prob_row_parts)}
        </div>
      </div>
      <div style="margin-bottom:4px;">{pill_html}</div>
      {price_section}
      {model_section}
      {form_section}
      {factor_section}
      {movement_section}
      {('<div class="sf-section"><div class="sf-section-title">Why we like it</div>' + reasons_html + '</div>') if reasons_html else ''}
      <div class="sf-section">{trust_section}</div>
      {links_html}
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
    event_date = signal.get("event_date") or ""
    market_end = signal.get("market_end_date")
    signal_created = signal.get("signal_created_at") or signal.get("created_at")
    market_updated = signal.get("market_updated_at")
    platform = str(signal.get("market_platform") or "").lower()
    market_label = "Open Market"
    if "kalshi" in platform:
        market_label = "Kalshi"
    elif platform:
        market_label = "Polymarket"
    market_links = render_link_buttons([
        (market_label, signal.get("market_url")),
        ("Trader Profile", signal.get("trader_url")),
        ("Source", signal.get("source_url")),
    ])

    consensus_traders = signal.get("consensus_traders") or []
    tag_values: list[str] = []
    for t in consensus_traders[:4]:
        size = t.get("size_usd")
        size_label = fmt_money(size) if size is not None else "$—"
        tag_values.append(f"{t.get('name') or DASH} · {size_label}")
    remaining = max(0, len(consensus_traders) - 4)
    if remaining:
        tag_values.append(f"+{remaining} more")
    tag_badges = " ".join(badge(f"[{tag}]", "muted") for tag in tag_values) if tag_values else badge("[—]", "muted")
    consensus_summary = (
        f"{signal.get('consensus_wallets', 0)} wallets · "
        f"{fmt_money(signal.get('consensus_total_size'))} tracked · "
        f"largest {signal.get('consensus_largest', DASH)} · "
        f"{signal.get('consensus_direction', DASH)}"
    )

    pills_html = " ".join(filter(None, [
        _pill(tier, tier_kind),
        _pill(f"SRC {source}", "purple" if source == "Falcon" else "cyan") if source else "",
        _pill(side, "green" if side in {"YES", "BUY"} else ("red" if side in {"NO", "SELL"} else "muted")) if side else "",
    ]))

    event_label = event_date or (fmt_event_time(market_end) if market_end else DASH)
    updated_label = fmt_relative(market_updated)

    # Sharp money block: hide entirely when no wallet alignment data exists.
    consensus_wallets = signal.get("consensus_wallets") or 0
    consensus_total = signal.get("consensus_total_size")
    consensus_direction = signal.get("consensus_direction") or ""
    consensus_largest = signal.get("consensus_largest") or ""
    largest_size = next(
        (t.get("size_usd") for t in consensus_traders if t.get("name") == consensus_largest),
        None,
    )
    alignment_pct = wallet_alignment_percent(consensus_wallets, consensus_wallets)
    # When the watchlist hasn't recorded an opposing side we treat all
    # tracked wallets as aligned (100%). That's not a fabrication — it
    # mirrors what `consensus_wallets` already counts.
    sharp_block = ""
    if consensus_wallets:
        align_label = "100%" if alignment_pct is None else f"{alignment_pct:.0f}%"
        size_label = format_money_short(consensus_total)
        largest_label = (
            f"{consensus_largest} · {format_money_short(largest_size)}"
            if consensus_largest and largest_size is not None
            else consensus_largest or DASH
        )
        consensus_side = consensus_direction or DASH
        sharp_block = (
            "<div class='sf-section'>"
            "<div class='sf-section-title'>Sharp Wallet Alignment</div>"
            f"<div class='sf-sharp-pct'>{align_label}</div>"
            "<div class='sf-sharp-meta'>"
            f"{consensus_wallets} tracked wallet{'s' if consensus_wallets != 1 else ''} · {size_label} exposure"
            "</div>"
            f"<div class='sf-sharp-meta'>Largest entry: {largest_label}</div>"
            f"<div class='sf-sharp-meta'>Consensus: {consensus_side}</div>"
            "</div>"
        )

    body = f"""
    <div class="sf-card {card_kind}">
      <div class="sf-card-head">
        <div>
          <div class="sf-card-title">{trader} <span class="sf-card-sub">{wallet}</span></div>
          <div class="sf-card-sub">{market}</div>
          <div class="sf-card-sub">{event_label} · signal {fmt_relative(signal_created)} · mkt {updated_label}</div>
        </div>
        <div class="sf-prob-row" style="margin-left:auto;">
          <div class='sf-prob-cell'>
            <div class='lbl'>Score</div>
            <div class='val'><span class='sf-score {score_class(score)}'>{fmt_score(score)}</span></div>
          </div>
          <div class='sf-prob-cell'>
            <div class='lbl'>Entry</div>
            <div class='val'>{entry}</div>
          </div>
          <div class='sf-prob-cell'>
            <div class='lbl'>Size</div>
            <div class='val'>{size}</div>
          </div>
        </div>
      </div>
      <div style="margin-bottom:4px;">{pills_html}</div>
      {sharp_block}
      {('<div class="sf-card-row sf-meta">' + reason + '</div>') if reason else ''}
      {market_links}
    </div>
    """
    st.markdown(body, unsafe_allow_html=True)

    if len(consensus_traders) > 4:
        expander_key = f"consensus-{signal.get('market_id')}-{signal.get('side')}-{signal.get('outcome')}"
        with st.expander("Consensus wallets (full list)", expanded=False, key=expander_key):
            st.markdown(
                " ".join(
                    badge(
                        f"[{t.get('name') or DASH} · {fmt_money(t.get('size_usd')) if t.get('size_usd') is not None else '$—'}]",
                        "muted",
                    )
                    for t in consensus_traders
                ),
                unsafe_allow_html=True,
            )


def render_score_distribution(edges: list[dict[str, Any]], *, threshold: float = SCORE_HIGH_CONV_MIN) -> None:
    """Render score-bucket bars + summary stats (top, median, std, count
    above threshold). Helps explain why few high-conviction cards appear."""
    if not edges:
        render_empty_state(
            "NO SCORE DATA",
            "Run the MLB edge scan to populate score distribution.",
        )
        return
    scores = []
    for e in edges:
        s = _as_float(e.get("score"))
        if s is not None:
            scores.append(s)
    if not scores:
        render_empty_state(
            "NO SCORE DATA",
            "Edges exist but none carry a numeric score.",
        )
        return
    counts = score_distribution_fn(scores)
    max_count = max(counts.values()) or 1
    rows: list[str] = []
    for _, _, label in SCORE_BUCKETS:
        n = counts.get(label, 0)
        width = int(round(100 * n / max_count)) if max_count else 0
        rows.append(
            "<div class='sf-bucket-row'>"
            f"<span class='lbl'>{label}</span>"
            f"<span class='sf-bucket-bar'><span style='width:{width}%'></span></span>"
            f"<span class='val'>{n}</span>"
            "</div>"
        )
    top = max(scores)
    sorted_scores = sorted(scores)
    mid = len(sorted_scores) // 2
    median = (
        sorted_scores[mid]
        if len(sorted_scores) % 2
        else 0.5 * (sorted_scores[mid - 1] + sorted_scores[mid])
    )
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    stdev = variance ** 0.5
    above = sum(1 for s in scores if s >= threshold)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Top score today", f"{top:.1f}")
    metric_cols[1].metric("Median score", f"{median:.1f}")
    metric_cols[2].metric("Std deviation", f"{stdev:.1f}")
    metric_cols[3].metric(
        f"Above {int(threshold)}",
        above,
        delta=f"of {len(scores)}",
        delta_color="off",
    )
    st.markdown(
        "<div class='sf-card'>"
        "<div class='sf-section-title'>Score Distribution</div>"
        + "".join(rows)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_why_no_high_conviction(
    edges: list[dict[str, Any]], *, threshold: float = SCORE_HIGH_CONV_MIN
) -> None:
    """Compact diagnostic block — only renders when the high-conviction
    count is zero. Counts come from the returned edges so we never invent
    numbers."""
    if not edges:
        return
    high_conv = [e for e in edges if (_as_float(e.get("score")) or 0.0) >= threshold]
    if high_conv:
        return
    scores = [s for s in (_as_float(e.get("score")) for e in edges) if s is not None]
    top = max(scores) if scores else 0.0
    downgraded = sum(1 for e in edges if (e.get("warnings") or []))
    missing_history = sum(
        1
        for e in edges
        for w in (e.get("warnings") or [])
        if "statcast" in str(w).lower() or "history" in str(w).lower()
    )
    high_chase = sum(1 for e in edges if str(e.get("chase_risk") or "").lower() == "high")

    body = (
        "<div class='sf-card'>"
        "<div class='sf-section-title'>Why no high-conviction edges?</div>"
        f"<div class='sf-card-row'>Top score today: <b>{top:.1f}</b></div>"
        f"<div class='sf-card-row'>Required threshold: <b>{int(threshold)}</b></div>"
        f"<div class='sf-card-row'>Edges downgraded by warnings: <b>{downgraded}</b></div>"
        f"<div class='sf-card-row'>Edges missing history: <b>{missing_history}</b></div>"
        f"<div class='sf-card-row'>High chase risk: <b>{high_chase}</b></div>"
        "</div>"
    )
    st.markdown(body, unsafe_allow_html=True)


def render_empty_state(title: str, body: str, *, actions: list[tuple[str, callable]] | None = None) -> None:
    """Friendly empty state — title, body, optional action buttons."""
    checked = fmt_dt_mst(datetime.now(timezone.utc))
    instance_id = next(_EMPTY_STATE_COUNTER)
    st.markdown(
        f"<div class='sf-card'><div class='sf-card-title'>{title}</div>"
        f"<div class='sf-card-row sf-meta'>{body}</div>"
        f"<div class='sf-card-row sf-meta'>Last checked: {checked}</div></div>",
        unsafe_allow_html=True,
    )
    if not actions:
        return
    cols = st.columns(len(actions))
    for idx, (col, (label, fn)) in enumerate(zip(cols, actions)):
        with col:
            key_label = re.sub(r"[^a-zA-Z0-9_-]+", "-", label).strip("-").lower()
            if st.button(label, key=f"empty-{instance_id}-{idx}-{key_label}", use_container_width=True):
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
        payload = api_get("/health", timeout=HEALTH_TIMEOUT)
    except ApiError as exc:
        render_api_error(exc, prefix="Backend test failed")
        return
    st.success(
        f"Backend OK · env={payload.get('environment')} · "
        f"timestamp={payload.get('timestamp', '?')}"
    )


def action_update_mlb_closing_lines() -> None:
    with st.spinner("Updating MLB closing lines..."):
        try:
            result = api_post(
                "/mlb/debug/closing-lines/run",
                timeout=MLB_RUN_TIMEOUT,
            )
        except ApiError as exc:
            render_api_error(exc, prefix="Closing-line update failed")
            return
    status = result.get("status") or "ok"
    st.success(f"Closing-line update finished ({status}).")
    st.cache_data.clear()
    st.rerun()


def action_grade_mlb_results() -> None:
    with st.spinner("Grading MLB results..."):
        try:
            result = api_post("/mlb/debug/grade-results/run", timeout=MLB_RUN_TIMEOUT)
        except ApiError as exc:
            render_api_error(exc, prefix="MLB grading failed")
            return
    status = result.get("status") or "ok"
    st.success(f"MLB grading finished ({status}).")
    st.cache_data.clear()
    st.rerun()


def action_sync_pnl_wallets() -> None:
    with st.spinner("Syncing personal wallet P&L cache..."):
        try:
            result = api_post("/pnl/sync", timeout=SCAN_TIMEOUT)
        except ApiError as exc:
            render_api_error(exc, prefix="P&L wallet sync failed")
            return
    st.cache_data.clear()
    warnings = result.get("warnings") or []
    st.success(
        f"P&L sync: {result.get('new_trades', 0)} new fills, "
        f"{result.get('positions_rebuilt', 0)} positions rebuilt ({result.get('mode')})."
    )
    for warning in warnings[:3]:
        st.warning(warning)
    st.rerun()


# =============================================================================
# Cached fetchers
# =============================================================================


@st.cache_data(ttl=10, show_spinner=False)
def fetch_health() -> dict[str, Any] | None:
    try:
        payload = api_get("/health", timeout=HEALTH_TIMEOUT)
        payload["_frontend_state"] = "healthy"
        return payload
    except ApiError as exc:
        st.session_state.setdefault("_fetch_errors", {})["/health"] = exc
        if exc.is_timeout:
            fallback_ok = backend_fallback_probe()
            return {
                "ok": False,
                "status": "waking" if fallback_ok else "cold_start_timeout",
                "_frontend_state": "waking",
                "_fallback_probe_ok": fallback_ok,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        return None


@st.cache_data(ttl=20, show_spinner=False)
def fetch_ready() -> dict[str, Any]:
    return safe_get("/ready", default={})


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
def fetch_odds_providers() -> dict[str, Any]:
    return safe_get("/mlb/debug/odds-providers", default={})


@st.cache_data(ttl=30, show_spinner=False)
def fetch_odds_event_match() -> dict[str, Any]:
    return safe_get("/mlb/debug/odds/event-match", default={})


@st.cache_data(ttl=30, show_spinner=False)
def fetch_market_validation() -> dict[str, Any]:
    return safe_get("/mlb/debug/market-validation", default={})


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


@st.cache_data(ttl=10, show_spinner=False)
def fetch_pnl_tracker() -> dict[str, Any]:
    return safe_get("/pnl/tracker", default={"summary": {}, "open_positions": [], "closed_positions": []})


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
        consensus_traders = []
        for sig in grouped:
            name = sig.get("trader_nickname") or shorten_wallet(sig.get("wallet"))
            consensus_traders.append(
                {
                    "name": name or DASH,
                    "size_usd": _as_float(sig.get("size_usd")),
                }
            )
        consensus_traders = sorted(
            consensus_traders,
            key=lambda t: (t.get("size_usd") or 0.0),
            reverse=True,
        )
        total_tracked = sum(t.get("size_usd") or 0.0 for t in consensus_traders)
        largest = consensus_traders[0]["name"] if consensus_traders else DASH
        direction = " ".join(
            [str(first.get("side") or ""), str(first.get("outcome") or _market_label(first))]
        ).strip()
        position = dict(latest_sig)
        position["score"] = max(_as_float(s.get("score")) or 0.0 for s in grouped)
        position["confidence"] = max(_as_float(s.get("confidence")) or 0.0 for s in grouped)
        position["signal_type"] = " + ".join(types)
        position["entry_price"] = avg_entry
        position["size_usd"] = total_size
        position["signal_count"] = len(grouped)
        position["market_url"] = _market_url(first)
        position["trader_url"] = _trader_url(first)
        position["market_end_date"] = first.get("market_end_date")
        position["market_created_at"] = first.get("market_created_at")
        position["market_updated_at"] = first.get("market_updated_at")
        position["signal_created_at"] = latest_sig.get("created_at")
        position["consensus_traders"] = consensus_traders
        position["consensus_total_size"] = total_tracked
        position["consensus_wallets"] = len(consensus_traders)
        position["consensus_largest"] = largest
        position["consensus_direction"] = direction or DASH
        position.update(_parse_market_parts(first))
        positions.append(position)
    return sorted(positions, key=lambda p: (p.get("score") or 0.0), reverse=True)


def position_watchlist_notes(position: dict[str, Any]) -> str:
    notes: list[str] = []
    wallets = int(position.get("consensus_wallets") or 0)
    events = int(position.get("signal_count") or 0)
    total_size = _as_float(position.get("consensus_total_size")) or _as_float(position.get("size_usd")) or 0.0
    signal_type = str(position.get("signal_type") or "")
    largest = position.get("consensus_largest") or DASH

    if wallets >= 3:
        notes.append(f"{wallets} tracked wallets aligned")
    elif wallets == 2:
        notes.append("2-wallet consensus")
    elif wallets == 1:
        notes.append(f"single wallet: {largest}")

    if events > wallets and events > 1:
        notes.append(f"{events} aggregated events")
    if total_size >= 2500:
        notes.append(f"large tracked size {fmt_money(total_size)}")
    elif total_size >= 1000:
        notes.append(f"meaningful size {fmt_money(total_size)}")

    signal_bits = []
    if "multi_wallet_consensus" in signal_type:
        signal_bits.append("consensus")
    if "size_threshold" in signal_type:
        signal_bits.append("size threshold")
    if "trusted_wallet_entry" in signal_type:
        signal_bits.append("trusted entry")
    if signal_bits:
        notes.append("signals: " + ", ".join(signal_bits))

    if not notes:
        notes.append(position.get("reason") or "tracked wallet activity")
    return " | ".join(notes)


# =============================================================================
# Bail early if backend offline
# =============================================================================

st.session_state.setdefault("_fetch_errors", {})
health = fetch_health()

if health and health.get("_frontend_state") == "waking":
    attempts = int(st.session_state.get("_backend_wake_attempts", 0))
    st.session_state["_backend_wake_attempts"] = attempts + 1
    st.markdown(
        f"""
        <div class='sf-header'>
          <div class='sf-brand'>
            <span class='sf-brand-mark'>â—†</span>
            <span class='sf-brand-name'>SIGNALFORGE</span>
            <span class='sf-brand-tagline'>Prediction Market + MLB Edge Terminal</span>
          </div>
          <div>{badge("WAKING BACKEND", "purple")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        f"Backend URL: **{API_BASE}**\n\n"
        "The `/health` request timed out during the first backend probe. "
        "This usually means Render is cold-starting; retrying automatically."
    )
    if attempts < 4:
        time.sleep(min(3 + attempts * 2, 10))
        st.cache_data.clear()
        st.rerun()
    st.warning("Backend is still waking or degraded. Refresh in a moment if this does not advance.")
    st.stop()

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
ready = fetch_ready() or {}
traders = fetch_traders()
signals_all = fetch_signals(limit=500)
positions_all = aggregate_signals_to_positions(signals_all)
alerts_all = fetch_alerts(limit=200)
mlb_edges_all = fetch_mlb_edges(limit=100)
mlb_daily_card = fetch_mlb_daily_card()
mlb_sources = fetch_mlb_sources()
mlb_performance = fetch_mlb_performance()
pnl_payload = fetch_pnl_tracker()
odds_cache_payload = fetch_odds_cache()
odds_providers_payload = fetch_odds_providers()
event_match_payload = fetch_odds_event_match()
pitcher_props_payload = fetch_pitcher_props()
market_validation_payload = fetch_market_validation()

providers_block = ready.get("providers", {}) or {}
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

# Current time context (MST/UTC) used across the dashboard.
now_utc = datetime.now(timezone.utc)
now_local = now_utc.astimezone(TZ_MST)


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
    # Off by default — Pass-action edges should not dominate the terminal.
    # Flip on for full-slate inspection in MLB Terminal and Wallet Flow.
    show_pass_candidates = st.checkbox(
        "Show pass candidates in terminal", value=False
    )
    debug_mode = st.checkbox("Show raw JSON in Debug tab", value=True)

    st.markdown("<div class='sf-divider'></div>", unsafe_allow_html=True)
    st.markdown("**Connection**")
    st.markdown(f"<div class='sf-meta'>API URL: <code>{API_BASE}</code></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sf-meta'>Backend: {status_badge(True, ok_label='ok', bad_label='down')}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='sf-meta'>Last health check: {fmt_dt_mst(health.get('timestamp'))}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='sf-meta'>MST: {fmt_dt_mst(now_utc)}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='sf-meta'>UTC: {fmt_dt_utc(now_utc)}</div>",
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
        if not show_pass_candidates and score < 65:
            continue
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
tz_mst = ZoneInfo("America/Phoenix")
prev_refresh = st.session_state.get("_last_dashboard_refresh_at")
st.session_state["_last_dashboard_refresh_at"] = now_utc.isoformat()
last_refresh_age = fmt_relative(prev_refresh, now=now_utc) if prev_refresh else "just now"
local_label = now_local.strftime("%b %d, %Y %I:%M %p MST").replace(" 0", " ")
health_label = fmt_dt_mst(health.get("timestamp"))

st.markdown(
    f"""
    <div class='sf-header'>
      <div class='sf-brand'>
        <span class='sf-brand-mark'>◆</span>
        <span class='sf-brand-name'>SIGNALFORGE</span>
        <span class='sf-brand-tagline'>Prediction Market + MLB Edge Terminal</span>
      </div>
      <div>
        <span class='pulse-dot'></span>{backend_badge}{mkt_badge}{odds_cache_badge}
        <div class='sf-meta' style='margin-top:6px;'>MST: {local_label}</div>
                <div class='sf-meta'>Last refresh: {last_refresh_age}</div>
                <div class='sf-meta'>Backend health: {health_label}</div>
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

render_pnl_summary_cards(pnl_payload, fmt_money=fmt_money, fmt_num=fmt_num)


# =============================================================================
# Tab navigation
# =============================================================================

(
    tab_command,
    tab_mlb,
    tab_wallet,
    tab_pnl,
    tab_perf,
    tab_odds,
    tab_watchlist,
    tab_alerts,
    tab_health,
) = st.tabs([
    "Command Center",
    "MLB Terminal",
    "Wallet Flow",
    "P&L Tracker",
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
    # Compact live ribbon — replaces the old bulky System Health card. Bulky
    # provider diagnostics now live only in Odds Cache + Health/Debug.
    provs_for_ribbon = providers_block
    sources_online_n = sum(
        1 for key in ("falcon", "odds_api", "weather_api", "mlb_stats_api")
        if (provs_for_ribbon.get(key) or {}).get("configured")
    )
    rc_for_ribbon = (mlb_sources.get("row_counts") or {})
    odds_cache_seg = (
        f"<span class='ok'>Odds {odds_cache_status.title()}</span>"
        if odds_cache_status == "fresh"
        else (
            f"<span class='warn'>Odds {odds_cache_status.title()}</span>"
            if odds_cache_status in {"empty", "stale"}
            else f"<span class='info'>Odds {odds_cache_status.title()}</span>"
        )
    )
    last_refresh_seg = compact_time_ago(prev_refresh, now=now_utc) if prev_refresh else "just now"
    ribbon_html = (
        "<div class='sf-ribbon'>"
        f"<span class='seg'><span class='pulse-dot'></span><span class='ok'>LIVE</span></span>"
        f"<span class='sep'>·</span>"
        f"<span class='seg'><span class='lbl'>Games</span>{rc_for_ribbon.get('mlb_games', 0)}</span>"
        f"<span class='sep'>·</span>"
        f"<span class='seg'>{odds_cache_seg}</span>"
        f"<span class='sep'>·</span>"
        f"<span class='seg'><span class='lbl'>Sources</span>{sources_online_n}</span>"
        f"<span class='sep'>·</span>"
        f"<span class='seg'><span class='lbl'>Edges</span>{len(mlb_edges_all)}</span>"
        f"<span class='sep'>·</span>"
        f"<span class='seg'><span class='lbl'>High Conv</span>{len(high_conviction)}</span>"
        f"<span class='sep'>·</span>"
        f"<span class='seg'><span class='lbl'>Refresh</span>{last_refresh_seg}</span>"
        "</div>"
    )
    st.markdown(ribbon_html, unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="medium")

    with left:
        st.markdown("### TOP ACTIONABLE OPPORTUNITIES")
        top_decisions = sorted(
            [
                e for e in mlb_edges_all
                if (e.get("score") or 0) >= SCORE_ACTIONABLE_MIN
                and str(e.get("action") or "").lower() != "pass"
                and not e.get("odds_stale")
            ],
            key=lambda e: e.get("score") or 0,
            reverse=True,
        )[:5]
        if top_decisions:
            for edge in top_decisions:
                render_edge_card(edge)
        else:
            render_empty_state(
                "NO ACTIONABLE OPPORTUNITIES",
                "System is monitoring. Check MLB Terminal for watchlist candidates.",
                actions=[
                    ("Run MLB edge scan", action_run_mlb_edge_scan),
                    ("Refresh odds cache", action_refresh_odds_cache),
                ],
            )
        # If nothing crossed the high-conviction bar, explain why using
        # only real counts derived from the returned edges.
        render_why_no_high_conviction(mlb_edges_all)

        st.markdown("### Watchlist Candidates")
        watchlist_candidates = [
            e for e in mlb_edges_all
            if SCORE_ACTIONABLE_MIN <= (e.get("score") or 0) < SCORE_STRONG_MIN
            and str(e.get("action") or "").lower().startswith("watch")
        ][:5]
        if watchlist_candidates:
            for edge in watchlist_candidates:
                render_edge_card(edge)
        else:
            render_empty_state(
                "MARKET SILENT",
                f"No watchlist candidates in the {SCORE_ACTIONABLE_MIN}-{SCORE_STRONG_MIN - 1} band right now.",
                actions=[("Refresh odds cache", action_refresh_odds_cache)],
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
        st.markdown("### Market Pulse")
        rc = (mlb_sources.get("row_counts") or {})
        edges_with_warnings = sum(1 for e in mlb_edges_all if e.get("warnings"))

        def _chip(label: str, value: Any, kind: str = "") -> str:
            return (
                f"<div class='sf-chip {kind}'><span class='lbl'>{label}</span>"
                f"<span class='val'>{value}</span></div>"
            )

        chip_kind_for_count = lambda n, ok_min=1: ("ok" if n >= ok_min else "")  # noqa: E731
        chips = "".join([
            _chip("Games", rc.get("mlb_games", 0), chip_kind_for_count(rc.get("mlb_games", 0))),
            _chip("Edges", len(mlb_edges_all), chip_kind_for_count(len(mlb_edges_all))),
            _chip("High Conv", len(high_conviction), "ok" if high_conviction else ""),
            _chip("Odds Snaps", rc.get("odds_snapshots", 0), "info"),
            _chip("Prop Snaps", rc.get("pitcher_prop_snapshots", 0), "info"),
            _chip("Warnings", edges_with_warnings, "warn" if edges_with_warnings else ""),
            _chip("Missing Odds", len(missing_odds_edges), "warn" if missing_odds_edges else ""),
        ])
        st.markdown(f"<div class='sf-chips'>{chips}</div>", unsafe_allow_html=True)

        st.markdown("### Recent Alerts")
        recent_sent = [a for a in alerts_all if a.get("status") == "sent"][:5]
        if recent_sent:
            for a in recent_sent:
                channel = a.get("channel") or "?"
                ch_badge = badge(channel, "purple" if channel == "discord" else "cyan")
                st.markdown(
                    "<div class='sf-card'>"
                    + f"<div class='sf-card-row'>{ch_badge}<span class='sf-meta'> · {fmt_dt_mst(a.get('created_at'))}</span></div>"
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
                        "NO QUALIFYING EDGES",
                        "No qualifying edges in this band for today's slate.",
                    )
                else:
                    for row in rows[:3]:
                        render_edge_card(row)
    else:
        render_empty_state(
            "DAILY CARD MISSING",
            "Run the MLB edge scan to materialize the daily card.",
            actions=[("Run MLB edge scan", action_run_mlb_edge_scan)],
        )

    st.markdown("### Score Distribution")
    render_score_distribution(mlb_edges_all)

    st.markdown("### All Edges")
    if mlb_edges_all:
        # Sidebar toggle decides whether the terminal hides Pass-action edges.
        # Off by default to keep the terminal decision-first; flip on for
        # full-slate inspection.
        terminal_rows = (
            mlb_edges_all
            if show_pass_candidates
            else [e for e in mlb_edges_all if str(e.get("action") or "").lower() != "pass"]
        )
        if not terminal_rows:
            render_empty_state(
                "NO NON-PASS EDGES",
                "All current edges grade as Pass. Toggle 'Show pass candidates' in the sidebar to inspect them.",
            )
        else:
            df_edges = pd.DataFrame([
                {
                    "score": e.get("score"),
                    "label": confidence_label_fn(e.get("score"), e.get("action"), e.get("confidence"))[0],
                    "action": e.get("action"),
                    "confidence": confidence_word(e.get("confidence"))[0],
                    "type": (e.get("edge_type") or "").replace("_", " "),
                    "market": e.get("market"),
                    "side": (e.get("side") or "").title(),
                    "line": e.get("line"),
                    "best_book": e.get("best_book") or DASH,
                    "best_price": american_from_price(e.get("best_price")) or DASH,
                    "implied_prob": (
                        f"{american_to_implied_probability(e.get('best_price')) * 100:.1f}%"
                        if american_to_implied_probability(e.get("best_price")) is not None
                        else DASH
                    ),
                    "chase_risk": e.get("chase_risk"),
                    "warnings": "; ".join((e.get("warnings") or [])[:3]),
                }
                for e in terminal_rows
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
                    "best_price": st.column_config.TextColumn("best price (US)"),
                    "implied_prob": st.column_config.TextColumn("implied %"),
                },
            )
    else:
        render_empty_state(
            "NO MLB EDGES",
            "Run the MLB edge scan. If still empty, check the Odds Cache tab.",
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
# P&L Tracker
# =============================================================================

with tab_pnl:
    render_pnl_tracker(
        pnl_payload,
        sync_action=action_sync_pnl_wallets,
        fmt_money=fmt_money,
        fmt_num=fmt_num,
        fmt_pct=fmt_pct,
    )


# =============================================================================
# Performance / CLV
# =============================================================================

with tab_perf:
    st.markdown("### Performance Window")
    window = st.selectbox(
        "Date range",
        ["Last 7 days", "Today", "Yesterday", "Last 30 days", "All time", "Custom"],
        index=0,
    )
    if window == "Custom":
        st.caption("Custom date filtering is not yet wired; showing all available data.")
    else:
        st.caption("Backend does not yet expose date filters; showing all available data.")

    perf_actions = st.columns([1, 1, 6])
    with perf_actions[0]:
        if st.button("Update closing lines", use_container_width=True):
            action_update_mlb_closing_lines()
    with perf_actions[1]:
        if st.button("Grade MLB results", use_container_width=True):
            action_grade_mlb_results()

    st.markdown("### Today's Score Distribution")
    render_score_distribution(mlb_edges_all)
    render_why_no_high_conviction(mlb_edges_all)

    graded = perf_summary.get("graded_edges") or 0
    if not graded:
        render_empty_state(
            "WAITING FOR GRADED RESULTS",
            "No graded edges yet. Update closing lines, then grade MLB results.",
            actions=[
                ("Update closing lines", action_update_mlb_closing_lines),
                ("Grade MLB results", action_grade_mlb_results),
            ],
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
            render_empty_state("NO EDGE-TYPE BREAKDOWN", "Need at least one graded edge per type.")

        st.markdown("### By score band")
        by_band = mlb_performance.get("by_score_band") or []
        if by_band:
            df_band = pd.DataFrame(by_band).fillna(DASH)
            st.dataframe(df_band, use_container_width=True, hide_index=True,
                         height=min(220, 60 + 32 * len(df_band)))
        else:
            render_empty_state("NO SCORE-BAND BREAKDOWN", "Grade more edges to populate.")

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
        providers_diag = odds_providers_payload or {}
        primary_diag = providers_diag.get("primary") or {}
        backup_diag = providers_diag.get("backup") or {}
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

        st.markdown("### Provider diagnostics")
        diag_cols = st.columns(3)
        diag_cols[0].metric("Last provider", providers_diag.get("last_provider_used") or DASH)
        diag_cols[1].metric("Primary events", primary_diag.get("events_fetched", 0))
        diag_cols[2].metric("Backup events", backup_diag.get("events_fetched", 0))

        st.markdown(
            "<div class='sf-card'>"
            + f"<div class='sf-card-row'><span class='k'>Primary:</span>{primary_diag.get('name') or 'Odds-API'}"
            + f" · enabled={bool(primary_diag.get('enabled'))}"
            + f" · last success={fmt_dt(primary_diag.get('last_success_at'))}"
            + "</div>"
            + f"<div class='sf-card-row'><span class='k'>Totals found:</span>{primary_diag.get('totals_found', 0)}"
            + f" · Pitcher props found:{primary_diag.get('pitcher_props_found', 0)}"
            + "</div>"
            + f"<div class='sf-card-row'><span class='k'>Last error:</span>"
            + f"{primary_diag.get('last_error') or DASH}"
            + f" · at {fmt_dt(primary_diag.get('last_error_at'))}"
            + "</div>"
            + "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='sf-card'>"
            + f"<div class='sf-card-row'><span class='k'>Backup:</span>{backup_diag.get('name') or 'SportsGameOdds'}"
            + f" · enabled={bool(backup_diag.get('enabled'))}"
            + f" · last success={fmt_dt(backup_diag.get('last_success_at'))}"
            + "</div>"
            + f"<div class='sf-card-row'><span class='k'>Totals found:</span>{backup_diag.get('totals_found', 0)}"
            + f" · Pitcher props found:{backup_diag.get('pitcher_props_found', 0)}"
            + "</div>"
            + f"<div class='sf-card-row'><span class='k'>Last error:</span>"
            + f"{backup_diag.get('last_error') or DASH}"
            + f" · at {fmt_dt(backup_diag.get('last_error_at'))}"
            + "</div>"
            + "</div>",
            unsafe_allow_html=True,
        )

        recent_errors = providers_diag.get("last_errors") or []
        if recent_errors:
            st.markdown("**Recent provider errors**")
            st.code("\n".join(str(err) for err in recent_errors[-5:]), language="text")

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

    st.markdown("### Active wallet positions")
    watchlist_positions = sorted(
        positions_all,
        key=lambda p: (
            _as_float(p.get("confidence")) or 0.0,
            _as_float(p.get("score")) or 0.0,
            _as_float(p.get("consensus_total_size")) or _as_float(p.get("size_usd")) or 0.0,
        ),
        reverse=True,
    )
    if watchlist_positions:
        rows = []
        for p in watchlist_positions:
            wallet_disp = p.get("wallet") if show_full_wallet else shorten_wallet(p.get("wallet"))
            rows.append({
                "confidence": p.get("confidence") or 0.0,
                "score": p.get("score") or 0.0,
                "trader": p.get("trader_nickname") or DASH,
                "wallet": wallet_disp or DASH,
                "league": p.get("league") or DASH,
                "matchup": p.get("matchup") or _market_label(p),
                "contract": p.get("contract") or DASH,
                "position": " ".join(
                    str(part) for part in (p.get("side"), p.get("outcome")) if part
                ) or p.get("consensus_direction") or DASH,
                "avg_entry": p.get("entry_price"),
                "tracked_size": p.get("consensus_total_size") or p.get("size_usd"),
                "wallets": p.get("consensus_wallets") or 1,
                "events": p.get("signal_count") or 1,
                "source": p.get("source") or DASH,
                "notes": position_watchlist_notes(p),
                "market": p.get("market_url"),
                "trader_profile": p.get("trader_url"),
            })
        df_watch_positions = pd.DataFrame(rows).fillna(DASH)
        st.dataframe(
            df_watch_positions,
            use_container_width=True,
            hide_index=True,
            height=min(620, 60 + 32 * len(df_watch_positions)),
            column_config={
                "confidence": st.column_config.ProgressColumn(
                    "confidence", min_value=0, max_value=100, format="%.1f"
                ),
                "score": st.column_config.ProgressColumn(
                    "score", min_value=0, max_value=100, format="%.1f"
                ),
                "avg_entry": st.column_config.NumberColumn("avg entry", format="%.3f"),
                "tracked_size": st.column_config.NumberColumn("tracked size", format="$%.0f"),
                "wallets": st.column_config.NumberColumn("wallets", format="%d"),
                "events": st.column_config.NumberColumn("events", format="%d"),
                "market": st.column_config.LinkColumn("market", display_text="open"),
                "trader_profile": st.column_config.LinkColumn("trader", display_text="profile"),
            },
        )
    else:
        render_empty_state(
            "No active wallet positions.",
            "Run a wallet scan to populate tracked wallet positions.",
            actions=[("Run wallet scan", action_run_wallet_scan)],
        )

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

    st.markdown("### Market validation")
    mv = market_validation_payload or {}
    mv1, mv2, mv3, mv4 = st.columns(4)
    mv1.metric("Rejected markets", mv.get("rejected_markets", 0))
    mv2.metric("Invalid odds", mv.get("invalid_odds", 0))
    mv3.metric("Malformed lines", mv.get("malformed_lines", 0))
    mv4.metric("Provider mismatches", mv.get("provider_mismatches", 0))
    mv_samples = mv.get("samples") or {}
    sample_blocks = [
        ("Rejected markets", mv_samples.get("rejected_markets") or []),
        ("Invalid odds", mv_samples.get("invalid_odds") or []),
        ("Malformed lines", mv_samples.get("malformed_lines") or []),
        ("Provider mismatches", mv_samples.get("provider_mismatches") or []),
    ]
    for title, rows in sample_blocks:
        if not rows:
            continue
        st.markdown(f"**{title} (latest)**")
        df_mv = pd.DataFrame(rows).fillna("")
        st.dataframe(df_mv.head(12), use_container_width=True, hide_index=True)

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
        with st.expander("/mlb/debug/odds-providers", expanded=False):
            st.json(odds_providers_payload)
        with st.expander("/mlb/debug/odds/event-match", expanded=False):
            st.json(event_match_payload)
        with st.expander("/mlb/debug/market-validation", expanded=False):
            st.json(market_validation_payload)
        with st.expander("/dashboard-summary", expanded=False):
            st.json(summary)
    else:
        st.caption("Raw JSON hidden. Enable 'Show raw JSON' in the sidebar to expose payloads.")
