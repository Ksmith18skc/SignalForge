"""SignalForge — Institutional sports-betting intelligence terminal.

Streamlit front-end for the SignalForge FastAPI backend. The design target
is "Bloomberg Terminal meets high-end sportsbook trading desk": dense,
decision-first, terminal-styled. No backend logic lives here — every fact
on the page is fetched from a SignalForge endpoint.

Backend URL: $SIGNALFORGE_API_URL (default http://localhost:8000).
Launch: `streamlit run dashboard.py`.
"""

from __future__ import annotations

import html
import json
import logging
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

# Dashboard-side debug logger. Lines tagged sf.dash trace rerun triggers,
# button clicks, job lifecycle events, backend request boundaries, and any
# active_job session_state mutations — exactly what we need to debug the
# "why is the spinner running forever?" failure mode.
logger = logging.getLogger("signalforge.dashboard")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s sf.dash: %(message)s")
    )
    logger.addHandler(_handler)
logger.setLevel(os.environ.get("SIGNALFORGE_DASHBOARD_LOG_LEVEL", "INFO").upper())
logger.propagate = False

from app.components.pnl_dashboard import render_pnl_summary_cards, render_pnl_tracker
from app.services import wallet_market_resolver as wmr
from app.utils.dashboard_format import (
    SCORE_ACTIONABLE_MIN,
    SCORE_BUCKETS,
    SCORE_HIGH_CONV_MIN,
    SCORE_STRONG_MIN,
    american_from_price,
    american_to_implied_probability,
    best_executable_edge,
    build_consensus_wallets,
    compact_time_ago,
    consensus_wallets_chips_html,
    conviction_tier,
    edge_risk_flags,
    edge_source_stack,
    executable_edge_rows,
    format_score_contributions,
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
    wallet_consensus_groups,
)

API_BASE = os.environ.get("SIGNALFORGE_API_URL", "http://localhost:8000").rstrip("/")
# Keep normal reads tight (10s) so a stalled backend can't hang the dashboard.
# /health gets its own longer budget *only* on the first wake-up attempt so
# Render's cold start (~30-45s) doesn't masquerade as "backend offline".
DEFAULT_TIMEOUT = 10.0
HEALTH_TIMEOUT = 45.0
HEALTH_WARM_TIMEOUT = 8.0
SCAN_TIMEOUT = 90.0
MLB_RUN_TIMEOUT = 180.0
# When set (via env or sidebar toggle) the dashboard renders shell + empty data
# without ever calling the backend. Useful for design work and for unblocking
# the UI when Render is degraded.
OFFLINE_MODE = os.environ.get("SIGNALFORGE_OFFLINE_MODE", "").lower() in {"1", "true", "yes"}
RETRY_PATH_PREFIXES = (
    "/health",
    "/api/status",
    "/ready",
    "/run-scan",
    "/mlb/debug/odds-cache",
    "/mlb/edges/run",
)
_EMPTY_STATE_COUNTER = count()
# Per-render monotonic counter for widget keys that would otherwise collide
# when the same signal/position is rendered in more than one tab (Wallets +
# Positions both call render_wallet_card) or aggregated across signals.
_WIDGET_KEY_COUNTER = count()


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
.sf-link { color: var(--cyan); text-decoration: none; }
.sf-link:hover { text-decoration: underline; }
.sf-source-stack { margin: 6px 0; display: flex; flex-wrap: wrap; gap: 4px; }
.sf-best-edge { margin-top: 6px; font-size: 0.82rem; }
.sf-contrib-row { display: flex; justify-content: space-between; font-size: 0.82rem; margin: 1px 0; }
.sf-contrib-pts { font-variant-numeric: tabular-nums; font-weight: 600; }
.sf-contrib-pts.green { color: var(--green); }
.sf-contrib-pts.red { color: var(--red); }
.sf-contrib-pts.muted { color: var(--muted); }
.sf-technical { margin-top: 6px; }
.sf-technical > summary { cursor: pointer; list-style: revert; }
.sf-wallet-row { font-size: 0.82rem; margin: 2px 0; }
.sf-wallet-debug { margin-top: 6px; }
.sf-wallet-debug summary { cursor: pointer; }
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
                    r = c.post(path, params=params, json=json, timeout=timeout)
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


def api_post(
    path: str,
    json: Any = None,
    timeout: float = DEFAULT_TIMEOUT,
    params: dict[str, Any] | None = None,
) -> Any:
    return _request_json("POST", path, params=params, json=json, timeout=timeout)


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
    stashed in session_state so the Health tab can surface it.

    In OFFLINE_MODE we skip the network entirely so the dashboard renders
    cards/tabs against the supplied defaults instead of waiting on timeouts.
    """
    if OFFLINE_MODE:
        return default
    try:
        return api_get(path, params=params, timeout=DEFAULT_TIMEOUT)
    except ApiError as exc:
        errors = st.session_state.setdefault("_fetch_errors", {})
        errors[path] = exc
        return default


# =============================================================================
# Formatting + tier helpers
# =============================================================================

DASH = "—"
TZ_MST = ZoneInfo("America/Phoenix")
CARD_DATE = datetime.now(TZ_MST).date().isoformat()


def _resolve_perf_window(choice: str) -> tuple[int | None, str | None]:
    """Map a selectbox label to (days, single_date) for the performance API.

    Dates are Arizona/MST so the rollover lines up with the backend default
    (``app.services.mlb_performance.arizona_today``).
    """
    today_az = datetime.now(TZ_MST).date()
    if choice == "Today":
        return None, today_az.isoformat()
    if choice == "Yesterday":
        return None, (today_az - timedelta(days=1)).isoformat()
    if choice == "Last 7 days":
        return 7, None
    if choice == "Last 30 days":
        return 30, None
    if choice == "All time":
        return None, None
    return 7, None


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


def edge_prediction_score(edge: dict[str, Any]) -> float:
    value = edge.get("prediction_score")
    if value is None:
        value = edge.get("score")
    return _as_float(value) or 0.0


def edge_execution_score(edge: dict[str, Any]) -> float:
    return _as_float(edge.get("execution_score")) or 0.0


def edge_wallet_alignment_score(edge: dict[str, Any]) -> float:
    return _as_float((edge.get("factors") or {}).get("wallet_alignment")) or 0.0


def edge_legacy_score(edge: dict[str, Any]) -> float:
    value = edge.get("legacy_score")
    if value is None:
        value = edge.get("score")
    return _as_float(value) or 0.0


def edge_decision_sort_key(edge: dict[str, Any]) -> tuple[float, float, float]:
    return (
        edge_prediction_score(edge),
        edge_wallet_alignment_score(edge),
        edge_execution_score(edge),
    )


def edge_pricing_sort_key(edge: dict[str, Any]) -> tuple[float, float, float]:
    factors = edge.get("factors") or {}
    price_edge = (
        _as_float(factors.get("sportsbook_price_edge"))
        or _as_float(factors.get("price_edge"))
        or _as_float(factors.get("odds_edge"))
        or 0.0
    )
    return (edge_legacy_score(edge), price_edge, edge_execution_score(edge))


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


_ALERT_SLUG_RE = re.compile(r"\b(mlb-[a-z0-9\-]+)", re.IGNORECASE)


def _edge_matchup_index(edges: list[dict[str, Any]]) -> dict[frozenset[str], str]:
    """Map each today-card edge's team pair -> a short matchup label, so a
    Falcon alert for the same game can point back at its edge card."""
    index: dict[frozenset[str], str] = {}
    for edge in edges or []:
        away = wmr.fullname_to_abbr(edge.get("away_team"))
        home = wmr.fullname_to_abbr(edge.get("home_team"))
        if not away or not home:
            continue
        pair = frozenset({away, home})
        index.setdefault(pair, f"{team_short(edge.get('away_team'))} @ {team_short(edge.get('home_team'))}")
    return index


def _alert_edge_reference(message: str | None, edge_matchups: dict[frozenset[str], str]) -> str:
    """Return a small '→ see edge card' note when the alert's market slug
    resolves to a game that has an edge on today's card; else empty string."""
    if not message or not edge_matchups:
        return ""
    match = _ALERT_SLUG_RE.search(message)
    if not match:
        return ""
    parsed = wmr.parse_market_slug(match.group(1).lower())
    if parsed is None:
        return ""
    label = edge_matchups.get(parsed.team_pair())
    if not label:
        return ""
    return f"<div class='sf-card-row sf-meta'>→ see edge card: {label}</div>"


_WALLET_TAG_KIND = {
    "WALLET CONFIRMED": "green",
    "ELITE AGREEMENT": "gold",
    "ELITE DISAGREEMENT": "red",
    "CROWDED SIDE": "purple",
    "NO WALLET DATA": "muted",
}


def _wallet_row(w: dict[str, Any]) -> str:
    name = w.get("trader_name") or "wallet"
    profile = w.get("profile_url")
    name_html = (
        f'<a class="sf-link" href="{profile}" target="_blank" rel="noopener">{name}</a>'
        if profile
        else name
    )
    tier = str(w.get("tier") or "").lower()
    tier_kind = {"elite": "gold", "trusted": "green"}.get(tier, "muted")
    size = fmt_money(w.get("size_usd"))
    entry = w.get("avg_entry")
    entry_str = f" @ {entry:.2f}" if isinstance(entry, (int, float)) else ""
    market = w.get("market_url")
    mkt_html = (
        f' · <a class="sf-link" href="{market}" target="_blank" rel="noopener">market</a>'
        if market
        else ""
    )
    return (
        f"<div class='sf-wallet-row'>{name_html} {_pill(tier or 'neutral', tier_kind)} "
        f"<span class='sf-meta'>{size}{entry_str}{mkt_html}</span></div>"
    )


def render_wallet_flow_section(edge: dict[str, Any]) -> str:
    """`Wallet Flow Confirmation` — tracked-wallet consensus + contributors."""
    ctx = edge.get("wallet_context") or None
    if not ctx:
        return ""

    tags = ctx.get("tags") or []
    tag_html = " ".join(_pill(t, _WALLET_TAG_KIND.get(t, "muted")) for t in tags)

    tracked = int(ctx.get("tracked_wallet_count") or 0)
    if tracked == 0:
        # NO WALLET DATA: show the tag + the reason so the card is honest.
        reason = ((ctx.get("debug") or {}).get("no_match_reason")) or "no tracked-wallet activity"
        return (
            "<div class='sf-section'>"
            "<div class='sf-section-title'>Wallet Flow Confirmation</div>"
            f"<div style='margin-bottom:4px;'>{tag_html}</div>"
            f"<div class='sf-meta'>{reason}</div>"
            f"{_render_wallet_debug(ctx)}"
            "</div>"
        )

    consensus = ctx.get("consensus_pct")
    consensus_str = f"{consensus:.0f}%" if isinstance(consensus, (int, float)) else DASH
    aligned_exp = fmt_money(ctx.get("aligned_exposure_usd"))
    opposing_exp = fmt_money(ctx.get("opposing_exposure_usd"))

    summary = (
        "<div class='sf-price-grid'>"
        f"<div class='sf-price-cell'><div class='lbl'>Consensus</div><div class='val'>{consensus_str}</div></div>"
        f"<div class='sf-price-cell'><div class='lbl'>Aligned</div><div class='val'>{aligned_exp}</div></div>"
        f"<div class='sf-price-cell'><div class='lbl'>Opposing</div><div class='val'>{opposing_exp}</div></div>"
        f"<div class='sf-price-cell'><div class='lbl'>Wallets</div><div class='val'>{tracked}</div></div>"
        "</div>"
    )

    aligned = (ctx.get("aligned_wallets") or [])[:3]
    opposing = (ctx.get("opposing_wallets") or [])[:3]
    aligned_html = ""
    if aligned:
        aligned_html = (
            "<div class='sf-meta' style='margin-top:6px;'>Aligned wallets</div>"
            + "".join(_wallet_row(w) for w in aligned)
        )
    opposing_html = ""
    if opposing:
        opposing_html = (
            "<div class='sf-meta' style='margin-top:6px;'>Opposing wallets</div>"
            + "".join(_wallet_row(w) for w in opposing)
        )

    return (
        "<div class='sf-section'>"
        "<div class='sf-section-title'>Wallet Flow Confirmation</div>"
        f"<div style='margin-bottom:4px;'>{tag_html}</div>"
        f"{summary}{aligned_html}{opposing_html}"
        f"{_render_wallet_debug(ctx)}"
        "</div>"
    )


def _render_wallet_debug(ctx: dict[str, Any]) -> str:
    """Expandable join explainability — uses a native <details> block so it
    stays inside the card HTML rather than a separate Streamlit widget."""
    debug = ctx.get("debug") or {}
    if not debug:
        return ""
    key = debug.get("normalized_key") or {}
    matched = debug.get("matched_slugs") or []
    rows = [
        ("Normalized key", f"{key.get('league')} {key.get('away_abbr')}@{key.get('home_abbr')} "
                           f"{key.get('market_type')} {key.get('line')} {key.get('outcome')}"),
        ("Sportsbook event", debug.get("sportsbook_event_id") or DASH),
        ("Line tolerance", debug.get("line_tolerance")),
        ("Candidate markets", debug.get("candidate_markets_considered")),
        ("Matched market(s)", ", ".join(matched) if matched else DASH),
    ]
    if debug.get("no_match_reason"):
        rows.append(("No-match reason", debug.get("no_match_reason")))
    body = "".join(
        f"<div class='sf-meta'><b>{label}:</b> {value}</div>" for label, value in rows
    )
    return (
        "<details class='sf-wallet-debug'>"
        "<summary class='sf-meta'>Wallet join debug</summary>"
        f"{body}</details>"
    )


def _render_source_stack(edge: dict[str, Any]) -> str:
    """#1 Edge source stack — where the edge comes from (top-level pills)."""
    sources = edge_source_stack(edge)
    if not sources:
        return ""
    pills = " ".join(_pill(label, kind) for label, kind in sources)
    return f"<div class='sf-source-stack'>{pills}</div>"


def _pct(value: Any) -> str:
    v = _as_float(value)
    return f"{v * 100:.1f}%" if v is not None else DASH


def _render_execution(edge: dict[str, Any]) -> str:
    """#2 Execution — venue prices, sportsbook implied, SF fair, best edge."""
    rows = executable_edge_rows(edge)
    best = best_executable_edge(edge)
    sf_fair = _as_float(edge.get("sf_fair_probability") or edge.get("calibrated_probability"))
    factors = edge.get("factors") or {}
    sportsbook_edge = (
        _as_float(factors.get("sportsbook_price_edge"))
        or _as_float(factors.get("price_edge"))
        or _as_float(factors.get("odds_edge"))
    )

    cells: list[str] = []
    if edge.get("execution_score") is not None:
        cells.append(
            "<div class='sf-price-cell'><div class='lbl'>Execution Score</div>"
            f"<div class='val'><span class='sf-score {score_class(edge.get('execution_score'))}'>"
            f"{fmt_score(edge.get('execution_score'))}</span></div></div>"
        )
    if sportsbook_edge is not None:
        cells.append(
            "<div class='sf-price-cell'><div class='lbl'>Sportsbook Price Edge</div>"
            f"<div class='val'>{sportsbook_edge:.1f}</div></div>"
        )
    have_pm = any("Polymarket" in r["venue"] or "Kalshi" in r["venue"] for r in rows)
    for r in rows:
        price = r.get("price")
        price_str = f"{price:.2f}" if isinstance(price, (int, float)) else DASH
        venue = html.escape(str(r["venue"]))
        url = r.get("url")
        venue_html = f"<a class='sf-link' href='{url}' target='_blank' rel='noopener'>{venue}</a>" if url else venue
        cells.append(
            f"<div class='sf-price-cell'><div class='lbl'>{venue_html}</div>"
            f"<div class='val'>{price_str} · {_pct(r.get('implied_prob'))}</div></div>"
        )
    if not have_pm:
        cells.append(
            "<div class='sf-price-cell'><div class='lbl'>Kalshi / Polymarket</div>"
            "<div class='val sf-meta'>not listed</div></div>"
        )
    cells.append(
        "<div class='sf-price-cell'><div class='lbl'>SF fair prob</div>"
        f"<div class='val purple'>{_pct(sf_fair) if sf_fair is not None else 'uncalibrated'}</div></div>"
    )

    best_html = ""
    if best:
        kind = "green" if best["edge_pct"] > 0 else "red"
        best_venue = html.escape(str(best["venue"]))
        best_pill = _pill(f"{best['edge_pct']:+.1f} pts @ {best_venue}", kind)
        best_html = f"<div class='sf-best-edge'>Best executable edge: {best_pill}</div>"

    return (
        "<div class='sf-section'><div class='sf-section-title'>Execution</div>"
        f"<div class='sf-price-grid'>{''.join(cells)}</div>{best_html}</div>"
    )


_BREAKDOWN_LABELS = {
    "projection_edge": "Projection edge",
    "wallet_alignment": "Wallet alignment",
    "pitcher_matchup": "Pitcher matchup",
    "environment": "Environment",
    "model_confidence": "Model confidence",
    "sportsbook_price_edge": "Sportsbook price edge",
    "line_movement": "Line movement",
    "clv_signal": "CLV signal",
    "market_quality": "Market quality",
}


def _render_axis_breakdown(
    edge: dict[str, Any],
    *,
    axis: str,
    title: str,
    score_key: str,
    breakdown_key: str,
) -> str:
    breakdown = edge.get(breakdown_key) or {}
    if not isinstance(breakdown, dict) or not breakdown:
        return ""
    factors = edge.get("factors") or {}
    rows: list[str] = []
    for name, contribution in breakdown.items():
        try:
            pts = float(contribution)
        except (TypeError, ValueError):
            continue
        raw = factors.get(name)
        if raw is None and name == "sportsbook_price_edge":
            raw = factors.get("price_edge") or factors.get("odds_edge")
        raw_float = _as_float(raw)
        raw_label = (
            f"<span class='sf-meta'>{raw_float:.1f}</span>"
            if raw_float is not None
            else ""
        )
        kind = "pos" if pts >= 0 else "neg"
        label = _BREAKDOWN_LABELS.get(name, name.replace("_", " ").title())
        rows.append(
            "<div class='sf-contrib-row'>"
            f"<span class='lbl'>{html.escape(label)}</span>"
            f"{raw_label}<span class='sf-contrib-pts {kind}'>{pts:+.1f}</span>"
            "</div>"
        )
    if not rows:
        return ""
    score = edge.get(score_key)
    score_line = (
        f"<div class='sf-meta'>Baseline 50 + factor contributions; {axis} score {fmt_score(score)}</div>"
        if score is not None
        else "<div class='sf-meta'>Baseline 50 + contributions</div>"
    )
    return (
        f"<div class='sf-section'><div class='sf-section-title'>{title}</div>"
        + "".join(rows)
        + score_line
        + "</div>"
    )


def _render_prediction_breakdown(edge: dict[str, Any]) -> str:
    return _render_axis_breakdown(
        edge,
        axis="prediction",
        title="Prediction Breakdown",
        score_key="prediction_score",
        breakdown_key="prediction_breakdown",
    )


def _render_execution_breakdown(edge: dict[str, Any]) -> str:
    return _render_axis_breakdown(
        edge,
        axis="execution",
        title="Execution Breakdown",
        score_key="execution_score",
        breakdown_key="execution_breakdown",
    )


def _render_history(edge: dict[str, Any]) -> str:
    """#4 Historical regime performance — record of similar graded edges."""
    band = edge.get("score_band_performance") or {}
    sample = int(band.get("sample_size") or 0)
    if sample == 0:
        return (
            "<div class='sf-section'><div class='sf-section-title'>Historical performance</div>"
            "<div class='sf-meta'>Not enough graded history yet for similar signals.</div></div>"
        )
    wr = band.get("win_rate")
    roi = band.get("roi_units")
    clv = band.get("avg_clv_points")
    grid = (
        "<div class='sf-price-grid'>"
        f"<div class='sf-price-cell'><div class='lbl'>Win rate</div><div class='val'>{_pct(wr)}</div></div>"
        f"<div class='sf-price-cell'><div class='lbl'>ROI (u/edge)</div><div class='val'>{roi if roi is not None else DASH}</div></div>"
        f"<div class='sf-price-cell'><div class='lbl'>Avg CLV</div><div class='val'>{clv if clv is not None else DASH}</div></div>"
        f"<div class='sf-price-cell'><div class='lbl'>Sample</div><div class='val'>{sample}</div></div>"
        "</div>"
    )
    return (
        "<div class='sf-section'><div class='sf-section-title'>Historical performance "
        f"<span class='sf-meta'>· {html.escape(str(band.get('edge_type') or ''))} {html.escape(str(band.get('score_band') or ''))}</span>"
        f"</div>{grid}</div>"
    )


def _render_score_interpretation(edge: dict[str, Any]) -> str:
    """#5 Score interpretation — conviction tier + calibrated hit probability."""
    score = edge.get("prediction_score") if edge.get("prediction_score") is not None else edge.get("score")
    tier_label, tier_kind = conviction_tier(score)
    calibrated = _as_float(edge.get("calibrated_probability"))
    band = edge.get("score_band_performance") or {}
    cal_str = _pct(calibrated) if calibrated is not None else "uncalibrated"
    record = ""
    if int(band.get("sample_size") or 0) > 0:
        record = (
            f"<span class='sf-meta'> · band record {band.get('wins', 0)}-"
            f"{band.get('losses', 0)}-{band.get('pushes', 0)}</span>"
        )
    return (
        "<div class='sf-section'><div class='sf-section-title'>Prediction interpretation</div>"
        f"<div>{_pill(tier_label, tier_kind)} "
        f"<span class='sf-meta'>calibrated hit prob: {cal_str}</span>{record}</div></div>"
    )


def _render_risk_flags(edge: dict[str, Any]) -> str:
    """#6 Risk factors — explicit chips, only those that apply."""
    flags = edge_risk_flags(edge)
    if not flags:
        return (
            "<div class='sf-section'><div class='sf-section-title'>Risk factors</div>"
            "<div class='sf-meta'>No notable risk flags.</div></div>"
        )
    chips = " ".join(_pill(label, kind) for label, kind in flags)
    return (
        "<div class='sf-section'><div class='sf-section-title'>Risk factors</div>"
        f"<div>{chips}</div></div>"
    )


def _render_score_decomposition(edge: dict[str, Any]) -> str:
    """#7 Score decomposition — additive +/- point contributions."""
    wc = edge.get("wallet_context") or {}
    rows = format_score_contributions(
        edge.get("score_contributions"),
        wallet_adjustment=wc.get("confidence_adjustment"),
    )
    if not rows:
        return ""
    items = "".join(
        f"<div class='sf-contrib-row'><span class='lbl'>{html.escape(label)}</span>"
        f"<span class='sf-contrib-pts {kind}'>{pts:+.1f}</span></div>"
        for label, pts, kind in rows
    )
    base = "<div class='sf-meta'>Baseline 50 + contributions ≈ score</div>"
    return (
        "<div class='sf-section'><div class='sf-section-title'>Score decomposition</div>"
        f"{items}{base}</div>"
    )


_EXECUTION_FACTOR_KEYS = {
    "odds_edge",
    "price_edge",
    "sportsbook_price_edge",
    "movement",
    "line_movement",
    "clv_signal",
    "market_quality",
}


def _non_execution_factors(edge: dict[str, Any]) -> dict[str, Any]:
    factors = edge.get("factors") or {}
    if edge.get("prediction_score") is None:
        return factors
    return {
        key: value
        for key, value in factors.items()
        if key not in _EXECUTION_FACTOR_KEYS
    }


def render_edge_card(edge: dict[str, Any]) -> None:
    """Premium edge card. Single source of truth for the visual hierarchy
    used across Command Center, MLB Terminal, Daily Card."""
    score = edge.get("prediction_score") if edge.get("prediction_score") is not None else edge.get("score")
    prediction_score = edge.get("prediction_score")
    execution_score = edge.get("execution_score")
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
    if prediction_score is not None or execution_score is not None:
        prob_row_parts: list[str] = [
            f"<div class='sf-prob-cell'><div class='lbl'>Prediction Score</div>"
            f"<div class='val'><span class='sf-score {score_class(score)}'>{fmt_score(score)}</span></div></div>",
            f"<div class='sf-prob-cell'><div class='lbl'>Execution Score</div>"
            f"<div class='val'><span class='sf-score {score_class(execution_score)}'>{fmt_score(execution_score)}</span></div></div>",
        ]
    else:
        prob_row_parts = [
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

    reasons = (edge.get("reasons") or [])[:3]
    factors = edge.get("factors") or {}

    # --- Prioritized sections (#10 visual hierarchy) ---
    # Top-of-card status badges (Wallet Confirmed / Cheap Price Trap /
    # Prediction Market Listed / Stale Odds / etc.) replace the older
    # technical pill stack so the homepage reads like a trader card.
    primary_badges = _primary_badges(edge)
    # Structured Market/Model/Edge/Wallet trader block — sits at the top
    # so the executable line, projection, and edge-vs-line are visible
    # without expanding the technical-factors block.
    trader_debug = _trader_debug_block(edge)
    source_stack = _render_source_stack(edge)          # 1
    execution_section = _render_execution(edge)        # 2
    wallet_section = render_wallet_flow_section(edge)  # 3
    history_section = _render_history(edge)            # 4
    interp_section = _render_score_interpretation(edge)  # 5
    risk_section = _render_risk_flags(edge)            # 6
    prediction_breakdown = _render_prediction_breakdown(edge)
    execution_breakdown = _render_execution_breakdown(edge)
    time_block = render_time_context(edge)             # 8
    links_html = render_link_buttons([                 # 9
        ("Market", (edge.get("wallet_context") or {}).get("execution", {}).get("market_url") if isinstance((edge.get("wallet_context") or {}).get("execution"), dict) else None),
        ("Sportsbook", edge.get("source_url")),
        ("Edge detail", edge.get("market_url")),
    ])

    reasons_html = ""
    if reasons:
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
                + "".join(f"<li>{html.escape(str(r))}</li>" for r in deduped[:3])
                + "</ul>"
            )

    # Technical factors demoted into a collapsed block (was the primary view).
    technical = (
        _render_model_vs_market(edge)
        + _render_recent_form(edge)
        + render_factor_bars(_non_execution_factors(edge))
        + ("<div class='sf-section'><div class='sf-section-title'>Why we like it</div>" + reasons_html + "</div>" if reasons_html else "")
        + "<div class='sf-section'>" + render_trust_tags(edge, odds_source=odds_source, fallback=fallback) + "</div>"
    )
    technical_block = (
        "<details class='sf-technical'><summary class='sf-section-title'>Technical factors</summary>"
        f"{technical}</details>"
    )

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
      {primary_badges}
      {trader_debug}
      {source_stack}
      {execution_section}
      {wallet_section}
      {history_section}
      {interp_section}
      {risk_section}
      {prediction_breakdown}
      {execution_breakdown}
      {links_html}
      {technical_block}
    </div>
    """
    st.markdown(body, unsafe_allow_html=True)


def _trader_card_title(signal: dict[str, Any]) -> str:
    """Build the "TOR @ BAL — Over 8.5" headline.

    The title is composed from the slug whenever the slug parses into
    a recognized matchup + market_type, because the slug always carries
    the *executable* market line. The DB-stored market_title is the
    fallback path — historically it sometimes inherited a model
    projection (e.g. "mlb hou@tex total 9.3 Over") from upstream
    ingestion bugs, so we use it only when slug parsing fails.
    """
    slug = signal.get("market_slug") or ""
    parsed = wmr.parse_market_slug(slug) if slug else None
    if parsed and parsed.away_abbr and parsed.home_abbr:
        matchup = f"{parsed.away_abbr.upper()} @ {parsed.home_abbr.upper()}"
        side = (signal.get("side") or signal.get("outcome") or "").strip()
        side_label = side.title() if side else ""
        line_str = ""
        if parsed.line is not None:
            f = float(parsed.line)
            line_str = f"{int(round(f))}" if abs(f - round(f)) < 1e-9 else f"{f:.1f}"
        if parsed.market_type == "total":
            if side_label and line_str:
                return f"{matchup} — {side_label} {line_str}"
            return f"{matchup} — Total"
        if parsed.market_type == "spread":
            if side_label and line_str:
                return f"{matchup} — Spread {side_label} {line_str}"
            return f"{matchup} — Spread"
        if parsed.market_type == "moneyline":
            return f"{matchup} — Moneyline {side_label}".strip()
        return matchup
    return signal.get("market_title") or signal.get("market_slug") or DASH


_PRIMARY_BADGE_STYLES: dict[str, str] = {
    # Operator-friendly labels — replaces the older technical-jargon
    # pills (SRC FALCON / WALLET CONFIRMED / etc.) on the homepage.
    "Wallet Confirmed":          "green",
    "Prediction Market Listed":  "purple",
    "Cheap Price Trap":          "red",
    "Stale Odds":                "red",
    "No Wallet Data":            "muted",
    "Needs Review":              "red",
    "Line Mismatch":             "red",
    "Chase Risk":                "red",
}


def _primary_badges(signal: dict[str, Any]) -> str:
    """Top-of-card status pills that summarize the decision picture.

    Replaces the dense technical badge stack on the homepage cards
    with six labels an operator can read at a glance. Computed from
    the same fields the score uses — never invented.
    """
    badges: list[str] = []
    consensus = int(signal.get("consensus_wallets") or 0)
    if consensus >= 1:
        badges.append("Wallet Confirmed")
    elif signal.get("source") or signal.get("market_slug"):
        # We had a chance to measure wallet flow and didn't find any —
        # that's "no wallet data", not silence.
        badges.append("No Wallet Data")

    if signal.get("cheap_price_trap"):
        badges.append("Cheap Price Trap")

    if signal.get("market_url") and "polymarket.com" in str(signal.get("market_url")):
        badges.append("Prediction Market Listed")

    if signal.get("odds_stale"):
        badges.append("Stale Odds")
    if signal.get("warnings"):
        badges.append("Needs Review")

    if not badges:
        return ""
    chunks = "".join(
        f"<span class='sf-badge sf-badge-{_PRIMARY_BADGE_STYLES.get(label, 'muted')}'>"
        f"{html.escape(label)}</span>"
        for label in badges
    )
    return f"<div class='sf-card-row' style='margin-top:2px;'>{chunks}</div>"


def _trader_debug_block(signal: dict[str, Any]) -> str:
    """Render the structured trader-card lines requested in the spec::

        Market: TOR @ BAL Over 8.5
        Model: 9.77
        Edge vs line: +1.27 runs
        Best executable price: <book> <price> / <implied>
        Wallet confirmation: yes/no

    Plus a collapsed ``<details>`` "Debug details" block holding the
    join key + low-level internals an operator only needs when
    something looks wrong.

    All values are pulled from the signal/edge dict directly — no
    field is invented. A row is omitted when its value isn't available,
    so the block never displays "—" or a fabricated default.
    """
    slug = signal.get("market_slug") or ""
    parsed = wmr.parse_market_slug(slug) if slug else None

    market_line = signal.get("market_line")
    if market_line is None and parsed and parsed.line is not None:
        market_line = parsed.line

    model_projection = (
        signal.get("model_projection")
        or signal.get("projected_total")
        or signal.get("model_projected_total")
    )

    side = (signal.get("side") or signal.get("outcome") or "").strip().title()
    matchup = ""
    if parsed and parsed.away_abbr and parsed.home_abbr:
        matchup = f"{parsed.away_abbr.upper()} @ {parsed.home_abbr.upper()}"

    visible_rows: list[tuple[str, str]] = []
    debug_rows: list[tuple[str, str]] = []

    # Market line — exact sportsbook value, no extra rounding so a
    # 10.0 line displays as "10.0" exactly as listed.
    if market_line is not None:
        line_disp = f"{float(market_line):.1f}".rstrip("0").rstrip(".")
        if "." not in line_disp:
            line_disp = f"{line_disp}.0"
        market_label = f"{matchup} {side} {line_disp}".strip() if matchup else f"{side} {line_disp}".strip()
        visible_rows.append(("Market", market_label))
    elif matchup:
        visible_rows.append(("Market", f"{matchup} {side}".strip()))

    # Model projection — 2 decimals.
    if model_projection is not None:
        try:
            visible_rows.append(("Model", f"{float(model_projection):.2f}"))
        except (TypeError, ValueError):
            pass

    # Edge vs line — model − market, 2 decimals, signed.
    if model_projection is not None and market_line is not None:
        try:
            delta = float(model_projection) - float(market_line)
            sign = "+" if delta >= 0 else ""
            unit = " runs" if (parsed and parsed.market_type == "total") else ""
            visible_rows.append(("Edge vs line", f"{sign}{delta:.2f}{unit}"))
        except (TypeError, ValueError):
            pass

    # Best executable price — straight pass-through from the
    # backend-supplied execution block; not derived locally.
    exec_block = (signal.get("execution") or signal.get("wallet_context") or {}).get("execution") \
        if isinstance(signal.get("wallet_context"), dict) else signal.get("execution")
    if isinstance(exec_block, dict):
        book = exec_block.get("book") or exec_block.get("platform") or ""
        price = exec_block.get("price") or exec_block.get("side_price")
        implied = exec_block.get("implied_prob") or exec_block.get("implied_probability")
        if price is not None or implied is not None:
            price_part = f"{price:+d}" if isinstance(price, int) else (
                str(price) if price is not None else ""
            )
            implied_part = (
                f"{float(implied) * 100:.1f}%" if isinstance(implied, (int, float)) else ""
            )
            parts = [p for p in (book, price_part, implied_part) if p]
            if parts:
                visible_rows.append(("Best executable price", " · ".join(parts)))

    # Wallet confirmation — derived from consensus_wallets count.
    consensus = int(signal.get("consensus_wallets") or 0)
    if consensus:
        visible_rows.append(("Wallet confirmation", f"yes ({consensus} tracked)"))
    elif exec_block or market_line is not None:
        # Only show "no" when we have a confirmed market context;
        # otherwise the row is misleading (it's not "no", we just
        # didn't measure).
        visible_rows.append(("Wallet confirmation", "no"))

    # Prediction-market match status.
    pm_status = signal.get("prediction_market_status")
    if pm_status == "not_listed" or (
        signal.get("source") and not signal.get("market_url") and slug == ""
    ):
        visible_rows.append(("Prediction market", "not listed"))

    # ----- Debug-only fields (collapsed by default) ----------------------
    # Internal join key — colon-separated, never appears in a URL.
    if parsed:
        side_for_key = (signal.get("side") or signal.get("outcome") or "").strip().lower()
        key = wmr.internal_market_key(parsed, side=side_for_key)
        if key:
            debug_rows.append(("Internal key", key))
    if slug:
        debug_rows.append(("Market slug", slug))
    if signal.get("market_id"):
        debug_rows.append(("Market ID", str(signal.get("market_id"))))
    candidate_count = signal.get("candidate_markets_count") or signal.get("candidate_count")
    if candidate_count is not None:
        debug_rows.append(("Candidate markets", str(candidate_count)))
    matched_slug = signal.get("matched_market_slug")
    if matched_slug:
        debug_rows.append(("Matched market slug", str(matched_slug)))
    line_tol = signal.get("line_tolerance")
    if line_tol is not None:
        debug_rows.append(("Line tolerance", str(line_tol)))
    sb_event = signal.get("sportsbook_event") or signal.get("source_url")
    if sb_event:
        debug_rows.append(("Sportsbook event", str(sb_event)))
    wallet_join = signal.get("wallet_join_debug")
    if wallet_join:
        debug_rows.append(("Wallet join debug", str(wallet_join)))

    if not visible_rows and not debug_rows:
        return ""

    visible_body = "".join(
        f"<div class='sf-card-row sf-meta'><span class='k'>{k}:</span> "
        f"{html.escape(str(v))}</div>"
        for k, v in visible_rows
    )
    debug_body = ""
    if debug_rows:
        inner = "".join(
            f"<div class='sf-card-row sf-meta'><span class='k'>{k}:</span> "
            f"{html.escape(str(v))}</div>"
            for k, v in debug_rows
        )
        # Native <details> — collapsed by default so the homepage card
        # stays clean while the join debug remains one click away.
        debug_body = (
            "<details class='sf-card-debug' style='margin-top:4px;'>"
            "<summary class='sf-meta' style='cursor:pointer;'>Debug details</summary>"
            f"<div style='margin-top:4px;'>{inner}</div>"
            "</details>"
        )
    return f"<div class='sf-section'>{visible_body}{debug_body}</div>"


def render_wallet_card(signal: dict[str, Any]) -> None:
    score = signal.get("score")
    tier, tier_kind = tier_for_score(score)
    card_kind = card_kind_for_tier(tier)

    trader = signal.get("trader_nickname") or DASH
    wallet = shorten_wallet(signal.get("wallet"))
    # Title: prefer a clean trader-card line derived from the slug
    # (always carries the executable market line). Falls back to the
    # backend-supplied title only when the slug isn't recognized — that
    # path is the one that historically leaked model projections into
    # the title.
    market = _trader_card_title(signal)
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
    # Collapsed summary: top 3 unique wallets by total size (one chip each).
    consensus_chips = consensus_wallets_chips_html(consensus_traders, limit=3)
    extra_wallets = max(0, len(consensus_traders) - 3)
    if consensus_chips and extra_wallets:
        consensus_chips = consensus_chips.replace(
            "</div>", f"<span class='sf-badge sf-badge-muted'>+{extra_wallets} more</span></div>"
        )

    # Source-stack style indicators derived from the signal's own consensus
    # data (the watchlist card is wallet-first, so these are honest, not edge
    # fields borrowed from the MLB cards).
    _wallets = int(signal.get("consensus_wallets") or 0)
    source_pills = [
        _pill(tier, tier_kind),
        _pill(f"SRC {source}", "purple" if source == "Falcon" else "cyan") if source else "",
        _pill(side, "green" if side in {"YES", "BUY"} else ("red" if side in {"NO", "SELL"} else "muted")) if side else "",
    ]
    if _wallets >= 2:
        source_pills.append(_pill("WALLET CONFIRMED", "green"))
    if _wallets >= 5:
        source_pills.append(_pill("CROWDED CONSENSUS", "purple"))
    pills_html = " ".join(filter(None, source_pills))

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

    # Top-of-card primary badges (Wallet Confirmed / Cheap Price Trap /
    # Stale Odds / etc.). Same operator-friendly set the edge card uses,
    # so the homepage reads consistently across signal sources.
    primary_badges_html = _primary_badges(signal)
    # Structured trader-card details block — sits below the headline
    # market label and surfaces Market vs Model vs Edge vs Wallet
    # confirmation without ever conflating the executable line with the
    # model projection.
    trader_block = _trader_debug_block(signal)

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
      {primary_badges_html}
      <div style="margin-bottom:4px;">{pills_html}</div>
      {trader_block}
      {sharp_block}
      {consensus_chips}
      {('<div class="sf-card-row sf-meta">' + reason + '</div>') if reason else ''}
      {market_links}
    </div>
    """
    st.markdown(body, unsafe_allow_html=True)

    if len(consensus_traders) > 3:
        # market_id+side+outcome is not unique: the same signal renders in both
        # the Wallets tab and the Positions tab, and aggregated positions can
        # share a side/outcome across signals. Append a monotonic suffix.
        expander_key = (
            f"consensus-{signal.get('market_id')}-{signal.get('side')}-"
            f"{signal.get('outcome')}-{next(_WIDGET_KEY_COUNTER)}"
        )
        with st.expander(
            f"Consensus wallets (full list · {len(consensus_traders)})",
            expanded=False,
            key=expander_key,
        ):
            # One clean HTML block, rendered once — no nested/escaped badges.
            st.markdown(
                consensus_wallets_chips_html(consensus_traders),
                unsafe_allow_html=True,
            )


def render_score_distribution(
    edges: list[dict[str, Any]],
    *,
    threshold: float = SCORE_HIGH_CONV_MIN,
    score_key: str = "score",
    label: str = "Score",
) -> None:
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
        value = e.get(score_key)
        if value is None and score_key != "score":
            value = e.get("score")
        s = _as_float(value)
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
    metric_cols[0].metric(f"Top {label.lower()} today", f"{top:.1f}")
    metric_cols[1].metric(f"Median {label.lower()}", f"{median:.1f}")
    metric_cols[2].metric("Std deviation", f"{stdev:.1f}")
    metric_cols[3].metric(
        f"Above {int(threshold)}",
        above,
        delta=f"of {len(scores)}",
        delta_color="off",
    )
    st.markdown(
        "<div class='sf-card'>"
        f"<div class='sf-section-title'>{label} Distribution</div>"
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
    high_conv = [e for e in edges if edge_prediction_score(e) >= threshold]
    if high_conv:
        return
    scores = [edge_prediction_score(e) for e in edges]
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
        f"<div class='sf-card-row'>Top prediction score today: <b>{top:.1f}</b></div>"
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
# Background-job state machine
# =============================================================================
#
# Replaces ad-hoc `with st.spinner(...)` blocks + scattered reruns with a
# single typed `active_job` dict in session_state. The command center owns
# one small status line (render_active_job_panel) that always tells the
# operator exactly what is in flight — or "Idle" if nothing is. Buttons
# disable themselves while their job is running so a double-click can't
# spawn a duplicate scan.
#
# active_job shape:
#     {
#         "name": "wallet_scan",
#         "label": "Wallet scan",
#         "started_at": "2026-05-28T20:11:43+00:00",
#         "status": "Posting /run-scan...",
#         "last_heartbeat": "2026-05-28T20:11:45+00:00",
#     }
#
# last_job carries the same fields plus finished_at, success, result_message.

JOB_TIMEOUT_SECONDS = int(os.environ.get("SIGNALFORGE_DASHBOARD_JOB_TIMEOUT", "240"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_log(event: str, **fields: Any) -> None:
    """Structured trace for rerun/click/job/backend events."""
    if not fields:
        logger.info(event)
        return
    payload = " ".join(f"{k}={v!r}" for k, v in fields.items())
    logger.info("%s %s", event, payload)


def _active_job() -> dict[str, Any] | None:
    return st.session_state.get("active_job")


def is_job_running(name: str | None = None) -> bool:
    job = _active_job()
    if not job:
        return False
    return name is None or job.get("name") == name


def clear_stale_job(timeout_seconds: int = JOB_TIMEOUT_SECONDS) -> None:
    """Drop active_job if it's older than timeout_seconds.

    The backend job itself may keep running — but the *dashboard's* tracking
    should never get stuck on a flag set minutes ago and never cleared
    (e.g. the user closed the tab mid-call, or a network timeout aborted
    the script before finish_job ran).
    """
    job = _active_job()
    if not job:
        return
    try:
        started = datetime.fromisoformat(job["started_at"])
    except (KeyError, ValueError):
        _job_log("clear_stale_job.invalid_started_at", job=job)
        st.session_state.pop("active_job", None)
        return
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    if elapsed > timeout_seconds:
        _job_log(
            "clear_stale_job.timeout",
            name=job.get("name"), elapsed=round(elapsed, 1),
            timeout=timeout_seconds,
        )
        st.session_state["last_job"] = {
            **job,
            "finished_at": _now_iso(),
            "success": False,
            "result_message": (
                f"Marked stale after {int(elapsed)}s "
                f"(timeout={timeout_seconds}s). Backend may still be running — "
                "refresh once it completes."
            ),
        }
        st.session_state.pop("active_job", None)


def start_job(name: str, label: str | None = None, *, status: str = "Starting...") -> bool:
    """Acquire the single-job lock. Returns False if `name` is already running."""
    clear_stale_job()
    if is_job_running(name):
        _job_log("start_job.duplicate", name=name)
        return False
    job = {
        "name": name,
        "label": label or name.replace("_", " ").title(),
        "started_at": _now_iso(),
        "status": status,
        "last_heartbeat": _now_iso(),
    }
    st.session_state["active_job"] = job
    _job_log("start_job", **job)
    return True


def update_job_status(message: str) -> None:
    job = _active_job()
    if not job:
        return
    job["status"] = message
    job["last_heartbeat"] = _now_iso()
    st.session_state["active_job"] = job
    _job_log("update_job_status", name=job.get("name"), status=message)


def finish_job(*, success: bool, result_message: str) -> None:
    job = _active_job()
    if not job:
        _job_log("finish_job.no_active_job", success=success, message=result_message)
        return
    finished = {
        **job,
        "finished_at": _now_iso(),
        "success": bool(success),
        "result_message": result_message,
    }
    st.session_state["last_job"] = finished
    st.session_state.pop("active_job", None)
    _job_log(
        "finish_job",
        name=finished.get("name"), success=success, message=result_message,
    )


def render_active_job_panel() -> None:
    """Single source of truth for 'what is running right now'.

    Renders inline as a small card under the command-center action bar so
    the dashboard never has to fall back on the page-wide Streamlit
    spinner overlay. When there is no active job and no remembered last
    job, it prints 'Idle' — exactly so the operator can tell the page is
    at rest rather than silently doing work.
    """
    clear_stale_job()
    job = _active_job()
    last = st.session_state.get("last_job") or {}
    if job:
        try:
            elapsed = (
                datetime.now(timezone.utc)
                - datetime.fromisoformat(job["started_at"])
            ).total_seconds()
        except (KeyError, ValueError):
            elapsed = 0.0
        st.markdown(
            f"<div class='sf-card purple' style='padding:8px 12px;margin-top:6px;'>"
            f"<div class='sf-card-row'>"
            f"<span class='k'>Background activity:</span> "
            f"<b>{html.escape(str(job.get('label') or job.get('name')))}</b> · "
            f"running {elapsed:.0f}s · "
            f"{html.escape(str(job.get('status') or ''))}</div>"
            f"<div class='sf-meta'>"
            f"Started: {fmt_dt_mst(job.get('started_at'))} · "
            f"Last heartbeat: {fmt_dt_mst(job.get('last_heartbeat'))}"
            f"</div></div>",
            unsafe_allow_html=True,
        )
        return
    if last:
        tone = "green" if last.get("success") else "red"
        st.markdown(
            f"<div class='sf-card {tone}' style='padding:8px 12px;margin-top:6px;'>"
            f"<div class='sf-card-row'>"
            f"<span class='k'>Background activity:</span> Idle. "
            f"Last: <b>{html.escape(str(last.get('label') or last.get('name')))}</b> · "
            f"{'OK' if last.get('success') else 'FAILED'} · "
            f"{html.escape(str(last.get('result_message') or ''))}</div>"
            f"<div class='sf-meta'>Finished: {fmt_dt_mst(last.get('finished_at'))}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        "<div class='sf-card' style='padding:8px 12px;margin-top:6px;'>"
        "<div class='sf-card-row'>"
        "<span class='k'>Background activity:</span> Idle."
        "</div></div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# Action handlers (sidebar + buttons)
# =============================================================================


def action_run_wallet_scan() -> None:
    if not start_job("wallet_scan", "Wallet scan", status="Posting /run-scan..."):
        st.toast("Wallet scan is already running.")
        return
    result: dict[str, Any] = {}
    with st.status("Wallet scan: contacting backend…", expanded=False, state="running") as status_box:
        _job_log("backend_request_start", job="wallet_scan", path="/run-scan", date=selected_card_date)
        try:
            result = api_post("/run-scan", timeout=SCAN_TIMEOUT, params={"date": selected_card_date})
        except ApiError as exc:
            _job_log("backend_request_end", job="wallet_scan", ok=False, error=str(exc))
            finish_job(success=False, result_message=f"Wallet scan failed: {exc}")
            status_box.update(label="Wallet scan failed", state="error")
            render_api_error(exc, prefix="Wallet scan failed")
            return
        _job_log("backend_request_end", job="wallet_scan", ok=True)
    state = result.get("state")
    if state == "running":
        finish_job(success=True, result_message="Backend worker queued the scan.")
        st.cache_data.clear()
        _job_log("rerun_trigger", source="action_run_wallet_scan.queued")
        st.rerun()
        return
    if state == "finished" and result.get("result"):
        result = result.get("result") or {}
    summary = (
        f"{result.get('markets_for_card_date', 0)}/{result.get('markets_seen', 0)} markets · "
        f"positions={result.get('positions_written', result.get('new_signals', 0))} · "
        f"alerts={result.get('alerts_written', result.get('new_alerts', 0))}"
    )
    finish_job(success=True, result_message=summary)
    st.toast(f"Wallet scan {result.get('generated_for_date') or selected_card_date}: {summary}")
    st.cache_data.clear()
    _job_log("rerun_trigger", source="action_run_wallet_scan.complete")
    st.rerun()


def action_run_mlb_edge_scan(*, force_stale: bool = False) -> None:
    if not start_job("mlb_edge_scan", "MLB edge scan", status="Posting /mlb/edges/run..."):
        st.toast("MLB edge scan is already running.")
        return
    params: dict[str, Any] = {"game_date": selected_card_date}
    if force_stale:
        # Operator override: skip the freshness gate so the engine runs
        # off whatever's in the odds cache. Used when the Render proxy
        # ate a previous refresh or when only BPP-fallback K cards are
        # needed and game-total freshness doesn't matter.
        params["force_stale"] = True
    result: dict[str, Any] = {}
    with st.status("MLB edge scan: running engine…", expanded=False, state="running") as status_box:
        _job_log("backend_request_start", job="mlb_edge_scan", path="/mlb/edges/run", date=selected_card_date, force_stale=force_stale)
        try:
            result = api_post("/mlb/edges/run", timeout=MLB_RUN_TIMEOUT, params=params)
        except ApiError as exc:
            _job_log("backend_request_end", job="mlb_edge_scan", ok=False, error=str(exc))
            # Surface the actual backend exception detail (truncated) in
            # the activity panel so future 502s aren't opaque. The full
            # body still goes through render_api_error.
            short_detail = exc.short_body() or str(exc)
            finish_job(
                success=False,
                result_message=f"MLB edge scan failed: {short_detail[:240]}",
            )
            status_box.update(label="MLB edge scan failed", state="error")
            render_api_error(exc, prefix="MLB edge scan failed")
            return
        _job_log("backend_request_end", job="mlb_edge_scan", ok=True)
    if str(result.get("status") or "").lower() == "blocked":
        reason = result.get("reason") or "Odds cache stale; refresh required before edge scan."
        # Pair the warning with the escape hatch so the operator can
        # rerun in one click when they've already refreshed and know
        # the cache is intentionally stale (e.g. pre-game scan only).
        finish_job(success=False, result_message=f"Blocked: {reason}")
        st.warning(reason)
        st.caption(
            "Click **Run scan anyway (use stale odds)** in the MLB "
            "Terminal control bar to bypass this gate — pitcher-K "
            "fallback cards will still build from BallparkPal."
        )
        st.cache_data.clear()
        _job_log("rerun_trigger", source="action_run_mlb_edge_scan.blocked")
        st.rerun()
        return
    generated_for = result.get("generated_for_date") or result.get("date") or "?"
    written = int(result.get("snapshots_written") or result.get("edges") or 0)
    preserved = int(result.get("snapshots_preserved_from_prior_dates") or 0)
    summary = (
        f"{generated_for}: {written} snapshot(s) across {result.get('games', 0)} game(s) "
        f"(odds events: {result.get('odds_events', 0)}, preserved: {preserved})"
    )
    finish_job(success=True, result_message=summary)
    st.toast(f"MLB scan complete · {summary}")
    st.cache_data.clear()
    _job_log("rerun_trigger", source="action_run_mlb_edge_scan.complete")
    st.rerun()


def action_refresh_odds_cache() -> None:
    if not start_job("odds_cache_refresh", "Odds cache refresh", status="Posting /mlb/debug/odds-cache/refresh..."):
        st.toast("Odds cache refresh is already running.")
        return
    result: dict[str, Any] = {}
    with st.status("Odds cache: refreshing…", expanded=False, state="running") as status_box:
        _job_log("backend_request_start", job="odds_cache_refresh", path="/mlb/debug/odds-cache/refresh", date=selected_card_date)
        try:
            result = api_post(
                "/mlb/debug/odds-cache/refresh",
                timeout=MLB_RUN_TIMEOUT,
                params={"game_date": selected_card_date},
            )
        except ApiError as exc:
            _job_log("backend_request_end", job="odds_cache_refresh", ok=False, error=str(exc))
            finish_job(success=False, result_message=f"Odds cache refresh failed: {exc}")
            status_box.update(label="Odds cache refresh failed", state="error")
            render_api_error(exc, prefix="Odds cache refresh failed")
            return
        _job_log("backend_request_end", job="odds_cache_refresh", ok=True)
    summary = (
        f"events={result.get('events_fetched', 0)}, "
        f"odds_calls={result.get('odds_calls', 0)}, "
        f"rate_limited={result.get('rate_limited', 0)}"
    )
    finish_job(success=True, result_message=summary)
    st.toast(f"Cache refresh · {summary}")
    st.cache_data.clear()
    _job_log("rerun_trigger", source="action_refresh_odds_cache")
    st.rerun()


def action_test_backend() -> None:
    if not start_job("backend_test", "Backend test", status="Calling /health..."):
        st.toast("Backend test is already running.")
        return
    with st.status("Backend test: calling /health…", expanded=False, state="running") as status_box:
        _job_log("backend_request_start", job="backend_test", path="/health")
        try:
            payload = api_get("/health", timeout=HEALTH_TIMEOUT)
        except ApiError as exc:
            _job_log("backend_request_end", job="backend_test", ok=False, error=str(exc))
            finish_job(success=False, result_message=f"Backend test failed: {exc}")
            status_box.update(label="Backend test failed", state="error")
            render_api_error(exc, prefix="Backend test failed")
            return
        _job_log("backend_request_end", job="backend_test", ok=True)
    summary = (
        f"OK · env={payload.get('environment')} · ts={payload.get('timestamp', '?')}"
    )
    finish_job(success=True, result_message=summary)
    status_box.update(label=f"Backend {summary}", state="complete")


def action_update_mlb_closing_lines(date: str | None = None) -> None:
    if not start_job("mlb_closing_lines", "MLB closing-lines update", status="Posting /mlb/debug/closing-lines/run..."):
        st.toast("Closing-lines update is already running.")
        return
    params: dict[str, Any] = {}
    if date:
        params["date"] = date
    result: dict[str, Any] = {}
    with st.status("MLB closing lines: updating…", expanded=False, state="running") as status_box:
        _job_log("backend_request_start", job="mlb_closing_lines", path="/mlb/debug/closing-lines/run", date=date)
        try:
            result = api_post(
                "/mlb/debug/closing-lines/run",
                timeout=MLB_RUN_TIMEOUT,
                params=params or None,
            )
        except ApiError as exc:
            _job_log("backend_request_end", job="mlb_closing_lines", ok=False, error=str(exc))
            finish_job(success=False, result_message=f"Closing-line update failed: {exc}")
            status_box.update(label="Closing-line update failed", state="error")
            render_api_error(exc, prefix="Closing-line update failed")
            return
        _job_log("backend_request_end", job="mlb_closing_lines", ok=True)
    updated = int(result.get("closing_lines_updated") or 0)
    candidates = int(result.get("candidates") or 0)
    skipped = int(result.get("skipped") or 0)
    failed = int(result.get("failed") or 0)
    reason = result.get("reason")
    date_suffix = f" for {date}" if date else ""
    if updated:
        st.success(
            f"Closing lines updated{date_suffix}: {updated} of {candidates} candidate edge(s). "
            f"Skipped {skipped}, failed {failed}."
        )
    elif reason:
        st.warning(f"Closing-line update finished{date_suffix} — no rows updated. {reason}")
    else:
        st.info(
            f"Closing-line update finished{date_suffix}: 0 updated, {candidates} candidate(s), "
            f"{skipped} skipped, {failed} failed."
        )
    summary = (
        f"updated={updated}, candidates={candidates}, skipped={skipped}, failed={failed}"
    )
    finish_job(success=True, result_message=summary)
    st.cache_data.clear()
    _job_log("rerun_trigger", source="action_update_mlb_closing_lines")
    st.rerun()


def action_grade_mlb_results(date: str | None = None) -> None:
    if not start_job("mlb_grade_results", "MLB grading", status="Posting /mlb/debug/grade-results/run..."):
        st.toast("MLB grading is already running.")
        return
    params: dict[str, Any] = {}
    if date:
        params["date"] = date
    result: dict[str, Any] = {}
    with st.status("MLB grading: running…", expanded=False, state="running") as status_box:
        _job_log("backend_request_start", job="mlb_grade_results", path="/mlb/debug/grade-results/run", date=date)
        try:
            result = api_post(
                "/mlb/debug/grade-results/run",
                timeout=MLB_RUN_TIMEOUT,
                params=params or None,
            )
        except ApiError as exc:
            _job_log("backend_request_end", job="mlb_grade_results", ok=False, error=str(exc))
            finish_job(success=False, result_message=f"MLB grading failed: {exc}")
            status_box.update(label="MLB grading failed", state="error")
            render_api_error(exc, prefix="MLB grading failed")
            return
        _job_log("backend_request_end", job="mlb_grade_results", ok=True)
    graded = int(result.get("graded") or 0)
    candidates = int(result.get("candidates") or 0)
    finals = int(result.get("finals_found") or 0)
    persisted = int(result.get("graded_from_persisted") or 0)
    live = int(result.get("graded_from_live") or 0)
    skipped_not_final = int(result.get("skipped_not_final") or 0)
    failed = int(result.get("failed") or 0)
    ingestion = result.get("ingestion") or {}
    upserted = int(ingestion.get("upserted") or 0)
    reason = result.get("reason")
    date_suffix = f" for {date}" if date else ""
    if graded:
        st.success(
            f"Graded {graded} edge(s){date_suffix} across {finals} final game(s) "
            f"(persisted: {persisted}, live: {live}). Ingested {upserted} new "
            f"final-score row(s). Candidates: {candidates}, not-yet-final: "
            f"{skipped_not_final}, failed: {failed}."
        )
    elif reason:
        st.warning(
            f"MLB grading finished{date_suffix} — no edges graded. {reason} "
            f"(Ingested {upserted} final-score row(s).)"
        )
    else:
        st.info(
            f"MLB grading finished{date_suffix}: candidates={candidates}, "
            f"finals={finals}, not_final={skipped_not_final}, failed={failed}, "
            f"ingested={upserted}."
        )
    summary = (
        f"graded={graded}, candidates={candidates}, finals={finals}, "
        f"persisted={persisted}, live={live}, ingested={upserted}, "
        f"not_final={skipped_not_final}, failed={failed}"
    )
    finish_job(success=True, result_message=summary)
    st.cache_data.clear()
    _job_log("rerun_trigger", source="action_grade_mlb_results")
    st.rerun()


def action_sync_pnl_wallets() -> None:
    if not start_job("pnl_sync", "P&L wallet sync", status="Posting /pnl/sync..."):
        st.toast("P&L sync is already running.")
        return
    result: dict[str, Any] = {}
    with st.status("P&L sync: contacting backend…", expanded=False, state="running") as status_box:
        _job_log("backend_request_start", job="pnl_sync", path="/pnl/sync")
        try:
            result = api_post("/pnl/sync", timeout=SCAN_TIMEOUT)
        except ApiError as exc:
            _job_log("backend_request_end", job="pnl_sync", ok=False, error=str(exc))
            finish_job(success=False, result_message=f"P&L sync failed: {exc}")
            status_box.update(label="P&L sync failed", state="error")
            render_api_error(exc, prefix="P&L wallet sync failed")
            return
        _job_log("backend_request_end", job="pnl_sync", ok=True)
    warnings = result.get("warnings") or []
    summary = (
        f"new_trades={result.get('new_trades', 0)}, "
        f"positions_rebuilt={result.get('positions_rebuilt', 0)} "
        f"({result.get('mode')})"
    )
    finish_job(success=True, result_message=summary)
    for warning in warnings[:3]:
        st.warning(warning)
    st.cache_data.clear()
    _job_log("rerun_trigger", source="action_sync_pnl_wallets")
    st.rerun()


# =============================================================================
# Cached fetchers
# =============================================================================


@st.cache_data(ttl=10, show_spinner=False)
def fetch_health() -> dict[str, Any] | None:
    # First attempt of the session gets the cold-start budget (~45s) because
    # Render needs that long to wake the worker. Subsequent calls use the
    # tight warm timeout so a half-dead backend can't stall the dashboard.
    warmed = bool(st.session_state.get("_backend_warmed"))
    timeout = HEALTH_WARM_TIMEOUT if warmed else HEALTH_TIMEOUT
    try:
        payload = api_get("/health", timeout=timeout)
        payload["_frontend_state"] = "healthy"
        st.session_state["_backend_warmed"] = True
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


@st.cache_data(ttl=5, show_spinner=False)
def fetch_scan_status() -> dict[str, Any]:
    return safe_get("/run-scan/status", default={"state": "unknown"})


@st.cache_data(ttl=10, show_spinner=False)
def fetch_summary(card_date: str = CARD_DATE) -> dict[str, Any]:
    return safe_get("/dashboard-summary", default={}, params={"date": card_date})


@st.cache_data(ttl=10, show_spinner=False)
def fetch_traders() -> list[dict[str, Any]]:
    return safe_get("/traders", default=[])


@st.cache_data(ttl=10, show_spinner=False)
def fetch_signals(limit: int = 500, card_date: str = CARD_DATE) -> list[dict[str, Any]]:
    return safe_get(
        "/signals",
        default=[],
        params={
            "limit": limit,
            "date": card_date,
            "active_only": True,
            "exclude_resolved": True,
        },
    )


@st.cache_data(ttl=10, show_spinner=False)
def fetch_tracked_wallet_positions(limit: int = 500) -> list[dict[str, Any]]:
    """All tracked-wallet positions across **every** date.

    The dashboard's ``fetch_signals`` is scoped to today's card date,
    which is what the Top Signals view needs — but the Wallet Flow tab
    has historically come up empty for operators whose tracked wallets
    are active on future games or in markets without a recognized
    date. This fetcher uses ``history=True`` to bypass the card-date
    scope so the Wallet Flow tab can show every tracked-wallet
    position the backend has, regardless of which slate it's on.
    """
    return safe_get(
        "/signals",
        default=[],
        params={"limit": limit, "history": True},
    )


@st.cache_data(ttl=10, show_spinner=False)
def fetch_tracked_wallet_live_positions(
    card_date: str = CARD_DATE,
) -> list[dict[str, Any]]:
    """Raw tracked-wallet Trade rows for ``card_date``.

    Bypasses the Signal pipeline so positions show up even when the
    score threshold or market-date normalizer would have dropped them
    from the curated signal feed. The Command Center panel
    "Tracked Wallet Live Positions" feeds off this — it's the source of
    truth for "do my tracked wallets have any open positions today?"
    that the dashboard NEEDS to answer truthfully.
    """
    return safe_get(
        "/tracked-wallet-positions",
        default=[],
        params={"date": card_date},
    )


@st.cache_data(ttl=30, show_spinner=False)
def fetch_tracked_wallet_debug(card_date: str = CARD_DATE) -> dict[str, Any]:
    """Per-rejection diagnostics for the Wallet Flow debug panel."""
    return safe_get(
        "/tracked-wallet-positions/debug",
        default={
            "raw_recent_trades": 0,
            "accepted_for_card_date": 0,
            "rejected": 0,
            "rejection_reasons": {},
            "top_rejected_examples": [],
        },
        params={"date": card_date, "limit": 50},
    )


@st.cache_data(ttl=10, show_spinner=False)
def fetch_alerts(limit: int = 200, card_date: str = CARD_DATE) -> list[dict[str, Any]]:
    return safe_get("/alerts", default=[], params={"limit": limit, "date": card_date})


@st.cache_data(ttl=10, show_spinner=False)
def fetch_historical_alerts(limit: int = 200) -> list[dict[str, Any]]:
    return safe_get("/alerts", default=[], params={"limit": limit, "history": True})


@st.cache_data(ttl=30, show_spinner=False)
def fetch_mlb_edges(limit: int = 100, card_date: str = CARD_DATE) -> list[dict[str, Any]]:
    return safe_get(
        "/mlb/edges/today",
        default=[],
        params={"limit": limit, "game_date": card_date},
    )


@st.cache_data(ttl=30, show_spinner=False)
def fetch_mlb_daily_card(card_date: str = CARD_DATE) -> dict[str, Any] | None:
    return safe_get("/mlb/daily-card", default=None, params={"game_date": card_date})


@st.cache_data(ttl=30, show_spinner=False)
def fetch_mlb_sources() -> dict[str, Any]:
    return safe_get("/mlb/debug/sources", default={})


@st.cache_data(ttl=30, show_spinner=False)
def fetch_odds_cache() -> dict[str, Any]:
    return safe_get("/mlb/debug/odds-cache", default={})


@st.cache_data(ttl=30, show_spinner=False)
def fetch_odds_providers() -> dict[str, Any]:
    return safe_get("/odds/providers/health", default={})


@st.cache_data(ttl=30, show_spinner=False)
def fetch_odds_event_match(card_date: str = CARD_DATE) -> dict[str, Any]:
    return safe_get("/mlb/debug/odds/event-match", default={}, params={"game_date": card_date})


@st.cache_data(ttl=10, show_spinner=False)
def fetch_dashboard_debug(card_date: str = CARD_DATE) -> dict[str, Any]:
    return safe_get("/dashboard/debug", default={}, params={"date": card_date})


@st.cache_data(ttl=30, show_spinner=False)
def fetch_market_validation() -> dict[str, Any]:
    return safe_get("/mlb/debug/market-validation", default={})


@st.cache_data(ttl=30, show_spinner=False)
def fetch_pitcher_props(limit: int = 100) -> dict[str, Any]:
    return safe_get("/mlb/debug/pitcher-props", default={"count": 0, "rows": []},
                    params={"limit": limit})


@st.cache_data(ttl=30, show_spinner=False)
def fetch_mlb_performance(
    days: int | None = None,
    date: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if days is not None:
        params["days"] = days
    if date:
        params["date"] = date
    out: dict[str, Any] = {
        "summary": safe_get("/mlb/performance/summary", default={}, params=params or None),
        "by_market": safe_get("/mlb/performance/by-market", default=[], params=params or None),
        "by_score_band": safe_get("/mlb/performance/by-score-band", default=[], params=params or None),
        "by_prediction_score_band": safe_get(
            "/mlb/performance/by-score-axis",
            default=[],
            params={**params, "axis": "prediction"} if params else {"axis": "prediction"},
        ),
        "by_execution_score_band": safe_get(
            "/mlb/performance/by-score-axis",
            default=[],
            params={**params, "axis": "execution"} if params else {"axis": "execution"},
        ),
        "clv": safe_get("/mlb/performance/clv", default={}, params=params or None),
        # Research-upgrade additions. Each endpoint degrades to {} or [] when
        # absent so older backends keep rendering the dashboard.
        "research_health": safe_get(
            "/mlb/performance/research-health", default={}, params=params or None,
        ),
        "by_side": safe_get(
            "/mlb/performance/by-side", default={}, params=params or None,
        ),
        "projection_calibration": safe_get(
            "/mlb/performance/projection-calibration", default={}, params=params or None,
        ),
        "by_projection_bucket": safe_get(
            "/mlb/performance/by-projection-bucket", default=[], params=params or None,
        ),
        "by_timing": safe_get(
            "/mlb/performance/by-timing", default=[], params=params or None,
        ),
        "factor_attribution": safe_get(
            "/mlb/performance/factor-attribution", default=[], params=params or None,
        ),
        # Factor-distribution audit + score-attribution report drive the
        # "is this factor dead weight?" diagnostics on the perf tab. Both
        # degrade to empty payloads on older backends.
        "factor_distribution": safe_get(
            "/mlb/performance/factor-distribution",
            default={"factors": [], "summary": {}},
            params=params or None,
        ),
        "score_attribution": safe_get(
            "/mlb/performance/score-attribution",
            default={"factors": []},
            params=params or None,
        ),
        # Rolling per-side performance is window-agnostic (the engine's
        # 14-day lookback) so it does NOT use the current perf-tab params.
        "recent_side_performance": safe_get(
            "/mlb/performance/recent-side-performance",
            default={"sides": {}},
        ),
    }
    return out


@st.cache_data(ttl=10, show_spinner=False)
def fetch_pnl_tracker() -> dict[str, Any]:
    return safe_get("/pnl/tracker", default={"summary": {}, "open_positions": [], "closed_positions": []})


@st.cache_data(ttl=60, show_spinner=False)
def fetch_ballparkpal_snapshots(slate_date: str | None = None) -> dict[str, Any]:
    """Read BallparkPal cache via the API. Pure read — Playwright is never
    invoked here. Cached for 60s so flipping subtabs feels instant.
    """
    params = {"slate_date": slate_date} if slate_date else None
    return safe_get(
        "/ballparkpal/snapshots",
        default={"slate_date": slate_date, "pages": {}, "labels": {}},
        params=params,
    )


# =============================================================================
# Signal → position aggregation (kept from prior dashboard)
# =============================================================================


def _market_label(sig: dict[str, Any]) -> str:
    return sig.get("market_title") or f"market#{sig.get('market_id')}"


def _market_slug(sig: dict[str, Any]) -> str:
    return sig.get("market_slug") or str(sig.get("market_title") or "")


def _market_url(sig: dict[str, Any]) -> str | None:
    """Resolve a click-through URL for a signal/edge.

    Delegates to ``wallet_market_resolver.market_url_for`` so Polymarket
    URLs always land on the event page rather than a (non-existent)
    line-specific event slug. The ``source_url`` already captured by
    the provider, if any, wins over the derived URL.
    """
    captured = sig.get("market_url") or sig.get("source_url")
    if captured and "polymarket.com/event/" in str(captured):
        # Trust an authoritative captured URL over a derived one — same
        # fallback policy as ``polymarket_event_url``.
        return str(captured)
    slug = _market_slug(sig)
    platform = str(sig.get("market_platform") or sig.get("source") or "").lower()
    return wmr.market_url_for(slug, platform)


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
        # One consensus row per unique wallet — fills are aggregated, never
        # rendered as separate entries.
        consensus_traders = build_consensus_wallets(grouped)
        # `size_usd` kept as an alias of total for downstream card formatting.
        for t in consensus_traders:
            t["size_usd"] = t.get("total_size_usd")
        total_tracked = sum(t.get("total_size_usd") or 0.0 for t in consensus_traders)
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

# Degraded-mode contract: the dashboard NEVER hard-stops on a backend miss.
# fetch_health() may return None (offline), {"_frontend_state": "waking"} (cold
# start), or a healthy payload. Each pre-fetch below uses safe_get() which
# already absorbs ApiError and returns the supplied default — so missing
# endpoints render as empty cards, not a blank page.
if OFFLINE_MODE:
    health = {
        "ok": False,
        "status": "offline_mode",
        "_frontend_state": "offline",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
else:
    health = fetch_health()

backend_state: str
if OFFLINE_MODE:
    backend_state = "offline"
elif health is None:
    backend_state = "offline"
elif health.get("_frontend_state") == "waking":
    backend_state = "waking"
else:
    backend_state = "healthy"

if backend_state == "waking":
    st.warning(
        f"Backend at **{API_BASE}** is waking up (Render cold start can take ~30-60s). "
        "Rendering the dashboard shell with empty data — refresh in a moment for live values."
    )
elif backend_state == "offline":
    err = st.session_state["_fetch_errors"].get("/health")
    if OFFLINE_MODE:
        st.info(
            "OFFLINE MODE — SignalForge backend calls are disabled. "
            "Dashboard shell renders with empty/mock data only."
        )
    else:
        st.error(
            f"Backend at **{API_BASE}** is unreachable. Showing dashboard shell with "
            "empty data. Start the backend locally with `uvicorn app.main:app --reload` "
            "or check Render logs."
        )
        if isinstance(err, ApiError):
            with st.expander("Show backend error"):
                render_api_error(err, prefix="Cannot reach SignalForge backend")


# =============================================================================
# Pre-fetch everything once per render
# =============================================================================

selected_card_date = CARD_DATE
summary = fetch_summary(selected_card_date) or {}
ready = fetch_ready() or {}
scan_status_payload = fetch_scan_status() or {}
traders = fetch_traders()
signals_all = fetch_signals(limit=500, card_date=selected_card_date)
positions_all = aggregate_signals_to_positions(signals_all)
# All-time tracked wallet positions (no card-date scope). The Wallet
# Flow tab consumes these so a tracked wallet that's only active on
# upcoming-slate or no-slate markets still surfaces.
all_wallet_signals = fetch_tracked_wallet_positions(limit=500)
all_wallet_positions = aggregate_signals_to_positions(all_wallet_signals)
# Raw tracked-wallet trade rows. Critical: this bypasses the Signal
# pipeline so positions appear even when the signal engine dropped them
# for score-threshold or market-date normalization reasons. The
# Command Center NEVER shows "no current-card wallet flow" while this
# list is non-empty.
tracked_wallet_live_positions = fetch_tracked_wallet_live_positions(
    card_date=selected_card_date,
)
tracked_wallet_debug = fetch_tracked_wallet_debug(card_date=selected_card_date)
alerts_all = fetch_alerts(limit=200, card_date=selected_card_date)
historical_alerts = fetch_historical_alerts(limit=200)
mlb_edges_all = fetch_mlb_edges(limit=100, card_date=selected_card_date)
mlb_daily_card = fetch_mlb_daily_card(selected_card_date)
mlb_sources = fetch_mlb_sources()
_perf_choice = st.session_state.get("perf_window_choice", "Last 7 days")
_perf_window_days, _perf_window_date = _resolve_perf_window(_perf_choice)
mlb_performance = fetch_mlb_performance(days=_perf_window_days, date=_perf_window_date)
pnl_payload = fetch_pnl_tracker()
odds_cache_payload = fetch_odds_cache()
odds_providers_payload = fetch_odds_providers()
event_match_payload = fetch_odds_event_match(selected_card_date)
pitcher_props_payload = fetch_pitcher_props()
market_validation_payload = fetch_market_validation()
dashboard_debug_payload = fetch_dashboard_debug(selected_card_date)

providers_block = ready.get("providers", {}) or {}
falcon_info = providers_block.get("falcon", {}) or {}
if isinstance(falcon_info, bool):
    falcon_info = {"configured": falcon_info, "healthy": False, "calls": 0, "successes": 0}
odds_block = providers_block.get("odds_api", {}) or {}
odds_cache_meta = odds_block.get("cache", {}) or {}

# Computed metrics used both in the header strip and the Command Center.
mlb_actionable = [e for e in mlb_edges_all if str(e.get("action") or "").lower() != "pass"]
high_conviction = [e for e in mlb_edges_all if edge_prediction_score(e) >= 85]
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
        "Score", min_value=0, max_value=100, value=(0, 100), step=5,
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
        "Show pass candidates in MLB terminal", value=False
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
wallet_filters_hide_current = bool(positions_all and not filtered_positions)


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
                <div class='sf-meta'>Card date: {selected_card_date}</div>
                <div class='sf-meta'>Last refresh: {last_refresh_age}</div>
                <div class='sf-meta'>Backend health: {health_label}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Controls live in a compact expander so the homepage stays
# decision-focused. Each button is disabled while its own dashboard-side
# job is in flight so a double-click can't spawn a duplicate scan. The
# wallet-scan button also respects the backend's own scan-status flag —
# the worker tracks long-running scans across reloads, and we don't want
# to launch a second one on top.
wallet_scan_active = is_job_running("wallet_scan") or scan_status_payload.get("state") == "running"
mlb_edge_active = is_job_running("mlb_edge_scan")
odds_cache_active = is_job_running("odds_cache_refresh")
backend_test_active = is_job_running("backend_test")

# Auto-expand only when a job is in flight — the operator wants to see
# the control bar while something is running, but otherwise the
# collapsed bar keeps the focus on top signals.
any_job_active = wallet_scan_active or mlb_edge_active or odds_cache_active or backend_test_active
with st.expander(
    "Controls", expanded=any_job_active,
):
    action_cols = st.columns([1, 1, 1, 1, 6])
    with action_cols[0]:
        scan_label = "Wallet scan running…" if wallet_scan_active else "Run wallet scan"
        if st.button(
            scan_label, use_container_width=True, type="primary",
            disabled=wallet_scan_active, key="cc_btn_wallet_scan",
        ):
            _job_log("button_click", name="run_wallet_scan")
            action_run_wallet_scan()
    with action_cols[1]:
        mlb_label = "MLB edge scan running…" if mlb_edge_active else "Run MLB edge scan"
        if st.button(
            mlb_label, use_container_width=True, type="primary",
            disabled=mlb_edge_active, key="cc_btn_mlb_edge_scan",
        ):
            _job_log("button_click", name="run_mlb_edge_scan")
            action_run_mlb_edge_scan()
        # Escape hatch: bypass the stale-odds gate. Pitcher-K cards
        # built from BallparkPal fallback don't need fresh sportsbook
        # odds at all, so blocking the entire scan on stale game-total
        # odds buries actionable K signal.
        if st.button(
            "Run scan anyway (use stale odds)",
            use_container_width=True,
            disabled=mlb_edge_active,
            key="cc_btn_mlb_edge_scan_force",
            help=(
                "Bypass the stale-odds gate. Game-total edges may price "
                "off cached odds; pitcher-K cards still build from "
                "BallparkPal fallback regardless of sportsbook freshness."
            ),
        ):
            _job_log("button_click", name="run_mlb_edge_scan_force")
            action_run_mlb_edge_scan(force_stale=True)
    with action_cols[2]:
        odds_label = "Refreshing odds…" if odds_cache_active else "Refresh odds cache"
        if st.button(
            odds_label, use_container_width=True,
            disabled=odds_cache_active, key="cc_btn_refresh_odds_cache",
        ):
            _job_log("button_click", name="refresh_odds_cache")
            action_refresh_odds_cache()
    with action_cols[3]:
        test_label = "Testing backend…" if backend_test_active else "Test backend"
        if st.button(
            test_label, use_container_width=True,
            disabled=backend_test_active, key="cc_btn_test_backend",
        ):
            _job_log("button_click", name="test_backend")
            action_test_backend()

# Single scoped status line — replaces every page-wide `st.spinner` overlay
# the dashboard used to throw up during background work.
render_active_job_panel()

def _render_wallet_scan_status(payload: dict[str, Any]) -> None:
    """Operator-facing wallet-scan visibility panel.

    Single source of truth for "what is the wallet scan actually doing
    right now?" — shows the phase, live counters, per-wallet diagnostics
    table, watchdog deadline, and end-of-run summary. Backed by
    ``/run-scan/status`` (reaped on every read so a wedged scan can never
    sit on "running" forever).
    """
    state = str(payload.get("state") or "idle").lower()
    if state == "idle":
        return
    started_at = payload.get("started_at")
    timeout_at = payload.get("timeout_at")
    summary = payload.get("summary")
    progress = payload.get("progress") or {}
    per_wallet = progress.get("per_wallet") or []
    phase = progress.get("phase") or DASH

    # Top banner — colored by state, names the phase + deadline. Never
    # shows just "running" — always says what stage and when it expires.
    if state == "running":
        st.info(
            f"**Wallet scan running** · phase: `{phase}` · started "
            f"{fmt_relative(started_at)} · watchdog timeout at "
            f"{fmt_dt_mst(timeout_at) or DASH}. "
            "Counters below update live; if nothing is moving, click "
            "**Reset scan state** or wait for the watchdog to mark it stale."
        )
    elif state == "timeout":
        st.error(
            f"**Wallet scan timed out** · {payload.get('error') or 'no detail'} "
            f"· phase at timeout: `{phase}`"
        )
    elif state == "failed":
        st.warning(
            f"**Wallet scan failed**: {payload.get('error') or 'unknown error'}"
        )
    elif state == "finished":
        st.success(f"**Wallet scan finished.** {summary or ''}".strip())

    # Live counters — wallets / markets / positions / rejections / errors.
    cols = st.columns(6)
    cols[0].metric(
        "Wallets scanned",
        f"{progress.get('wallets_scanned', 0)} / {progress.get('wallets_loaded', 0)}",
    )
    cols[1].metric("Raw positions", progress.get("raw_positions_found", 0))
    cols[2].metric("Active after filters", progress.get("active_positions", 0))
    cols[3].metric("Markets checked", progress.get("markets_checked", 0))
    cols[4].metric("API errors", progress.get("api_errors", 0))
    cols[5].metric("Rate-limit hits", progress.get("rate_limited", 0))

    rej_cols = st.columns(3)
    rej_cols[0].metric(
        "Rejected stale", progress.get("positions_rejected_stale", 0)
    )
    rej_cols[1].metric(
        "Rejected date mismatch",
        progress.get("positions_rejected_date_mismatch", 0),
    )
    rej_cols[2].metric(
        "Rejected market-key mismatch",
        progress.get("positions_rejected_market_key_mismatch", 0),
    )

    # Per-wallet debug table — wallet, address, request status, raw / active
    # counts, last seen market, error. The table the user explicitly asked
    # for so "0 Active" is always explainable per-wallet.
    if per_wallet:
        st.markdown("**Per-wallet diagnostics**")
        df_rows: list[dict[str, Any]] = []
        for row in per_wallet:
            df_rows.append(
                {
                    "nickname": row.get("nickname") or DASH,
                    "address": shorten_wallet(row.get("address")) or DASH,
                    "status": row.get("status") or DASH,
                    "raw_positions": int(row.get("raw_positions") or 0),
                    "active_positions": int(row.get("active_positions") or 0),
                    "last_market": row.get("last_market") or DASH,
                    "error": (row.get("error") or "")[:160] or DASH,
                }
            )
        st.dataframe(
            pd.DataFrame(df_rows).fillna(DASH),
            use_container_width=True, hide_index=True,
            height=min(360, 60 + 30 * len(df_rows)),
        )

    if summary and state != "finished":
        st.caption(summary)


_render_wallet_scan_status(scan_status_payload)

# Operator controls: dry-run diagnostics + manual reset. Sit just under
# the visibility panel so the operator never has to hunt for them when a
# scan looks wedged.
scan_ctl_cols = st.columns([1, 1, 6])
with scan_ctl_cols[0]:
    if st.button(
        "Run scan diagnostics",
        key="cc_btn_scan_diagnostics",
        use_container_width=True,
        help=(
            "Dry-run probe — confirms tracked-wallet count, primary "
            "provider reachability, and a sample of raw positions. "
            "Does NOT trigger a full scan."
        ),
    ):
        _job_log("button_click", name="run_scan_diagnostics")
        with st.status("Running scan diagnostics…", expanded=True, state="running") as diag_box:
            try:
                diag = api_get("/run-scan/diagnostics", params={"sample": 5}, timeout=30.0)
                diag_box.update(label="Scan diagnostics complete", state="complete")
            except ApiError as exc:
                diag = None
                diag_box.update(label="Scan diagnostics failed", state="error")
                render_api_error(exc, prefix="Scan diagnostics failed")
        if diag is not None:
            dcols = st.columns(3)
            dcols[0].metric("Tracked wallets", diag.get("tracked_wallets", 0))
            dcols[1].metric(
                "Provider reachable",
                "yes" if diag.get("provider_reachable") else "no",
            )
            dcols[2].metric("Sample positions", diag.get("sample_count", 0))
            if diag.get("provider_error"):
                st.warning(f"Provider error: {diag['provider_error']}")
            sample = diag.get("sample_positions") or []
            if sample:
                st.markdown("**Sample raw positions (first wallet)**")
                st.dataframe(
                    pd.DataFrame(sample).fillna(DASH),
                    use_container_width=True, hide_index=True,
                    height=min(240, 60 + 30 * len(sample)),
                )
            else:
                st.caption("No sample positions returned for the first wallet.")
with scan_ctl_cols[1]:
    reset_disabled = scan_status_payload.get("state") not in {"running", "timeout", "failed"}
    if st.button(
        "Reset scan state",
        key="cc_btn_reset_scan_state",
        use_container_width=True,
        disabled=reset_disabled,
        help=(
            "Force the wallet-scan status back to idle. The watchdog "
            "does this automatically after 3 minutes, but this button "
            "lets you do it immediately if a scan looks wedged."
        ),
    ):
        _job_log("button_click", name="reset_scan_state")
        try:
            api_post("/run-scan/reset", json={})
            st.success("Scan state reset to idle.")
        except ApiError as exc:
            render_api_error(exc, prefix="Reset failed")
        st.cache_data.clear()
        _job_log("rerun_trigger", source="reset_scan_state")
        st.rerun()

stale_positions_hidden = int(dashboard_debug_payload.get("stale_wallet_positions_hidden") or 0)
stale_alerts_hidden = int(dashboard_debug_payload.get("stale_alerts_hidden") or 0)
archived_total = stale_positions_hidden + stale_alerts_hidden
if archived_total:
    # Neutral note — the previous wording made historical retention sound
    # like an outage. Operators just need to know how many records the
    # date filter is currently scoping out.
    st.caption(
        f"Showing today's opportunities only. "
        f"{archived_total:,} historical records archived "
        f"(scoped out by card date {selected_card_date})."
    )

if not mlb_edges_all and odds_cache_status in {"stale", "empty"}:
    st.warning("Odds cache stale; refresh required before edge scan.")
mlb_blocked_by_stale_odds = not mlb_edges_all and odds_cache_status in {"stale", "empty"}


# Decision-focused KPI strip — only the four numbers the operator
# needs to answer "what should I bet right now?".
#   Games Today      — slate size, the universe we're picking from
#   Edges Found      — total MLB edges in today's scan
#   Wallet Signals   — tracked-wallet activity for today's card
#   High Conviction  — edges above the strong-signal threshold
#
# The previous seven-column strip carried P&L, CLV, Falcon health, odds
# cache — useful, but operator-facing detail that belongs in the
# dedicated P&L Tracker / Performance / Odds Cache / Health tabs. P&L
# summary cards moved into `tab_pnl` (see below) so the homepage stays
# focused on opportunities, not portfolio state.
games_today_count = int((mlb_sources.get("row_counts") or {}).get("mlb_games", 0))
wallet_signals_count = len(positions_all)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Games Today", games_today_count)
m2.metric("Edges Found", len(mlb_edges_all))
m3.metric("Wallet Signals", wallet_signals_count)
m4.metric("High Conviction", len(high_conviction))


# =============================================================================
# Tab navigation
# =============================================================================

(
    tab_command,
    tab_mlb,
    tab_pricing_edge,
    tab_wallet,
    tab_pnl,
    tab_perf,
    tab_odds,
    tab_bpp,
    tab_watchlist,
    tab_alerts,
    tab_health,
) = st.tabs([
    "Command Center",
    "MLB Terminal",
    "Pricing Edge",
    "Wallet Flow",
    "P&L Tracker",
    "Performance / CLV",
    "Odds Cache",
    "BallparkPal",
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

    # -----------------------------------------------------------------------
    # Tracked Wallet Live Positions — always shown when raw positions
    # exist, regardless of whether they survived the Signal pipeline's
    # card-date / score-threshold filters. This is the panel that
    # guarantees the dashboard never shows "no current-card wallet flow"
    # while the scanner is reporting active positions.
    # -----------------------------------------------------------------------
    from app.services.tracked_wallet_positions import (
        classify_wallet_against_edges,
        edges_indexed_by_key,
    )

    edge_index_for_wallets = edges_indexed_by_key(mlb_edges_all or [])
    classified_live_positions = [
        classify_wallet_against_edges(pos, edge_index_for_wallets)
        for pos in (tracked_wallet_live_positions or [])
    ]
    live_count = len(classified_live_positions)
    matched_count = sum(
        1 for p in classified_live_positions
        if p.get("edge_match_kind") in {"exact_line", "matchup_date"}
    )
    wallet_only_count = sum(
        1 for p in classified_live_positions
        if p.get("edge_match_kind") == "wallet_only"
    )
    st.markdown(
        f"### Tracked Wallet Live Positions — {live_count} found today"
    )
    if classified_live_positions:
        st.caption(
            f"{matched_count} match an MLB edge · {wallet_only_count} "
            "wallet-only (no matching edge) · sourced from raw Trade rows, "
            "no score threshold."
        )
        live_rows = []
        for pos in classified_live_positions:
            kind = pos.get("edge_match_kind") or "wallet_only"
            # Renamed from ``badge`` — the prior name shadowed the
            # module-level ``badge()`` helper, which later sections
            # (Recent Alerts, Wallet Flow tab) need to call.
            match_status = {
                "exact_line": "✅ Wallet Confirmed (exact line)",
                "matchup_date": "✅ Wallet Confirmed (matchup)",
                "sport_date": "wallet near edge",
                "wallet_only": "wallet-only",
            }.get(kind, kind)
            live_rows.append({
                "status": match_status,
                "trader": pos.get("wallet_nickname") or DASH,
                "sport": pos.get("sport") or DASH,
                "matchup": pos.get("market_title") or pos.get("market_slug") or DASH,
                "side": pos.get("side") or DASH,
                "outcome": pos.get("outcome") or DASH,
                "entry_price": pos.get("entry_price"),
                "current_yes": pos.get("current_yes_price"),
                "size_usd": pos.get("size_usd"),
                "opened_at": pos.get("opened_at") or DASH,
                "event_date": pos.get("parsed_event_date") or DASH,
                "normalized_key": pos.get("normalized_market_key") or DASH,
                "platform": pos.get("market_platform") or DASH,
                "market_slug": pos.get("market_slug") or DASH,
            })
        st.dataframe(
            pd.DataFrame(live_rows).fillna(DASH),
            use_container_width=True,
            hide_index=True,
            height=min(360, 60 + 32 * len(live_rows)),
            column_config={
                "entry_price": st.column_config.NumberColumn("entry", format="%.3f"),
                "current_yes": st.column_config.NumberColumn("current yes", format="%.3f"),
                "size_usd": st.column_config.NumberColumn("size $", format="$%.0f"),
            },
        )
    else:
        st.info(
            "No tracked-wallet trades in the last 36h for "
            f"{selected_card_date}. The scanner debug panel below shows "
            "exactly why each candidate row was rejected — open it before "
            "blaming the data."
        )

    # -----------------------------------------------------------------------
    # Wallet-Confirmed Edge Cards — the strict join: only edges where
    # a tracked wallet has a current position. This is what the
    # operator looks at when they want "wallet+model agree."
    # -----------------------------------------------------------------------
    confirmed_edges_seen: set[int] = set()
    confirmed_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for pos in classified_live_positions:
        matched_edge = pos.get("matched_edge")
        if not matched_edge:
            continue
        if pos.get("edge_match_kind") not in {"exact_line", "matchup_date"}:
            continue
        edge_id = matched_edge.get("id")
        if edge_id is not None and edge_id in confirmed_edges_seen:
            continue
        if edge_id is not None:
            confirmed_edges_seen.add(edge_id)
        confirmed_pairs.append((pos, matched_edge))

    st.markdown(
        f"### Wallet-Confirmed Edge Cards — {len(confirmed_pairs)} match"
        + ("" if len(confirmed_pairs) == 1 else "es")
    )
    if confirmed_pairs:
        confirmed_rows = []
        for pos, edge in confirmed_pairs:
            confirmed_rows.append({
                "trader": pos.get("wallet_nickname") or DASH,
                "wallet_side": pos.get("side") or DASH,
                "market": edge.get("market") or DASH,
                "edge_side": (edge.get("side") or "").title(),
                "prediction": edge.get("prediction_score"),
                "execution": edge.get("execution_score"),
                "size_usd": pos.get("size_usd"),
                "match_kind": pos.get("edge_match_kind"),
            })
        st.dataframe(
            pd.DataFrame(confirmed_rows).fillna(DASH),
            use_container_width=True,
            hide_index=True,
            height=min(280, 60 + 32 * len(confirmed_rows)),
            column_config={
                "prediction": st.column_config.NumberColumn("prediction", format="%.1f"),
                "execution": st.column_config.NumberColumn("execution", format="%.1f"),
                "size_usd": st.column_config.NumberColumn("size $", format="$%.0f"),
            },
        )
    else:
        st.caption(
            "No wallet-confirmed edges right now. Wallet-only positions "
            "still appear in the panel above."
        )

    # -----------------------------------------------------------------------
    # Wallet Flow debug panel — counts plus worked examples for every
    # rejection so the operator can see exactly why a row was hidden.
    # -----------------------------------------------------------------------
    with st.expander(
        f"Wallet Flow debug panel · "
        f"{tracked_wallet_debug.get('raw_recent_trades') or 0} raw / "
        f"{tracked_wallet_debug.get('accepted_for_card_date') or 0} accepted / "
        f"{tracked_wallet_debug.get('rejected') or 0} rejected",
        expanded=False,
    ):
        dbg_cols = st.columns(4)
        dbg_cols[0].metric("Raw recent trades (36h)", tracked_wallet_debug.get("raw_recent_trades") or 0)
        dbg_cols[1].metric("Accepted for card_date", tracked_wallet_debug.get("accepted_for_card_date") or 0)
        dbg_cols[2].metric("Rejected", tracked_wallet_debug.get("rejected") or 0)
        dbg_cols[3].metric("Displayed", live_count)
        reasons = tracked_wallet_debug.get("rejection_reasons") or {}
        if reasons:
            st.markdown("**Rejection reason histogram**")
            st.dataframe(
                pd.DataFrame(
                    [{"reason": k, "count": v} for k, v in reasons.items()]
                ).sort_values("count", ascending=False),
                use_container_width=True, hide_index=True,
            )
        examples = tracked_wallet_debug.get("top_rejected_examples") or []
        if examples:
            st.markdown("**Top 10 rejected examples**")
            st.dataframe(
                pd.DataFrame(examples[:10]).fillna(DASH),
                use_container_width=True, hide_index=True,
            )
        elif (tracked_wallet_debug.get("rejected") or 0) == 0:
            st.caption("No rejections in window.")

    left, right = st.columns([3, 2], gap="medium")

    with left:
        st.markdown("### Top Signals Today")
        top_decisions = sorted(
            [
                e for e in mlb_edges_all
                if edge_prediction_score(e) >= SCORE_ACTIONABLE_MIN
                and str(e.get("action") or "").lower() != "pass"
                and not e.get("odds_stale")
            ],
            key=edge_decision_sort_key,
            reverse=True,
        )[:5]
        if top_decisions:
            for edge in top_decisions:
                render_edge_card(edge)
        else:
            # Per spec: distinct empty-state for the high-conviction
            # section vs. the lower-priority watchlist section below.
            if mlb_blocked_by_stale_odds:
                render_empty_state(
                    "No actionable signals",
                    "Odds cache is stale — refresh odds before running the edge scan.",
                    actions=[
                        ("Refresh odds cache", action_refresh_odds_cache),
                        ("Run MLB edge scan", action_run_mlb_edge_scan),
                    ],
                )
            elif not mlb_edges_all:
                render_empty_state(
                    "No actionable signals",
                    "No edges have been generated for today. Run the MLB edge scan or refresh the odds cache.",
                    actions=[
                        ("Run MLB edge scan", action_run_mlb_edge_scan),
                        ("Refresh odds cache", action_refresh_odds_cache),
                    ],
                )
            else:
                st.info(
                    "No high-conviction signals yet. Showing watchlist candidates below."
                )
        # If nothing crossed the high-conviction bar, explain why using
        # only real counts derived from the returned edges.
        render_why_no_high_conviction(mlb_edges_all)

        st.markdown("### Watchlist Candidates")
        watchlist_candidates = [
            e for e in sorted(mlb_edges_all, key=edge_decision_sort_key, reverse=True)
            if SCORE_ACTIONABLE_MIN <= edge_prediction_score(e) < SCORE_STRONG_MIN
            and str(e.get("action") or "").lower().startswith("watch")
        ][:5]
        if watchlist_candidates:
            for edge in watchlist_candidates:
                render_edge_card(edge)
        else:
            render_empty_state(
                "MARKET SILENT",
                (
                    "Odds cache stale; refresh odds before running the edge scan."
                    if mlb_blocked_by_stale_odds
                    else f"No watchlist candidates in the {SCORE_ACTIONABLE_MIN}-{SCORE_STRONG_MIN - 1} band right now."
                ),
                actions=[("Refresh odds cache", action_refresh_odds_cache)],
            )

        st.markdown("### Highest Conviction Wallet Flow")
        top_wallets = [p for p in filtered_positions if (p.get("score") or 0) >= score_min][:5]
        if top_wallets:
            for sig in top_wallets:
                render_wallet_card(sig)
        elif wallet_filters_hide_current:
            render_empty_state(
                "Current-card wallet flow hidden by filters.",
                f"{len(positions_all)} current-card signal(s) exist. "
                "Lower the score filter or clear sidebar filters.",
            )
        elif len(classified_live_positions) > 0:
            # Curated signal feed is empty, but raw wallet activity is
            # not — point the operator at the panel above instead of
            # pretending nothing is happening.
            render_empty_state(
                "No HIGH-conviction wallet flow — but raw activity exists.",
                f"{len(classified_live_positions)} tracked-wallet position(s) "
                f"are open for {selected_card_date}. None cleared the "
                "Signal pipeline's score threshold, so they're surfaced "
                "in 'Tracked Wallet Live Positions' above. Open the Wallet "
                "Flow debug panel for the rejection breakdown.",
            )
        else:
            render_empty_state(
                "No current-card wallet flow found.",
                f"No {selected_card_date} wallet flow meets the live-card filters.",
            )

    with right:
        st.markdown("### Market Pulse")
        # Slimmed to four operator-facing numbers — raw counts (odds
        # snaps, prop snaps, source health) live in Odds Cache / Health.
        edges_with_warnings = sum(1 for e in mlb_edges_all if e.get("warnings"))
        last_refresh_label = (
            compact_time_ago(prev_refresh, now=now_utc) if prev_refresh else "just now"
        )

        def _pulse_row(label: str, value: Any, kind: str = "") -> str:
            kind_cls = f" {kind}" if kind else ""
            return (
                "<div class='sf-card-row sf-meta'>"
                f"<span class='k'>{label}:</span> "
                f"<span class='val{kind_cls}'>{value}</span>"
                "</div>"
            )

        review_kind = "warn" if edges_with_warnings else ""
        missing_kind = "warn" if missing_odds_edges else ""
        pulse_html = (
            "<div class='sf-card'>"
            + _pulse_row("Edges found", len(mlb_edges_all))
            + _pulse_row("Need review", edges_with_warnings, review_kind)
            + _pulse_row("Missing odds", len(missing_odds_edges), missing_kind)
            + _pulse_row("Last refreshed", last_refresh_label)
            + "</div>"
        )
        st.markdown(pulse_html, unsafe_allow_html=True)

        st.markdown("### Recent Alerts")
        recent_sent = [a for a in alerts_all if a.get("status") == "sent"][:5]
        if recent_sent:
            edge_matchups = _edge_matchup_index(mlb_edges_all)
            for a in recent_sent:
                channel = a.get("channel") or "?"
                ch_badge = badge(channel, "purple" if channel == "discord" else "cyan")
                edge_link = _alert_edge_reference(a.get("message"), edge_matchups)
                st.markdown(
                    "<div class='sf-card'>"
                    + f"<div class='sf-card-row'>{ch_badge}<span class='sf-meta'> · {fmt_dt_mst(a.get('created_at'))}</span></div>"
                    + f"<div class='sf-card-row'>{(a.get('message') or '')[:160]}</div>"
                    + edge_link
                    + "</div>",
                    unsafe_allow_html=True,
                )
        else:
            render_empty_state(
                "No current-card alerts.",
                f"No alerts have been dispatched for {selected_card_date}.",
            )


# =============================================================================
# MLB Terminal
# =============================================================================

with tab_mlb:
    # Daily card hero — three columns of edge cards.
    arizona_today_iso = datetime.now(TZ_MST).date().isoformat()
    is_stale_card = bool(mlb_daily_card and mlb_daily_card.get("is_stale"))
    card_date_label = (mlb_daily_card or {}).get("card_date") or "—"
    requested_date_label = (mlb_daily_card or {}).get("requested_date") or arizona_today_iso

    st.markdown("### Daily Card")
    if is_stale_card:
        st.warning(
            f"⚠️ No MLB edge scan has been run for **{requested_date_label}** "
            f"(Arizona today). Showing the most recent card from **{card_date_label}**. "
            "Click *Run MLB edge scan* to refresh today's edges, high conviction plays, "
            "and near misses."
        )
        action_cols = st.columns([1, 5])
        with action_cols[0]:
            if st.button("Run MLB edge scan", use_container_width=True, key="mlb_run_scan_stale"):
                action_run_mlb_edge_scan()
    elif mlb_daily_card:
        st.caption(
            f"Showing card for **{card_date_label}** · Arizona today: **{arizona_today_iso}**"
        )

    if mlb_daily_card:
        hero_cols = st.columns(3, gap="medium")
        sections = [
            ("Top Game Totals", mlb_daily_card.get("top_game_totals") or []),
            ("Top Pitcher Ks", mlb_daily_card.get("top_pitcher_strikeouts") or []),
            ("Near Misses", mlb_daily_card.get("near_misses") or []),
        ]
        # Pitcher K stage-by-stage diagnostics — used both for a stage-
        # specific empty-state message and for the diagnostics panel
        # rendered below the hero strip.
        _dq_summary = (mlb_daily_card or {}).get("data_quality_summary") or {}
        pitcher_k_diag = _dq_summary.get("pitcher_k") or {}
        pitcher_k_empty_msg = (
            _dq_summary.get("pitcher_k_empty_state_message")
            or "No qualifying edges in this band for today's slate."
        )
        for col, (title, rows) in zip(hero_cols, sections):
            with col:
                stale_suffix = " (stale)" if is_stale_card else ""
                st.markdown(f"<h3>{title}{stale_suffix}</h3>", unsafe_allow_html=True)
                if not rows:
                    if mlb_blocked_by_stale_odds:
                        body = "Odds cache stale; refresh odds before running the edge scan."
                    elif is_stale_card:
                        body = (
                            f"No saved edges for {requested_date_label}. "
                            "Run the scan to populate."
                        )
                    elif title == "Top Pitcher Ks":
                        # Stage-specific message tells the operator
                        # which funnel stage collapsed (no projections /
                        # no odds / no matches / no edge / etc).
                        body = pitcher_k_empty_msg
                    else:
                        body = "No qualifying edges in this band for today's slate."
                    render_empty_state("NO QUALIFYING EDGES", body)
                else:
                    for row in rows[:3]:
                        render_edge_card(row)
    else:
        render_empty_state(
            "DAILY CARD MISSING",
            f"No MLB daily card has ever been generated. Run the MLB edge scan to "
            f"materialize {arizona_today_iso}'s card.",
            actions=[("Run MLB edge scan", action_run_mlb_edge_scan)],
        )

    # ----------------------------------------------------------------------
    # Pitcher K scan diagnostics — exposed inline (NOT in an expander
    # buried at the bottom) so the operator can see funnel collapse
    # in the same place they're seeing "no cards."
    # ----------------------------------------------------------------------
    if pitcher_k_diag:
        st.markdown("### Pitcher K Scan Diagnostics")
        d_cols = st.columns(4)
        d_cols[0].metric(
            "Strikeout projections loaded",
            pitcher_k_diag.get("strikeout_projections_loaded", 0),
        )
        d_cols[1].metric(
            "Sportsbook K props loaded",
            pitcher_k_diag.get("sportsbook_pitcher_k_props_loaded", 0),
        )
        d_cols[2].metric(
            "Pitcher names matched (sportsbook)",
            pitcher_k_diag.get("pitcher_names_matched_sportsbook", 0),
        )
        d_cols[3].metric(
            "Pitcher names matched (BPP fallback)",
            pitcher_k_diag.get("pitcher_names_matched_ballparkpal", 0),
        )
        d_cols2 = st.columns(4)
        d_cols2[0].metric(
            "Candidates built (sportsbook)",
            pitcher_k_diag.get("candidates_built_from_sportsbook", 0),
        )
        d_cols2[1].metric(
            "Candidates built (BPP fallback)",
            pitcher_k_diag.get("candidates_built_from_ballparkpal_fallback", 0),
        )
        d_cols2[2].metric(
            "Rejected: missing odds",
            pitcher_k_diag.get("candidates_rejected_missing_odds", 0),
        )
        d_cols2[3].metric(
            "Rejected: K-edge < 0.15",
            pitcher_k_diag.get("candidates_rejected_by_threshold", 0),
        )
        d_cols3 = st.columns(4)
        d_cols3[0].metric(
            "Promoted: watchlist (≥0.15)",
            pitcher_k_diag.get("candidates_promoted_watchlist", 0),
        )
        d_cols3[1].metric(
            "Promoted: candidate (≥0.35)",
            pitcher_k_diag.get("candidates_promoted_candidate", 0),
        )
        d_cols3[2].metric(
            "Promoted: strong (≥0.65)",
            pitcher_k_diag.get("candidates_promoted_strong", 0),
        )
        d_cols3[3].metric(
            "Cards rendered",
            pitcher_k_diag.get("cards_rendered", 0),
        )
        unmatched = pitcher_k_diag.get("unmatched_pitcher_examples") or []
        if unmatched:
            with st.expander(
                f"Unmatched pitcher examples ({len(unmatched)})",
                expanded=False,
            ):
                st.dataframe(
                    pd.DataFrame(unmatched).fillna(DASH),
                    use_container_width=True, hide_index=True,
                )
        fb_examples = pitcher_k_diag.get("fallback_card_examples") or []
        if fb_examples:
            with st.expander(
                f"BallparkPal fallback cards built ({len(fb_examples)})",
                expanded=False,
            ):
                st.dataframe(
                    pd.DataFrame(fb_examples).fillna(DASH),
                    use_container_width=True, hide_index=True,
                )

    st.markdown("### Score Distribution")
    if mlb_blocked_by_stale_odds:
        render_empty_state(
            "NO SCORE DATA",
            "Odds cache stale; refresh odds before running the edge scan.",
            actions=[("Refresh odds cache", action_refresh_odds_cache)],
        )
    else:
        render_score_distribution(
            mlb_edges_all,
            score_key="prediction_score",
            label="Prediction Score",
        )

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
        # Watchlist ranking: prediction_score → wallet_alignment →
        # execution_score. The original legacy_score lives on the
        # Pricing Edge tab for the sportsbook-edge-first view.
        terminal_rows = sorted(
            terminal_rows,
            key=edge_decision_sort_key,
            reverse=True,
        )
        if not terminal_rows:
            render_empty_state(
                "NO NON-PASS EDGES",
                "All current edges grade as Pass. Toggle 'Show pass candidates' in the sidebar to inspect them.",
            )
        else:
            df_edges = pd.DataFrame([
                {
                    "prediction": e.get("prediction_score"),
                    "execution": e.get("execution_score"),
                    "trap": "⚠ trap" if e.get("cheap_price_trap") else "",
                    "wallet_align": (e.get("factors") or {}).get("wallet_alignment"),
                    "label": confidence_label_fn(
                        e.get("prediction_score") if e.get("prediction_score") is not None else e.get("score"),
                        e.get("action"),
                        e.get("confidence"),
                    )[0],
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
                    "prediction": st.column_config.NumberColumn("prediction", format="%.1f"),
                    "execution": st.column_config.NumberColumn("execution", format="%.1f"),
                    "wallet_align": st.column_config.NumberColumn("wallet aligned", format="%.1f"),
                    "trap": st.column_config.TextColumn("trap"),
                    "line": st.column_config.NumberColumn("line", format="%.1f"),
                    "best_price": st.column_config.TextColumn("best price (US)"),
                    "implied_prob": st.column_config.TextColumn("implied %"),
                },
            )
            st.caption(
                "Watchlist sort order: prediction_score → wallet_alignment → "
                "execution_score. 'trap' fires when execution ≥70 but "
                "prediction <65 (high price, weak model)."
            )
    else:
        render_empty_state(
            "NO MLB EDGES",
            (
                "Odds cache stale; refresh odds before running the edge scan."
                if mlb_blocked_by_stale_odds
                else "Run the MLB edge scan. If still empty, check the Odds Cache tab."
            ),
            actions=[
                ("Refresh odds cache", action_refresh_odds_cache)
                if mlb_blocked_by_stale_odds
                else ("Run MLB edge scan", action_run_mlb_edge_scan)
            ],
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
# Pricing Edge
# =============================================================================

with tab_pricing_edge:
    st.markdown("### Pricing Edge View")
    st.caption(
        "Legacy sportsbook-edge-heavy view. Ranking here uses legacy_score, "
        "sportsbook price edge, and execution_score. It does not drive the "
        "main watchlist ranking."
    )
    pricing_rows = sorted(mlb_edges_all, key=edge_pricing_sort_key, reverse=True)
    if not pricing_rows:
        render_empty_state(
            "NO PRICING EDGES",
            (
                "Odds cache stale; refresh odds before running the edge scan."
                if mlb_blocked_by_stale_odds
                else "Run the MLB edge scan to populate pricing edges."
            ),
            actions=[("Refresh odds cache", action_refresh_odds_cache)],
        )
    else:
        df_pricing = pd.DataFrame([
            {
                "legacy_score": edge_legacy_score(e),
                "sportsbook_price_edge": (
                    _as_float((e.get("factors") or {}).get("sportsbook_price_edge"))
                    or _as_float((e.get("factors") or {}).get("price_edge"))
                    or _as_float((e.get("factors") or {}).get("odds_edge"))
                ),
                "execution": e.get("execution_score"),
                "prediction": e.get("prediction_score"),
                "trap": "trap" if e.get("cheap_price_trap") else "",
                "market": e.get("market"),
                "side": (e.get("side") or "").title(),
                "line": e.get("line"),
                "best_book": e.get("best_book") or DASH,
                "best_price": american_from_price(e.get("best_price")) or DASH,
                "line_movement": (e.get("factors") or {}).get("line_movement")
                    or (e.get("factors") or {}).get("movement"),
                "clv_signal": (e.get("factors") or {}).get("clv_signal"),
                "market_quality": (e.get("factors") or {}).get("market_quality"),
                "action": e.get("action"),
            }
            for e in pricing_rows
        ]).fillna(DASH)
        st.dataframe(
            df_pricing,
            use_container_width=True,
            hide_index=True,
            height=min(520, 60 + 32 * len(df_pricing)),
            column_config={
                "legacy_score": st.column_config.NumberColumn("legacy score", format="%.1f"),
                "sportsbook_price_edge": st.column_config.NumberColumn("sportsbook edge", format="%.1f"),
                "execution": st.column_config.NumberColumn("execution", format="%.1f"),
                "prediction": st.column_config.NumberColumn("prediction", format="%.1f"),
                "line": st.column_config.NumberColumn("line", format="%.1f"),
                "best_price": st.column_config.TextColumn("best price (US)"),
                "line_movement": st.column_config.NumberColumn("line movement", format="%.1f"),
                "clv_signal": st.column_config.NumberColumn("CLV signal", format="%.1f"),
                "market_quality": st.column_config.NumberColumn("market quality", format="%.1f"),
            },
        )

        st.markdown("### Top Pricing Cards")
        for e in pricing_rows[:5]:
            factors = e.get("factors") or {}
            title = html.escape(str(format_card_title(e)))
            legacy = edge_legacy_score(e)
            price_edge = (
                _as_float(factors.get("sportsbook_price_edge"))
                or _as_float(factors.get("price_edge"))
                or _as_float(factors.get("odds_edge"))
            )
            trap = _pill("Cheap Price Trap", "red") if e.get("cheap_price_trap") else ""
            body = (
                "<div class='sf-card'>"
                "<div class='sf-card-head'>"
                f"<div><div class='sf-card-title'>{title}</div>"
                f"<div class='sf-card-sub'>{html.escape(str(e.get('market') or DASH))}</div></div>"
                "<div class='sf-prob-row' style='margin-left:auto;'>"
                f"<div class='sf-prob-cell'><div class='lbl'>Legacy Score</div>"
                f"<div class='val'><span class='sf-score {score_class(legacy)}'>{fmt_score(legacy)}</span></div></div>"
                f"<div class='sf-prob-cell'><div class='lbl'>Sportsbook Edge</div>"
                f"<div class='val'>{fmt_score(price_edge)}</div></div>"
                "</div></div>"
                f"<div class='sf-card-row'>{trap}</div>"
                f"{render_market_price_block(e)}"
                f"{_render_movement_clv(e)}"
                f"{render_factor_bars(factors)}"
                "</div>"
            )
            st.markdown(body, unsafe_allow_html=True)


# =============================================================================
# Wallet Flow
# =============================================================================

def _positions_dataframe(positions: list[dict[str, Any]]) -> "pd.DataFrame":
    """Standard sortable table for tracked wallet positions."""
    rows = []
    for s in positions:
        wallet_disp = s.get("wallet") if show_full_wallet else shorten_wallet(s.get("wallet"))
        tier = tier_for_score(s.get("score"))[0]
        rows.append({
            "score": _as_float(s.get("score")) or 0.0,
            "tier": tier,
            "trader": s.get("trader_nickname") or DASH,
            "wallet": wallet_disp or DASH,
            "league": s.get("league") or DASH,
            "matchup": s.get("matchup") or _market_label(s),
            "contract": s.get("contract") or DASH,
            "side": s.get("side") or DASH,
            "outcome": s.get("outcome") or DASH,
            "avg_entry": _as_float(s.get("entry_price")),
            "size_usd": _as_float(s.get("size_usd")),
            "events": s.get("signal_count") or 1,
            "event_date": s.get("event_date") or DASH,
            "source": s.get("source") or DASH,
            "market": s.get("market_url"),
        })
    return pd.DataFrame(rows).fillna(DASH)


with tab_wallet:
    # --- Source toggle: current card vs all dates -----------------------
    # ``filtered_positions`` is card-date-scoped, which is what the rest
    # of the dashboard wants. Wallet Flow defaults to *all-date* tracked
    # positions because operators who tracked a wallet specifically to
    # watch upcoming-slate plays would otherwise see an empty tab.
    scope_cols = st.columns([2, 1, 1])
    with scope_cols[0]:
        position_scope = st.radio(
            "Position scope",
            options=["All tracked dates", f"Today's card ({selected_card_date})"],
            horizontal=True,
            key="wallet_flow_scope",
            help=(
                "Default: every tracked-wallet position across every date "
                "the backend has. Switch to today's card to see only "
                "positions on the current slate."
            ),
        )
    using_all_dates = position_scope.startswith("All")
    source_positions = all_wallet_positions if using_all_dates else positions_all
    scoped_positions = sorted(
        [p for p in source_positions if (p.get("score") or 0.0) >= score_min],
        key=lambda p: (p.get("score") or 0.0, p.get("consensus_total_size") or 0.0),
        reverse=True,
    )

    # --- Diagnostic banner explaining "where did my positions go?" -----
    n_tracked_wallets = len({
        (s.get("wallet") or s.get("trader_nickname"))
        for s in all_wallet_signals
    } - {None, ""})
    diag_cols = st.columns(4)
    diag_cols[0].metric("Tracked wallets w/ positions", n_tracked_wallets)
    diag_cols[1].metric("All-date positions", len(all_wallet_positions))
    diag_cols[2].metric(f"Today ({selected_card_date})", len(positions_all))
    diag_cols[3].metric("Showing", len(scoped_positions))
    if not scoped_positions:
        if not all_wallet_positions:
            st.warning(
                "No tracked-wallet positions in the database. Run **Wallet scan** "
                "from the Controls bar to ingest fresh trades — if a scan just ran "
                "and this is still empty, the backend returned zero positions."
            )
        elif not using_all_dates and not positions_all:
            st.info(
                f"You have **{len(all_wallet_positions)}** tracked-wallet position(s) "
                f"across all dates, but none on the **{selected_card_date}** card. "
                "Switch the scope toggle to **All tracked dates** to see them."
            )
        elif scoped_positions == [] and wallet_filters_hide_current:
            st.info(
                "Sidebar filters are hiding every position. Lower the score "
                "filter or pick `(all)` for trader / league / source / contract."
            )

    # --- Tracked-Wallet Consensus (>=2 wallets aligned) ----------------
    st.markdown("### Tracked-Wallet Consensus")
    st.caption(
        "Markets where two or more tracked wallets are on the same side. "
        "Sorted by wallet count → total size → mean score."
    )
    consensus_positions = wallet_consensus_groups(scoped_positions, min_wallets=2)
    if consensus_positions:
        for pos in consensus_positions[:10]:
            render_wallet_card(pos)
    else:
        st.caption(
            "No consensus yet — every tracked-wallet position above is held "
            "by a single wallet. Cards still appear in the sections below."
        )

    # --- Highest Conviction (single-wallet, ranked by score) ------------
    st.markdown("### Highest Conviction Positions")
    st.caption(
        "Top tracked-wallet positions by score. Includes single-wallet "
        "plays so a one-wallet conviction signal doesn't get hidden by "
        "the consensus filter above."
    )
    top_positions = scoped_positions[:10]
    if top_positions:
        for pos in top_positions:
            render_wallet_card(pos)
    else:
        # Diagnostic banner above already explains why; no need to repeat.
        pass

    # --- All Tracked Positions (sortable, no card limit) ----------------
    st.markdown(f"### All Tracked Positions ({len(scoped_positions)})")
    st.caption(
        "Click the score column header to sort. Dataframe view of every "
        "tracked-wallet position in the current scope."
    )
    if scoped_positions:
        df_pos = _positions_dataframe(scoped_positions)
        st.dataframe(
            df_pos,
            use_container_width=True,
            hide_index=True,
            height=min(560, 60 + 32 * len(df_pos)),
            column_config={
                "score": st.column_config.ProgressColumn(
                    "score", min_value=0, max_value=100, format="%.1f",
                ),
                "avg_entry": st.column_config.NumberColumn("avg entry", format="%.3f"),
                "size_usd": st.column_config.NumberColumn("size USD", format="$%.0f"),
                "market": st.column_config.LinkColumn("market", display_text="open"),
            },
        )


# =============================================================================
# P&L Tracker
# =============================================================================

with tab_pnl:
    # Summary cards moved off the homepage so the first screen stays
    # focused on opportunities, not portfolio state — but they remain
    # the natural lead-in for the dedicated P&L tab.
    render_pnl_summary_cards(pnl_payload, fmt_money=fmt_money, fmt_num=fmt_num)
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


def _md_table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    """Render a list of dicts as a GitHub-flavored markdown table.

    LLM-friendly: numeric values keep their precision, missing cells are
    rendered as '—' so the model can see they were absent rather than zero.
    """
    if not rows:
        return "_(no rows)_\n"
    if columns is None:
        seen: dict[str, None] = {}
        for row in rows:
            for k in row.keys():
                seen.setdefault(k, None)
        columns = list(seen.keys())
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body_lines: list[str] = []
    for row in rows:
        cells: list[str] = []
        for col in columns:
            value = row.get(col)
            if value is None or value == "":
                cells.append("—")
            elif isinstance(value, float):
                cells.append(f"{value:.4f}".rstrip("0").rstrip("."))
            elif isinstance(value, (list, dict)):
                cells.append(json.dumps(value, default=str))
            else:
                cells.append(str(value))
        body_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body_lines]) + "\n"


def _md_kv(items: list[tuple[str, Any]]) -> str:
    """Render key/value pairs as a markdown bullet list."""
    lines: list[str] = []
    for key, value in items:
        if isinstance(value, float):
            shown = f"{value:.6f}".rstrip("0").rstrip(".")
        elif value is None:
            shown = "—"
        else:
            shown = str(value)
        lines.append(f"- **{key}**: {shown}")
    return "\n".join(lines) + "\n"


def build_performance_export_markdown(
    *,
    mlb_performance: dict[str, Any],
    perf_summary: dict[str, Any],
    clv_block: dict[str, Any],
    diagnostics: dict[str, Any],
    window_label: str,
    perf_date: str | None,
    backend_start: str | None,
    backend_end: str | None,
    arizona_today_iso: str,
) -> str:
    """Serialize the Performance tab into a single markdown document.

    Designed for paste-into-LLM workflows: every section that the tab
    renders visually has a corresponding markdown block here, with the
    raw `mlb_performance` JSON appended as a final appendix so the model
    can fall back to structured data when a summary is ambiguous.
    """
    generated_at = datetime.now(timezone.utc).isoformat()
    parts: list[str] = []
    parts.append("# SignalForge — MLB Performance Export\n")
    parts.append(
        _md_kv([
            ("Generated at (UTC)", generated_at),
            ("Window selection", window_label),
            ("Selected date (Arizona)", perf_date or "n/a (multi-day window)"),
            ("Backend window start", backend_start or "all available"),
            ("Backend window end", backend_end or "all available"),
            ("Arizona today", arizona_today_iso),
            ("Graded edges in window", perf_summary.get("graded_edges") or 0),
            ("Average prediction score", perf_summary.get("average_prediction_score")),
            ("Average execution score", perf_summary.get("average_execution_score")),
            ("Average legacy score", perf_summary.get("average_legacy_score")),
        ])
    )

    # --- Diagnostics ------------------------------------------------------
    parts.append("\n## Diagnostics\n")
    parts.append(
        _md_kv([
            ("Candidate edge snapshots", diagnostics.get("snapshot_count", 0)),
            ("Closing lines", diagnostics.get("closing_line_count", 0)),
            ("Graded edges", diagnostics.get("graded_edge_count", 0)),
            ("Persisted final scores", diagnostics.get("persisted_final_score_count", 0)),
            ("Live finals found", diagnostics.get("live_final_count", 0)),
            ("Final score count (max of sources)", diagnostics.get("final_score_count", 0)),
            ("Last graded at", diagnostics.get("last_graded_at") or "—"),
            ("Reason (if blocked)", diagnostics.get("reason") or "—"),
        ])
    )

    # --- Research Health -------------------------------------------------
    health = mlb_performance.get("research_health") or {}
    parts.append("\n## Research Health\n")
    parts.append(
        _md_kv([
            ("Positive CLV rate", health.get("positive_clv_rate")),
            ("Avg CLV points", health.get("average_clv_points")),
            ("Avg CLV %", health.get("average_clv_percent")),
            ("ROI units", health.get("roi_units")),
            ("Win rate", health.get("win_rate")),
            ("Graded sample size", health.get("graded_sample_size")),
            ("Sample size tier", health.get("sample_size_tier")),
            ("Sample size label", health.get("sample_size_label")),
        ])
    )

    # --- CLV Overview ----------------------------------------------------
    parts.append("\n## CLV Overview\n")
    parts.append(
        _md_kv([
            ("Avg CLV points", clv_block.get("average_clv_points")),
            ("Avg CLV %", clv_block.get("average_clv_percent")),
            ("Positive CLV rate", clv_block.get("positive_clv_rate")),
            ("Edges with CLV", clv_block.get("edges_with_clv") or 0),
            ("Missing CLV count", clv_block.get("missing_clv_count") or 0),
        ])
    )
    by_side_clv = clv_block.get("by_side") or {}
    by_edge_type_clv = clv_block.get("by_edge_type") or {}
    clv_rows: list[dict[str, Any]] = []
    for side_name, payload in by_side_clv.items():
        clv_rows.append({
            "scope": f"side: {side_name}",
            "count": payload.get("count"),
            "avg_clv_points": payload.get("average_clv_points"),
            "avg_clv_percent": payload.get("average_clv_percent"),
            "positive_clv_rate": payload.get("positive_clv_rate"),
        })
    for etype, payload in by_edge_type_clv.items():
        clv_rows.append({
            "scope": f"edge: {etype}",
            "count": payload.get("count"),
            "avg_clv_points": payload.get("average_clv_points"),
            "avg_clv_percent": payload.get("average_clv_percent"),
            "positive_clv_rate": payload.get("positive_clv_rate"),
        })
    if clv_rows:
        parts.append("\n### CLV by Scope\n")
        parts.append(_md_table(clv_rows))

    # --- Over vs Under ---------------------------------------------------
    side_block = mlb_performance.get("by_side") or {}
    over_stats = side_block.get("over") or {}
    under_stats = side_block.get("under") or {}
    parts.append("\n## Over vs Under Split (game_total)\n")
    if side_block.get("directional_bias_warning"):
        parts.append(f"> Directional bias warning: {side_block['directional_bias_warning']}\n")
    parts.append(
        _md_table([
            {
                "side": "over",
                "count": over_stats.get("count") or 0,
                "win_rate": over_stats.get("win_rate"),
                "roi_units": over_stats.get("roi_units"),
                "avg_score": over_stats.get("average_score"),
                "avg_clv_points": over_stats.get("average_clv_points"),
            },
            {
                "side": "under",
                "count": under_stats.get("count") or 0,
                "win_rate": under_stats.get("win_rate"),
                "roi_units": under_stats.get("roi_units"),
                "avg_score": under_stats.get("average_score"),
                "avg_clv_points": under_stats.get("average_clv_points"),
            },
        ])
    )

    # --- Legacy Score Band Performance -----------------------------------
    parts.append("\n## Legacy Score Band Performance\n")
    by_band = mlb_performance.get("by_score_band") or []
    if by_band:
        parts.append(_md_table(by_band))
        unstable_bands = [
            row.get("score_band") for row in by_band
            if not row.get("stable", True) and (row.get("graded_edges") or 0) > 0
        ]
        if unstable_bands:
            parts.append(
                f"\n> Unstable bands (<30 graded): {', '.join(map(str, unstable_bands))}\n"
            )
    else:
        parts.append("_No score-band breakdown yet._\n")

    parts.append("\n## Prediction Score Band Performance\n")
    by_prediction_band = mlb_performance.get("by_prediction_score_band") or []
    parts.append(
        _md_table(by_prediction_band)
        if by_prediction_band
        else "_No prediction-score breakdown yet._\n"
    )

    parts.append("\n## Execution Score Band Performance\n")
    by_execution_band = mlb_performance.get("by_execution_score_band") or []
    parts.append(
        _md_table(by_execution_band)
        if by_execution_band
        else "_No execution-score breakdown yet._\n"
    )

    # --- Projection Calibration ------------------------------------------
    cal = mlb_performance.get("projection_calibration") or {}
    parts.append("\n## Projection Calibration\n")
    for w in cal.get("warnings") or []:
        parts.append(f"> Warning: {w}\n")
    parts.append(
        _md_kv([
            ("Avg model projected total", cal.get("avg_model_projected_total")),
            ("Avg market entry total", cal.get("avg_market_entry_total")),
            ("Avg closing total", cal.get("avg_closing_total")),
            ("Avg actual total", cal.get("avg_actual_total")),
            ("Avg projection error (signed)", cal.get("avg_projection_error")),
            ("Avg |projection error|", cal.get("avg_absolute_projection_error")),
            ("Rows with projection", cal.get("rows_with_projection") or 0),
            ("Graded game_totals", cal.get("graded_game_totals") or 0),
        ])
    )

    # --- Projection Buckets ----------------------------------------------
    bucket_rows = mlb_performance.get("by_projection_bucket") or []
    parts.append("\n## Projection Buckets\n")
    parts.append(_md_table(bucket_rows) if bucket_rows else "_No projection-bucket data yet._\n")

    # --- Timing Analytics ------------------------------------------------
    timing_rows = mlb_performance.get("by_timing") or []
    parts.append("\n## Timing Analytics\n")
    parts.append(_md_table(timing_rows) if timing_rows else "_No timing data yet._\n")

    # --- ROI by Edge Type ------------------------------------------------
    by_market = mlb_performance.get("by_market") or []
    parts.append("\n## ROI by Edge Type\n")
    parts.append(_md_table(by_market) if by_market else "_No edge-type breakdown yet._\n")

    # --- Factor Attribution ----------------------------------------------
    factor_rows = mlb_performance.get("factor_attribution") or []
    parts.append("\n## Factor Attribution\n")
    if factor_rows:
        parts.append(_md_table(factor_rows))
        unstable_factors = [r.get("factor") for r in factor_rows if r.get("unstable")]
        if unstable_factors:
            parts.append(
                f"\n> Unstable factors (sample <50): {', '.join(map(str, unstable_factors))}\n"
            )
    else:
        parts.append("_No factor attribution available._\n")

    # --- Factor Distribution Audit ---------------------------------------
    fdist = mlb_performance.get("factor_distribution") or {}
    parts.append("\n## Factor Distribution Audit\n")
    fdist_rows = fdist.get("factors") or []
    fdist_summary = fdist.get("summary") or {}
    if fdist_rows:
        parts.append(_md_table(fdist_rows))
        stuck = fdist_summary.get("stuck_at_neutral_factors") or []
        no_info = fdist_summary.get("no_information_factors") or []
        if stuck:
            stuck_labels = ", ".join(
                f"{r.get('factor')} ({(r.get('rate') or 0)*100:.0f}%)" for r in stuck
            )
            parts.append(
                f"\n> Stuck at neutral 50 ≥95%: {stuck_labels}\n"
            )
        if no_info:
            no_info_labels = ", ".join(r.get("factor") for r in no_info)
            parts.append(
                f"\n> No detectable information (low variance + weak CLV corr): "
                f"{no_info_labels}\n"
            )
    else:
        parts.append("_No factor-distribution data yet._\n")

    # --- Score Attribution -----------------------------------------------
    sattr = mlb_performance.get("score_attribution") or {}
    parts.append("\n## Score Attribution\n")
    sattr_rows = sattr.get("factors") or []
    if sattr_rows:
        parts.append(_md_table(sattr_rows))
        parts.append(
            f"\n_Total absolute contribution across all factors: "
            f"{sattr.get('total_absolute_contribution_points') or 0:.1f} points._\n"
        )
    else:
        parts.append("_No score-attribution data yet._\n")

    # --- Recent Side Performance + Engine Penalty ------------------------
    rsp = mlb_performance.get("recent_side_performance") or {}
    rsp_sides = rsp.get("sides") or {}
    parts.append("\n## Recent Side Performance & Engine Penalty\n")
    if rsp_sides:
        penalty_rows: list[dict[str, Any]] = []
        for side_name, payload in rsp_sides.items():
            penalty_rows.append({
                "side": side_name,
                "sample": payload.get("sample_size"),
                "decided": payload.get("decided"),
                "wins": payload.get("wins"),
                "losses": payload.get("losses"),
                "win_rate": payload.get("win_rate"),
                "roi_units": payload.get("roi_units"),
                "engine_penalty_points": payload.get("penalty_points"),
            })
        parts.append(_md_table(penalty_rows))
        parts.append(
            f"\n_Window: last {rsp.get('lookback_days')} days "
            f"({rsp.get('window_start')} → {rsp.get('window_end')})._\n"
        )
    else:
        parts.append("_No rolling side-performance data yet._\n")

    # --- CLV Leaders -----------------------------------------------------
    top_pos = clv_block.get("top_positive") or []
    top_neg = clv_block.get("top_negative") or []
    parts.append("\n## Raw Graded Edges — CLV Leaders\n")
    parts.append("\n### Top positive CLV\n")
    parts.append(_md_table(top_pos) if top_pos else "_None._\n")
    parts.append("\n### Top negative CLV\n")
    parts.append(_md_table(top_neg) if top_neg else "_None._\n")

    # --- Raw payload appendix --------------------------------------------
    # Keep the full JSON so the LLM can recompute any aggregate the
    # human-readable summary lossily renders (e.g. small floats, missing
    # nested counts). Marked as a separate section so a human can skip it.
    parts.append("\n## Appendix — Raw mlb_performance JSON\n")
    parts.append("```json\n")
    parts.append(json.dumps(mlb_performance, indent=2, default=str, sort_keys=True))
    parts.append("\n```\n")
    return "".join(parts)


with tab_perf:
    st.markdown("### Performance Window")
    window_options = ["Last 7 days", "Today", "Yesterday", "Last 30 days", "All time"]
    window = st.selectbox(
        "Date range",
        window_options,
        index=window_options.index(_perf_choice) if _perf_choice in window_options else 0,
        key="perf_window_choice",
    )
    perf_days, perf_date = _resolve_perf_window(window)
    diagnostics = perf_summary.get("diagnostics") or {}
    backend_window = perf_summary.get("window") or {}
    backend_start = backend_window.get("start_date") or diagnostics.get("start_date")
    backend_end = backend_window.get("end_date") or diagnostics.get("end_date")
    arizona_today_iso = perf_summary.get("arizona_today") or datetime.now(TZ_MST).date().isoformat()
    range_label = (
        f"{backend_start} → {backend_end}" if backend_start and backend_end else "all available data"
    )
    st.caption(f"Arizona today: **{arizona_today_iso}** · backend window: **{range_label}**")

    perf_actions = st.columns([1, 1, 1.2, 4.8])
    action_date = perf_date  # only pass to backend when a single day is selected
    with perf_actions[0]:
        if st.button("Update closing lines", use_container_width=True):
            action_update_mlb_closing_lines(date=action_date)
    with perf_actions[1]:
        if st.button("Grade MLB results", use_container_width=True):
            action_grade_mlb_results(date=action_date)
    with perf_actions[2]:
        export_markdown = build_performance_export_markdown(
            mlb_performance=mlb_performance,
            perf_summary=perf_summary,
            clv_block=clv_block,
            diagnostics=diagnostics,
            window_label=window,
            perf_date=perf_date,
            backend_start=backend_start,
            backend_end=backend_end,
            arizona_today_iso=arizona_today_iso,
        )
        export_label_date = perf_date or arizona_today_iso
        export_window_slug = re.sub(r"[^a-z0-9]+", "_", window.lower()).strip("_")
        st.download_button(
            "Download summary",
            data=export_markdown.encode("utf-8"),
            file_name=(
                f"signalforge_mlb_performance_{export_label_date}_{export_window_slug}.md"
            ),
            mime="text/markdown",
            use_container_width=True,
            help=(
                "Markdown export of every analytics section on this tab — "
                "paste into ChatGPT/Claude for tuning + model training."
            ),
        )

    # --- Debug visibility panel so the user can tell WHY the tab is empty ----
    with st.expander("Performance debug panel", expanded=not (perf_summary.get("graded_edges") or 0)):
        dbg_row1 = st.columns(4)
        dbg_row1[0].metric("Selected date", perf_date or backend_start or arizona_today_iso)
        dbg_row1[1].metric("Candidate edge snapshots", diagnostics.get("snapshot_count", 0))
        dbg_row1[2].metric("Closing lines", diagnostics.get("closing_line_count", 0))
        dbg_row1[3].metric("Graded edges", diagnostics.get("graded_edge_count", 0))
        dbg_row2 = st.columns(4)
        dbg_row2[0].metric(
            "Persisted final scores", diagnostics.get("persisted_final_score_count", 0),
        )
        dbg_row2[1].metric("Live finals found", diagnostics.get("live_final_count", 0))
        dbg_row2[2].metric(
            "Backend window",
            f"{backend_start or 'all'} → {backend_end or 'all'}",
        )
        dbg_row2[3].metric(
            "Last graded at", diagnostics.get("last_graded_at") or "—",
        )
        st.caption(
            f"Selected: **{window}** · Arizona today: **{arizona_today_iso}** · "
            f"backend dates: **{backend_start or 'all'} → {backend_end or 'all'}**"
        )
        if diagnostics.get("reason"):
            st.info(diagnostics["reason"])
        if perf_date:
            if st.button(
                "Ingest persisted final scores for selected date",
                use_container_width=False,
                key="ingest_final_scores_btn",
            ):
                with st.spinner(f"Ingesting final scores for {perf_date}..."):
                    try:
                        ingest_result = api_post(
                            "/mlb/debug/final-scores/ingest",
                            params={"date": perf_date},
                            timeout=MLB_RUN_TIMEOUT,
                        )
                    except ApiError as exc:
                        render_api_error(exc, prefix="Final-score ingestion failed")
                        ingest_result = None
                if ingest_result is not None:
                    if ingest_result.get("error"):
                        st.warning(
                            f"Final-score ingestion error for {perf_date}: "
                            f"{ingest_result['error']}"
                        )
                    else:
                        st.success(
                            f"Final scores for {perf_date}: "
                            f"games_seen={ingest_result.get('games_seen', 0)}, "
                            f"finals_found={ingest_result.get('finals_found', 0)}, "
                            f"upserted={ingest_result.get('upserted', 0)}."
                        )
                    st.cache_data.clear()
                    st.rerun()

    st.markdown(f"### Score Distribution · {range_label}")
    render_score_distribution(mlb_edges_all)
    render_why_no_high_conviction(mlb_edges_all)

    snapshot_count = int(diagnostics.get("snapshot_count") or 0)
    final_score_count = int(diagnostics.get("final_score_count") or 0)
    graded = int(perf_summary.get("graded_edges") or 0)
    if snapshot_count == 0:
        target = perf_date or backend_start or arizona_today_iso
        render_empty_state(
            "NO SAVED EDGE SNAPSHOTS",
            f"No saved edge snapshots for {target}. Run an MLB edge scan during game "
            "day to enable grading tomorrow.",
            actions=[
                ("Update closing lines", lambda: action_update_mlb_closing_lines(date=action_date)),
                ("Grade MLB results", lambda: action_grade_mlb_results(date=action_date)),
            ],
        )
    elif graded == 0:
        body = (
            diagnostics.get("reason")
            or (
                f"{snapshot_count} edge snapshot(s) found but {final_score_count} final score(s) "
                "are ingested. Click 'Grade MLB results' once games are final."
            )
        )
        render_empty_state(
            "WAITING FOR GRADED RESULTS",
            body,
            actions=[
                ("Update closing lines", lambda: action_update_mlb_closing_lines(date=action_date)),
                ("Grade MLB results", lambda: action_grade_mlb_results(date=action_date)),
            ],
        )
    if graded:
        # ------------------------------------------------------------------
        # Research Mode filter. Applies to the displayed analytics only —
        # the raw stored grading history is never mutated by this control.
        # ------------------------------------------------------------------
        research_modes = [
            "All candidates",
            "65+ only",
            "75+ only",
            "85+ only",
            "Paper only / watchlist",
        ]
        rmode = st.selectbox(
            "Research Mode",
            research_modes,
            index=0,
            key="perf_research_mode",
            help=(
                "Filtering changes research view only. It does not alter "
                "stored grading history."
            ),
        )
        st.caption(
            "Filtering changes research view only. It does not alter stored "
            "grading history."
        )

        def _score_min_from_mode(mode: str) -> float | None:
            """Translate the Research Mode selector to a minimum score floor."""
            if mode == "65+ only":
                return 65.0
            if mode == "75+ only":
                return 75.0
            if mode == "85+ only":
                return 85.0
            if mode == "Paper only / watchlist":
                # Watchlist tier = scores 55-64, neither weak nor playable.
                return 55.0
            return None

        score_min = _score_min_from_mode(rmode)
        paper_only = rmode == "Paper only / watchlist"

        def _band_in_scope(band: str) -> bool:
            """Whether a score band passes the active Research Mode filter."""
            if rmode == "All candidates":
                return True
            if rmode == "Paper only / watchlist":
                return band == "55-64"
            if score_min is None:
                return True
            order = ["<55", "55-64", "65-74", "75-84", "85+"]
            try:
                return order.index(band) >= order.index(
                    "85+" if score_min >= 85 else "75-84" if score_min >= 75
                    else "65-74" if score_min >= 65 else "<55"
                )
            except ValueError:
                return True

        # ------------------------------------------------------------------
        # 1. Research Health — CLV-first headline. ROI/win-rate intentionally
        # sit below CLV because they're noisier on small samples.
        # ------------------------------------------------------------------
        health = mlb_performance.get("research_health") or {}
        st.markdown("### Research Health")
        h1, h2, h3, h4, h5, h6 = st.columns(6)
        h1.metric("Positive CLV rate", fmt_pct(health.get("positive_clv_rate")))
        h2.metric("Avg CLV points", fmt_num(health.get("average_clv_points"), fmt="{:+.3f}"))
        h3.metric("Avg CLV %", fmt_pct(health.get("average_clv_percent")))
        h4.metric("ROI units", fmt_num(health.get("roi_units"), fmt="{:+.2f}"))
        h5.metric("Win rate", fmt_pct(health.get("win_rate")))
        h6.metric("Graded sample", health.get("graded_sample_size") or graded)
        st.caption(
            "Avg prediction score: "
            f"{fmt_num(perf_summary.get('average_prediction_score'), fmt='{:.1f}')} | "
            "Avg execution score: "
            f"{fmt_num(perf_summary.get('average_execution_score'), fmt='{:.1f}')} | "
            "Avg legacy score: "
            f"{fmt_num(perf_summary.get('average_legacy_score'), fmt='{:.1f}')}"
        )

        # ------------------------------------------------------------------
        # 2. Sample Size Warning — surface the confidence tier inline rather
        # than burying it inside a tooltip.
        # ------------------------------------------------------------------
        st.markdown("### Sample Size")
        tier = health.get("sample_size_tier") or "exploratory"
        sample_label = health.get("sample_size_label") or "exploratory only"
        sample_n = health.get("graded_sample_size") or graded
        if tier == "exploratory":
            st.warning(
                f"Sample size {sample_n} — **{sample_label}**. "
                "Do not draw firm conclusions; ROI and band-level win rates "
                "are likely dominated by variance."
            )
        elif tier == "early":
            st.info(f"Sample size {sample_n} — **{sample_label}**.")
        else:
            st.success(f"Sample size {sample_n} — **{sample_label}**.")

        # ------------------------------------------------------------------
        # 3. CLV Overview — primary research metric. Side / band / edge-type
        # CLV breakdowns sit here.
        # ------------------------------------------------------------------
        st.markdown("### CLV Overview")
        co1, co2, co3, co4 = st.columns(4)
        co1.metric("Avg CLV points", fmt_num(clv_block.get("average_clv_points"), fmt="{:+.3f}"))
        co2.metric("Avg CLV %", fmt_pct(clv_block.get("average_clv_percent")))
        co3.metric("Positive CLV rate", fmt_pct(clv_block.get("positive_clv_rate")))
        co4.metric("Edges w/ CLV", clv_block.get("edges_with_clv") or 0)
        missing_clv = clv_block.get("missing_clv_count") or 0
        if missing_clv:
            st.caption(
                f"{missing_clv} graded edge(s) have no CLV — usually missing "
                "closing line. Run 'Update closing lines' before grading."
            )
        by_side_clv = (clv_block.get("by_side") or {})
        by_edge_type_clv = (clv_block.get("by_edge_type") or {})
        clv_rows = []
        for side_name, payload in by_side_clv.items():
            clv_rows.append({
                "scope": f"side: {side_name}",
                "count": payload.get("count"),
                "avg_clv_points": payload.get("average_clv_points"),
                "avg_clv_percent": payload.get("average_clv_percent"),
                "positive_clv_rate": payload.get("positive_clv_rate"),
            })
        for etype, payload in by_edge_type_clv.items():
            clv_rows.append({
                "scope": f"edge: {etype}",
                "count": payload.get("count"),
                "avg_clv_points": payload.get("average_clv_points"),
                "avg_clv_percent": payload.get("average_clv_percent"),
                "positive_clv_rate": payload.get("positive_clv_rate"),
            })
        if clv_rows:
            st.dataframe(
                pd.DataFrame(clv_rows).fillna(DASH),
                use_container_width=True, hide_index=True,
                height=min(280, 60 + 32 * len(clv_rows)),
            )

        # ------------------------------------------------------------------
        # 4. Over vs Under Split — directional bias diagnostic.
        # ------------------------------------------------------------------
        st.markdown("### Over vs Under Split (game_total)")
        side_block = mlb_performance.get("by_side") or {}
        over_stats = side_block.get("over") or {}
        under_stats = side_block.get("under") or {}
        if side_block.get("directional_bias_warning"):
            st.warning(side_block["directional_bias_warning"])
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("**Over**")
            st.metric("Count", over_stats.get("count") or 0)
            st.metric("Win rate", fmt_pct(over_stats.get("win_rate")))
            st.metric("ROI units", fmt_num(over_stats.get("roi_units"), fmt="{:+.2f}"))
            st.metric("Avg score", fmt_num(over_stats.get("average_score"), fmt="{:.1f}"))
            st.metric("Avg CLV", fmt_num(over_stats.get("average_clv_points"), fmt="{:+.3f}"))
        with s2:
            st.markdown("**Under**")
            st.metric("Count", under_stats.get("count") or 0)
            st.metric("Win rate", fmt_pct(under_stats.get("win_rate")))
            st.metric("ROI units", fmt_num(under_stats.get("roi_units"), fmt="{:+.2f}"))
            st.metric("Avg score", fmt_num(under_stats.get("average_score"), fmt="{:.1f}"))
            st.metric("Avg CLV", fmt_num(under_stats.get("average_clv_points"), fmt="{:+.3f}"))

        # ------------------------------------------------------------------
        # 5. Score Band Performance — five-band segmentation with stability
        # flag. Bands with <30 graded edges render in a muted style.
        # ------------------------------------------------------------------
        st.markdown("### Legacy Score Band Performance")
        by_band = mlb_performance.get("by_score_band") or []
        if rmode != "All candidates":
            by_band = [row for row in by_band if _band_in_scope(row.get("score_band") or "")]
        if by_band:
            df_band = pd.DataFrame(by_band).fillna(DASH)
            st.dataframe(
                df_band, use_container_width=True, hide_index=True,
                height=min(280, 60 + 32 * len(df_band)),
            )
            unstable_bands = [
                row["score_band"] for row in by_band
                if not row.get("stable", True) and (row.get("graded_edges") or 0) > 0
            ]
            if unstable_bands:
                st.caption(
                    "Bands with <30 graded edges shouldn't be treated as profitable: "
                    + ", ".join(unstable_bands)
                )
        else:
            render_empty_state("NO SCORE-BAND BREAKDOWN", "Grade more edges to populate.")

        # ------------------------------------------------------------------
        # 6. Projection Calibration — diagnoses model_proj vs market_close
        # vs actual_total. Warnings only fire on absolute miss > 0.75 runs.
        # ------------------------------------------------------------------
        st.markdown("### Prediction vs Execution Score Performance")
        axis_tabs = st.tabs(["Prediction Score", "Execution Score"])
        for axis_tab, axis_key in zip(
            axis_tabs,
            ("by_prediction_score_band", "by_execution_score_band"),
        ):
            with axis_tab:
                axis_rows = mlb_performance.get(axis_key) or []
                if rmode != "All candidates":
                    axis_rows = [
                        row for row in axis_rows
                        if _band_in_scope(row.get("score_band") or "")
                    ]
                if axis_rows:
                    st.dataframe(
                        pd.DataFrame(axis_rows).fillna(DASH),
                        use_container_width=True,
                        hide_index=True,
                        height=min(280, 60 + 32 * len(axis_rows)),
                    )
                else:
                    st.caption("No graded rows for this score axis yet.")

        st.markdown("### Projection Calibration")
        cal = mlb_performance.get("projection_calibration") or {}
        for w in cal.get("warnings") or []:
            st.warning(w)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Avg model proj.", fmt_num(cal.get("avg_model_projected_total"), fmt="{:.2f}"))
        c2.metric("Avg entry total", fmt_num(cal.get("avg_market_entry_total"), fmt="{:.2f}"))
        c3.metric("Avg close total", fmt_num(cal.get("avg_closing_total"), fmt="{:.2f}"))
        c4.metric("Avg actual total", fmt_num(cal.get("avg_actual_total"), fmt="{:.2f}"))
        c5.metric("Avg proj. error", fmt_num(cal.get("avg_projection_error"), fmt="{:+.2f}"))
        c6.metric("Avg |proj. error|", fmt_num(cal.get("avg_absolute_projection_error"), fmt="{:.2f}"))
        rows_with_proj = cal.get("rows_with_projection") or 0
        graded_gt = cal.get("graded_game_totals") or 0
        if graded_gt and rows_with_proj == 0:
            st.caption(
                "Model projection field not yet populated by the scan pipeline — "
                "calibration falls back to market entry total. Wire "
                "`MlbEdge.model_projected_total` at scan time to enable full "
                "calibration."
            )

        # ------------------------------------------------------------------
        # 7. Projection Buckets — answers "are 10+ projected totals failing?"
        # ------------------------------------------------------------------
        st.markdown("### Projection Buckets")
        bucket_rows = mlb_performance.get("by_projection_bucket") or []
        if bucket_rows and any((r.get("graded_edges") or 0) > 0 for r in bucket_rows):
            df_bucket = pd.DataFrame(bucket_rows).fillna(DASH)
            st.dataframe(
                df_bucket, use_container_width=True, hide_index=True,
                height=min(280, 60 + 32 * len(df_bucket)),
            )
        else:
            st.caption("No projection-bucket data yet.")

        # ------------------------------------------------------------------
        # 8. Timing Analytics — detect late signals via hours-before-game.
        # ------------------------------------------------------------------
        st.markdown("### Timing Analytics")
        timing_rows = mlb_performance.get("by_timing") or []
        if timing_rows and any((r.get("graded_edges") or 0) > 0 for r in timing_rows):
            df_timing = pd.DataFrame(timing_rows).fillna(DASH)
            st.dataframe(
                df_timing, use_container_width=True, hide_index=True,
                height=min(260, 60 + 32 * len(df_timing)),
            )
        else:
            st.caption("No timing data yet (requires game start_time + edge created_at).")

        # ------------------------------------------------------------------
        # 9. ROI by Edge Type — kept from the prior dashboard.
        # ------------------------------------------------------------------
        st.markdown("### ROI by Edge Type")
        by_market = mlb_performance.get("by_market") or []
        if by_market:
            df_market = pd.DataFrame(by_market).fillna(DASH)
            st.dataframe(
                df_market, use_container_width=True, hide_index=True,
                height=min(280, 60 + 32 * len(df_market)),
            )
            try:
                chart_df = pd.DataFrame(by_market).set_index("edge_type")[["roi_units"]]
                if not chart_df.empty:
                    st.bar_chart(chart_df, height=220)
            except Exception:
                pass
        else:
            render_empty_state("NO EDGE-TYPE BREAKDOWN", "Need at least one graded edge per type.")

        # ------------------------------------------------------------------
        # 10. Factor Attribution — avg-on-wins, avg-on-losses, correlations.
        # Factors marked unstable (sample <50) are flagged in their row.
        # ------------------------------------------------------------------
        st.markdown("### Factor Attribution")
        factor_rows = mlb_performance.get("factor_attribution") or []
        if factor_rows:
            df_factors = pd.DataFrame(factor_rows).fillna(DASH)
            st.dataframe(
                df_factors, use_container_width=True, hide_index=True,
                height=min(320, 60 + 32 * len(df_factors)),
            )
            unstable_factors = [r["factor"] for r in factor_rows if r.get("unstable")]
            if unstable_factors:
                st.caption(
                    "Unstable factors (sample <50): " + ", ".join(unstable_factors)
                )
        else:
            st.caption("No factor attribution available.")

        # ------------------------------------------------------------------
        # 10b. Factor Distribution Audit — stuck-at-50 / no-information
        # callouts. Pre-tuning diagnostic: answers "is this factor actually
        # carrying signal or is its producer a stub?"
        # ------------------------------------------------------------------
        st.markdown("### Factor Distribution Audit")
        fdist = mlb_performance.get("factor_distribution") or {}
        fdist_rows = fdist.get("factors") or []
        fdist_summary = fdist.get("summary") or {}
        if fdist_rows:
            df_fdist = pd.DataFrame(fdist_rows).fillna(DASH)
            st.dataframe(
                df_fdist, use_container_width=True, hide_index=True,
                height=min(360, 60 + 32 * len(df_fdist)),
            )
            stuck = fdist_summary.get("stuck_at_neutral_factors") or []
            no_info = fdist_summary.get("no_information_factors") or []
            if stuck:
                stuck_labels = ", ".join(
                    f"{r['factor']} ({(r.get('rate') or 0)*100:.0f}%)" for r in stuck
                )
                st.warning(
                    f"Factors stuck at the neutral 50 sentinel ≥95% of the "
                    f"time: {stuck_labels}. Upstream producer is likely a stub."
                )
            if no_info:
                no_info_labels = ", ".join(r["factor"] for r in no_info)
                st.warning(
                    f"Factors with no detectable information "
                    f"(low variance + weak CLV correlation): {no_info_labels}."
                )
        else:
            st.caption("No factor-distribution data yet (need graded edges).")

        # ------------------------------------------------------------------
        # 10c. Score Attribution — share of score movement per factor.
        # Tells the operator where the score is actually coming from
        # (e.g. "82% of the score's swing comes from data_quality" → bad).
        # ------------------------------------------------------------------
        st.markdown("### Score Attribution")
        sattr = mlb_performance.get("score_attribution") or {}
        sattr_rows = sattr.get("factors") or []
        if sattr_rows:
            df_sattr = pd.DataFrame(sattr_rows).fillna(DASH)
            st.dataframe(
                df_sattr, use_container_width=True, hide_index=True,
                height=min(320, 60 + 32 * len(df_sattr)),
            )
            st.caption(
                f"Total |contribution| across all factors in window: "
                f"{sattr.get('total_absolute_contribution_points') or 0:.1f} "
                f"points · contribution_share sums to 1.0 across factors."
            )
        else:
            st.caption("No score-attribution data yet (need graded edges).")

        # ------------------------------------------------------------------
        # 10d. Over/Under bias diagnostics — rolling 14d perf + the engine's
        # per-side score penalty. Diagnoses whether the model is currently
        # being throttled on either side.
        # ------------------------------------------------------------------
        st.markdown("### Recent Side Performance & Engine Penalty")
        rsp = mlb_performance.get("recent_side_performance") or {}
        rsp_sides = rsp.get("sides") or {}
        if rsp_sides:
            penalty_rows = [
                {
                    "side": side_name,
                    "sample": payload.get("sample_size"),
                    "decided": payload.get("decided"),
                    "wins": payload.get("wins"),
                    "losses": payload.get("losses"),
                    "win_rate": payload.get("win_rate"),
                    "roi_units": payload.get("roi_units"),
                    "engine_penalty_points": payload.get("penalty_points"),
                }
                for side_name, payload in rsp_sides.items()
            ]
            st.dataframe(
                pd.DataFrame(penalty_rows).fillna(DASH),
                use_container_width=True, hide_index=True,
                height=min(180, 60 + 32 * len(penalty_rows)),
            )
            st.caption(
                f"Window: last {rsp.get('lookback_days')} days "
                f"({rsp.get('window_start')} → {rsp.get('window_end')}). "
                "Penalty points are subtracted from the candidate score at "
                "scan time when sample ≥ floor and win rate < 0.45."
            )
            active_penalties = [
                f"{side_name}: −{payload.get('penalty_points'):.1f}"
                for side_name, payload in rsp_sides.items()
                if (payload.get("penalty_points") or 0) > 0
            ]
            if active_penalties:
                st.warning(
                    "Active side penalties on next scan: " + ", ".join(active_penalties)
                )
        else:
            st.caption("No rolling side-performance data yet.")

        # ------------------------------------------------------------------
        # 11. Raw Graded Edges + CLV leaders for spot-checks.
        # ------------------------------------------------------------------
        st.markdown("### Raw Graded Edges — CLV leaders")
        clv_cols = st.columns(2)
        with clv_cols[0]:
            st.markdown("**Top positive CLV**")
            top_pos = clv_block.get("top_positive") or []
            if top_pos:
                st.dataframe(
                    pd.DataFrame(top_pos).fillna(DASH), use_container_width=True,
                    hide_index=True, height=min(280, 60 + 32 * len(top_pos)),
                )
            else:
                st.caption("None")
        with clv_cols[1]:
            st.markdown("**Top negative CLV**")
            top_neg = clv_block.get("top_negative") or []
            if top_neg:
                st.dataframe(
                    pd.DataFrame(top_neg).fillna(DASH), use_container_width=True,
                    hide_index=True, height=min(280, 60 + 32 * len(top_neg)),
                )
            else:
                st.caption("None")


# =============================================================================
# BallparkPal — manual CSV upload is the primary ingestion path. The scraper
# (scripts/update_ballparkpal_cache.py) remains available behind an Advanced
# expander for local workstations that can actually open a headed browser.
# Hosted environments (Render) can't open a real window, which is why the
# legacy Playwright-only flow used to stick in RUNNING_LOGIN for hours.
# =============================================================================

# Fallback list used when the backend returns no labels (e.g. offline mode).
PAGE_DEFAULT_OPTIONS = ("positive_ev", "strikeouts", "hr_zone", "hits", "game_sims")
# Backend job states that block the dashboard's "is something running?" checks.
ACTIVE_JOB_STATUSES = {"queued", "running"}
# Map raw backend state → operator-friendly label so the dashboard never
# shows "RUNNING LOGIN" without saying what that means.
BPP_DISPLAY_STATE_LABELS = {
    "idle": "Idle",
    "waiting_for_login": "Waiting for login",
    "processing": "Processing",
    "complete": "Complete",
    "failed": "Failed",
}

with tab_bpp:
    bpp_payload = fetch_ballparkpal_snapshots(slate_date=None)
    bpp_pages = (bpp_payload.get("pages") or {})
    bpp_labels = (bpp_payload.get("labels") or {})

    def _bpp_page(key: str) -> dict[str, Any]:
        return bpp_pages.get(key) or {}

    (
        bpp_overview_tab,
        bpp_ev_tab,
        bpp_k_tab,
        bpp_hr_tab,
        bpp_hits_tab,
        bpp_sims_tab,
        bpp_debug_tab,
    ) = st.tabs([
        "Overview",
        "Positive EV",
        "Strikeouts",
        "Home Run Zone",
        "Hits",
        "Game Sims",
        "Raw Snapshots",
    ])

    # ---- Job control: refresh + login + operator logs --------------------
    # Live job state. We poll /ballparkpal/jobs (uncached) so the panel
    # reflects the running job in real time without re-rendering the
    # whole dashboard.
    try:
        bpp_jobs_payload = api_get_once("/ballparkpal/jobs", timeout=8.0) or {}
    except Exception:  # noqa: BLE001
        bpp_jobs_payload = {}
    bpp_active_job = bpp_jobs_payload.get("active")
    bpp_has_profile = bool(bpp_jobs_payload.get("has_profile"))
    bpp_recent_jobs = bpp_jobs_payload.get("jobs") or []

    # ---- Overview --------------------------------------------------------
    with bpp_overview_tab:
        st.markdown("### BallparkPal Cache Overview")
        st.caption(
            "**Manual CSV upload is the primary path.** Export each page from "
            "BallparkPal, upload below, and the MLB edge scan reads the "
            "snapshots exactly the same way it would after a scrape. "
            "Playwright automation is optional and lives in the Advanced "
            "expander."
        )

        # ----- Stuck-job recovery -----------------------------------------
        # A login job that's been running 5+ minutes on a hosted environment
        # is never going to finish — Playwright can't open a headed browser
        # on Render. The backend now reaps these automatically (5-minute
        # cap) but the operator may want to clear them immediately. This is
        # the manual escape hatch.
        if bpp_active_job:
            active_age = float(bpp_active_job.get("duration_seconds") or 0)
            tone = "🟥" if active_age > 300 else "🟧"
            stuck_msg = (
                f"{tone} Background job **{bpp_active_job.get('job_id')}** "
                f"(mode={bpp_active_job.get('mode')}) has been "
                f"{bpp_active_job.get('status')} for {active_age:.0f}s."
            )
            if active_age > 300:
                stuck_msg += " The backend will reap it automatically — click below to clear it now."
            st.warning(stuck_msg)
            if st.button(
                "Cancel stuck job / reset BallparkPal job state",
                type="primary",
                key="bpp_cancel_stuck_job",
                use_container_width=False,
            ):
                _job_log("button_click", name="bpp_cancel_stuck_job", job_id=bpp_active_job.get("job_id"))
                try:
                    cleared = api_post("/ballparkpal/jobs/reset", json={})
                    st.success(
                        f"Cleared {cleared.get('count', 0)} stuck job(s). "
                        f"You can now upload a CSV or trigger a new run."
                    )
                except ApiError as exc:
                    render_api_error(exc, prefix="Could not reset job state")
                st.cache_data.clear()
                _job_log("rerun_trigger", source="bpp_cancel_stuck_job")
                st.rerun()

        # ----- Manual CSV upload (PRIMARY PATH) ---------------------------
        st.markdown("#### Manual CSV Upload (recommended)")
        st.caption(
            "Drop one or more CSVs exported from BallparkPal. For each file, "
            "pick the page it came from — the parser uses the same column "
            "aliasing as the scraper, so the resulting cache rows are "
            "indistinguishable from an automated refresh."
        )

        # File uploader. ``key`` is bumped after a successful upload (see
        # ``bpp_uploader_nonce``) so the widget resets — otherwise the same
        # files would keep getting re-uploaded on every rerun.
        uploader_nonce = st.session_state.setdefault("bpp_uploader_nonce", 0)
        uploaded_files = st.file_uploader(
            "Upload CSV files (one per BallparkPal page)",
            type=["csv", "tsv", "txt"],
            accept_multiple_files=True,
            key=f"bpp_csv_uploader_{uploader_nonce}",
            help=(
                "Strikeout Center, Positive EV, Home Run Zone, Hits, and "
                "Game Simulations are the supported pages. The mapping "
                "dropdown below each file controls which page the rows "
                "land in."
            ),
        )

        # Slate date applies to every file in the batch — BallparkPal
        # exports are slate-specific so it's fair to pick one date.
        upload_date_col, upload_btn_col = st.columns([1, 1])
        with upload_date_col:
            upload_slate_date = st.text_input(
                "Slate date (applies to all files)",
                value=selected_card_date,
                key="bpp_upload_slate_date",
                help="YYYY-MM-DD. Defaults to the card date selected in the sidebar.",
            )
        # Status placeholder for upload feedback — keeps it scoped to the
        # CSV section instead of a page-wide spinner.
        upload_state_placeholder = st.empty()

        # Per-file controls: page mapping dropdown + name.
        upload_specs: list[tuple[Any, str]] = []
        page_options = list(bpp_labels.keys()) or list(PAGE_DEFAULT_OPTIONS)
        if uploaded_files:
            st.caption("Map each uploaded file to its BallparkPal page:")
            for file_obj in uploaded_files:
                file_cols = st.columns([3, 2])
                with file_cols[0]:
                    st.markdown(
                        f"<div class='sf-meta'>📄 <b>{html.escape(file_obj.name)}</b> "
                        f"· {file_obj.size or 0} bytes</div>",
                        unsafe_allow_html=True,
                    )
                with file_cols[1]:
                    # Best-effort autodetect based on filename hints —
                    # "strikeout-center.csv" → strikeouts. Operator can
                    # override.
                    lower = file_obj.name.lower()
                    if "strike" in lower:
                        autodetect = "strikeouts"
                    elif "positive" in lower or "ev" in lower:
                        autodetect = "positive_ev"
                    elif "home" in lower or "hr" in lower:
                        autodetect = "hr_zone"
                    elif "hit" in lower:
                        autodetect = "hits"
                    elif "sim" in lower or "game" in lower:
                        autodetect = "game_sims"
                    else:
                        autodetect = page_options[0] if page_options else "positive_ev"
                    default_idx = (
                        page_options.index(autodetect) if autodetect in page_options else 0
                    )
                    page_choice = st.selectbox(
                        "Page",
                        options=page_options,
                        index=default_idx,
                        format_func=lambda k: bpp_labels.get(k, k),
                        key=f"bpp_csv_page_{uploader_nonce}_{file_obj.name}",
                    )
                upload_specs.append((file_obj, page_choice))

        process_clicked = st.button(
            "Process uploaded CSVs",
            type="primary",
            disabled=not upload_specs,
            key="bpp_process_csv_uploads",
            use_container_width=False,
        )

        if process_clicked and upload_specs:
            _job_log(
                "button_click",
                name="bpp_process_csv_uploads",
                file_count=len(upload_specs),
            )
            upload_state_placeholder.empty()
            with upload_state_placeholder.container():
                with st.status(
                    f"Processing {len(upload_specs)} CSV upload(s)…",
                    expanded=True, state="running",
                ) as status_box:
                    success_count = 0
                    failures: list[tuple[str, str]] = []
                    upload_results: list[dict[str, Any]] = []
                    for file_obj, page_key in upload_specs:
                        try:
                            file_obj.seek(0)
                        except Exception:  # noqa: BLE001
                            pass
                        raw_bytes = file_obj.read()
                        _job_log(
                            "backend_request_start",
                            job="bpp_csv_upload",
                            page=page_key,
                            filename=file_obj.name,
                            bytes=len(raw_bytes),
                        )
                        try:
                            with httpx.Client(base_url=API_BASE, timeout=60.0) as client:
                                resp = client.post(
                                    "/ballparkpal/upload-csv",
                                    data={"page": page_key, "slate_date": upload_slate_date.strip() or ""},
                                    files={"file": (file_obj.name, raw_bytes, "text/csv")},
                                )
                            if resp.status_code >= 400:
                                err_text = resp.text[:600]
                                _job_log(
                                    "backend_request_end",
                                    job="bpp_csv_upload", ok=False,
                                    status=resp.status_code,
                                    filename=file_obj.name,
                                )
                                failures.append((file_obj.name, err_text))
                                continue
                            payload = resp.json()
                        except (httpx.HTTPError, ValueError) as exc:
                            _job_log(
                                "backend_request_end",
                                job="bpp_csv_upload", ok=False,
                                error=str(exc), filename=file_obj.name,
                            )
                            failures.append((file_obj.name, str(exc)))
                            continue
                        _job_log(
                            "backend_request_end",
                            job="bpp_csv_upload", ok=True,
                            filename=file_obj.name,
                            parsed_rows=payload.get("parsed_row_count"),
                        )
                        success_count += 1
                        upload_results.append(payload)
                    if success_count and not failures:
                        status_box.update(
                            label=f"Processed {success_count} CSV(s) — cache marked fresh.",
                            state="complete",
                        )
                    elif success_count:
                        status_box.update(
                            label=f"Processed {success_count} of {len(upload_specs)} CSV(s) with errors.",
                            state="error",
                        )
                    else:
                        status_box.update(
                            label="No CSVs processed — see errors below.",
                            state="error",
                        )

            # Result previews — raw headers, raw rows, parsed rows, parse
            # warnings, rejection reasons. Expanded by default when
            # something looks wrong (zero parsed rows or generic fallback)
            # so the operator doesn't have to hunt for the diagnostics.
            for payload in upload_results:
                p_label = bpp_labels.get(payload.get("page"), payload.get("page"))
                parsed_rows_n = int(payload.get("parsed_row_count") or 0)
                raw_rows_n = int(payload.get("raw_row_count") or 0)
                used_fallback = bool(payload.get("used_generic_fallback"))
                indicator = "✅"
                if parsed_rows_n == 0:
                    indicator = "❌"
                elif used_fallback or (payload.get("rejection_reasons") or []):
                    indicator = "⚠️"
                with st.expander(
                    f"{indicator} {p_label} · {payload.get('filename')} "
                    f"(raw_rows={raw_rows_n}, parsed_rows={parsed_rows_n})",
                    expanded=(parsed_rows_n == 0 or used_fallback),
                ):
                    # Detection summary
                    det_idx = payload.get("detected_header_row_index")
                    det_score = payload.get("header_detection_score")
                    st.markdown(
                        f"**Header row:** index `{det_idx}` "
                        f"(detection score `{det_score}`) · "
                        f"**Generic fallback:** "
                        f"{'yes' if used_fallback else 'no'}"
                    )
                    raw_headers = payload.get("raw_headers") or []
                    canon_headers = payload.get("canonical_headers") or []
                    st.markdown(
                        f"**Detected raw headers:** `{', '.join(raw_headers) or DASH}`"
                    )
                    st.markdown(
                        f"**Canonical (aliased) headers:** "
                        f"`{', '.join(c for c in canon_headers if c) or DASH}`"
                    )

                    # First 20 RAW rows — what the file actually looks
                    # like before header detection / parsing.
                    raw_preview = payload.get("raw_rows_preview") or []
                    if raw_preview:
                        st.markdown("**First 20 raw CSV rows (before parsing):**")
                        st.dataframe(
                            pd.DataFrame(raw_preview).fillna(DASH),
                            use_container_width=True, hide_index=False,
                            height=min(420, 60 + 24 * len(raw_preview)),
                        )

                    # Warnings + parse errors
                    for warning in payload.get("warnings") or []:
                        st.warning(warning)

                    # Per-row rejection reasons — only shown when at
                    # least one row was rejected by the strict parser
                    rejections = payload.get("rejection_reasons") or []
                    if rejections:
                        st.markdown(
                            f"**Rejected rows: {len(rejections)} of "
                            f"{raw_rows_n}.** Each row below is shown with "
                            "the required field(s) it was missing."
                        )
                        rej_rows = [
                            {
                                "row_index": r.get("row_index"),
                                "missing": ", ".join(r.get("missing") or []),
                                "sample": "; ".join(
                                    f"{k}={v}" for k, v in (r.get("sample") or {}).items()
                                ),
                            }
                            for r in rejections[:50]
                        ]
                        st.dataframe(
                            pd.DataFrame(rej_rows).fillna(DASH),
                            use_container_width=True, hide_index=True,
                            height=min(280, 60 + 28 * len(rej_rows)),
                        )

                    # Parsed-row preview (first 10 canonical rows)
                    rows_preview = payload.get("rows_preview") or []
                    if rows_preview:
                        label = "Preview (first 10 parsed rows)"
                        if used_fallback:
                            label += " — generic fallback (canonical headers as keys)"
                        st.markdown(f"**{label}:**")
                        st.dataframe(
                            pd.DataFrame(rows_preview).fillna(DASH),
                            use_container_width=True, hide_index=True,
                            height=min(280, 60 + 32 * len(rows_preview)),
                        )
                    else:
                        st.error(
                            "Zero rows parsed. Inspect the raw rows + "
                            "rejection reasons above to see why."
                        )
            for filename, err_text in failures:
                with st.expander(f"❌ {filename} — failed", expanded=True):
                    st.code(err_text, language="text")

            if success_count:
                # Bump the uploader nonce so the file_uploader widget
                # resets — otherwise the same files are visible on every
                # rerun and would re-upload if the operator clicked
                # Process again.
                st.session_state["bpp_uploader_nonce"] = uploader_nonce + 1
                st.cache_data.clear()
                _job_log("rerun_trigger", source="bpp_csv_upload_success", success_count=success_count)
                st.rerun()

        # ----- Advanced: Playwright (off by default) ----------------------
        with st.expander(
            "🔧 Advanced · Browser automation (Playwright login + scrape)",
            expanded=False,
        ):
            st.caption(
                "Playwright automation works only on a workstation that can "
                "actually open a headed browser. On Render or any hosted "
                "container, the login flow hangs forever — use the manual "
                "CSV upload above instead."
            )
            ctrl = st.columns([1.2, 1.2, 2, 1.4, 1.4])
            with ctrl[0]:
                launch_login = st.button(
                    "Launch Login Browser",
                    use_container_width=True,
                    disabled=bool(bpp_active_job),
                    help="Local-only. Opens Playwright headed so you can sign in.",
                    key="bpp_advanced_launch_login",
                )
            with ctrl[1]:
                refresh_clicked = st.button(
                    "Refresh BallparkPal Data",
                    use_container_width=True,
                    disabled=bool(bpp_active_job) or not bpp_has_profile,
                    help=(
                        "Run the scrape against cached login session."
                        if bpp_has_profile
                        else "Run login initialization first."
                    ),
                    key="bpp_advanced_refresh",
                )
            with ctrl[2]:
                sel_pages = st.multiselect(
                    "Pages",
                    options=list(bpp_labels.keys()) or list(PAGE_DEFAULT_OPTIONS),
                    default=list(bpp_labels.keys()) or list(PAGE_DEFAULT_OPTIONS),
                    format_func=lambda k: bpp_labels.get(k, k),
                    key="bpp_pages_selector",
                )
            with ctrl[3]:
                sel_date = st.text_input(
                    "Slate date",
                    value="today",
                    key="bpp_date_input",
                    help="YYYY-MM-DD, 'today', or 'yesterday'.",
                )
            with ctrl[4]:
                headless_choice = st.checkbox(
                    "Headless",
                    value=True,
                    key="bpp_headless_choice",
                    help="Uncheck to watch Playwright run.",
                )

            if not bpp_has_profile and not bpp_active_job:
                st.info(
                    "No persisted browser session found. If you're running "
                    "locally, click **Launch Login Browser** to sign in. On "
                    "Render, stick with the CSV upload above."
                )

            if launch_login:
                _job_log("button_click", name="bpp_launch_login")
                try:
                    resp = api_post("/ballparkpal/login", json={})
                    st.session_state["bpp_active_job_id"] = resp.get("job_id")
                    st.success(
                        f"Login browser launched (job {resp.get('job_id')}). "
                        "Complete the sign-in in the opened window, then click 'Finish Login'."
                    )
                except ApiError as exc:
                    render_api_error(exc, prefix="Failed to start login")
                _job_log("rerun_trigger", source="bpp_advanced_launch_login")
                st.rerun()

            if refresh_clicked:
                _job_log("button_click", name="bpp_advanced_refresh")
                try:
                    resp = api_post(
                        "/ballparkpal/refresh",
                        json={
                            "pages": sel_pages,
                            "slate_date": sel_date.strip() or None,
                            "headless": bool(headless_choice),
                        },
                    )
                    st.session_state["bpp_active_job_id"] = resp.get("job_id")
                    st.success(f"Refresh queued (job {resp.get('job_id')}).")
                except ApiError as exc:
                    render_api_error(exc, prefix="Failed to start refresh")
                _job_log("rerun_trigger", source="bpp_advanced_refresh")
                st.rerun()

            # Active job panel + opt-in auto-refresh. Lives inside the
            # Advanced expander since manual CSV upload makes this panel
            # purely informational for most operators.
            if bpp_active_job:
                job = bpp_active_job
                mode = str(job.get("mode") or "").lower()
                job_status = str(job.get("status") or "").lower()
                duration = float(job.get("duration_seconds") or 0)
                # Friendly status labels — never just "RUNNING LOGIN" with
                # no context. Each maps to one of: idle / waiting /
                # processing / complete / failed.
                if job_status not in ACTIVE_JOB_STATUSES:
                    display_state = "complete" if job_status == "success" else "failed"
                elif mode == "login":
                    display_state = "waiting_for_login"
                else:
                    display_state = "processing"
                state_label = BPP_DISPLAY_STATE_LABELS.get(display_state, display_state)
                jc = st.columns([1, 1, 1, 1])
                jc[0].metric("Job", job.get("job_id") or DASH)
                jc[1].metric("State", state_label)
                jc[2].metric("Mode", str(job.get("mode") or "?").upper())
                jc[3].metric("Duration", f"{duration:.1f}s")
                if mode == "login":
                    if st.button(
                        "Finish Login (close browser)",
                        key=f"bpp_signal_{job.get('job_id')}",
                    ):
                        _job_log("button_click", name="bpp_finish_login", job_id=job.get("job_id"))
                        try:
                            api_post(f"/ballparkpal/jobs/{job['job_id']}/signal", json={})
                            st.success("Sent finish signal. Browser should close shortly.")
                        except ApiError as exc:
                            render_api_error(exc, prefix="Signal failed")
                        _job_log("rerun_trigger", source="bpp_finish_login")
                        st.rerun()
                else:
                    st.info(
                        "BallparkPal refresh job is running on the backend. "
                        "Click **Refresh status** to update, or tick "
                        "**Auto-refresh** to poll every 8s."
                    )

                poll_cols = st.columns([1, 1, 4])
                with poll_cols[0]:
                    if st.button(
                        "Refresh status", key="bpp_refresh_status",
                        use_container_width=True,
                    ):
                        _job_log("button_click", name="bpp_refresh_status", job_id=job.get("job_id"))
                        _job_log("rerun_trigger", source="bpp_refresh_status")
                        st.rerun()
                with poll_cols[1]:
                    bpp_auto_refresh = st.checkbox(
                        "Auto-refresh (8s)",
                        value=False,
                        key="bpp_auto_refresh",
                        help=(
                            "Off by default. While checked, this panel polls "
                            "the backend every 8 seconds — the only "
                            "auto-rerun in the dashboard. Uncheck to stop."
                        ),
                    )

                logs_text = job.get("logs") or ""
                if logs_text:
                    with st.expander("Live logs", expanded=True):
                        st.code(logs_text[-4000:], language="text")

                if bpp_auto_refresh:
                    st.caption(
                        f"Auto-refreshing every 8s while bpp job "
                        f"{job.get('job_id')} is active. Uncheck to stop."
                    )
                    import time as _time
                    _time.sleep(8.0)
                    _job_log(
                        "rerun_trigger",
                        source="bpp_auto_refresh",
                        job_id=job.get("job_id"),
                    )
                    st.rerun()
            elif st.session_state.get("bpp_active_job_id"):
                # Job recently finished — drop the cache + clear the marker
                # so subtabs see fresh snapshots.
                try:
                    st.cache_data.clear()
                except Exception:
                    pass
                st.session_state.pop("bpp_active_job_id", None)

        # ----- Operator log panel -----------------------------------------
        with st.expander("Operator logs / recent jobs", expanded=False):
            if not bpp_recent_jobs:
                st.caption("No jobs run yet.")
            else:
                recent_rows = []
                for j in bpp_recent_jobs:
                    recent_rows.append({
                        "job_id": j.get("job_id"),
                        "mode": j.get("mode"),
                        "status": j.get("status"),
                        "started_at": j.get("started_at"),
                        "finished_at": j.get("finished_at"),
                        "duration_s": j.get("duration_seconds"),
                        "pages": ", ".join(j.get("pages") or []) or DASH,
                        "return_code": j.get("return_code"),
                        "error": j.get("error_message") or DASH,
                    })
                st.dataframe(
                    pd.DataFrame(recent_rows).fillna(DASH),
                    use_container_width=True, hide_index=True,
                    height=min(280, 60 + 32 * len(recent_rows)),
                )
                last_logs = (bpp_recent_jobs[0] or {}).get("logs") or ""
                if last_logs:
                    st.markdown("**Last job logs (tail)**")
                    st.code(last_logs[-4000:], language="text")

        st.markdown("#### Cached pages")
        if not bpp_pages:
            render_empty_state(
                "NO BALLPARKPAL CACHE",
                "Upload a CSV per page above, or (locally) run "
                "`python scripts/update_ballparkpal_cache.py --pages "
                "positive_ev,strikeouts,hr_zone,hits,game_sims --date today`.",
            )
        else:
            ov_rows: list[dict[str, Any]] = []
            stale_any = False
            login_required = False
            for key, label in bpp_labels.items():
                page = _bpp_page(key)
                if page.get("status") == "login_required":
                    login_required = True
                if page.get("stale"):
                    stale_any = True
                src = page.get("source") or "—"
                # Show "manual_csv · filename" so the overview makes the
                # source unmistakable. Defensive on missing filename.
                if src == "manual_csv" and page.get("filename"):
                    src_display = f"manual_csv · {page.get('filename')}"
                else:
                    src_display = src
                ov_rows.append(
                    {
                        "page": label,
                        "source": src_display,
                        "status": page.get("status") or "missing",
                        "rows": page.get("row_count") or 0,
                        "slate_date": page.get("slate_date") or DASH,
                        "fetched_at": page.get("fetched_at") or DASH,
                        "last_updated_text": page.get("last_updated_text") or DASH,
                        "stale": "yes" if page.get("stale") else "no",
                        "error": page.get("error_message") or DASH,
                    }
                )
            if login_required:
                st.warning(
                    "BallparkPal session expired in the (advanced) scraper "
                    "path. Use the manual CSV upload above — it never needs "
                    "a session."
                )
            if stale_any:
                st.warning(
                    "Some pages have not been refreshed in the last 24h. "
                    "Upload a fresh CSV or re-run the scraper locally."
                )
            st.dataframe(
                pd.DataFrame(ov_rows).fillna(DASH),
                use_container_width=True, hide_index=True,
                height=min(320, 60 + 32 * len(ov_rows)),
            )

    # ---- Positive EV -----------------------------------------------------
    with bpp_ev_tab:
        ev = _bpp_page("positive_ev")
        rows = ev.get("rows") or []
        st.markdown("### Positive EV")
        st.caption(
            f"{ev.get('row_count') or 0} rows · last updated "
            f"{ev.get('last_updated_text') or DASH}"
        )
        if not rows:
            st.info("No Positive EV rows cached yet.")
        else:
            df_ev = pd.DataFrame(rows)
            with st.expander("Filters", expanded=False):
                col_a, col_b, col_c, col_d = st.columns(4)
                markets = sorted({str(r.get("market") or "") for r in rows if r.get("market")})
                teams = sorted({str(r.get("team") or "") for r in rows if r.get("team")})
                books = sorted({str(r.get("book") or "") for r in rows if r.get("book")})
                sel_market = col_a.multiselect("Market", markets, default=[])
                sel_team = col_b.multiselect("Team", teams, default=[])
                sel_book = col_c.multiselect("Book", books, default=[])
                min_delta = col_d.number_input(
                    "Min BP delta (%)", value=0.0, step=0.5,
                    help="Filter to rows where ballparkpal_delta ≥ this.",
                )

            def _row_passes(r: dict[str, Any]) -> bool:
                if sel_market and r.get("market") not in sel_market:
                    return False
                if sel_team and r.get("team") not in sel_team:
                    return False
                if sel_book and r.get("book") not in sel_book:
                    return False
                bp_delta = r.get("ballparkpal_delta")
                if min_delta and (bp_delta is None or bp_delta < min_delta):
                    return False
                return True

            filtered = [r for r in rows if _row_passes(r)]
            df_ev_filtered = pd.DataFrame(filtered) if filtered else pd.DataFrame(columns=df_ev.columns)
            # Highlight rows where both sportsbook-vs-consensus AND
            # sportsbook-vs-BPP deltas are positive — that's the "double
            # confirmation" the EV page is meant to surface.
            cs_delta = df_ev_filtered.get("consensus_delta")
            bp_delta = df_ev_filtered.get("ballparkpal_delta")
            highlight_n = 0
            if cs_delta is not None and bp_delta is not None:
                try:
                    highlight_n = int(((cs_delta > 0) & (bp_delta > 0)).sum())
                except Exception:
                    highlight_n = 0
            st.caption(
                f"{len(filtered)} of {len(rows)} rows after filters · "
                f"{highlight_n} with both consensus & BPP edge positive."
            )
            if not df_ev_filtered.empty:
                st.dataframe(
                    df_ev_filtered.fillna(DASH),
                    use_container_width=True, hide_index=True,
                    height=min(560, 60 + 32 * len(df_ev_filtered)),
                )

    # ---- Strikeouts ------------------------------------------------------
    with bpp_k_tab:
        k_payload = _bpp_page("strikeouts")
        k_rows = k_payload.get("rows") or []
        st.markdown("### Strikeout Center")
        st.caption(
            f"{k_payload.get('row_count') or 0} pitchers · last updated "
            f"{k_payload.get('last_updated_text') or DASH}"
        )
        if not k_rows:
            st.info("No strikeout rows cached yet.")
        else:
            # Compare against SignalForge pitcher_strikeouts edges from the
            # already-loaded MLB-edges list. We do not refetch.
            sf_k_by_pitcher: dict[str, dict[str, Any]] = {}
            for edge in mlb_edges_all or []:
                if str(edge.get("edge_type") or "") != "pitcher_strikeouts":
                    continue
                market = str(edge.get("market") or "")
                pitcher_name = market.split(" Over ")[0].split(" Under ")[0].strip()
                if pitcher_name:
                    key = "".join(ch for ch in pitcher_name.lower() if ch.isalnum())
                    sf_k_by_pitcher[key] = edge

            display = []
            for row in k_rows:
                pname = row.get("pitcher") or ""
                key = "".join(ch for ch in pname.lower() if ch.isalnum())
                sf_edge = sf_k_by_pitcher.get(key) or {}
                line = sf_edge.get("line")
                k_proj = row.get("projected_k")
                gap = (
                    round(float(k_proj) - float(line), 2)
                    if k_proj is not None and line is not None
                    else None
                )
                display.append({
                    **row,
                    "sf_line": line,
                    "sf_side": sf_edge.get("side"),
                    "k_gap_vs_sf_line": gap,
                })
            st.dataframe(
                pd.DataFrame(display).fillna(DASH),
                use_container_width=True, hide_index=True,
                height=min(560, 60 + 32 * len(display)),
            )

    # ---- Home Run Zone ---------------------------------------------------
    with bpp_hr_tab:
        hr_payload = _bpp_page("hr_zone")
        meta = hr_payload.get("meta") or {}
        st.markdown("### Home Run Zone")
        st.caption(
            f"last updated {hr_payload.get('last_updated_text') or DASH}"
        )
        sub_totals = meta.get("totals") or []
        sub_game = meta.get("by_game") or []
        sub_team = meta.get("by_team") or []
        sub_hitters = meta.get("hitters") or []
        if not any([sub_totals, sub_game, sub_team, sub_hitters]):
            st.info("No HR Zone tables cached yet.")
        if sub_totals:
            st.markdown("**Park totals**")
            st.dataframe(
                pd.DataFrame(sub_totals).fillna(DASH),
                use_container_width=True, hide_index=True,
                height=min(320, 60 + 32 * len(sub_totals)),
            )
        if sub_game:
            st.markdown("**By game**")
            st.dataframe(
                pd.DataFrame(sub_game).fillna(DASH),
                use_container_width=True, hide_index=True,
                height=min(320, 60 + 32 * len(sub_game)),
            )
        if sub_team:
            st.markdown("**By team**")
            st.dataframe(
                pd.DataFrame(sub_team).fillna(DASH),
                use_container_width=True, hide_index=True,
                height=min(320, 60 + 32 * len(sub_team)),
            )
        if sub_hitters:
            st.markdown("**Hitters**")
            st.dataframe(
                pd.DataFrame(sub_hitters).fillna(DASH),
                use_container_width=True, hide_index=True,
                height=min(420, 60 + 32 * len(sub_hitters)),
            )

    # ---- Hits ------------------------------------------------------------
    with bpp_hits_tab:
        hits_payload = _bpp_page("hits")
        hits_rows = hits_payload.get("rows") or []
        st.markdown("### Hits")
        st.caption(
            f"{hits_payload.get('row_count') or 0} batters · last updated "
            f"{hits_payload.get('last_updated_text') or DASH}"
        )
        if not hits_rows:
            st.info("No hits rows cached yet.")
        else:
            st.dataframe(
                pd.DataFrame(hits_rows).fillna(DASH),
                use_container_width=True, hide_index=True,
                height=min(560, 60 + 32 * len(hits_rows)),
            )

    # ---- Game Sims -------------------------------------------------------
    with bpp_sims_tab:
        sims_payload = _bpp_page("game_sims")
        sims_rows = sims_payload.get("rows") or []
        st.markdown("### Game Simulations")
        st.caption(
            f"{sims_payload.get('row_count') or 0} games · last updated "
            f"{sims_payload.get('last_updated_text') or DASH}"
        )
        if not sims_rows:
            st.info("No game-sim rows cached yet.")
        else:
            # Join against SignalForge game_total edges already in memory.
            # We use (home, away) keyed on uppercase team codes; mismatches
            # leave the SF columns blank rather than guessing.
            sf_totals: dict[tuple[str, str], dict[str, Any]] = {}
            for edge in mlb_edges_all or []:
                if str(edge.get("edge_type") or "") != "game_total":
                    continue
                home = str(edge.get("home_team") or "").upper()
                away = str(edge.get("away_team") or "").upper()
                if home and away:
                    sf_totals.setdefault((home, away), edge)
            display = []
            for row in sims_rows:
                home = str(row.get("home_team") or "").upper()
                away = str(row.get("away_team") or "").upper()
                sf_edge = sf_totals.get((home, away)) or {}
                market_total = sf_edge.get("line")
                sf_proj = (
                    sf_edge.get("model_projected_total")
                    or sf_edge.get("projected_total")
                )
                bpp_total = row.get("projected_total")
                display.append({
                    **row,
                    "market_total": market_total,
                    "sf_projected_total": sf_proj,
                    "bpp_minus_market": (
                        round(float(bpp_total) - float(market_total), 2)
                        if bpp_total is not None and market_total is not None
                        else None
                    ),
                    "sf_minus_bpp": (
                        round(float(sf_proj) - float(bpp_total), 2)
                        if sf_proj is not None and bpp_total is not None
                        else None
                    ),
                    "sf_minus_market": (
                        round(float(sf_proj) - float(market_total), 2)
                        if sf_proj is not None and market_total is not None
                        else None
                    ),
                })
            st.dataframe(
                pd.DataFrame(display).fillna(DASH),
                use_container_width=True, hide_index=True,
                height=min(560, 60 + 32 * len(display)),
            )

    # ---- Raw snapshots / debug ------------------------------------------
    with bpp_debug_tab:
        st.markdown("### Raw Snapshots / Debug")
        st.caption(
            "Run `python scripts/update_ballparkpal_cache.py --login` first "
            "if no session has been saved. Then `--pages "
            "positive_ev,strikeouts,hr_zone,hits,game_sims --date today` "
            "writes one snapshot row per page to `ballparkpal_snapshots`."
        )
        debug_rows = []
        for key, label in bpp_labels.items():
            page = _bpp_page(key)
            debug_rows.append({
                "page": label,
                "status": page.get("status") or "missing",
                "rows": page.get("row_count") or 0,
                "slate_date": page.get("slate_date") or DASH,
                "fetched_at": page.get("fetched_at") or DASH,
                "source_url": page.get("source_url") or DASH,
                "raw_html_path": page.get("raw_html_path") or DASH,
                "warnings": ", ".join(page.get("warnings") or []) or DASH,
                "error_message": page.get("error_message") or DASH,
            })
        if debug_rows:
            st.dataframe(
                pd.DataFrame(debug_rows).fillna(DASH),
                use_container_width=True, hide_index=True,
                height=min(360, 60 + 32 * len(debug_rows)),
            )


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
        diag_cols[0].metric("Active provider", providers_diag.get("last_provider_used") or providers_diag.get("provider") or DASH)
        diag_cols[1].metric("Primary events", primary_diag.get("events_fetched", 0))
        diag_cols[2].metric("Backup events", backup_diag.get("events_fetched", 0))

        st.markdown(
            "<div class='sf-card'>"
            + f"<div class='sf-card-row'><span class='k'>Primary:</span>{primary_diag.get('name') or 'Odds-API'}"
            + f" · enabled={bool(primary_diag.get('enabled'))}"
            + f" · api key={'yes' if primary_diag.get('api_key_present', True) else 'no'}"
            + f" · last success={fmt_dt(primary_diag.get('last_success_at'))}"
            + "</div>"
            + f"<div class='sf-card-row'><span class='k'>Totals found:</span>{primary_diag.get('totals_found', 0)}"
            + f" · Pitcher props found:{primary_diag.get('pitcher_props_found', 0)}"
            + "</div>"
            + f"<div class='sf-card-row'><span class='k'>Last error:</span>"
            + f"{primary_diag.get('last_error') or DASH}"
            + f" · at {fmt_dt(primary_diag.get('last_error_at'))}"
            + f" · cooldown until {fmt_dt(primary_diag.get('cooldown_until'))}"
            + f" · strategy {primary_diag.get('last_successful_strategy') or DASH}"
            + "</div>"
            + "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='sf-card'>"
            + f"<div class='sf-card-row'><span class='k'>Backup:</span>{backup_diag.get('name') or 'SportsGameOdds'}"
            + f" · enabled={bool(backup_diag.get('enabled'))}"
            + f" · api key={'yes' if backup_diag.get('api_key_present', True) else 'no'}"
            + f" · last success={fmt_dt(backup_diag.get('last_success_at'))}"
            + "</div>"
            + f"<div class='sf-card-row'><span class='k'>Totals found:</span>{backup_diag.get('totals_found', 0)}"
            + f" · Pitcher props found:{backup_diag.get('pitcher_props_found', 0)}"
            + "</div>"
            + f"<div class='sf-card-row'><span class='k'>Last error:</span>"
            + f"{backup_diag.get('last_error') or DASH}"
            + f" · at {fmt_dt(backup_diag.get('last_error_at'))}"
            + f" · cooldown until {fmt_dt(backup_diag.get('cooldown_until'))}"
            + f" · strategy {backup_diag.get('last_successful_strategy') or DASH}"
            + "</div>"
            + "</div>",
            unsafe_allow_html=True,
        )

        recent_errors = providers_diag.get("last_errors") or []
        if recent_errors:
            st.markdown("**Recent provider errors**")
            st.code("\n".join(str(err) for err in recent_errors[-5:]), language="text")

        if primary_diag.get("plan_limit_warning"):
            st.warning(str(primary_diag.get("plan_limit_warning")))

        if backup_diag.get("cooldown_until"):
            st.warning(f"SportsGameOdds rate-limited until {fmt_dt(backup_diag.get('cooldown_until'))}")

        if odds_cache_status in {"stale", "empty"}:
            st.warning(
                "Odds refresh failed or cache is stale. The edge scan will refuse stale odds unless you opt into a force-stale run."
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
            "No current-card wallet flow found.",
            f"No active wallet positions were found for {selected_card_date}.",
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
    ac1.metric("Today alerts", len(alerts_all))
    ac2.metric("Sent", len(sent))
    ac3.metric("Skipped", len(skipped))
    ac4.metric("Failed", len(failed))

    if not alerts_all:
        render_empty_state(
            "No current-card alerts.",
            f"No alerts have been dispatched for {selected_card_date}.",
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

    historical_only = [
        a for a in historical_alerts
        if (a.get("generated_for_date") or "") != selected_card_date
    ]
    with st.expander("Historical alerts", expanded=False):
        if not historical_only:
            st.caption("No historical alerts found.")
        else:
            rows = []
            for a in historical_only:
                rows.append({
                    "generated_for_date": a.get("generated_for_date") or DASH,
                    "created_at": fmt_dt(a.get("created_at")),
                    "channel": a.get("channel"),
                    "status": a.get("status"),
                    "signal_id": a.get("signal_id"),
                    "message": (a.get("message") or "")[:240],
                })
            st.warning("Historical / stale, not today's card")
            st.dataframe(
                pd.DataFrame(rows).fillna(DASH),
                use_container_width=True,
                hide_index=True,
                height=min(420, 60 + 32 * len(rows)),
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

    # --- Falcon adaptive learning controls ---------------------------------
    with st.expander("Falcon adaptive learning", expanded=False):
        agents_payload = safe_get("/falcon/agents", default={})
        learning_stats = safe_get("/falcon/learning/stats", default={})
        scheduler_status = safe_get("/falcon/learning/scheduler", default={})
        agents = agents_payload.get("agents") or []

        cols = st.columns(4)
        cols[0].metric("Wired agents", len(agents))
        cols[1].metric(
            "Tracked wallets (learned)",
            len(learning_stats.get("wallets") or []),
        )
        cols[2].metric(
            "Adaptive factor rows",
            len(learning_stats.get("adaptive_weights") or []),
        )
        cols[3].metric(
            "Calibration bands",
            len(learning_stats.get("calibration_bands") or []),
        )

        ctrl = st.columns(3)
        with ctrl[0]:
            if st.button("Backfill tracked wallets", use_container_width=True):
                with st.spinner("Calling Wallet 360 for every tracked wallet..."):
                    try:
                        result = api_post(
                            "/falcon/learning/backfill", timeout=MLB_RUN_TIMEOUT,
                        )
                    except ApiError as exc:
                        render_api_error(exc, prefix="Falcon backfill failed")
                        result = None
                if result is not None:
                    st.success(
                        f"Backfilled {result.get('wallets_backfilled', 0)} "
                        f"of {result.get('wallets_seen', 0)} wallets · "
                        f"specialisations={result.get('specialisations_written', 0)} · "
                        f"unavailable={result.get('wallets_unavailable', 0)}"
                    )
                    if result.get("errors"):
                        with st.expander("Backfill errors"):
                            for err in result["errors"]:
                                st.code(err, language="text")
                    st.cache_data.clear()
                    st.rerun()
        with ctrl[1]:
            if st.button("Recompute learning", use_container_width=True):
                with st.spinner("Recomputing weights, bands, tiers..."):
                    try:
                        result = api_post(
                            "/falcon/learning/recompute", timeout=MLB_RUN_TIMEOUT,
                        )
                    except ApiError as exc:
                        render_api_error(exc, prefix="Falcon recompute failed")
                        result = None
                if result is not None:
                    fw = (result.get("factor_weights") or {})
                    cb = (result.get("calibration") or {})
                    tr = (result.get("tiers") or {})
                    st.success(
                        f"Recomputed · weights updated={fw.get('weights_updated', 0)} "
                        f"(below min sample={fw.get('weights_below_min_sample', 0)}) · "
                        f"bands={cb.get('bands_updated', 0)} · "
                        f"tiers written={tr.get('tiers_written', 0)}"
                    )
                    st.cache_data.clear()
                    st.rerun()
        with ctrl[2]:
            if st.button("Trigger scheduler tick", use_container_width=True):
                with st.spinner("Running one scheduler tick..."):
                    try:
                        result = api_post(
                            "/falcon/learning/scheduler/tick", timeout=MLB_RUN_TIMEOUT,
                        )
                    except ApiError as exc:
                        render_api_error(exc, prefix="Scheduler tick failed")
                        result = None
                if result is not None:
                    if result.get("error"):
                        st.warning(f"Scheduler tick error: {result['error']}")
                    else:
                        st.success("Scheduler tick complete.")
                    st.cache_data.clear()

        st.markdown("**Wired Falcon agents**")
        if agents:
            agent_rows = [
                {"name": a["name"], "label": a["label"], "id": a["id"]}
                for a in agents
            ]
            st.dataframe(
                pd.DataFrame(agent_rows),
                use_container_width=True, hide_index=True,
                height=min(360, 60 + 28 * len(agent_rows)),
            )
        else:
            st.caption("Agent registry unavailable.")

        st.markdown("**Adaptive factor weights**")
        weights = learning_stats.get("adaptive_weights") or []
        if weights:
            st.dataframe(
                pd.DataFrame(weights),
                use_container_width=True, hide_index=True,
                height=min(320, 60 + 28 * len(weights)),
            )
        else:
            st.caption("No graded factor attribution yet — weights are using the static prior.")

        st.markdown("**Calibration bands**")
        bands = learning_stats.get("calibration_bands") or []
        if bands:
            st.dataframe(
                pd.DataFrame(bands),
                use_container_width=True, hide_index=True,
                height=min(280, 60 + 28 * len(bands)),
            )
        else:
            st.caption("Not enough graded signals to calibrate yet.")

        st.markdown("**Tracked wallets (learned)**")
        wallets_rows = learning_stats.get("wallets") or []
        if wallets_rows:
            st.dataframe(
                pd.DataFrame(wallets_rows),
                use_container_width=True, hide_index=True,
                height=min(360, 60 + 28 * len(wallets_rows)),
            )
        else:
            st.caption("No wallet stats yet — run backfill or wait for first graded signal.")

        st.markdown("**Scheduler status**")
        st.json(scheduler_status, expanded=False)

        st.markdown("**Per-signal Falcon Intelligence**")
        st.caption(
            "Enter a signal ID to see the full factor breakdown, contributing "
            "wallets, regime snapshot, and calibrated probability."
        )
        explain_id = st.text_input("Signal ID", key="falcon_explain_signal_id")
        if explain_id.strip():
            try:
                sig_id_int = int(explain_id.strip())
            except ValueError:
                st.warning("Signal ID must be an integer.")
            else:
                explain_payload = safe_get(
                    f"/falcon/learning/explain/{sig_id_int}", default=None,
                )
                if not explain_payload:
                    st.info(f"No learning rows for signal #{sig_id_int} (not graded yet, or signal not found).")
                else:
                    head = st.columns(3)
                    head[0].metric("Raw score", fmt_num(explain_payload.get("raw_score"), fmt="{:.1f}"))
                    cal = explain_payload.get("calibrated_probability")
                    head[1].metric(
                        "Calibrated prob",
                        fmt_pct(cal) if cal is not None else "—",
                    )
                    wallets_block = explain_payload.get("wallets") or {}
                    head[2].metric(
                        "Elite disagreement",
                        wallets_block.get("elite_disagreement_count", 0),
                    )

                    factors_block = explain_payload.get("factors") or {}
                    factor_rows = factors_block.get("rows") or []
                    if factor_rows:
                        st.markdown("Factors")
                        flat = []
                        for fr in factor_rows:
                            adaptive = fr.get("adaptive") or {}
                            flat.append({
                                "factor": fr.get("factor_name"),
                                "value": fr.get("value"),
                                "weight": fr.get("weight"),
                                "adaptive_weight": adaptive.get("current_weight"),
                                "predictive_power": adaptive.get("predictive_power"),
                                "sample": adaptive.get("sample_size"),
                            })
                        st.dataframe(pd.DataFrame(flat), use_container_width=True, hide_index=True)

                    wallet_rows = wallets_block.get("rows") or []
                    if wallet_rows:
                        st.markdown("Contributing wallets")
                        st.dataframe(
                            pd.DataFrame([
                                {
                                    "wallet": (w.get("wallet_address") or "")[:16] + "…",
                                    "side": w.get("side"),
                                    "tier": w.get("tier"),
                                    "win_rate": w.get("win_rate"),
                                    "roi": w.get("roi"),
                                    "avg_clv": w.get("avg_clv"),
                                    "sample": w.get("sample_size"),
                                }
                                for w in wallet_rows
                            ]),
                            use_container_width=True, hide_index=True,
                        )

                    regime_block = explain_payload.get("regime") or {}
                    if regime_block.get("available"):
                        st.markdown("Market regime")
                        st.json(regime_block.get("summary") or {}, expanded=False)
                    if explain_payload.get("conflict"):
                        st.markdown("Conflict / contrarian flags")
                        st.json(explain_payload["conflict"], expanded=False)

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

    st.markdown("### Card-Date Debug")
    debug_rows = [
        ("Arizona today", dashboard_debug_payload.get("arizona_today")),
        ("Selected card date", dashboard_debug_payload.get("selected_card_date")),
        ("Latest wallet scan generated_for_date", dashboard_debug_payload.get("latest_wallet_scan_generated_for_date")),
        ("Latest MLB edge generated_for_date", dashboard_debug_payload.get("latest_mlb_edge_generated_for_date")),
        ("Latest alert generated_for_date", dashboard_debug_payload.get("latest_alert_generated_for_date")),
        ("Today wallet positions", dashboard_debug_payload.get("today_wallet_positions")),
        ("Stale wallet positions hidden", dashboard_debug_payload.get("stale_wallet_positions_hidden")),
        ("Today alerts", dashboard_debug_payload.get("today_alerts")),
        ("Stale alerts hidden", dashboard_debug_payload.get("stale_alerts_hidden")),
        (
            "Odds cache age",
            f"{dashboard_debug_payload.get('odds_cache_age_minutes')} min"
            if dashboard_debug_payload.get("odds_cache_age_minutes") is not None
            else DASH,
        ),
    ]
    st.dataframe(
        pd.DataFrame(debug_rows, columns=["field", "value"]).fillna(DASH),
        use_container_width=True,
        hide_index=True,
        height=390,
    )

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
        with st.expander("/dashboard/debug", expanded=False):
            st.json(dashboard_debug_payload)
    else:
        st.caption("Raw JSON hidden. Enable 'Show raw JSON' in the sidebar to expose payloads.")
