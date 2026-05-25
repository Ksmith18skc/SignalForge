"""SignalForge — Streamlit dashboard.

Talks to the FastAPI backend at SIGNALFORGE_API_URL (default http://localhost:8000).
Run with:
    streamlit run dashboard.py
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API_BASE = os.environ.get("SIGNALFORGE_API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = 10.0


# ----------------------------------------------------------------------------
# Page + dark quant theme
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="SignalForge",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --sf-bg: #0b0e14;
        --sf-panel: #11141c;
        --sf-border: #1f2430;
        --sf-text: #d6deeb;
        --sf-muted: #6b7a90;
        --sf-accent: #00d4ff;
        --sf-up: #34d399;
        --sf-down: #f87171;
        --sf-warn: #fbbf24;
      }
      .stApp {
        background-color: var(--sf-bg);
        color: var(--sf-text);
        font-family: 'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace;
      }
      section[data-testid="stSidebar"] {
        background-color: var(--sf-panel);
        border-right: 1px solid var(--sf-border);
      }
      div[data-testid="stMetric"] {
        background-color: var(--sf-panel);
        padding: 1rem 1.25rem;
        border: 1px solid var(--sf-border);
        border-radius: 6px;
      }
      div[data-testid="stMetric"] label {
        color: var(--sf-muted) !important;
        text-transform: uppercase;
        font-size: 0.7rem !important;
        letter-spacing: 0.1em;
      }
      div[data-testid="stMetricValue"] {
        color: var(--sf-text) !important;
        font-weight: 600;
      }
      h1, h2, h3, h4 { color: var(--sf-text); }
      .sf-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }
      .sf-badge-mock {
        background-color: rgba(251, 191, 36, 0.12);
        color: var(--sf-warn);
        border: 1px solid var(--sf-warn);
      }
      .sf-badge-falcon {
        background-color: rgba(52, 211, 153, 0.12);
        color: var(--sf-up);
        border: 1px solid var(--sf-up);
      }
      .sf-badge-partial {
        background-color: rgba(0, 212, 255, 0.12);
        color: var(--sf-accent);
        border: 1px solid var(--sf-accent);
      }
      .sf-badge-offline {
        background-color: rgba(248, 113, 113, 0.12);
        color: var(--sf-down);
        border: 1px solid var(--sf-down);
      }
      .sf-meta {
        color: var(--sf-muted);
        font-size: 0.8rem;
        margin-top: 0.4rem;
      }
      div[data-testid="stDataFrame"] {
        border: 1px solid var(--sf-border);
        border-radius: 4px;
      }
      .stButton>button[kind="primary"] {
        background-color: var(--sf-accent);
        color: var(--sf-bg);
        border: none;
        font-weight: 700;
        letter-spacing: 0.05em;
      }
      .stButton>button[kind="primary"]:hover {
        background-color: #00b8e0;
        color: var(--sf-bg);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# API client — each fetch is cached for 10s to keep the UI snappy
# ----------------------------------------------------------------------------


def _client() -> httpx.Client:
    return httpx.Client(base_url=API_BASE, timeout=REQUEST_TIMEOUT)


@st.cache_data(ttl=10, show_spinner=False)
def fetch_health() -> dict[str, Any] | None:
    try:
        with _client() as c:
            r = c.get("/health")
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError:
        return None


@st.cache_data(ttl=10, show_spinner=False)
def fetch_summary() -> dict[str, Any] | None:
    try:
        with _client() as c:
            r = c.get("/dashboard-summary")
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError:
        return None


@st.cache_data(ttl=10, show_spinner=False)
def fetch_traders() -> list[dict[str, Any]]:
    try:
        with _client() as c:
            r = c.get("/traders")
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError:
        return []


@st.cache_data(ttl=10, show_spinner=False)
def fetch_signals(limit: int = 500) -> list[dict[str, Any]]:
    try:
        with _client() as c:
            r = c.get("/signals", params={"limit": limit})
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError:
        return []


@st.cache_data(ttl=10, show_spinner=False)
def fetch_alerts(limit: int = 200) -> list[dict[str, Any]]:
    try:
        with _client() as c:
            r = c.get("/alerts", params={"limit": limit})
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError:
        return []


def create_trader(payload: dict[str, Any]) -> dict[str, Any]:
    with _client() as c:
        r = c.post("/traders", json=payload)
        r.raise_for_status()
        return r.json()


def delete_trader(trader_id: int) -> None:
    with _client() as c:
        r = c.delete(f"/traders/{trader_id}")
        r.raise_for_status()


def trigger_scan() -> dict[str, Any]:
    with _client() as c:
        r = c.post("/run-scan", timeout=60.0)
        r.raise_for_status()
        return r.json()


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _truncate_wallet(w: str | None, n: int = 14) -> str:
    if not w:
        return "—"
    return w if len(w) <= n else f"{w[:n]}…"


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


def _format_line_token(token: str | None) -> str:
    if not token:
        return ""
    return token.replace("pt", ".")


def _parse_market_parts(sig: dict[str, Any]) -> dict[str, str]:
    slug = _market_slug(sig)
    parts = slug.split("-") if slug else []
    date_idx = next(
        (
            i
            for i in range(max(len(parts) - 2, 0))
            if re.fullmatch(r"20\d{2}", parts[i] or "")
            and re.fullmatch(r"\d{2}", parts[i + 1] or "")
            and re.fullmatch(r"\d{2}", parts[i + 2] or "")
        ),
        None,
    )
    if date_idx is None or date_idx < 2:
        return {
            "league": "",
            "matchup": _market_label(sig),
            "contract": "",
            "event_date": "",
        }

    league = parts[0].upper()
    teams = [team.upper() for team in parts[1:date_idx]]
    matchup = " vs ".join(teams) if teams else _market_label(sig)
    event_date = "-".join(parts[date_idx:date_idx + 3])
    contract_parts = parts[date_idx + 3:]

    contract = "Moneyline"
    position_hint = ""
    if contract_parts[:1] == ["total"]:
        contract = f"Total {_format_line_token(contract_parts[1] if len(contract_parts) > 1 else '')}".strip()
    elif contract_parts[:1] == ["spread"]:
        position_hint = contract_parts[1].title() if len(contract_parts) > 1 else ""
        line = _format_line_token(contract_parts[2] if len(contract_parts) > 2 else "")
        contract = f"Spread {line}".strip()

    return {
        "league": league,
        "matchup": matchup,
        "contract": contract,
        "event_date": event_date,
        "position_hint": position_hint,
    }


def _position_label(sig: dict[str, Any]) -> str:
    outcome = sig.get("outcome")
    if outcome:
        return str(outcome)
    hint = sig.get("position_hint")
    if hint:
        return str(hint)
    contract = str(sig.get("contract") or "")
    if contract.startswith("Total"):
        return "Missing outcome"
    if contract == "Moneyline":
        return "Missing outcome"
    return "Missing outcome"


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def aggregate_signals_to_positions(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse repeated signal events into one row per trader/market/side.

    The scanner emits signal events on every pass. For dashboard use, a
    position-like view is more useful: de-dupe identical events, then aggregate
    all unique entries for the same trader + market + side.
    """
    latest_by_event: dict[tuple[Any, ...], dict[str, Any]] = {}
    for s in signals:
        event_key = (
            s.get("source"),
            s.get("trader_id"),
            s.get("wallet"),
            s.get("trader_nickname"),
            s.get("market_id"),
            _market_label(s),
            s.get("side"),
            s.get("outcome"),
            s.get("signal_type"),
            round(_as_float(s.get("entry_price")) or 0.0, 8),
            round(_as_float(s.get("size_usd")) or 0.0, 2),
            s.get("reason"),
        )
        existing = latest_by_event.get(event_key)
        if existing is None or str(s.get("created_at") or "") > str(existing.get("created_at") or ""):
            latest_by_event[event_key] = s

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for s in latest_by_event.values():
        position_key = (
            s.get("source"),
            s.get("trader_id"),
            s.get("wallet"),
            s.get("trader_nickname"),
            s.get("market_id"),
            _market_label(s),
            s.get("side"),
            s.get("outcome"),
        )
        groups.setdefault(position_key, []).append(s)

    positions: list[dict[str, Any]] = []
    for grouped in groups.values():
        first = grouped[0]
        unique_fills: dict[tuple[float, float], tuple[float | None, float]] = {}
        for s in grouped:
            entry = _as_float(s.get("entry_price"))
            size = _as_float(s.get("size_usd")) or 0.0
            fill_key = (round(entry or 0.0, 8), round(size, 2))
            unique_fills.setdefault(fill_key, (entry, size))

        entries = [entry for entry, _size in unique_fills.values()]
        sizes = [size for _entry, size in unique_fills.values()]
        total_size = sum(sizes)
        weighted_entry_num = sum(
            (entry or 0.0) * size
            for entry, size in zip(entries, sizes, strict=False)
            if entry is not None and size > 0
        )
        weighted_entry_den = sum(
            size
            for entry, size in zip(entries, sizes, strict=False)
            if entry is not None and size > 0
        )
        fallback_entries = [entry for entry in entries if entry is not None]
        avg_entry = (
            weighted_entry_num / weighted_entry_den
            if weighted_entry_den
            else (sum(fallback_entries) / len(fallback_entries) if fallback_entries else None)
        )
        signal_types = sorted({str(s.get("signal_type")) for s in grouped if s.get("signal_type")})
        scores = [_as_float(s.get("score")) or 0.0 for s in grouped]
        latest = max(grouped, key=lambda s: str(s.get("created_at") or ""))

        position = dict(latest)
        position["created_at"] = latest.get("created_at")
        position["score"] = max(scores) if scores else 0.0
        position["confidence"] = max((_as_float(s.get("confidence")) or 0.0 for s in grouped), default=0.0)
        position["signal_type"] = " + ".join(signal_types)
        position["entry_price"] = avg_entry
        position["size_usd"] = total_size
        position["signal_count"] = len(grouped)
        position["market_url"] = _market_url(first)
        position["trader_url"] = _trader_url(first)
        position.update(_parse_market_parts(first))
        outcomes = sorted({str(s.get("outcome")) for s in grouped if s.get("outcome")})
        position["outcome"] = " + ".join(outcomes) if outcomes else first.get("outcome")
        position["position"] = _position_label(position)
        position["reason"] = (
            f"Aggregated {len(grouped)} unique signal event"
            f"{'' if len(grouped) == 1 else 's'} for {_market_label(first)} "
            f"({first.get('side') or '-'})"
        )
        positions.append(position)

    return sorted(
        positions,
        key=lambda s: (s.get("score") or 0.0, str(s.get("created_at") or "")),
        reverse=True,
    )


# ----------------------------------------------------------------------------
# Bail early if backend offline
# ----------------------------------------------------------------------------

health = fetch_health()

if health is None:
    st.title("SignalForge")
    st.markdown(
        '<span class="sf-badge sf-badge-offline">Backend offline</span>',
        unsafe_allow_html=True,
    )
    st.error(
        f"Cannot reach the SignalForge backend at **{API_BASE}**.\n\n"
        "Start it in another terminal with:\n\n"
        "```bash\n"
        "uvicorn app.main:app --reload\n"
        "```"
    )
    st.stop()

# Pre-fetch everything once so each tab/section is fast.
summary = fetch_summary() or {}
traders = fetch_traders()
signals_all = fetch_signals(limit=500)
positions_all = aggregate_signals_to_positions(signals_all)
alerts_all = fetch_alerts(limit=200)


# ----------------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Filters")

    score_min, score_max = st.slider(
        "Signal score", min_value=0, max_value=100, value=(0, 100), step=5
    )

    trader_options = ["(all)"] + sorted({t["nickname"] for t in traders})
    selected_trader = st.selectbox("Trader", trader_options, index=0)

    market_options = ["(all)"] + sorted({_market_label(s) for s in positions_all})
    selected_market = st.selectbox("Market", market_options, index=0)

    source_options = ["(all)", "Falcon", "PolymarketAnalytics", "Polycopy", "Mock"]
    selected_source = st.selectbox("Source", source_options, index=0)

    st.markdown("---")
    st.markdown(f"<div class='sf-meta'>API: {API_BASE}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='sf-meta'>Refreshed {datetime.now(timezone.utc).isoformat(timespec='seconds')}</div>",
        unsafe_allow_html=True,
    )


def apply_filters(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for s in signals:
        score = s.get("score") or 0.0
        if not (score_min <= score <= score_max):
            continue
        if selected_trader != "(all)" and s.get("trader_nickname") != selected_trader:
            continue
        if selected_market != "(all)" and _market_label(s) != selected_market:
            continue
        if selected_source != "(all)" and s.get("source") != selected_source:
            continue
        out.append(s)
    return out


filtered_positions = apply_filters(positions_all)


# ----------------------------------------------------------------------------
# Header — title, source badge, copy-mode meta, scan button
# ----------------------------------------------------------------------------

st.title("SignalForge")
st.caption("Prediction-market intelligence — alert-only MVP. Not financial advice.")

hcol1, hcol2, hcol3 = st.columns([2, 3, 1])

falcon_info = health.get("providers", {}).get("falcon", {}) or {}
# `falcon_info` may be a bool in older deployments; normalize.
if isinstance(falcon_info, bool):
    falcon_info = {"configured": falcon_info, "healthy": False, "calls": 0, "successes": 0}
falcon_configured = bool(falcon_info.get("configured"))
falcon_calls = int(falcon_info.get("last_scan_calls") or falcon_info.get("calls") or 0)
falcon_ok = int(falcon_info.get("last_scan_successes") or falcon_info.get("successes") or 0)

# Three-state badge based on the last completed scan:
#   Falcon          — every wallet/trade call succeeded
#   Partial Falcon  — some succeeded, some fell back to mock
#   Mock            — no key, or no successes (or no scan yet)
with hcol1:
    if not falcon_configured:
        badge_html = (
            '<span class="sf-badge sf-badge-mock">Source: Mock · no Falcon key</span>'
        )
    elif falcon_calls == 0:
        badge_html = (
            '<span class="sf-badge sf-badge-mock">'
            'Source: Mock · Falcon configured, awaiting first scan</span>'
        )
    elif falcon_ok == 0:
        badge_html = (
            f'<span class="sf-badge sf-badge-mock">'
            f'Source: Mock · Falcon failing (0/{falcon_calls})</span>'
        )
    elif falcon_ok == falcon_calls:
        badge_html = (
            f'<span class="sf-badge sf-badge-falcon">'
            f'Source: Falcon · {falcon_ok}/{falcon_calls} ok</span>'
        )
    else:
        badge_html = (
            f'<span class="sf-badge sf-badge-partial">'
            f'Source: Partial Falcon · {falcon_ok}/{falcon_calls} ok</span>'
        )
    st.markdown(badge_html, unsafe_allow_html=True)

with hcol2:
    provs = health.get("providers", {})
    configured = [
        name for name, info in provs.items()
        if (isinstance(info, dict) and info.get("configured")) or info is True
    ] or ["—"]
    st.markdown(
        f"<div class='sf-meta'>"
        f"env={health['environment']} · copy_mode={health['default_copy_mode']} "
        f"· auto_trade={health['auto_trading_enabled']} "
        f"· configured=[{', '.join(configured)}]"
        f"</div>",
        unsafe_allow_html=True,
    )
    if falcon_configured and falcon_calls > 0 and falcon_ok < falcon_calls:
        last_err = falcon_info.get("last_error") or ""
        last_code = falcon_info.get("last_status_code")
        st.markdown(
            f"<div class='sf-meta'>"
            f"Falcon last call: HTTP {last_code} · "
            f"endpoint <code>{falcon_info.get('last_endpoint') or '—'}</code> · "
            f"err: <code>{(last_err[:80] + '…') if len(last_err) > 80 else last_err}</code>"
            f"</div>",
            unsafe_allow_html=True,
        )

with hcol3:
    if st.button("Run scan now", width="stretch", type="primary"):
        with st.spinner("Scanning..."):
            try:
                result = trigger_scan()
                st.toast(
                    f"Scan complete: {result['new_signals']} signals, "
                    f"{result['new_alerts']} alerts, "
                    f"{result['duration_seconds']:.2f}s"
                )
                st.cache_data.clear()
                st.rerun()
            except httpx.HTTPError as exc:
                st.error(f"Scan failed: {exc}")

st.divider()


# ----------------------------------------------------------------------------
# Metric cards
# ----------------------------------------------------------------------------

health_meta = summary.get("watchlist_health", {})
sent_alerts = sum(1 for a in alerts_all if a.get("status") == "sent")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Active positions", len(positions_all))
m2.metric("Alerts dispatched", f"{sent_alerts} / {len(alerts_all)}")
m3.metric("Watched traders", len(traders))
m4.metric("Simulated PnL (USD)", f"${summary.get('simulated_pnl_usd', 0.0):,.2f}")

st.divider()


# ----------------------------------------------------------------------------
# Highest conviction
# ----------------------------------------------------------------------------

st.subheader("Highest conviction positions")

top_signals = sorted(filtered_positions, key=lambda s: s.get("score") or 0, reverse=True)[:10]
if not top_signals:
    st.info("No signals match the current filters.")
else:
    df_top = pd.DataFrame(
        [
            {
                "score": s.get("score") or 0.0,
                "confidence": s.get("confidence") or 0.0,
                "type": s.get("signal_type"),
                "events": s.get("signal_count") or 1,
                "source": s.get("source"),
                "trader": s.get("trader_url"),
                "name": s.get("trader_nickname") or "-",
                "league": s.get("league") or "-",
                "matchup": s.get("matchup") or _market_label(s),
                "contract": s.get("contract") or "-",
                "position": s.get("position") or _position_label(s),
                "event_date": s.get("event_date") or "-",
                "market": s.get("market_url"),
                "action": s.get("side") or "-",
                "avg_entry": s.get("entry_price"),
                "position_usd": s.get("size_usd"),
                "reason": s.get("reason"),
            }
            for s in top_signals
        ]
    )
    st.dataframe(
        df_top,
        width="stretch",
        hide_index=True,
        column_config={
            "score": st.column_config.ProgressColumn(
                "score", min_value=0, max_value=100, format="%.1f"
            ),
            "confidence": st.column_config.ProgressColumn(
                "confidence", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "trader": st.column_config.LinkColumn("trader", display_text="profile"),
            "market": st.column_config.LinkColumn("market", display_text="open"),
            "avg_entry": st.column_config.NumberColumn("avg entry", format="%.3f"),
            "position_usd": st.column_config.NumberColumn("position USD", format="$%.0f"),
        },
    )

st.divider()


# ----------------------------------------------------------------------------
# Tabs: signals / traders / alerts / health
# ----------------------------------------------------------------------------

tab_signals, tab_traders, tab_alerts, tab_health = st.tabs(
    ["Positions", "Watched Traders", "Alerts", "Health"]
)

with tab_signals:
    st.markdown(f"**{len(filtered_positions)}** positions match current filters")
    if filtered_positions:
        df_sig = pd.DataFrame(
            [
                {
                    "created_at": s.get("created_at"),
                    "score": s.get("score") or 0.0,
                    "confidence": s.get("confidence") or 0.0,
                    "type": s.get("signal_type"),
                    "events": s.get("signal_count") or 1,
                    "source": s.get("source"),
                    "trader": s.get("trader_url"),
                    "name": s.get("trader_nickname") or "-",
                    "wallet": _truncate_wallet(s.get("wallet")),
                    "league": s.get("league") or "-",
                    "matchup": s.get("matchup") or _market_label(s),
                    "contract": s.get("contract") or "-",
                    "position": s.get("position") or _position_label(s),
                    "event_date": s.get("event_date") or "-",
                    "market": s.get("market_url"),
                    "action": s.get("side") or "-",
                    "avg_entry": s.get("entry_price"),
                    "position_usd": s.get("size_usd"),
                    "reason": s.get("reason"),
                }
                for s in filtered_positions
            ]
        )
        st.dataframe(
            df_sig,
            width="stretch",
            hide_index=True,
            column_config={
                "score": st.column_config.ProgressColumn(
                    "score", min_value=0, max_value=100, format="%.1f"
                ),
                "confidence": st.column_config.ProgressColumn(
                    "confidence", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "trader": st.column_config.LinkColumn("trader", display_text="profile"),
                "market": st.column_config.LinkColumn("market", display_text="open"),
                "avg_entry": st.column_config.NumberColumn("avg entry", format="%.3f"),
                "position_usd": st.column_config.NumberColumn("position USD", format="$%.0f"),
            },
        )
    else:
        st.info("No signals match. Try widening the score range or clearing filters.")

with tab_traders:
    st.markdown("#### Add wallet")
    with st.form("add_wallet_form", clear_on_submit=True):
        form_cols = st.columns([2, 2, 1])
        wallet_address = form_cols[0].text_input(
            "Wallet address",
            placeholder="0x...",
        ).strip()
        nickname = form_cols[1].text_input(
            "Nickname",
            placeholder="optional",
        ).strip()
        trust_score = form_cols[2].number_input(
            "Trust",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
        )

        tags_raw = st.text_input("Tags", placeholder="sports, macro, sharp")
        notes = st.text_area("Notes", height=80, placeholder="Why this wallet is worth tracking")
        scan_after_add = st.checkbox("Run scan after adding", value=True)

        submitted = st.form_submit_button("Add wallet", type="primary")
        if submitted:
            if not wallet_address.startswith("0x") or len(wallet_address) != 42:
                st.error("Enter a valid 0x wallet address.")
            else:
                tags = [tag.strip() for tag in tags_raw.split(",") if tag.strip()]
                payload = {
                    "nickname": nickname or wallet_address[:10],
                    "wallet_address": wallet_address,
                    "platform": "polymarket",
                    "trust_score": float(trust_score),
                    "tags": tags,
                    "notes": notes.strip() or None,
                    "copy_enabled": False,
                    "copy_mode": "alert_only",
                }
                try:
                    created = create_trader(payload)
                    if scan_after_add:
                        result = trigger_scan()
                        st.success(
                            f"Added {created['nickname']} and scanned: "
                            f"{result['new_signals']} new signals."
                        )
                    else:
                        st.success(f"Added {created['nickname']}.")
                    st.cache_data.clear()
                    st.rerun()
                except httpx.HTTPStatusError as exc:
                    detail = exc.response.text
                    if exc.response.status_code == 409:
                        st.error("That nickname is already in the watchlist.")
                    else:
                        st.error(f"Could not add wallet: {detail}")
                except httpx.HTTPError as exc:
                    st.error(f"Could not reach API: {exc}")

    st.divider()

    st.markdown("#### Remove wallets")
    if not traders:
        st.info("No wallets are currently tracked.")
    else:
        trader_by_label = {
            f"{t['nickname']} · {_truncate_wallet(t.get('wallet_address'))}": t
            for t in traders
        }
        remove_labels = st.multiselect(
            "Tracked wallets",
            options=list(trader_by_label),
            placeholder="Select wallets to remove",
        )
        confirm_remove = st.checkbox("Delete selected wallets and their signals")
        if st.button(
            "Remove selected",
            disabled=not remove_labels or not confirm_remove,
        ):
            try:
                for label in remove_labels:
                    delete_trader(int(trader_by_label[label]["id"]))
                st.success(f"Removed {len(remove_labels)} wallet(s).")
                st.cache_data.clear()
                st.rerun()
            except httpx.HTTPError as exc:
                st.error(f"Could not remove wallet: {exc}")

    st.divider()

    if not traders:
        st.warning("No traders seeded. Run `python -m scripts.seed` first.")
    else:
        df_t = pd.DataFrame(
            [
                {
                    "nickname": t["nickname"],
                    "wallet": _truncate_wallet(t.get("wallet_address")),
                    "platform": t.get("platform"),
                    "trust": t.get("trust_score") or 0.0,
                    "rank": t.get("trader_rank"),
                    "win_rate": t.get("win_rate"),
                    "total_pnl": t.get("total_pnl"),
                    "7d_return": t.get("seven_day_return"),
                    "positions": t.get("total_positions"),
                    "copy_mode": t.get("copy_mode"),
                    "tags": ", ".join(t.get("tags") or []),
                }
                for t in traders
            ]
        )
        st.dataframe(
            df_t,
            width="stretch",
            hide_index=True,
            column_config={
                "trust": st.column_config.ProgressColumn(
                    "trust", min_value=0, max_value=100, format="%.0f"
                ),
                "win_rate": st.column_config.NumberColumn("win rate", format="%.2f"),
                "total_pnl": st.column_config.NumberColumn("total PnL", format="$%.0f"),
                "7d_return": st.column_config.NumberColumn("7d return", format="%.2f"),
            },
        )

with tab_alerts:
    if not alerts_all:
        st.info("No alerts dispatched yet.")
    else:
        df_a = pd.DataFrame(
            [
                {
                    "created_at": a.get("created_at"),
                    "channel": a.get("channel"),
                    "status": a.get("status"),
                    "signal_id": a.get("signal_id"),
                    "message": a.get("message"),
                    "error": a.get("error") or "",
                }
                for a in alerts_all
            ]
        )
        st.dataframe(df_a, width="stretch", hide_index=True)

with tab_health:
    st.markdown("#### System")
    st.json(health)
    st.markdown("#### Watchlist")
    st.json(health_meta)
