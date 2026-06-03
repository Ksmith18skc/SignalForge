"""SignalForge — tracked-wallet consensus dashboard.

A clean, focused Streamlit front-end for the SignalForge FastAPI backend. The
single job of this page: surface markets where 2+ tracked wallets are aligned on
the same side, across ANY market — pure wallet-to-wallet consensus.

No backend logic lives here. Every fact on the page is fetched from a
SignalForge endpoint.

Backend URL: $SIGNALFORGE_API_URL (default http://localhost:8000).
Launch: `streamlit run dashboard.py`.
"""

from __future__ import annotations

import os
import re
from datetime import date, datetime, timezone

import httpx
import pandas as pd
import streamlit as st

from app.services import wallet_market_resolver as wmr
from app.services.card_date import arizona_today
from app.utils.dashboard_format import (
    compact_time_ago,
    format_money_short,
    short_addr,
    wallet_consensus_groups,
)

API_BASE = os.environ.get("SIGNALFORGE_API_URL", "http://localhost:8000").rstrip("/")
DEFAULT_TIMEOUT = 20.0

st.set_page_config(
    page_title="SignalForge — Wallet Consensus",
    page_icon="🛰️",
    layout="wide",
)

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #

st.markdown(
    """
    <style>
      .stApp { background: #0b0e14; }
      .sf-title { font-size: 1.6rem; font-weight: 700; color: #e6edf3; letter-spacing: -0.01em; }
      .sf-sub { color: #8b949e; font-size: 0.85rem; margin-top: -4px; }
      .sf-badge { display:inline-block; padding:3px 10px; border-radius:999px;
                  font-size:0.75rem; font-weight:600; }
      .sf-ok { background:#0f2e1d; color:#3fb950; border:1px solid #1f6f43; }
      .sf-warn { background:#3a2a0f; color:#e3b341; border:1px solid #7a5c1e; }
      .sf-card { background:#11161f; border:1px solid #1f2733; border-radius:12px;
                 padding:16px 18px; margin-bottom:14px; }
      .sf-card-head { display:flex; justify-content:space-between; align-items:baseline; }
      .sf-mkt { color:#e6edf3; font-weight:650; font-size:1.02rem; }
      .sf-side { color:#58a6ff; font-weight:600; font-size:0.85rem; }
      .sf-count { background:#13243b; color:#58a6ff; border:1px solid #1f4068;
                  border-radius:999px; padding:2px 10px; font-size:0.78rem; font-weight:700; }
      .sf-wallet { color:#adbac7; font-size:0.86rem; padding:2px 0; }
      .sf-wname { color:#e6edf3; font-weight:600; }
      .sf-dim { color:#6e7681; font-size:0.8rem; }
      a.sf-link { color:#58a6ff; text-decoration:none; font-size:0.82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=30, show_spinner=False)
def api_get(path: str, params: dict | None = None) -> object:
    with httpx.Client(base_url=API_BASE, timeout=DEFAULT_TIMEOUT) as c:
        r = c.get(path, params=params)
        r.raise_for_status()
        return r.json()


def api_post(path: str, params: dict | None = None) -> object:
    with httpx.Client(base_url=API_BASE, timeout=180.0) as c:
        r = c.post(path, params=params)
        r.raise_for_status()
        return r.json()


def safe_get(path: str, params: dict | None = None, default=None):
    try:
        return api_get(path, params=params)
    except Exception as exc:  # noqa: BLE001
        st.session_state["_last_api_error"] = f"{path}: {exc}"
        return default


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

ready = safe_get("/ready", default={}) or {}
falcon = ((ready.get("providers") or {}).get("falcon")) or {}
source_live = bool(falcon.get("configured") and falcon.get("healthy"))
if falcon.get("configured"):
    badge = (
        f"<span class='sf-badge sf-ok'>Falcon · {falcon.get('last_scan_successes', 0)}"
        f"/{falcon.get('last_scan_calls', 0)} ok</span>"
        if source_live
        else "<span class='sf-badge sf-warn'>Falcon configured · calls failing</span>"
    )
else:
    badge = "<span class='sf-badge sf-warn'>Mock — no Falcon key</span>"

scan = safe_get("/run-scan/status", default={}) or {}
last_scan = scan.get("last_finished_at") or scan.get("last_started_at")
last_scan_label = compact_time_ago(last_scan) if last_scan else "—"

c1, c2, c3 = st.columns([6, 2, 2])
with c1:
    st.markdown("<div class='sf-title'>🛰️ SignalForge — Wallet Consensus</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='sf-sub'>Markets where multiple tracked wallets agree · "
        f"Source: {badge} · Last scan: {last_scan_label}</div>",
        unsafe_allow_html=True,
    )
with c2:
    if st.button("🔄 Refresh", use_container_width=True):
        api_get.clear()
        st.rerun()
with c3:
    if st.button("▶ Run scan now", use_container_width=True, type="primary"):
        try:
            api_post("/run-scan")
            st.toast("Scan triggered — refresh in a few seconds.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Scan failed to start: {exc}")

st.divider()


# --------------------------------------------------------------------------- #
# Filters (sidebar) + Data
# --------------------------------------------------------------------------- #


def _opts(rows: list[dict], key: str) -> list[str]:
    return sorted({str(r.get(key)) for r in rows if r.get(key)})


try:
    _today = date.fromisoformat(arizona_today())
except Exception:  # noqa: BLE001
    _today = datetime.now(timezone.utc).date()

st.sidebar.header("Filters")

# A search tag like 'mlb-det-tb-2026-06-02' carries its own date — honor it so
# searching another slate "just works" without first changing the date picker.
search = st.sidebar.text_input(
    "Search event / wallet",
    placeholder="e.g. mlb-det-tb-2026-06-02 or HomeRunHazard",
    help="Substring match on market slug, title, normalized key, or wallet name.",
    key="f_search",
).strip()
_date_in_search = re.search(r"\d{4}-\d{2}-\d{2}", search)
search_date = _date_in_search.group(0) if _date_in_search else None

picked_date = st.sidebar.date_input("Card date", value=_today, key="f_date")
effective_date = search_date or picked_date.isoformat()
if search_date and search_date != picked_date.isoformat():
    st.sidebar.caption(f"📌 Using date **{search_date}** from your search tag.")

positions = safe_get("/tracked-wallet-positions", params={"date": effective_date}, default=[]) or []

sport_sel = st.sidebar.multiselect("Sport", _opts(positions, "sport"), key="f_sport")
type_sel = st.sidebar.multiselect(
    "Market type",
    _opts(positions, "market_subtype") or _opts(positions, "market_category"),
    key="f_type",
)
platform_sel = st.sidebar.multiselect("Platform", _opts(positions, "market_platform"), key="f_platform")
min_wallets = st.sidebar.slider("Min wallets aligned", 2, 8, 2, key="f_minw")
min_size = st.sidebar.number_input("Min total aligned size ($)", min_value=0, value=0, step=500, key="f_minsize")


def _passes(pos: dict) -> bool:
    if sport_sel and str(pos.get("sport")) not in sport_sel:
        return False
    mtype = pos.get("market_subtype") or pos.get("market_category")
    if type_sel and str(mtype) not in type_sel:
        return False
    if platform_sel and str(pos.get("market_platform")) not in platform_sel:
        return False
    if search:
        hay = " ".join(
            str(pos.get(k) or "")
            for k in (
                "market_slug", "market_title", "normalized_market_key",
                "matchup_date_key", "sport_date_key", "wallet_nickname",
            )
        ).lower()
        if search.lower() not in hay:
            return False
    return True


filtered = [p for p in positions if _passes(p)]
consensus = wallet_consensus_groups(filtered, min_wallets=min_wallets)
if min_size:
    consensus = [g for g in consensus if (g.get("consensus_total_size") or 0.0) >= min_size]

active_filters = bool(sport_sel or type_sel or platform_sel or search or min_size or min_wallets != 2)
if st.sidebar.button("Reset filters", use_container_width=True):
    for _k in ("f_search", "f_date", "f_sport", "f_type", "f_platform", "f_minw", "f_minsize"):
        st.session_state.pop(_k, None)
    st.rerun()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Tracked positions", len(filtered), delta=f"of {len(positions)}" if active_filters else None)
m2.metric("Aligned consensus markets", len(consensus))
m3.metric("Wallets in consensus", sum(g.get("consensus_wallets", 0) for g in consensus))
m4.metric(
    "Consensus size",
    format_money_short(sum(g.get("consensus_total_size", 0.0) for g in consensus)),
)
st.caption(f"Showing card date **{effective_date}**" + ("  ·  filters active" if active_filters else ""))

tab_consensus, tab_positions, tab_watchlist, tab_alerts, tab_diag = st.tabs(
    ["🤝 Aligned Consensus", "📋 All Positions", "👛 Watchlist", "🔔 Alerts", "🔎 Diagnostics"]
)


# --------------------------------------------------------------------------- #
# Aligned Consensus (primary)
# --------------------------------------------------------------------------- #


def _market_url(pos: dict) -> str | None:
    slug = pos.get("market_slug")
    if not slug:
        return None
    try:
        return wmr.market_url_for(slug, pos.get("market_platform"))
    except Exception:  # noqa: BLE001
        return None


def _aggregate_wallets(members: list[dict]) -> list[dict]:
    """Collapse per-fill rows into one row per wallet.

    A wallet usually fills a single position across many small trades; without
    this the roster repeats the same wallet dozens of times. We sum size and
    compute a size-weighted average entry so each wallet shows one true line.
    """
    by_wallet: dict[str, dict] = {}
    for m in members:
        name = (
            m.get("wallet_nickname")
            or m.get("trader_nickname")
            or short_addr(m.get("wallet_address") or m.get("wallet"))
        )
        key = (m.get("wallet_address") or m.get("wallet") or name or "").strip().lower()
        size = float(m.get("size_usd") or 0.0)
        price = m.get("entry_price")
        agg = by_wallet.setdefault(key, {"name": name, "size": 0.0, "notional": 0.0, "fills": 0})
        agg["size"] += size
        agg["fills"] += 1
        if price is not None:
            agg["notional"] += float(price) * size
    rows = []
    for agg in by_wallet.values():
        avg_entry = (agg["notional"] / agg["size"]) if agg["size"] > 0 and agg["notional"] else None
        rows.append({**agg, "avg_entry": avg_entry})
    rows.sort(key=lambda r: r["size"], reverse=True)
    return rows


with tab_consensus:
    if not consensus:
        st.info(
            "No aligned consensus right now — no market has 2+ tracked wallets on the "
            "same side. If you expected some, check the **Diagnostics** tab to see where "
            "positions dropped out of the pipeline, or hit **Run scan now**."
        )
    for g in consensus:
        title = g.get("market_title") or g.get("market_slug") or f"Market {g.get('market_id')}"
        side = " · ".join(x for x in [g.get("outcome"), g.get("side")] if x)
        n = g.get("consensus_wallets", 0)
        total = format_money_short(g.get("consensus_total_size", 0.0))
        url = _market_url(g)
        link = f"<a class='sf-link' href='{url}' target='_blank'>open market ↗</a>" if url else ""
        wallets = _aggregate_wallets(g.get("consensus_members", []))

        with st.container(border=True):
            st.markdown(
                f"""
                <div class='sf-card-head'>
                  <div><span class='sf-mkt'>{title}</span> &nbsp;
                       <span class='sf-side'>{side}</span></div>
                  <div><span class='sf-count'>{n} wallets aligned</span></div>
                </div>
                <div class='sf-dim' style='margin:6px 0 2px'>Total aligned size: {total} &nbsp; {link}</div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander(f"Show {n} aligned wallets", expanded=n <= 4):
                rows_html = ""
                for w in wallets:
                    size = format_money_short(w["size"])
                    entry_s = f" @ {w['avg_entry']:.2f}" if w["avg_entry"] is not None else ""
                    fills_s = (
                        f" <span class='sf-dim'>({w['fills']} fills)</span>"
                        if w["fills"] > 1 else ""
                    )
                    rows_html += (
                        f"<div class='sf-wallet'>• <span class='sf-wname'>{w['name']}</span>"
                        f" — {size}{entry_s}{fills_s}</div>"
                    )
                st.markdown(rows_html, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# All Positions
# --------------------------------------------------------------------------- #

with tab_positions:
    if not filtered:
        st.info(
            "No tracked-wallet positions match the current filters / date. "
            "Adjust the sidebar or try **Run scan now**."
        )
    else:
        df = pd.DataFrame(filtered)
        cols = [
            "wallet_nickname", "market_title", "side", "outcome",
            "entry_price", "size_usd", "current_yes_price", "current_no_price",
            "sport", "market_platform", "opened_at",
        ]
        cols = [c for c in cols if c in df.columns]
        df = df[cols].sort_values("size_usd", ascending=False) if "size_usd" in df else df[cols]
        st.dataframe(df, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Watchlist
# --------------------------------------------------------------------------- #

with tab_watchlist:
    traders = safe_get("/traders", default=[]) or []
    if not traders:
        st.info("No tracked wallets. Seed the watchlist: `python -m scripts.seed`.")
    else:
        tdf = pd.DataFrame(traders)
        if "wallet_address" in tdf.columns:
            tdf["profile"] = tdf["wallet_address"].map(
                lambda w: f"https://polymarketanalytics.com/traders/{w}#trades" if w else None
            )
        cols = [
            "nickname", "profile", "trust_score", "win_rate", "total_pnl",
            "trader_rank", "total_positions", "copy_mode", "wallet_address",
        ]
        cols = [c for c in cols if c in tdf.columns]
        tdf = tdf[cols].sort_values("trust_score", ascending=False) if "trust_score" in tdf else tdf[cols]
        st.dataframe(
            tdf,
            use_container_width=True,
            hide_index=True,
            column_config={
                "profile": st.column_config.LinkColumn(
                    "Profile", display_text="View ↗"
                ),
            },
        )


# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #

with tab_alerts:
    alerts = safe_get("/alerts", params={"limit": 50}, default=[]) or []
    if not alerts:
        st.info("No alerts dispatched for today's card.")
    else:
        for a in alerts:
            ts = compact_time_ago(a.get("created_at"))
            st.markdown(
                f"<div class='sf-card'><span class='sf-dim'>{ts} · "
                f"{a.get('channel')} · {a.get('status')}</span><br>{a.get('message','')}</div>",
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------- #
# Diagnostics — explain empty states
# --------------------------------------------------------------------------- #

with tab_diag:
    st.caption(
        "When scans succeed but consensus is empty, this funnel pinpoints the exact "
        "stage records disappear."
    )
    funnel = safe_get("/dashboard/pipeline-debug", default={}) or {}
    if funnel:
        st.markdown(f"**Drop stage:** {funnel.get('drop_stage', '—')}")
        st.json(funnel.get("funnel", {}))
        with st.expander("Full funnel detail"):
            st.json(funnel)
    with st.expander("Per-row rejection log (/tracked-wallet-positions/debug)"):
        st.json(safe_get("/tracked-wallet-positions/debug", default={}) or {})
    if st.session_state.get("_last_api_error"):
        st.warning(f"Last API error: {st.session_state['_last_api_error']}")

st.markdown(
    f"<div class='sf-dim' style='margin-top:18px'>API: <code>{API_BASE}</code> · "
    f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}</div>",
    unsafe_allow_html=True,
)
