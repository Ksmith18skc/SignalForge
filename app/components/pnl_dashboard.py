"""Streamlit UI for the personal P&L tracker."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import streamlit as st


def render_pnl_summary_cards(payload: dict[str, Any], *, fmt_money: Callable[[Any], str], fmt_num: Callable[..., str]) -> None:
    summary = payload.get("summary") or {}
    attribution = payload.get("attribution") or {}
    cols = st.columns(6)
    cols[0].metric("Total P&L", fmt_money(summary.get("total_pnl_usd")))
    cols[1].metric("Unrealized P&L", fmt_money(summary.get("unrealized_pnl_usd")))
    cols[2].metric("Realized P&L", fmt_money(summary.get("realized_pnl_usd")))
    cols[3].metric("SF-Trailed P&L", fmt_money(attribution.get("trailed_pnl_usd")))
    cols[4].metric("Avg CLV", fmt_num(attribution.get("average_clv_points"), fmt="{:+.3f}"))
    cols[5].metric("Open Exposure", fmt_money(summary.get("open_position_value_usd")))


def render_pnl_tracker(
    payload: dict[str, Any],
    *,
    sync_action: Callable[[], None],
    fmt_money: Callable[[Any], str],
    fmt_num: Callable[..., str],
    fmt_pct: Callable[[Any], str],
) -> None:
    mode = str(payload.get("mode") or "empty").upper()
    st.markdown(f"### P&L Tracker { _badge(mode, 'purple' if mode == 'MOCK' else 'green') }", unsafe_allow_html=True)
    warnings = payload.get("warnings") or []
    for warning in warnings[:4]:
        st.warning(warning)

    actions = st.columns([1, 5])
    with actions[0]:
        if st.button("Sync wallets", use_container_width=True, type="primary"):
            sync_action()

    render_pnl_summary_cards(payload, fmt_money=fmt_money, fmt_num=fmt_num)

    summary = payload.get("summary") or {}
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Portfolio value", fmt_money(summary.get("total_value_usd")))
    m2.metric("Cash / stable", fmt_money(summary.get("cash_usd")))
    m3.metric("ROI", fmt_pct((summary.get("roi_percent") or 0) / 100) if summary.get("roi_percent") is not None else "—")
    m4.metric("Daily P&L", fmt_money(summary.get("daily_pnl_usd")))
    m5.metric("Largest winner", fmt_money(summary.get("largest_winner_usd")))
    m6.metric("Largest loser", fmt_money(summary.get("largest_loser_usd")))

    st.markdown("### Smart Alerts")
    _render_alerts(payload.get("alerts") or [])

    st.markdown("### Filters")
    all_rows = (payload.get("open_positions") or []) + (payload.get("closed_positions") or [])
    f1, f2, f3, f4, f5 = st.columns(5)
    source = f1.selectbox("Source", _options(all_rows, "platform"))
    sport = f2.selectbox("Sport", _options(all_rows, "sport"))
    confidence = f3.selectbox("Confidence", _options(all_rows, "confidence_tier"))
    trailed = f4.selectbox("Trailed", ["(all)", "trailed", "not trailed"])
    status = f5.selectbox("Status", ["(all)", "open", "closed"])

    open_rows = _filter_rows(payload.get("open_positions") or [], source, sport, confidence, trailed, status)
    closed_rows = _filter_rows(payload.get("closed_positions") or [], source, sport, confidence, trailed, status)

    st.markdown("### Open Positions")
    _render_positions_table(open_rows, open_positions=True)

    st.markdown("### Closed / Settled Positions")
    _render_positions_table(closed_rows, open_positions=False)

    st.markdown("### Exposure")
    _render_exposure(payload.get("exposure") or {})

    st.markdown("### Missed Edges")
    _render_missed_edges(payload.get("missed_edges") or [])

    st.markdown("### SignalForge Attribution")
    _render_attribution(payload.get("attribution") or {}, fmt_money=fmt_money, fmt_num=fmt_num, fmt_pct=fmt_pct)


def _render_positions_table(rows: list[dict[str, Any]], *, open_positions: bool) -> None:
    if not rows:
        st.caption("No positions in this filter set.")
        return
    table = []
    for row in rows:
        table.append({
            "badges": " ".join(f"[{b}]" for b in row.get("badges", [])),
            "source": row.get("platform"),
            "sport": row.get("sport"),
            "market": row.get("market_title") or row.get("market_slug"),
            "side": row.get("side"),
            "outcome": row.get("outcome"),
            "shares": row.get("shares"),
            "entry": row.get("avg_entry_price"),
            "current": row.get("current_price"),
            "fair": row.get("fair_probability"),
            "edge entry": row.get("edge_at_entry"),
            "edge now": row.get("current_edge"),
            "CLV": row.get("clv_points"),
            "unrealized": row.get("unrealized_pnl_usd") if open_positions else None,
            "realized": row.get("realized_pnl_usd"),
            "total P&L": row.get("total_pnl_usd"),
            "confidence": row.get("confidence_tier"),
            "signal status": row.get("signal_status"),
        })
    df = pd.DataFrame(table)
    if not open_positions and "unrealized" in df.columns:
        df = df.drop(columns=["unrealized"])
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=min(520, 58 + 32 * len(df)),
        column_config={
            "shares": st.column_config.NumberColumn("shares", format="%.2f"),
            "entry": st.column_config.NumberColumn("entry", format="%.3f"),
            "current": st.column_config.NumberColumn("current", format="%.3f"),
            "fair": st.column_config.NumberColumn("SF fair", format="%.3f"),
            "edge entry": st.column_config.NumberColumn("edge entry", format="%+.3f"),
            "edge now": st.column_config.NumberColumn("edge now", format="%+.3f"),
            "CLV": st.column_config.NumberColumn("CLV", format="%+.3f"),
            "unrealized": st.column_config.NumberColumn("unrealized", format="$%.2f"),
            "realized": st.column_config.NumberColumn("realized", format="$%.2f"),
            "total P&L": st.column_config.NumberColumn("total P&L", format="$%.2f"),
        },
    )


def _render_missed_edges(rows: list[dict[str, Any]]) -> None:
    if not rows:
        st.caption("No missed actionable recommendations in the current lookback.")
        return
    df = pd.DataFrame([
        {
            "badges": " ".join(f"[{b}]" for b in r.get("badges", [])),
            "source": r.get("source"),
            "sport": r.get("sport"),
            "market": r.get("market_title") or r.get("market_slug"),
            "side": r.get("side"),
            "fair": r.get("fair_probability"),
            "callout price": r.get("market_price"),
            "edge": r.get("implied_edge"),
            "score": r.get("score"),
            "captured": r.get("captured_at"),
        }
        for r in rows
    ])
    st.dataframe(df, use_container_width=True, hide_index=True, height=min(360, 58 + 32 * len(df)))


def _render_alerts(rows: list[dict[str, Any]]) -> None:
    if not rows:
        st.caption("No active P&L alerts.")
        return
    for alert in rows[:6]:
        kind = "red" if alert.get("severity") in {"crit", "warn"} else "cyan"
        st.markdown(
            "<div class='sf-card'>"
            + f"<div class='sf-card-row'>{_badge(str(alert.get('severity') or 'info').upper(), kind)} "
            + f"<b>{alert.get('title') or ''}</b></div>"
            + f"<div class='sf-card-sub'>{alert.get('body') or ''}</div>"
            + "</div>",
            unsafe_allow_html=True,
        )


def _render_exposure(exposure: dict[str, Any]) -> None:
    cols = st.columns(3)
    for col, name in zip(cols, ["sport", "market", "source"]):
        with col:
            st.markdown(f"#### By {name.title()}")
            rows = exposure.get(name) or []
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=220)
            else:
                st.caption("No open exposure.")


def _render_attribution(
    attr: dict[str, Any],
    *,
    fmt_money: Callable[[Any], str],
    fmt_num: Callable[..., str],
    fmt_pct: Callable[[Any], str],
) -> None:
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Trailed P&L", fmt_money(attr.get("trailed_pnl_usd")))
    a2.metric("Non-dashboard P&L", fmt_money(attr.get("non_signal_pnl_usd")))
    a3.metric("Trailed win rate", fmt_pct(attr.get("trailed_win_rate")))
    a4.metric("Avg CLV", fmt_num(attr.get("average_clv_points"), fmt="{:+.3f}"))
    a5.metric("Trailed trades", attr.get("trailed_trade_count") or 0)
    best = attr.get("best_signal")
    worst = attr.get("worst_signal")
    c1, c2 = st.columns(2)
    c1.markdown(f"**Best signal:** {(best or {}).get('market') or '—'} ({fmt_money((best or {}).get('pnl_usd'))})")
    c2.markdown(f"**Worst signal:** {(worst or {}).get('market') or '—'} ({fmt_money((worst or {}).get('pnl_usd'))})")


def _filter_rows(
    rows: list[dict[str, Any]],
    source: str,
    sport: str,
    confidence: str,
    trailed: str,
    status: str,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if source != "(all)" and row.get("platform") != source:
            continue
        if sport != "(all)" and row.get("sport") != sport:
            continue
        if confidence != "(all)" and row.get("confidence_tier") != confidence:
            continue
        if status != "(all)" and row.get("status") != status:
            continue
        if trailed == "trailed" and not row.get("trailed_signalforge"):
            continue
        if trailed == "not trailed" and row.get("trailed_signalforge"):
            continue
        out.append(row)
    return out


def _options(rows: list[dict[str, Any]], key: str) -> list[str]:
    return ["(all)"] + sorted({str(r.get(key)) for r in rows if r.get(key)})


def _badge(text: str, kind: str = "muted") -> str:
    return f"<span class='sf-pill {kind}'>{text}</span>"
