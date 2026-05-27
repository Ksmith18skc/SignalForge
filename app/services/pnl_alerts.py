"""Smart alerts for personal wallet positions."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.services.pnl_tracker import exposure_breakdown
from app.services.position_matcher import find_missed_opportunities
from app.storage.pnl_store import MyPosition, PnlAlert, SignalAttribution


def refresh_pnl_alerts(db: Session) -> list[PnlAlert]:
    """Create idempotent, operator-facing P&L alerts."""
    settings = get_settings()
    positions = list(db.scalars(select(MyPosition)))
    open_positions = [p for p in positions if p.status == "open" and p.shares > 0]
    alerts: list[PnlAlert] = []

    for pos in open_positions:
        if pos.current_edge is not None and pos.current_edge <= settings.pnl_alert_negative_edge_threshold:
            alerts.append(_upsert_alert(
                db,
                dedupe_key=f"negative-edge:{pos.id}",
                alert_type="negative_edge",
                severity="warn",
                title="Wallet position no longer has positive edge",
                body=f"{pos.market_title or pos.market_slug} current edge {pos.current_edge:+.3f}.",
                wallet_id=pos.wallet_id,
                market_slug=pos.market_slug,
                my_position_id=pos.id,
            ))

    attrs = list(db.scalars(select(SignalAttribution)))
    for attr in attrs:
        if attr.clv_points is not None and attr.clv_points >= settings.pnl_alert_strong_clv_points:
            alerts.append(_upsert_alert(
                db,
                dedupe_key=f"strong-clv:{attr.id}",
                alert_type="strong_clv",
                severity="info",
                title="Strong CLV on SignalForge-trailed position",
                body=f"CLV {attr.clv_points:+.3f} points.",
                my_position_id=attr.my_position_id,
                recommendation_id=attr.recommendation_id,
            ))
        if attr.label == "missed_best_price":
            alerts.append(_upsert_alert(
                db,
                dedupe_key=f"bad-entry:{attr.id}",
                alert_type="bad_entry",
                severity="warn",
                title="Late or bad entry versus SignalForge callout",
                body="User entry was materially worse than the frozen callout price.",
                my_position_id=attr.my_position_id,
                recommendation_id=attr.recommendation_id,
            ))

    total_open = sum((p.current_value_usd or p.cost_basis_usd or 0.0) for p in open_positions)
    for slc in exposure_breakdown(open_positions, dimension="market", portfolio_value=total_open):
        if slc.portfolio_pct / 100.0 >= settings.pnl_alert_overexposure_pct:
            alerts.append(_upsert_alert(
                db,
                dedupe_key=f"overexposed-market:{slc.key}",
                alert_type="overexposure",
                severity="warn",
                title="Overexposed to one market",
                body=f"{slc.label} is {slc.portfolio_pct:.1f}% of open exposure.",
                market_slug=slc.key,
            ))
    for slc in exposure_breakdown(open_positions, dimension="sport", portfolio_value=total_open):
        if slc.portfolio_pct / 100.0 >= settings.pnl_alert_overexposure_sport_pct:
            alerts.append(_upsert_alert(
                db,
                dedupe_key=f"overexposed-sport:{slc.key}",
                alert_type="overexposure",
                severity="warn",
                title="Overexposed to one sport",
                body=f"{slc.label} is {slc.portfolio_pct:.1f}% of open exposure.",
            ))

    for rec in find_missed_opportunities(db):
        alerts.append(_upsert_alert(
            db,
            dedupe_key=f"missed-edge:{rec.id}",
            alert_type="missed_edge",
            severity="info",
            title="Recommended edge has no matching wallet position",
            body=f"{rec.market_title or rec.market_slug} remains in the missed-edge queue.",
            market_slug=rec.market_slug,
            recommendation_id=rec.id,
        ))
    db.flush()
    return alerts


def _upsert_alert(
    db: Session,
    *,
    dedupe_key: str,
    alert_type: str,
    severity: str,
    title: str,
    body: str,
    wallet_id: int | None = None,
    market_slug: str | None = None,
    my_position_id: int | None = None,
    recommendation_id: int | None = None,
) -> PnlAlert:
    alert = db.scalar(select(PnlAlert).where(PnlAlert.dedupe_key == dedupe_key))
    if alert is None:
        alert = PnlAlert(
            dedupe_key=dedupe_key,
            alert_type=alert_type,
            severity=severity,
            title=title,
            body=body,
            wallet_id=wallet_id,
            market_slug=market_slug,
            my_position_id=my_position_id,
            recommendation_id=recommendation_id,
        )
        db.add(alert)
    else:
        alert.severity = severity
        alert.title = title
        alert.body = body
    return alert
