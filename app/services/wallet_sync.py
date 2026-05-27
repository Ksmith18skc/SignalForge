"""Personal Kalshi/Polymarket wallet sync for the P&L tracker.

Live wallet adapters are deliberately conservative. If credentials or venue
fields are missing, the sync preserves cached rows and records a warning rather
than manufacturing a live P&L value. Mock mode is explicit in returned payloads.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.services import pnl_tracker, position_matcher
from app.services.pnl_alerts import refresh_pnl_alerts
from app.storage.pnl_store import (
    MyTrade,
    MyWallet,
    RecommendationSnapshot,
    WalletSnapshot,
    insert_trades,
    latest_snapshot,
    list_wallets,
    upsert_wallet,
    write_snapshot,
)


def sync_personal_wallets(db: Session) -> dict[str, Any]:
    """Sync configured personal wallets and refresh derived P&L tables."""
    settings = get_settings()
    use_mock = (
        settings.pnl_use_mock_wallet
        or not settings.has_polymarket_wallet_addresses()
        and not settings.has_kalshi_user_credentials()
    )
    position_matcher.snapshot_actionable_recommendations(db)

    warnings: list[str] = []
    wallets: list[MyWallet] = []
    inserted = 0

    if use_mock:
        wallet = upsert_wallet(
            db,
            platform="mock",
            address="mock:personal-wallet",
            label="Mock Personal Wallet",
        )
        wallets.append(wallet)
        inserted += _insert_mock_wallet_trades(db, wallet)
        db.flush()
        warnings.append("Mock wallet mode is active; values are test data, not live P&L.")
    else:
        for address in settings.polymarket_wallet_list():
            wallet = upsert_wallet(db, platform="polymarket", address=address)
            wallet.last_sync_error = (
                "Live Polymarket wallet fill sync is not configured in this build; "
                "showing cached data only."
            )
            warnings.append(wallet.last_sync_error)
            wallets.append(wallet)
        if settings.has_kalshi_user_credentials():
            wallet = upsert_wallet(
                db,
                platform="kalshi",
                address=f"kalshi:{settings.kalshi_user_api_key_id}",
                label="Kalshi Account",
            )
            wallet.last_sync_error = (
                "Live Kalshi user-position sync requires signed account endpoints; "
                "showing cached data only."
            )
            warnings.append(wallet.last_sync_error)
            wallets.append(wallet)

    rebuilt = 0
    for wallet in wallets:
        positions = pnl_tracker.rebuild_positions_for_wallet(db, wallet)
        _refresh_position_marks(db, positions)
        cash = _mock_cash_balance(wallet) if use_mock else None
        summary = pnl_tracker.compute_portfolio_summary(
            pnl_tracker.PnlInputs(
                positions=positions,
                cash_by_wallet={wallet.id: cash} if cash is not None else {},
                previous_total_value_usd=_previous_wallet_total(db, wallet),
            )
        )
        write_snapshot(
            db,
            WalletSnapshot(
                wallet_id=wallet.id,
                cash_balance_usd=cash,
                open_position_value_usd=summary.open_position_value_usd,
                realized_pnl_usd=summary.realized_pnl_usd,
                unrealized_pnl_usd=summary.unrealized_pnl_usd,
                total_value_usd=summary.total_value_usd,
                is_estimated=summary.is_estimated,
            ),
        )
        wallet.last_synced_at = datetime.utcnow()
        if use_mock:
            wallet.last_sync_error = None
        rebuilt += len(positions)

    position_matcher.match_trades_to_recommendations(db, wallet_ids=[w.id for w in wallets])
    refresh_pnl_alerts(db)
    db.flush()
    return {
        "mode": "mock" if use_mock else "cached_live",
        "wallets": len(wallets),
        "new_trades": inserted,
        "positions_rebuilt": rebuilt,
        "warnings": warnings,
    }


def ensure_pnl_demo_data_if_needed(db: Session) -> dict[str, Any] | None:
    """Create mock data only when no personal wallet cache exists."""
    if list_wallets(db, active_only=False):
        return None
    return sync_personal_wallets(db)


def _insert_mock_wallet_trades(db: Session, wallet: MyWallet) -> int:
    now = datetime.utcnow()
    rec = db.scalar(
        select(RecommendationSnapshot).order_by(RecommendationSnapshot.captured_at.desc()).limit(1)
    )
    market_slug = (rec.market_slug if rec else None) or "mock-mlb-total-under-8-5"
    market_title = (rec.market_title if rec else None) or "Mock MLB Total Under 8.5"
    outcome = (rec.outcome if rec else None) or "under"
    signal_price = rec.market_price if rec and rec.market_price is not None else 0.47

    trades = [
        MyTrade(
            wallet_id=wallet.id,
            external_id=f"mock-fill-{market_slug}-1",
            platform=wallet.platform,
            market_slug=market_slug,
            market_title=market_title,
            side="BUY",
            outcome=outcome,
            price=max(0.01, min(0.99, signal_price + 0.01)),
            size_shares=120.0,
            size_usd=round(120.0 * max(0.01, min(0.99, signal_price + 0.01)), 2),
            sport=(rec.sport if rec else None) or "mlb",
            event_date=(rec.event_date if rec else None),
            source="mock",
            timestamp=(rec.captured_at + timedelta(minutes=18)) if rec else now - timedelta(hours=2),
            raw={"mock": True, "assumption": "deterministic test fill"},
        ),
        MyTrade(
            wallet_id=wallet.id,
            external_id="mock-fill-non-signal-1",
            platform=wallet.platform,
            market_slug="mock-nba-series-price",
            market_title="Mock NBA Series Price",
            side="BUY",
            outcome="yes",
            price=0.38,
            size_shares=80.0,
            size_usd=30.40,
            sport="nba",
            event_date=None,
            source="mock",
            timestamp=now - timedelta(hours=5),
            raw={"mock": True, "assumption": "non-dashboard comparison trade"},
        ),
    ]
    return insert_trades(db, trades)


def _refresh_position_marks(db: Session, positions: list[Any]) -> None:
    for pos in positions:
        rec = position_matcher.latest_recommendation_for_position(
            db, market_slug=pos.market_slug, outcome=pos.outcome
        )
        if rec:
            pos.fair_probability = rec.fair_probability
            pos.confidence_tier = rec.confidence_tier
            pos.signal_status = (
                "still_actionable" if rec.threshold_status == "actionable" else "no_longer_actionable"
            )
            pos.edge_at_entry = pnl_tracker.compute_edge(rec.fair_probability, pos.avg_entry_price)
            current_price = rec.market_price
        else:
            current_price = pos.current_price
            pos.signal_status = "no_matching_recommendation"
            if current_price is None and pos.platform == "mock":
                current_price = max(0.01, min(0.99, (pos.avg_entry_price or 0.50) + 0.03))
        if current_price is None:
            # We have no reliable current mark; leave P&L estimated/stale.
            pos.current_price = None
            pos.current_value_usd = 0.0
            pos.unrealized_pnl_usd = 0.0
            pos.is_stale_price = True
            continue
        pos.current_price = current_price
        pos.current_value_usd = pnl_tracker.compute_position_value(pos.shares, current_price)
        pos.unrealized_pnl_usd = pnl_tracker.compute_unrealized_pnl(
            pos.shares, pos.avg_entry_price, current_price
        )
        pos.current_edge = pnl_tracker.compute_edge(pos.fair_probability, current_price)
        pos.is_stale_price = pnl_tracker.is_stale_price(pos.last_updated)


def _previous_wallet_total(db: Session, wallet: MyWallet) -> float | None:
    snap = latest_snapshot(db, wallet.id)
    return snap.total_value_usd if snap else None


def _mock_cash_balance(wallet: MyWallet) -> float:
    return 750.0 if wallet.platform == "mock" else 0.0
