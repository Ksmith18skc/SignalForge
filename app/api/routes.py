"""HTTP API routes."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import delete, desc, func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.models import Alert, Market, Position, Signal, Trade, Trader
from app.providers.falcon import FalconProvider, get_falcon_health
from app.schemas import (
    AlertOut,
    DashboardSummary,
    MarketOut,
    ScanResult,
    SignalOut,
    TraderCreate,
    TraderOut,
    WatchlistHealth,
)
from app.services.scanner import run_scan_once

router = APIRouter()


# ---------------------------- health ----------------------------------------


@router.get("/health")
def health() -> dict[str, object]:
    s = get_settings()
    falcon_health = get_falcon_health()
    # "Configured" means a key is set. "Healthy" means calls are actually
    # succeeding — these can disagree (wrong base URL, expired key, etc.).
    return {
        "status": "ok",
        "app": s.app_name,
        "environment": s.environment,
        "default_copy_mode": s.default_copy_mode,
        "auto_trading_enabled": s.enable_auto_trading,
        "providers": {
            "falcon": {
                "configured": s.has_falcon_credentials(),
                "healthy": falcon_health.healthy,
                "calls": falcon_health.calls,
                "successes": falcon_health.successes,
                "success_rate": round(falcon_health.success_rate, 3),
                "last_status_code": falcon_health.last_status_code,
                "last_error": falcon_health.last_error,
                "last_endpoint": falcon_health.last_endpoint,
                "last_scan_at": (
                    falcon_health.last_scan_at.isoformat()
                    if falcon_health.last_scan_at
                    else None
                ),
                "last_scan_calls": falcon_health.last_scan_calls,
                "last_scan_successes": falcon_health.last_scan_successes,
                "base_url": falcon_health.base_url or s.falcon_base_url,
            },
            "polymarket": {"configured": s.has_polymarket_credentials()},
            "kalshi": {"configured": s.has_kalshi_credentials()},
        },
        "alerts": {
            "console": {"configured": True},
            "discord": {"configured": bool(s.discord_webhook_url)},
            "telegram": {
                "configured": bool(s.telegram_bot_token and s.telegram_chat_id),
                "has_bot_token": bool(s.telegram_bot_token),
                "has_chat_id": bool(s.telegram_chat_id),
            },
            "email": {
                "configured": bool(
                    s.alert_email_to
                    and s.smtp_host
                    and (s.alert_email_from or s.smtp_username)
                ),
                "has_recipient": bool(s.alert_email_to),
                "has_smtp_host": bool(s.smtp_host),
                "has_from_address": bool(s.alert_email_from or s.smtp_username),
            },
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


# ---------------------------- traders ---------------------------------------


@router.get("/traders", response_model=list[TraderOut])
def list_traders(db: Session = Depends(get_db)) -> list[Trader]:
    return list(db.scalars(select(Trader).order_by(Trader.trust_score.desc())))


@router.post("/traders", response_model=TraderOut, status_code=status.HTTP_201_CREATED)
def create_trader(payload: TraderCreate, db: Session = Depends(get_db)) -> Trader:
    existing = db.scalar(select(Trader).where(Trader.nickname == payload.nickname))
    if existing:
        raise HTTPException(status_code=409, detail=f"trader '{payload.nickname}' already exists")

    if payload.wallet_address:
        existing_wallet = db.scalar(
            select(Trader).where(Trader.wallet_address == payload.wallet_address)
        )
        if existing_wallet:
            raise HTTPException(
                status_code=409,
                detail=f"wallet '{payload.wallet_address}' already exists",
            )

    trader = Trader(**payload.model_dump())
    # MVP guard: never allow live copy_mode through the API.
    if trader.copy_mode == "live" and not get_settings().enable_auto_trading:
        trader.copy_mode = "alert_only"

    db.add(trader)
    db.commit()
    db.refresh(trader)
    return trader


@router.delete(
    "/traders/{trader_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    response_class=Response,
)
def delete_trader(trader_id: int, db: Session = Depends(get_db)) -> None:
    trader = db.get(Trader, trader_id)
    if trader is None:
        raise HTTPException(status_code=404, detail=f"trader {trader_id} not found")

    signal_ids = list(
        db.scalars(select(Signal.id).where(Signal.trader_id == trader_id))
    )
    if signal_ids:
        db.execute(delete(Alert).where(Alert.signal_id.in_(signal_ids)))
        db.execute(delete(Signal).where(Signal.id.in_(signal_ids)))

    db.execute(delete(Position).where(Position.trader_id == trader_id))
    db.execute(delete(Trade).where(Trade.trader_id == trader_id))
    db.delete(trader)
    db.commit()
    return None


# ---------------------------- markets ---------------------------------------


@router.get("/markets", response_model=list[MarketOut])
def list_markets(db: Session = Depends(get_db), limit: int = 50) -> list[Market]:
    return list(
        db.scalars(
            select(Market).where(Market.is_active.is_(True)).limit(limit)
        )
    )


# ---------------------------- signals ---------------------------------------


def _enrich_signal(signal: Signal) -> SignalOut:
    base = SignalOut.model_validate(signal)
    if signal.trader:
        base.wallet = signal.trader.wallet_address
        base.trader_nickname = signal.trader.nickname
    if signal.market:
        base.market_title = signal.market.title
        base.market_slug = signal.market.slug
        base.market_platform = signal.market.platform
    return base


@router.get("/signals", response_model=list[SignalOut])
def list_signals(db: Session = Depends(get_db), limit: int = 50) -> list[SignalOut]:
    signals = list(
        db.scalars(select(Signal).order_by(desc(Signal.created_at)).limit(limit))
    )
    return [_enrich_signal(s) for s in signals]


# ---------------------------- alerts ----------------------------------------


@router.get("/alerts", response_model=list[AlertOut])
def list_alerts(db: Session = Depends(get_db), limit: int = 50) -> list[Alert]:
    return list(db.scalars(select(Alert).order_by(desc(Alert.created_at)).limit(limit)))


# ---------------------------- run-scan --------------------------------------


@router.post("/run-scan", response_model=ScanResult)
async def trigger_scan() -> ScanResult:
    return await run_scan_once()


# ---------------------------- falcon-test -----------------------------------

# LaBradfordSmith22 from the seeded watchlist — a known-good wallet to probe.
_DEFAULT_TEST_WALLET = "0x9495425feeb0c250accb89275c97587011b19a27"


@router.get("/falcon-test")
async def falcon_test(
    wallet: str = _DEFAULT_TEST_WALLET,
    agent_id: int = FalconProvider.AGENT_WALLET_360,
    window_days: int = 3,
) -> dict[str, object]:
    """Probe a single Falcon agent and return the raw response.

    Useful for verifying the API key works and discovering the actual response
    shape so the parser in FalconProvider can be tightened. Defaults to
    agent_id=581 (Wallet 360) against LaBradfordSmith22's wallet.
    """
    s = get_settings()
    if not s.has_falcon_credentials():
        raise HTTPException(
            status_code=400,
            detail="SIGNALFORGE_FALCON_API_KEY is not set",
        )

    falcon = FalconProvider(s.falcon_api_key, s.falcon_base_url)
    raw = await falcon.query_agent(
        agent_id,
        params={"proxy_wallet": wallet, "window_days": str(window_days)},
    )
    health = get_falcon_health()
    return {
        "agent_id": agent_id,
        "wallet": wallet,
        "window_days": window_days,
        "ok": raw is not None,
        "raw_response": raw,
        "falcon_health": {
            "last_status_code": health.last_status_code,
            "last_error": health.last_error,
            "last_endpoint": health.last_endpoint,
            "last_agent_id": health.last_agent_id,
        },
    }


# ---------------------------- dashboard -------------------------------------


@router.get("/dashboard-summary", response_model=DashboardSummary)
def dashboard_summary(db: Session = Depends(get_db)) -> DashboardSummary:
    active_signals_raw = list(
        db.scalars(
            select(Signal).order_by(desc(Signal.score), desc(Signal.created_at)).limit(10)
        )
    )
    active_signals = [_enrich_signal(s) for s in active_signals_raw]

    top_traders = list(
        db.scalars(
            select(Trader).order_by(desc(Trader.trust_score)).limit(10)
        )
    )

    # "Highest conviction" markets = those carrying the highest-scored signals.
    top_market_ids_query = (
        select(Signal.market_id, func.max(Signal.score).label("max_score"))
        .group_by(Signal.market_id)
        .order_by(desc("max_score"))
        .limit(10)
    )
    top_market_ids = [row[0] for row in db.execute(top_market_ids_query).all()]
    if top_market_ids:
        markets = list(db.scalars(select(Market).where(Market.id.in_(top_market_ids))))
        # preserve order from the score query
        order = {mid: idx for idx, mid in enumerate(top_market_ids)}
        markets.sort(key=lambda m: order.get(m.id, 999))
    else:
        markets = list(db.scalars(select(Market).limit(10)))

    recent_alerts = list(
        db.scalars(select(Alert).order_by(desc(Alert.created_at)).limit(10))
    )

    # Watchlist health: how complete is our trader enrichment?
    all_traders = list(db.scalars(select(Trader)))
    enriched = [t for t in all_traders if t.total_pnl is not None or t.trader_rank is not None]
    win_rates = [t.win_rate for t in all_traders if t.win_rate is not None]
    health = WatchlistHealth(
        total_traders=len(all_traders),
        enriched_traders=len(enriched),
        avg_trust_score=round(
            sum(t.trust_score for t in all_traders) / max(len(all_traders), 1), 2
        ),
        avg_win_rate=round(sum(win_rates) / len(win_rates), 4) if win_rates else None,
        enabled_for_copy=sum(1 for t in all_traders if t.copy_enabled),
    )

    # Simulated PnL is a placeholder — the MVP doesn't execute trades.
    simulated_pnl = 0.0

    return DashboardSummary(
        active_signals=active_signals,
        top_traders=top_traders,
        highest_conviction_markets=markets,
        recent_alerts=recent_alerts,
        simulated_pnl_usd=simulated_pnl,
        watchlist_health=health,
    )
