"""HTTP API routes."""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import delete, desc, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.models import Alert, Market, Signal, Trade, Trader
from app.providers.falcon import FalconProvider, get_falcon_health
from app.services.ingestion_health import get_ingestion_health
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
from app.services.scanner import (
    reset_scan_status,
    run_scan_diagnostics,
    run_scan_once,
    scan_status,
    trigger_manual_scan_background,
)
from app.services.card_date import arizona_today, market_card_date

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------- health ----------------------------------------


@router.get("/health")
def health() -> dict[str, object]:
    """Pure liveness probe — must return instantly, never touch DB or providers.

    Render uses this for its port scan. Anything that can block (DB DDL, provider
    metrics, cache refreshes) belongs on /ready instead.
    """
    return {
        "ok": True,
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/")
def root() -> dict[str, object]:
    return health()


@router.get("/api/status")
def api_status() -> dict[str, object]:
    return health()


@router.get("/ready")
def ready() -> dict[str, object]:
    """Dependency/cache readiness probe.

    Keep `/health` as a cheap process liveness endpoint. This route carries the
    heavier diagnostic payload that the dashboard and operators can inspect
    after the process is already awake.
    """
    s = get_settings()
    falcon_health = get_falcon_health()
    ingest = get_ingestion_health()
    # Importing here keeps /health independent of the bootstrap module so a
    # broken import in main.py can't take down the liveness probe.
    try:
        from app.db import db_init_status
        from app.main import get_bootstrap_state

        db_status = db_init_status()
        bootstrap = get_bootstrap_state()
    except Exception as exc:  # noqa: BLE001
        db_status = {"ready": False, "error": f"db_init_status import failed: {exc}"}
        bootstrap = {"error": f"bootstrap state import failed: {exc}"}
    # "Configured" means a key is set. "Healthy" means calls are actually
    # succeeding — these can disagree (wrong base URL, expired key, etc.).
    return {
        "status": "ok" if db_status.get("ready") else "warming",
        "app": s.app_name,
        "bootstrap": bootstrap,
        "db_initialized": db_status,
        "environment": s.environment,
        "default_copy_mode": s.default_copy_mode,
        "auto_trading_enabled": s.enable_auto_trading,
        "database": {"backend": _database_backend(s.database_url)},
        "ingestion": {
            "ingestion_failures": ingest.ingestion_failures,
            "db_rollbacks": ingest.db_rollbacks,
            "last_ingestion_error": ingest.last_ingestion_error,
            "last_ingestion_error_at": (
                ingest.last_ingestion_error_at.isoformat()
                if ingest.last_ingestion_error_at
                else None
            ),
            "last_rollback_at": (
                ingest.last_rollback_at.isoformat() if ingest.last_rollback_at else None
            ),
            "trades_inserted": ingest.trades_inserted,
            "trades_skipped_oversized": ingest.trades_skipped_oversized,
        },
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


def _database_backend(database_url: str) -> str:
    scheme = database_url.split(":", 1)[0].lower() if database_url else ""
    if scheme.startswith("sqlite"):
        return "sqlite"
    if scheme.startswith("postgres") or scheme.startswith("postgresql"):
        return "postgres"
    return "unknown"


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
    # Guard: never allow live copy_mode through the API.
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
        if signal.trader.wallet_address:
            base.trader_url = f"https://polymarketanalytics.com/traders/{signal.trader.wallet_address}"
    if signal.market:
        base.market_title = signal.market.title
        base.market_slug = signal.market.slug
        base.market_platform = signal.market.platform
        base.market_created_at = signal.market.created_at
        base.market_updated_at = signal.market.updated_at
        base.market_end_date = signal.market.end_date
        if signal.market.slug:
            # Single source of truth — keeps the URL event-level for Polymarket
            # so a click never lands on a stale market page that doesn't exist.
            from app.services.wallet_market_resolver import market_url_for as _market_url_for
            base.market_url = _market_url_for(signal.market.slug, signal.market.platform)
    return base


def _signal_matches_live_card(
    signal: Signal,
    *,
    card_date: str | None,
    active_only: bool,
    exclude_resolved: bool,
) -> bool:
    market = signal.market
    market_date = market_card_date(market)
    if card_date:
        if market_date and market_date != card_date:
            return False
        signal_date = signal.generated_for_date or market_date
        if signal_date != card_date:
            return False
    if active_only and market is not None and market.is_active is False:
        return False
    if exclude_resolved:
        if market is not None and market.is_active is False:
            return False
        if market_date and card_date and market_date != card_date:
            return False
    return True


def _live_signals(
    db: Session,
    *,
    limit: int,
    card_date: str | None,
    active_only: bool = True,
    exclude_resolved: bool = True,
) -> list[Signal]:
    fetch_limit = max(limit * 5, limit, 250)
    rows = list(
        db.scalars(
            select(Signal)
            .order_by(desc(Signal.score), desc(Signal.created_at))
            .limit(fetch_limit)
        )
    )
    out = [
        signal for signal in rows
        if _signal_matches_live_card(
            signal,
            card_date=card_date,
            active_only=active_only,
            exclude_resolved=exclude_resolved,
        )
    ]
    return out[:limit]


@router.get("/signals", response_model=list[SignalOut])
def list_signals(
    db: Session = Depends(get_db),
    limit: int = 50,
    date: str | None = None,
    active_only: bool = False,
    exclude_resolved: bool = False,
    history: bool = False,
) -> list[SignalOut]:
    target = None if history else (date or arizona_today())
    signals = _live_signals(
        db,
        limit=limit,
        card_date=target,
        active_only=active_only,
        exclude_resolved=exclude_resolved,
    )
    return [_enrich_signal(s) for s in signals]


# ---------------------------- tracked-wallet live positions ------------------
#
# These routes intentionally bypass the Signal pipeline. The "Tracked Wallet
# Live Positions" panel must surface raw Trade rows even when the signal engine
# dropped them for score-threshold or market-date normalization reasons —
# otherwise the dashboard renders an empty state while the scanner reports
# thousands of rejected rows.


@router.get("/tracked-wallet-positions")
def list_tracked_wallet_positions(
    db: Session = Depends(get_db),
    date: str | None = None,
) -> list[dict[str, object]]:
    """Raw tracked-wallet trades plausibly belonging to ``date``.

    No score threshold; sport-agnostic. The dashboard's Aligned Consensus and
    All Positions views feed off this so the operator always sees tracked-wallet
    positions when they exist, even if no Signal row was generated for them.
    """
    from app.services.tracked_wallet_positions import live_positions

    target = date or arizona_today()
    return live_positions(db, card_date=target)


@router.get("/tracked-wallet-positions/debug")
def tracked_wallet_positions_debug(
    db: Session = Depends(get_db),
    date: str | None = None,
    limit: int = 50,
) -> dict[str, object]:
    """Per-row rejection diagnostics for the diagnostics panel.

    Returns the count buckets (raw, accepted, rejected, reason histogram) plus
    up to ``limit`` worked examples with the wallet nickname, slug, parsed date,
    expected date, and normalized key so the operator can see EXACTLY why a row
    was dropped.
    """
    from app.services.tracked_wallet_positions import live_position_debug

    target = date or arizona_today()
    return live_position_debug(db, card_date=target, limit=max(0, int(limit)))


# ---------------------------- alerts ----------------------------------------


@router.get("/alerts", response_model=list[AlertOut])
def list_alerts(
    db: Session = Depends(get_db),
    limit: int = 50,
    date: str | None = None,
    history: bool = False,
) -> list[Alert]:
    target = None if history else (date or arizona_today())
    fetch_limit = max(limit * 5, limit, 250)
    rows = list(
        db.scalars(select(Alert).order_by(desc(Alert.created_at)).limit(fetch_limit))
    )
    if target:
        rows = [alert for alert in rows if _alert_matches_card_date(alert, target)]
    return rows[:limit]


def _alert_matches_card_date(alert: Alert, card_date: str) -> bool:
    signal = alert.signal
    market_date = market_card_date(signal.market) if signal else None
    if market_date and market_date != card_date:
        return False
    alert_date = alert.generated_for_date
    signal_date = signal.generated_for_date if signal else None
    return (alert_date or signal_date or market_date) == card_date


# ---------------------------- run scan --------------------------------------


@router.post("/run-scan")
async def trigger_scan(date: str | None = None) -> dict[str, object]:
    return trigger_manual_scan_background(card_date=date)


@router.get("/run-scan/status")
def trigger_scan_status() -> dict[str, object]:
    return scan_status()


@router.post("/run-scan/blocking", response_model=ScanResult)
async def trigger_scan_blocking(date: str | None = None) -> ScanResult:
    return await run_scan_once(card_date=date)


@router.get("/run-scan/diagnostics")
async def trigger_scan_diagnostics(sample: int = 5) -> dict[str, object]:
    """Cheap pre-flight probe.

    Confirms tracked-wallet count, primary provider reachability, and a sample
    of the first wallet's raw positions — *without* triggering a full scan.
    """
    return await run_scan_diagnostics(sample_size=max(1, min(int(sample or 5), 20)))


@router.post("/run-scan/reset")
def trigger_scan_reset() -> dict[str, object]:
    """Operator escape hatch — flip the manual scan status back to idle."""
    return reset_scan_status()


# ---------------------------- falcon ----------------------------------------

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
    shape. Defaults to agent_id=581 (Wallet 360) against a known wallet.
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
    falcon_health = get_falcon_health()
    return {
        "agent_id": agent_id,
        "wallet": wallet,
        "window_days": window_days,
        "ok": raw is not None,
        "raw_response": raw,
        "falcon_health": {
            "last_status_code": falcon_health.last_status_code,
            "last_error": falcon_health.last_error,
            "last_endpoint": falcon_health.last_endpoint,
            "last_agent_id": falcon_health.last_agent_id,
        },
    }


@router.get("/falcon/agents")
def falcon_agents_status() -> dict[str, object]:
    """Registry of wired Falcon agents + the live last-call telemetry."""
    from app.providers.falcon_agents import all_specs

    falcon_health = get_falcon_health()
    return {
        "agents": [
            {"name": spec.name, "label": spec.label, "id": spec.id}
            for spec in all_specs()
        ],
        "health": falcon_health.as_dict(),
    }


# ---------------------------- dashboard -------------------------------------


@router.get("/dashboard/debug")
def dashboard_debug(
    date: str | None = None,
    db: Session = Depends(get_db),
) -> dict[str, object]:
    selected = date or arizona_today()
    status_payload = scan_status()

    signals = list(db.scalars(select(Signal).order_by(desc(Signal.created_at)).limit(5000)))
    today_signals = [
        s for s in signals
        if _signal_matches_live_card(
            s, card_date=selected, active_only=True, exclude_resolved=True,
        )
    ]
    stale_signals = [
        s for s in signals
        if not _signal_matches_live_card(
            s, card_date=selected, active_only=True, exclude_resolved=True,
        )
    ]

    alerts = list(db.scalars(select(Alert).order_by(desc(Alert.created_at)).limit(5000)))
    today_alerts = [a for a in alerts if _alert_matches_card_date(a, selected)]
    stale_alerts = [a for a in alerts if not _alert_matches_card_date(a, selected)]

    latest_signal = signals[0] if signals else None
    latest_alert = alerts[0] if alerts else None

    return {
        "arizona_today": arizona_today(),
        "selected_card_date": selected,
        "latest_wallet_scan_generated_for_date": (
            ((status_payload.get("result") or {}).get("generated_for_date"))
            or status_payload.get("generated_for_date")
            or (latest_signal.generated_for_date if latest_signal else None)
        ),
        "latest_alert_generated_for_date": (
            latest_alert.generated_for_date if latest_alert else None
        ),
        "today_wallet_positions": len(today_signals),
        "stale_wallet_positions_hidden": len(stale_signals),
        "today_alerts": len(today_alerts),
        "stale_alerts_hidden": len(stale_alerts),
    }


@router.get("/dashboard/pipeline-debug")
def dashboard_pipeline_debug(
    date: str | None = None,
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """Full wallet-trades -> card-date -> consensus funnel.

    Returns a per-stage count for ``date`` (default: Arizona today) plus a
    ``drop_stage`` naming the earliest stage that hit zero. This is the "where
    did my data go?" panel: when scans succeed but the Aligned Consensus view is
    empty, this pinpoints the exact stage records disappear.
    """
    from app.services.pipeline_diagnostics import pipeline_funnel

    target = date or arizona_today()
    return pipeline_funnel(db, card_date=target)


@router.get("/dashboard-summary", response_model=DashboardSummary)
def dashboard_summary(
    db: Session = Depends(get_db),
    date: str | None = None,
    history: bool = False,
) -> DashboardSummary:
    target = None if history else (date or arizona_today())
    active_signals_raw = _live_signals(
        db,
        limit=10,
        card_date=target,
        active_only=True,
        exclude_resolved=True,
    )
    active_signals = [_enrich_signal(s) for s in active_signals_raw]

    top_traders = list(
        db.scalars(
            select(Trader).order_by(desc(Trader.trust_score)).limit(10)
        )
    )

    # "Highest conviction" markets = those carrying the highest-scored signals.
    top_market_ids = []
    seen_market_ids: set[int] = set()
    for signal in active_signals_raw:
        if signal.market_id in seen_market_ids:
            continue
        seen_market_ids.add(signal.market_id)
        top_market_ids.append(signal.market_id)
        if len(top_market_ids) >= 10:
            break
    if top_market_ids:
        markets = list(db.scalars(select(Market).where(Market.id.in_(top_market_ids))))
        # preserve order from the score query
        order = {mid: idx for idx, mid in enumerate(top_market_ids)}
        markets.sort(key=lambda m: order.get(m.id, 999))
    else:
        markets = list(db.scalars(select(Market).limit(10)))

    recent_alerts = list_alerts(db, limit=10, date=target, history=history)

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

    # Simulated PnL is a placeholder — no trades are executed.
    simulated_pnl = 0.0

    return DashboardSummary(
        active_signals=active_signals,
        top_traders=top_traders,
        highest_conviction_markets=markets,
        recent_alerts=recent_alerts,
        simulated_pnl_usd=simulated_pnl,
        watchlist_health=health,
    )
