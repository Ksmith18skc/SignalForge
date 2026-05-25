"""FalconProvider — Quickstart agent dispatcher only.

Falcon exposes a single POST endpoint that takes an `agent_id` + `params` and
returns whatever that agent produces. There are many `agent_id`s; SignalForge
currently wires three:

  * 584 — Leaderboard (top traders)
  * 581 — Wallet 360 (single-wallet recent activity)
  * 556 — Recent trades (single-wallet trade history)

Other capabilities the marketing site advertised (cross-market comparison,
sentiment signals, market metadata) don't have proven agent_ids yet — those
BaseProvider methods route directly to MockProvider so they don't make Falcon
HTTP calls and don't pollute the health success rate.

Use `/falcon-test` to dump the raw shape of any agent response, then refine the
parsers below to extract the fields the rest of SignalForge consumes.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from app.providers.base import BaseProvider, ProviderSource
from app.providers.mock import MockProvider

logger = logging.getLogger(__name__)


def _unwrap_first_row(envelope: Any) -> dict[str, Any]:
    """Walk Falcon's nested `{data: {results: [...]}}` envelope to the first row.

    Falcon agents return data in shapes like:
      {timestamp, params, pagination, data: {results: [{...row...}]}}
      {data: [{...row...}]}
      {results: [{...row...}]}

    This walks up to 4 levels of `data`/`results`/`summary`/`stats` keys and
    stops at the first leaf dict.
    """
    cur: Any = envelope
    for _ in range(4):
        if not isinstance(cur, dict):
            break
        for k in ("data", "results", "summary", "stats"):
            v = cur.get(k)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                cur = v[0]
                break
            if isinstance(v, dict):
                cur = v
                break
        else:
            break  # no wrapper key found — we're at the leaf
    return cur if isinstance(cur, dict) else {}


def _unwrap_rows(envelope: Any) -> list[dict[str, Any]]:
    """Walk Falcon's envelope to the list of rows."""
    cur: Any = envelope
    for _ in range(4):
        if not isinstance(cur, dict):
            break
        for k in ("data", "results", "rows", "trades"):
            v = cur.get(k)
            if isinstance(v, list):
                return [r for r in v if isinstance(r, dict)]
            if isinstance(v, dict):
                cur = v
                break
        else:
            break
    return []


def _agg_win_rate(perf_by_category: str | list[dict[str, Any]] | None) -> float | None:
    """Falcon returns `performance_by_category` as a JSON-encoded string. Parse
    it and compute a trade-count-weighted average win_rate."""
    if not perf_by_category:
        return None
    if isinstance(perf_by_category, str):
        try:
            perf_by_category = json.loads(perf_by_category)
        except (ValueError, TypeError):
            return None
    if not isinstance(perf_by_category, list):
        return None

    total_trades = 0
    weighted = 0.0
    for entry in perf_by_category:
        if not isinstance(entry, dict):
            continue
        trades = entry.get("total_trades") or 0
        wr = entry.get("win_rate")
        if not trades or wr is None:
            continue
        total_trades += trades
        weighted += float(wr) * trades
    return round(weighted / total_trades, 4) if total_trades else None


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _risk_level_score(value: Any) -> float:
    levels = {
        "LOW": 1.0,
        "MEDIUM": 0.65,
        "MODERATE": 0.65,
        "HIGH": 0.25,
        "VERY_HIGH": 0.1,
    }
    return levels.get(str(value or "").strip().upper(), 0.5)


def _wallet360_trust_score(row: dict[str, Any]) -> float:
    """Map Wallet 360 risk/performance metrics to a 0-100 trust score."""
    statistical_confidence = _clip(_to_float(row.get("statistical_confidence"), 0.5) or 0.5)
    risk_score = _risk_level_score(row.get("risk_level"))
    win_rate = _clip(
        _to_float(row.get("win_rate_last_30day"), None)
        or _to_float(row.get("win_rate"), 0.5)
        or 0.5
    )
    profit_factor = _clip((_to_float(row.get("profit_factor"), 1.0) or 1.0) / 2.0)
    roi = _clip(((_to_float(row.get("roi"), 0.0) or 0.0) + 25.0) / 75.0)
    trade_depth = _clip(((_to_float(row.get("total_trades"), 0.0) or 0.0) ** 0.5) / 100.0)
    diversity = _clip((_to_float(row.get("category_diversity_score"), 1.0) or 1.0) / 4.0)
    concentration = 1.0 - _clip(_to_float(row.get("market_concentration_ratio"), 0.5) or 0.5)

    penalty = 0.0
    for flag in (
        "suspicious_win_rate_flag",
        "sybil_risk_flag",
        "timing_anomaly_flag",
        "position_size_volatility_flag",
        "single_market_dependence_flag",
    ):
        if row.get(flag) is True:
            penalty += 0.08
    penalty += 0.15 * _clip((_to_float(row.get("sybil_risk_score"), 0.0) or 0.0) / 100.0)

    total = (
        0.25 * statistical_confidence
        + 0.20 * risk_score
        + 0.15 * win_rate
        + 0.12 * profit_factor
        + 0.10 * roi
        + 0.08 * trade_depth
        + 0.05 * diversity
        + 0.05 * concentration
        - penalty
    )
    return round(_clip(total) * 100.0, 2)


# --------------------------------------------------------------------------
# Health tracking — module-level so /health and dashboard can read it
# --------------------------------------------------------------------------


@dataclass
class FalconHealth:
    base_url: str = ""
    calls: int = 0
    successes: int = 0
    last_error: str | None = None
    last_status_code: int | None = None
    last_endpoint: str | None = None
    last_agent_id: int | None = None
    last_call_at: datetime | None = None
    # last completed scan window
    last_scan_at: datetime | None = None
    last_scan_calls: int = 0
    last_scan_successes: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / self.calls if self.calls else 0.0

    @property
    def healthy(self) -> bool:
        """True if Falcon is delivering data on this scan."""
        return self.calls > 0 and self.success_rate >= 0.5

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "calls": self.calls,
            "successes": self.successes,
            "success_rate": round(self.success_rate, 3),
            "healthy": self.healthy,
            "last_error": self.last_error,
            "last_status_code": self.last_status_code,
            "last_endpoint": self.last_endpoint,
            "last_agent_id": self.last_agent_id,
            "last_call_at": self.last_call_at.isoformat() if self.last_call_at else None,
            "last_scan_at": self.last_scan_at.isoformat() if self.last_scan_at else None,
            "last_scan_calls": self.last_scan_calls,
            "last_scan_successes": self.last_scan_successes,
        }


_health = FalconHealth()
_lock = threading.Lock()


def get_falcon_health() -> FalconHealth:
    """Snapshot of the latest Falcon call stats. Safe to call from any thread."""
    with _lock:
        return FalconHealth(**_health.__dict__)


def begin_scan_window(base_url: str) -> None:
    """Reset per-scan counters at the start of a scan pass."""
    with _lock:
        _health.base_url = base_url
        _health.calls = 0
        _health.successes = 0


def end_scan_window() -> tuple[int, int]:
    """Freeze the current per-scan counters into `last_scan_*` and return them."""
    with _lock:
        _health.last_scan_at = datetime.utcnow()
        _health.last_scan_calls = _health.calls
        _health.last_scan_successes = _health.successes
        return _health.calls, _health.successes


def _record(
    ok: bool,
    endpoint: str,
    agent_id: int | None = None,
    status: int | None = None,
    error: str | None = None,
) -> None:
    with _lock:
        _health.calls += 1
        _health.last_endpoint = endpoint
        _health.last_agent_id = agent_id
        _health.last_call_at = datetime.utcnow()
        if status is not None:
            _health.last_status_code = status
        if ok:
            _health.successes += 1
        else:
            _health.last_error = (error or "unknown")[:240]


# --------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------


def _is_wallet_address(s: str | None) -> bool:
    """0x-prefixed 40-char hex — enough to skip Falcon calls for nicknames."""
    return bool(s and s.startswith("0x") and len(s) == 42)


class FalconProvider(BaseProvider):
    source = ProviderSource.FALCON

    # The only endpoint we hit. Everything is dispatched via agent_id.
    PATH_AGENT = "/api/v2/semantic/retrieve/parameterized"

    # Known agents.
    AGENT_LEADERBOARD = 584
    AGENT_WALLET_360 = 581
    AGENT_RECENT_TRADES = 556

    def __init__(self, api_key: str | None, base_url: str, timeout: float = 10.0) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._fallback = MockProvider()
        with _lock:
            _health.base_url = self._base_url

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise RuntimeError(
                "FalconProvider invoked without an API key. "
                "Ingestion should fall back to MockProvider when credentials are missing."
            )
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "SignalForge/0.1",
        }

    async def query_agent(
        self,
        agent_id: int,
        params: dict[str, Any] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any] | None:
        """POST to /api/v2/semantic/retrieve/parameterized. Returns parsed JSON
        on success, None on any failure (HTTP error, non-JSON, or Falcon's
        200-with-error-body envelope). Failures are recorded in the health tracker.
        """
        body = {
            "agent_id": agent_id,
            "params": params or {},
            "pagination": {"limit": limit, "offset": offset},
            "formatter_config": {"format_type": "raw"},
        }
        url = f"{self._base_url}{self.PATH_AGENT}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=body, headers=self._headers())
            status = resp.status_code
            if status >= 400:
                _record(
                    False, self.PATH_AGENT, agent_id=agent_id,
                    status=status, error=resp.text[:200],
                )
                return None
            try:
                data = resp.json()
            except ValueError:
                _record(
                    False, self.PATH_AGENT, agent_id=agent_id,
                    status=status, error="non-JSON response",
                )
                return None
            # Falcon-style envelope error (HTTP 200 + error body).
            if isinstance(data, dict) and (
                data.get("error") is True
                or (isinstance(data.get("msg"), str) and "not found" in data["msg"].lower())
            ):
                msg = data.get("msg") or data.get("message") or str(data)[:200]
                _record(
                    False, self.PATH_AGENT, agent_id=agent_id,
                    status=status, error=msg[:200],
                )
                return None
            _record(True, self.PATH_AGENT, agent_id=agent_id, status=status)
            return data
        except httpx.HTTPError as exc:
            _record(
                False, self.PATH_AGENT, agent_id=agent_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            return None

    # ----------------------------------------------------------------------
    # Named Falcon agents
    # ----------------------------------------------------------------------

    async def get_leaderboard(
        self,
        window_days: int = 15,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any] | None:
        """Top traders by ROI / PnL (agent 584). Returns raw envelope or None."""
        return await self.query_agent(
            self.AGENT_LEADERBOARD,
            params={"window_days": str(window_days)},
            limit=limit,
            offset=offset,
        )

    async def get_wallet_360(
        self,
        wallet: str,
        window_days: int = 3,
        limit: int = 100,
    ) -> dict[str, Any] | None:
        """Wallet 360 deep-dive (agent 581). Returns raw envelope or None."""
        if not _is_wallet_address(wallet):
            return None
        return await self.query_agent(
            self.AGENT_WALLET_360,
            params={"proxy_wallet": wallet, "window_days": str(window_days)},
            limit=limit,
        )

    async def get_recent_trades(
        self,
        wallet: str,
        limit: int = 50,
    ) -> dict[str, Any] | None:
        """Recent wallet trades (agent 556). Returns raw envelope or None."""
        if not _is_wallet_address(wallet):
            return None
        return await self.query_agent(
            self.AGENT_RECENT_TRADES,
            params={"proxy_wallet": wallet},
            limit=limit,
        )

    # ----------------------------------------------------------------------
    # BaseProvider interface
    # ----------------------------------------------------------------------

    async def get_trader_stats(self, wallet: str) -> dict[str, Any]:
        """Falcon path: agent 581. Mock fallback if wallet isn't an address
        or the response shape doesn't carry what we need.

        Real Wallet 360 fields (per /falcon-test):
          pnl_last_30day, roi, num_markets_traded, risk_level,
          performance_by_category (JSON-encoded list of category breakdowns),
          ...
        We map these into SignalForge's trader columns.
        """
        if not _is_wallet_address(wallet):
            return await self._fallback.get_trader_stats(wallet)

        raw = await self.get_wallet_360(wallet)
        if not raw:
            return await self._fallback.get_trader_stats(wallet)

        row = _unwrap_first_row(raw)
        if not row:
            return await self._fallback.get_trader_stats(wallet)

        # Build a category-strength map from the per-category breakdown.
        perf = row.get("performance_by_category")
        if isinstance(perf, str):
            try:
                perf = json.loads(perf)
            except (ValueError, TypeError):
                perf = None
        category_strengths: dict[str, float] = {}
        if isinstance(perf, list):
            for entry in perf:
                if not isinstance(entry, dict):
                    continue
                cat = entry.get("category")
                wr = entry.get("win_rate")
                if cat and wr is not None:
                    category_strengths[cat] = round(float(wr), 4)

        return {
            "wallet": wallet,
            "trust_score": _wallet360_trust_score(row),
            "total_pnl": row.get("pnl_last_30day") or row.get("total_pnl") or row.get("pnl"),
            "win_rate": _agg_win_rate(perf),
            "seven_day_return": row.get("roi"),
            "trader_rank": row.get("rank") or row.get("trader_rank"),
            "total_positions": row.get("num_markets_traded")
            or row.get("markets_traded")
            or row.get("active_positions"),
            "net_worth": row.get("avg_market_exposure") or row.get("net_worth"),
            "category_strengths": category_strengths,
            "source": self.source.value,
            "_raw": raw,
        }

    async def get_trader_trades(self, wallet: str, limit: int = 50) -> list[dict[str, Any]]:
        """Falcon path: agent 556.

        When Falcon is configured as the primary provider, do not synthesize
        trades on failure. MockProvider is already used as the primary when no
        Falcon key exists; returning mock rows here would pollute live scans.
        """
        if not _is_wallet_address(wallet):
            return []

        raw = await self.get_recent_trades(wallet, limit=limit)
        if not raw:
            return []

        rows = _unwrap_rows(raw)
        trades: list[dict[str, Any]] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            try:
                trades.append(
                    {
                        "wallet": wallet,
                        "market_slug": r.get("market_slug") or r.get("slug"),
                        "market_title": r.get("market_title") or r.get("question") or r.get("title"),
                        "category": r.get("category"),
                        "side": str(r.get("side") or "BUY").upper(),
                        "outcome": (
                            str(r.get("outcome"))
                            if r.get("outcome") is not None
                            else None
                        ),
                        "price": float(r.get("price") or r.get("avg_price") or 0.5),
                        "size_usd": float(
                            r.get("size_usd") or r.get("notional") or r.get("size") or 0.0
                        ),
                        "timestamp": r.get("timestamp") or r.get("ts"),
                        "external_id": r.get("trade_id") or r.get("id"),
                        "source": self.source.value,
                    }
                )
            except (TypeError, ValueError) as exc:
                logger.debug("skipping malformed Falcon trade row: %s", exc)

        return trades

    # --- everything else: direct mock — no Falcon HTTP call -----------------
    # These have no proven agent_ids yet. Routing them through Falcon would
    # only generate failures that drag down the success rate without changing
    # the resulting (mock) data.

    async def get_market_data(self, market_slug: str) -> dict[str, Any]:
        return await self._fallback.get_market_data(market_slug)

    async def get_orderbook(self, market_slug: str) -> dict[str, Any]:
        return await self._fallback.get_orderbook(market_slug)

    async def get_cross_market_comparison(self, topic: str) -> list[dict[str, Any]]:
        return await self._fallback.get_cross_market_comparison(topic)

    async def get_sentiment_signals(self, market_slug: str) -> dict[str, Any]:
        return await self._fallback.get_sentiment_signals(market_slug)

    async def list_active_markets(self, limit: int = 25) -> list[dict[str, Any]]:
        return await self._fallback.list_active_markets(limit)
