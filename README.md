# SignalForge

> Tracked-wallet **consensus intelligence** for Polymarket / Kalshi.
> **Alert-only — no real trades are placed.**

SignalForge watches a curated list of smart wallets, ingests their trades through
a pluggable provider layer (Falcon, Polymarket, Kalshi), and surfaces the one
thing that matters: **markets where 2+ tracked wallets are aligned on the same
side** — across any market, any sport. That's the consensus the dashboard is
built around.

---

## ⚠️ Not financial advice

This is research / tooling. It does **not** execute trades. Defaults are
`ENABLE_AUTO_TRADING=false` and `DEFAULT_COPY_MODE=alert_only`, and no order
routing exists. Don't rely on this for investment decisions.

---

## Project layout

```
signalforge/
  app/
    main.py                 FastAPI app entry point
    config.py               Pydantic settings (loads .env)
    db.py                   SQLAlchemy engine + session
    models.py               ORM models (Trader, Market, Trade, Signal, Alert, MarketSnapshot)
    schemas.py              Pydantic API schemas
    providers/              Falcon (primary), Polymarket, Kalshi, Mock fallback
    services/
      ingestion.py          Provider -> DB normalization (Trade rows)
      scanner.py            Run-once + background loop
      signal_engine.py      Rule-based signal generation
      scoring.py            Weighted score (0-100)
      alerts.py             Console + Discord/Telegram/email channels
      tracked_wallet_positions.py   Raw tracked-wallet live positions (any market, no score gate)
      pipeline_diagnostics.py       "where did my data go?" funnel
      wallet_market_resolver.py     Market-slug parsing + URLs
    utils/dashboard_format.py       Pure display helpers + wallet_consensus_groups()
    api/routes.py           FastAPI routes
  scripts/
    seed.py                 Seed the watchlist
    run_worker.py           Standalone scanner loop
  dashboard.py              Streamlit wallet-consensus dashboard
  tests/
```

---

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate              # Windows (or: source .venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env                # optional — every credential is optional
```

Missing keys cause that provider to fall back to `MockProvider`, so SignalForge
runs end-to-end with zero config.

---

## Running it

Two terminals: one for the backend, one for the dashboard.

### 1. Seed the watchlist (once)

```bash
python -m scripts.seed
```

### 2. Run the backend

```bash
uvicorn app.main:app --reload
```

API at `http://localhost:8000`. The lifespan handler starts a background scanner
on `SIGNALFORGE_SCAN_INTERVAL_SECONDS` so trades + signals keep flowing.

Key endpoints:

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET | `/health` | Liveness probe |
| GET | `/ready` | Readiness + provider/alert-channel health |
| GET | `/traders` · POST · DELETE | The tracked-wallet watchlist |
| GET | `/markets` | Active markets |
| GET | `/signals` | Generated signals |
| GET | `/tracked-wallet-positions` | Raw tracked-wallet live positions (any market, no score gate) |
| GET | `/tracked-wallet-positions/debug` | Per-row rejection diagnostics |
| GET | `/alerts` | Dispatched alerts |
| POST | `/run-scan` | Trigger an ingest + signal + alert pass |
| GET | `/dashboard-summary` | Top signals, traders, markets, alerts, watchlist health |
| GET | `/dashboard/pipeline-debug` | Per-stage funnel + `drop_stage` (why is consensus empty?) |

Swagger UI at `http://localhost:8000/docs`.

### 3. Run the dashboard

```bash
streamlit run dashboard.py
```

Opens at `http://localhost:8501`. Point it at a non-default backend with
`SIGNALFORGE_API_URL=http://host:port streamlit run dashboard.py`.

Tabs:

- **Aligned Consensus** — the centerpiece: one card per market where 2+ tracked
  wallets share a side, with wallet names, sizes, entries, and a market link.
- **All Positions** — every tracked-wallet live position (any market).
- **Watchlist** — the tracked wallets with trust score / win rate / PnL.
- **Alerts** — recent dispatched alerts.
- **Diagnostics** — the pipeline funnel that explains an empty consensus view.

### Scanner-only mode

```bash
python -m scripts.run_worker
```

---

## How consensus works

`/tracked-wallet-positions` returns every tracked-wallet trade plausibly on
today's card — **sport-agnostic, no score threshold**. The dashboard feeds those
into `wallet_consensus_groups()` ([app/utils/dashboard_format.py](app/utils/dashboard_format.py)),
which groups by `(market_id, side, outcome)` and keeps groups where **≥ 2 distinct
wallets** agree. That's the aligned consensus — independent of any model edge.

If the view is empty, `/dashboard/pipeline-debug` names the earliest funnel stage
that hit zero (no trades → card-date mismatch → trade window → no alignment).

---

## How signals are scored

Each candidate produces a 0-100 score with default weights:

| Component | Weight |
| --------- | ------ |
| Wallet quality | 35% |
| Multi-wallet consensus | 25% |
| Liquidity | 15% |
| Entry timing | 15% |
| Price inefficiency | 10% |

Override any via `SIGNALFORGE_SCORING__*` env vars.

---

## Tests

```bash
pytest -q
```
