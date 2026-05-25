# SignalForge

> Semi-automated prediction-market intelligence for Polymarket / Kalshi.
> **Alert-only MVP — no real trades are placed.**

SignalForge watches a curated list of smart wallets, ingests trade + market
data through a pluggable provider layer (Falcon, Polymarket Analytics,
Polycopy, Kalshi), scores opportunities with a configurable weighted model,
and ships alerts to console / Discord / Telegram / email channels.

---

## ⚠️ Not financial advice

This project is research / tooling. It does **not** execute trades. The MVP
ships with `ENABLE_AUTO_TRADING=false` and `DEFAULT_COPY_MODE=alert_only`, and
the risk service force-downgrades any `live` request unless that flag is
flipped on. Even then, no order routing is wired up. Don't rely on this for
investment decisions.

---

## Project layout

```
signalforge/
  app/
    main.py                 FastAPI app entry point
    config.py               Pydantic settings (loads .env)
    db.py                   SQLAlchemy engine + session
    models.py               ORM models
    schemas.py              Pydantic API schemas
    providers/
      base.py               Provider interface
      falcon.py             Falcon API (primary)
      polymarket.py         Polymarket Analytics
      kalshi.py             Kalshi
      mock.py               Fallback synthetic data
    services/
      ingestion.py          Provider -> DB normalization
      scoring.py            Weighted score (0-100)
      risk.py               Position/daily/market caps
      signal_engine.py      Rule-based signal generation
      alerts.py             Console + placeholder channels
      scanner.py            Run-once + background loop
    api/
      routes.py             FastAPI routes
    utils/
      logging.py
  scripts/
    seed.py                 Seed the watchlist
    run_worker.py           Standalone scanner loop
  tests/
    test_scoring.py
    test_risk.py
    test_signal_engine.py
  dashboard.py              Streamlit dashboard (dark quant theme)
  .env.example
  requirements.txt
```

---

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                # then edit if you have API keys
```

Every credential in `.env` is optional. Missing keys cause that provider to
fall back to `MockProvider` so SignalForge runs end-to-end with zero config.

### Required `.env` values

**Zero required to run** — defaults in [`app/config.py`](app/config.py) cover
everything. The interesting ones to set:

```env
# storage (override only if you don't want the default SQLite file)
SIGNALFORGE_DATABASE_URL=sqlite:///./signalforge.db

# Falcon API — the only credential that changes the "Source" badge
# in the dashboard from "Mock" to "Falcon (live)"
SIGNALFORGE_FALCON_API_KEY=your-key-here
SIGNALFORGE_FALCON_BASE_URL=https://narrative.agent.heisenberg.so

# scanner cadence + signal noise floor
SIGNALFORGE_SCAN_INTERVAL_SECONDS=60
SIGNALFORGE_SIGNAL_SCORE_THRESHOLD=60

# trading posture — keep alert_only / false for the MVP
SIGNALFORGE_DEFAULT_COPY_MODE=alert_only
SIGNALFORGE_ENABLE_AUTO_TRADING=false
```

The full list (alerts, Polymarket, Kalshi, risk caps, scoring weights) is in
[`.env.example`](.env.example) and the [environment variables](#environment-variables)
table below.

> **Heads up:** every SignalForge env var starts with `SIGNALFORGE_` so it
> never collides with unrelated env vars in your shell (e.g. another project's
> `DATABASE_URL`).

---

## Seeding the watchlist

The seed script inserts the operator's Polymarket Analytics watchlist:

```bash
python -m scripts.seed
```

Seeded traders (default `copy_mode=alert_only`):
LaBradfordSmith22, surfandturf, HomeRunHazard, bananawoin, VeryLucky888,
Soarin22, ewelmealt, pinkblanket, bambambole, ooohhyeah.

---

## Running it

You'll typically want **two terminals**: one for the FastAPI backend and one
for the Streamlit dashboard. The seed step is one-time.

### 1. Seed the watchlist (once)

```bash
python -m scripts.seed
```

### 2. Run the backend

```bash
uvicorn app.main:app --reload
```

The API runs at `http://localhost:8000` and exposes:

| Method | Path                 | Purpose                                            |
| ------ | -------------------- | -------------------------------------------------- |
| GET    | `/health`            | Liveness + which providers have credentials       |
| GET    | `/traders`           | List watched traders                              |
| POST   | `/traders`           | Add a trader to the watchlist                     |
| GET    | `/markets`           | List active markets                               |
| GET    | `/signals`           | Recent generated signals                          |
| GET    | `/alerts`            | Recent dispatched alerts                          |
| POST   | `/run-scan`          | Trigger a full ingest + signal + alert pass       |
| GET    | `/dashboard-summary` | Top signals, traders, markets, alerts, health     |

Swagger UI at `http://localhost:8000/docs`. The FastAPI lifespan handler also
starts a background scanner thread on `SIGNALFORGE_SCAN_INTERVAL_SECONDS` so
signals keep flowing while the API runs.

### 3. Run the dashboard

In a second terminal:

```bash
streamlit run dashboard.py
```

Opens at `http://localhost:8501`. The dashboard fetches everything from the
FastAPI backend — start the backend first.

What you get:

- **Source badge** — green "Falcon (live)" when `SIGNALFORGE_FALCON_API_KEY`
  is set, amber "Mock — no Falcon key" otherwise.
- **Metric cards** — active signals, alerts dispatched, watched traders,
  simulated PnL.
- **Filters** (sidebar) — score range, trader, market, source.
- **Highest conviction** table — top 10 signals by score with progress bars.
- **Tabs** — Signals / Watched Traders / Alerts / Health (raw JSON).
- **Run scan now** button — triggers `POST /run-scan` and refreshes.

Point the dashboard at a non-default backend with
`SIGNALFORGE_API_URL=http://host:port streamlit run dashboard.py`.

### Scanner-only mode

If you don't want the HTTP server but still want signals generated on a loop:

```bash
python -m scripts.run_worker
```

---

## Tests

```bash
pytest -q
```

Tests cover scoring math, risk caps, and the signal-generation rules. Each test
runs against an in-memory SQLite database for isolation.

---

## Environment variables

Defined in [`.env.example`](.env.example). **Every var is prefixed with
`SIGNALFORGE_`** so it never collides with unrelated env vars in your shell
(e.g. another project's `DATABASE_URL`). Key flags:

| Variable                              | Default                                 | Notes                                                  |
| ------------------------------------- | --------------------------------------- | ------------------------------------------------------ |
| `SIGNALFORGE_ENVIRONMENT`             | `dev`                                   | `dev` / `staging` / `prod`                             |
| `SIGNALFORGE_DATABASE_URL`            | `sqlite:///./signalforge.db`            | Any SQLAlchemy URL                                     |
| `SIGNALFORGE_DEFAULT_COPY_MODE`       | `alert_only`                            | `disabled` / `alert_only` / `paper` / `live`           |
| `SIGNALFORGE_ENABLE_AUTO_TRADING`     | `false`                                 | Force-disabled in MVP — flip only when wiring brokers  |
| `SIGNALFORGE_SCAN_INTERVAL_SECONDS`   | `60`                                    | Background scanner cadence                             |
| `SIGNALFORGE_SIGNAL_SCORE_THRESHOLD`  | `60`                                    | Minimum score to persist + alert                       |
| `SIGNALFORGE_FALCON_API_KEY`          | _(blank)_                               | If missing, MockProvider is used as primary            |
| `SIGNALFORGE_FALCON_BASE_URL`         | `https://narrative.agent.heisenberg.so` |                                                        |
| `SIGNALFORGE_POLYMARKET_API_KEY`      | _(blank)_                               |                                                        |
| `SIGNALFORGE_KALSHI_API_KEY`          | _(blank)_                               |                                                        |
| `SIGNALFORGE_DISCORD_WEBHOOK_URL`     | _(blank)_                               | Placeholder — console alerts only in MVP               |
| `SIGNALFORGE_TELEGRAM_BOT_TOKEN`      | _(blank)_                               | Placeholder                                            |
| `SIGNALFORGE_ALERT_EMAIL_TO`          | _(blank)_                               | Placeholder                                            |

Nested settings use `__` as the separator
(e.g. `SIGNALFORGE_RISK__BANKROLL_USD=10000`).

---

## How signals are scored

Each signal candidate produces a `ScoreBreakdown` on a 0-100 scale with the
default weights:

| Component                | Weight |
| ------------------------ | ------ |
| Wallet quality           | 35%    |
| Multi-wallet consensus   | 25%    |
| Liquidity                | 15%    |
| Entry timing             | 15%    |
| Price inefficiency       | 10%    |

Override any of these via `SCORING__*` env vars without touching code.

Every signal records:
`source` (Falcon / PolymarketAnalytics / Polycopy / Mock), `wallet`,
`trader_nickname`, `market`, `side`, `entry_price`, `size_usd`,
`confidence`, and a human-readable `reason`.

---

## Wiring Falcon for real

`app/providers/falcon.py` is wired against the real Falcon endpoints at
`https://narrative.agent.heisenberg.so`:

| SignalForge call               | Falcon endpoint                                |
| ------------------------------ | ---------------------------------------------- |
| `get_trader_stats(wallet)`     | `POST /v2/traders/stats`                       |
| `get_trader_trades(wallet)`    | `POST /api/v2/semantic/retrieve/parameterized` with `agent_id=581` (Wallet 360) |
| `get_market_data(slug)`        | `POST /v2/markets/retrieve` with `agent_id=574` |
| `list_active_markets(limit)`   | `POST /v2/markets/retrieve` with `agent_id=574`, broad params |
| `get_cross_market_comparison`  | `POST /v2/cross/compare`                       |
| `get_sentiment_signals(slug)`  | `POST /v2/signals/sentiment`                   |
| `get_orderbook(slug)`          | _no documented endpoint yet → mock fallback_   |

If a call fails for any reason — wrong base URL, expired key, undocumented
endpoint — the method silently falls back to `MockProvider` so the rest of
the pipeline keeps producing signals. The scanner logs one summary line per
pass (`Falcon: 12/15 calls succeeded`) instead of one warning per call.

### Knowing whether Falcon is actually working

`GET /health` returns per-call health stats:

```json
{
  "providers": {
    "falcon": {
      "configured": true,
      "healthy": true,
      "calls": 47,
      "successes": 45,
      "success_rate": 0.957,
      "last_status_code": 200,
      "last_endpoint": "/v2/traders/stats",
      "last_scan_calls": 15,
      "last_scan_successes": 14
    }
  }
}
```

The dashboard badge reads this directly:

- **Green** "Source: Falcon · 14/15 ok" — calls are succeeding
- **Amber** "Source: Mock · Falcon configured but 0/15 ok" — key is set but
  every call is failing (with the HTTP status + last endpoint shown right
  underneath)
- **Amber** "Source: Mock · no Falcon key" — no key configured

### Extending to new Falcon agent_ids

To use a Falcon endpoint that isn't wired up yet (the docs mention many
more `agent_id`s than the two we use here), call the generic dispatcher:

```python
from app.providers.falcon import FalconProvider

falcon = FalconProvider(api_key=..., base_url=...)
data = await falcon.query_agent(
    agent_id=999,
    params={"some_param": "value"},
    limit=50,
)
```

This posts to `/api/v2/semantic/retrieve/parameterized` with the canonical
Quickstart body shape.
