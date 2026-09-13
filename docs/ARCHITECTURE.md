# Architecture

A modular, autonomous **multi-agent** trading system. Seven independent Python processes
("agents") never import or call each other. They coordinate through Redis: typed
Pydantic messages on named pub/sub channels for events, plus a few shared keys for state
(see [Shared Redis keys](#shared-redis-keys)). Any agent can be added, removed, or
restarted independently.

```
                         ┌──────────────────── Redis pub/sub bus ────────────────────┐
                         │                                                            │
 ┌───────────────┐  OHLCV│   ┌────────────────────┐  signal  ┌───────────────┐ proposal
 │ Market Data   │───────┼──▶│ Technical Analysis │─────────▶│  Decision     │───────┐
 │ (DataSource)  │       │   │ (indicators + ML)  │          │ (aggregate)   │       │
 └───────────────┘       │   └────────────────────┘          └───────────────┘       ▼
        ▲ live public    │   ┌────────────────────┐          ┌───────────────┐  ┌──────────┐
        │ keyless feed   │   │ Sentiment (stub)   │          │ Risk          │◀─┤ proposal │
        │                │   └────────────────────┘          │ (10 checks)   │  └──────────┘
 ┌───────────────┐       │                                    └───────┬───────┘
 │ Monitoring    │◀──────┘   heartbeats / health                      │ approved
 │ (health 8081) │                                           ┌────────▼────────┐
 └───────────────┘                                           │ Execution       │
        ▲                                                     │ (PaperBroker)   │
        │  FastAPI control plane (8000) + WS                  └─────────────────┘
        └───────────────────────────────────────────────────────────┘
```

## The agents

| Agent | Responsibility |
|-------|----------------|
| **Market Data** | Polls public OHLCV every 5 minutes via the pluggable `DataSource` (keyless CCXT by default), keeps only closed candles (the newest row an exchange returns is still forming), de-duplicates them, publishes them, and persists to TimescaleDB. Also publishes a ticker every 30 s. |
| **Technical Analysis** | Maintains rolling candle buffers, computes RSI / MACD / Bollinger / EMA (+ ATR, volume) into a confidence-scored `TechnicalSignal`. Optionally uses an XGBoost model (off by default — see below). |
| **Decision** | Aggregates signals and sizes a `TradeProposal`. The EMA-50 trend filter applies only to ML signals, which carry `ema_50_dist`; rule-based signals are not filtered. |
| **Risk** | Runs ordered checks (circuit breaker, halt, daily loss, drawdown, staleness, existing position, max positions, sizing, minimum size) and emits an approved/modified/rejected `RiskAssessment`. One position per symbol: a same-side proposal is rejected, an opposite-side proposal closes the position. Approval reserves the size; the fill reconciles it. |
| **Execution** | Routes approved trades to a broker behind the `ExecutionBroker` interface — `PaperBroker` (default, simulated fills with slippage + fees) or the optional local MT5 bridge. Checks stop-loss / take-profit every 30 s against the latest ticker price and publishes those closes like any other fill. |
| **Monitoring** | Subscribes to every heartbeat, detects stale agents, serves a health endpoint (`:8081`), and can alert. |
| **Sentiment** | A disabled stub (the seventh agent); ships off, idles with heartbeats. |

Plus a **FastAPI control plane** (`:8000`): REST + a WebSocket (`/ws/stream`) that fans out
the whole pipeline (signals → proposals → assessments → fills → heartbeats) to dashboards.

## Message channels

Channel names come from the `Channels` class only (never hardcoded strings):

- `market.ohlcv.{symbol}.{timeframe}` — closed candles
- `market.ticker.{symbol}` — last traded price (used by Execution for stop-loss / take-profit)
- `signal.technical.{symbol}` — technical signals
- `decision.proposal` · `risk.assessment` · `execution.result` — trade lifecycle
- `system.heartbeat` · `system.risk_override` · `system.alert` — system

Pub/sub is at-most-once: a message published while a subscriber is down is lost. Each
execution result carries a `result_id`, and Risk applies results idempotently.

## Shared Redis keys

Some state is shared through Redis keys rather than messages. These are the couplings to
keep in mind when changing an agent:

| Key | Written by | Read by | Lifetime |
|-----|-----------|---------|----------|
| `paper_portfolio:cash`, `paper_portfolio:positions` | PaperBroker | PaperBroker (restore), API | No TTL — state |
| `execution:balance`, `execution:positions` | Execution | API | 300 s |
| `execution:history` | Execution | API | Last 100 results |
| `signal:technical:{symbol}:latest` | Decision | Execution (fallback price) | 300 s |
| `portfolio:paper:value_usd` | Risk | Dashboards | No TTL |
| `system:trading_halted` | API `/control/halt`, Risk emergency halt | Every agent at startup | Until `/control/resume` |

Redis runs with `maxmemory-policy volatile-lru`, so only keys with a TTL can be evicted;
the no-TTL state keys above are never evicted.

## Control plane

`POST /control/*` (halt, resume, command) requires an `X-API-Key` header matching
`CONTROL_API_KEY` and is disabled until that is set. nginx does not proxy `/control`, so
the public site cannot reach it. A halt is latched in Redis: agents that restart come back
halted until `/control/resume` clears the latch and the agents are restarted.

## The two pluggable seams

The system is decoupled at exactly two points, which is what makes it keyless-by-default and
MT5-optional:

1. **Data** — [`data_sources/`](../data_sources): a `DataSource` interface with
   `PublicExchangeSource` (keyless CCXT public data, **default**) and `MT5Source`
   (local-only, **read-only**). Selected by `DATA_SOURCE`.
2. **Execution** — [`agents/execution/`](../agents/execution): an `ExecutionBroker` interface
   with `PaperBroker` (simulated, **default**) and `MT5Bridge` (local terminal, optional).
   Selected by `EXECUTION_BROKER`.

The public/cloud demo uses only the keyless + paper implementations, so it runs with **zero
secrets**. MT5 is never required and never used in the cloud.

## Signal source: rules vs. ML

The TA agent can generate signals from **rules** (weighted indicators) or an **XGBoost ML**
model. As of Phase 5 the default is **rules** (`TA_USE_ML_SIGNALS=false`): a rigorous
walk-forward evaluation showed the ML path does not beat the rule baseline out-of-sample and
overtrades. The ML code and models remain — opt back in with `TA_USE_ML_SIGNALS=true`. See
[EVALUATION.md](EVALUATION.md) and [MODEL_CHANGES.md](MODEL_CHANGES.md).

## Repository layout

```
agents/         # the seven agents + BaseAgent (lifecycle, heartbeat, circuit breaker)
core/           # shared infra: config, logging, messaging (Redis), models, db, metrics
data_sources/   # pluggable market-data layer (public exchange / MT5)
api/            # FastAPI control plane + WebSocket + dashboard endpoints
backtest/       # evaluation harness (engine, metrics, benchmarks, ml_eval, report)
dashboard/      # bundled React + Vite dashboard (served by nginx in compose)
data/sample/    # committed offline OHLCV sample data (keyless, reproducible)
scripts/        # seed / train / evaluate / healthcheck / start helpers
infra/          # deploy IaC: Terraform (Oracle free tier), cloud-init, Caddy
infrastructure/ # build: agent Dockerfile, Postgres init, Prometheus config
tests/          # unit + integration (incl. the keyless end-to-end pipeline test)
```

## Design rules (invariants)

1. Agents never import from each other — only from `core/`.
2. All inter-agent communication goes through Redis via `core/messaging.py` — pub/sub for
   events, a few documented shared keys for state.
3. All messages are Pydantic models from `core/models/`.
4. Channel names come from the `Channels` class only.
5. `TRADING_MODE=paper` is the safe default — never live by accident.
6. The Execution agent talks only to the `ExecutionBroker` interface.
7. All config via `get_settings()` — never `os.environ` directly in agents.
