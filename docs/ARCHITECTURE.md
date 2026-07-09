# Architecture

A modular, autonomous **multi-agent** trading system. Seven independent Python processes
("agents") communicate **only over Redis pub/sub** — no agent imports another. Every
message is a typed Pydantic model on a named channel, so any agent can be added, removed,
or restarted independently.

```
                         ┌──────────────────── Redis pub/sub bus ────────────────────┐
                         │                                                            │
 ┌───────────────┐  OHLCV│   ┌────────────────────┐  signal  ┌───────────────┐ proposal
 │ Market Data   │───────┼──▶│ Technical Analysis │─────────▶│  Decision     │───────┐
 │ (DataSource)  │       │   │ (indicators + ML)  │          │ (aggregate)   │       │
 └───────────────┘       │   └────────────────────┘          └───────────────┘       ▼
        ▲ live public    │   ┌────────────────────┐          ┌───────────────┐  ┌──────────┐
        │ keyless feed   │   │ Sentiment (stub)   │          │ Risk          │◀─┤ proposal │
        │                │   └────────────────────┘          │ (8 checks)    │  └──────────┘
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
| **Market Data** | Polls public OHLCV via the pluggable `DataSource` (keyless CCXT by default), normalises + de-duplicates candles, publishes them, and persists to TimescaleDB. |
| **Technical Analysis** | Maintains rolling candle buffers, computes RSI / MACD / Bollinger / EMA (+ ATR, volume) into a confidence-scored `TechnicalSignal`. Optionally uses an XGBoost model (off by default — see below). |
| **Decision** | Aggregates signals, applies a trend filter, and sizes a `TradeProposal`. |
| **Risk** | Runs eight checks (circuit breaker, halt, daily-loss, drawdown, max positions, staleness, min size, final sizing) and emits an approved/modified/rejected `RiskAssessment`. |
| **Execution** | Routes approved trades to a broker behind the `ExecutionBroker` interface — `PaperBroker` (default, simulated fills with slippage + fees) or the optional local MT5 bridge. |
| **Monitoring** | Subscribes to every heartbeat, detects stale agents, serves a health endpoint (`:8081`), and can alert. |
| **Sentiment** | A disabled stub (the seventh agent); ships off, idles with heartbeats. |

Plus a **FastAPI control plane** (`:8000`): REST + a WebSocket (`/ws/stream`) that fans out
the whole pipeline (signals → proposals → assessments → fills → heartbeats) to dashboards.

## Message channels

Channel names come from the `Channels` class only (never hardcoded strings):

- `market.ohlcv.{symbol}.{timeframe}` — candles
- `signal.technical.{symbol}` — technical signals
- `decision.proposal` · `risk.assessment` · `execution.result` — trade lifecycle
- `system.heartbeat` · `system.risk_override` · `system.alert` — system

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
2. All inter-agent communication is Redis pub/sub via `core/messaging.py`.
3. All messages are Pydantic models from `core/models/`.
4. Channel names come from the `Channels` class only.
5. `TRADING_MODE=paper` is the safe default — never live by accident.
6. The Execution agent talks only to the `ExecutionBroker` interface.
7. All config via `get_settings()` — never `os.environ` directly in agents.
