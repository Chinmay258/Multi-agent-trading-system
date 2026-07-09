# Operational Runbook

This runbook is the operator's reference for running, monitoring, and
intervening in the trading system. It assumes you've followed the
quickstart in the repository README and have a working environment.

If you only have time for one thing each morning, do the
[Daily supervision checklist](#daily-supervision-checklist).

---

## 1. System startup

### Full stack via Docker Compose (recommended)

```bash
docker-compose up -d postgres redis           # infra first
docker-compose up -d                          # all agents + API
docker-compose logs -f --tail=50 risk_agent   # verify risk agent log spam
```

Bring up the observability profile if you also want Prometheus + Grafana:

```bash
docker-compose --profile monitoring up -d prometheus grafana
```

### Single-process launcher (for development)

```bash
python -m scripts.start
```

This boots every agent and the FastAPI server inside one event loop.
Useful for iterating on agent code without container restart churn.
**Do not use in production** — you lose per-agent restart isolation.

### Pre-flight

```bash
python scripts/healthcheck.py
```

Exits 0 if all four checks (Redis, Postgres, MonitoringAgent, FastAPI)
pass. Use this from cron, container `HEALTHCHECK`, and post-deploy
smoke scripts.

---

## 2. Verifying paper trading is active

Three independent sources should agree. If any disagree, **stop and
investigate** before letting trading continue.

1. Settings:
   ```bash
   curl -s http://localhost:8000/health | jq .trading_mode
   ```
   Must report `"paper"`.

2. Startup banner — the launcher prints `trading_mode : paper` on stdout.

3. ExecutionAgent log line on startup:
   ```
   "paper_broker_connected" initial_balance_usd=10000.0
   ```

If `TRADING_MODE=live` is set unexpectedly, `PaperBroker.connect()`
raises `RuntimeError` immediately and the ExecutionAgent quarantines —
this is the intended safety net.

---

## 3. Reading agent health

| Endpoint | Port | Returns |
| --- | --- | --- |
| `GET /health`   | 8081 | MonitoringAgent — system-wide health summary |
| `GET /agents`   | 8081 | Full per-agent heartbeat registry |
| `GET /health`   | 8000 | FastAPI — same data, exposed for ops tools |

A `200` from MonitoringAgent means **every registered agent has sent a
heartbeat within `heartbeat_timeout_seconds`** (default 30s). A `503`
means at least one agent is stale; the payload lists which ones.

```bash
# At-a-glance status
curl -s http://localhost:8081/health | jq '{status, stale_agents, agent_count}'
```

---

## 4. Pause / resume / halt trading

| Action | API call | Effect |
| --- | --- | --- |
| Pause   | `POST /control/command {"command":"PAUSE"}`    | Agents stop processing but stay alive |
| Resume  | `POST /control/command {"command":"RESUME"}`   | Agents resume from paused state |
| Halt    | `POST /control/halt`                            | Publishes `RiskOverride(requires_human_reset=true)`; ExecutionAgent stops placing orders until restarted |

Halt is **load-bearing** — it is the same path the Risk agent itself
takes when a limit is breached. After a halt:

1. Identify the cause (`GET /agents`, ExecutionAgent logs, Slack alerts).
2. Resolve the cause (close positions, adjust limits, fix the bug).
3. Restart the affected agents to clear `_trading_halted`.

There is intentionally no "resume from halt" API. Resetting a circuit
breaker requires the operator to restart agents — that ceremony is the
last guard before a runaway loop costs real money.

---

## 5. Interpreting Prometheus metrics

Scrape from `http://<host>:9090/metrics` (port from
`MONITORING_PROMETHEUS_PORT`).

| Metric | Type | What it tells you |
| --- | --- | --- |
| `trading_agent_up{agent}` | Gauge | 1 if agent is running, 0 if stopped |
| `trading_agent_errors_total{agent,error_type}` | Counter | Rate of unhandled exceptions; spikes mean trouble |
| `trading_messages_published_total{agent,channel}` | Counter | Bus throughput; flatlining = upstream stall |
| `trading_messages_consumed_total{agent,channel}` | Counter | Downstream consumption; lag = backed-up agent |
| `trading_orders_placed_total{symbol,side,mode}` | Counter | Order flow by symbol/side; per-side imbalance is interesting |
| `trading_orders_rejected_total{symbol,reason}` | Counter | Rejection rate; spikes by reason point at root cause |
| `trading_portfolio_value_usd` | Gauge | Cash + open position cost basis |
| `trading_portfolio_daily_pnl_usd` | Gauge | Daily realised PnL, reset at UTC midnight |
| `trading_open_positions_count` | Gauge | How many positions are open right now |
| `trading_signal_confidence{symbol}` | Gauge | Confidence of the most recent TA signal |
| `trading_data_age_seconds{symbol}` | Gauge | Seconds since the last candle for this symbol |
| `trading_signal_generation_seconds{symbol}` | Histogram | Latency of indicator math + signal scoring |
| `trading_order_fill_latency_seconds{mode}` | Histogram | Paper or live fill latency (paper is simulated) |
| `trading_indicator_computation_seconds` | Histogram | Per-indicator compute time |

Useful PromQL starters:

```promql
# All agents up?
min(trading_agent_up) == 1

# Error rate per agent over 5m
sum by (agent) (rate(trading_agent_errors_total[5m]))

# p95 signal latency
histogram_quantile(0.95, sum(rate(trading_signal_generation_seconds_bucket[5m])) by (le, symbol))

# Rejection breakdown
sum by (reason) (rate(trading_orders_rejected_total[15m]))
```

---

## 6. Daily supervision checklist

Run through this once per morning (or per shift). It takes ~3 minutes.

1. **All agents up.** `min(trading_agent_up) == 1` — anything else is
   either a crash or a config drift.
2. **Data freshness.** `max(trading_data_age_seconds) < 60` — primary
   timeframe is 1m, anything older means the feed is stalling.
3. **Daily PnL within limits.** `trading_portfolio_daily_pnl_usd`
   should be > `-5% × paper_initial_balance_usd`. If it's negative and
   large, expect the circuit breaker to trip soon.
4. **Circuit breaker not tripped.** `GET /agents` payload for
   `risk_agent` includes `circuit_breaker_tripped: false`.
5. **No quarantined agents in last 24h.** Search structured logs for
   `agent_quarantined` events; one means the agent exhausted its
   restart budget and is now down.

If any check fails, escalate per [§7](#incident-response).

---

## 7. Incident response

### Agent crash

**Detect:** `trading_agent_up{agent="X"} == 0` for more than 30s, or
`agent_quarantined` log event.

**Diagnose:** `docker logs trading_<name>` (Compose deploy) or the
launcher stdout (single-process). Look for the last
`agent_error` event before the crash — it carries `error_type`,
`context`, and `consecutive_errors`.

**Action:**
- If transient (network blip, exchange flap), `docker-compose restart <name>`.
- If the error is deterministic (config, bug), fix the cause and redeploy.
- If you can't fix in 15 minutes and the agent is on the trade path,
  halt trading (§4) so risk doesn't degrade silently.

### Stale market data

**Detect:** `trading_data_age_seconds > 60` for any symbol.

**Diagnose:** check MarketDataAgent logs for `poll_loop` errors;
hit the exchange's status page; verify the symbol is still listed.

**Action:** if the exchange is down, halt trading — stale data plus an
unchanged position is the worst combination. Resume only when freshness
recovers.

### Daily loss limit hit

**Detect:** Risk agent emits `emergency_halt_triggered` with
`reason="Daily loss limit breached"`. ExecutionAgent stops placing
orders within one heartbeat.

**Diagnose:** review the last hour of execution results; look for
unusual fill prices, runaway sizing, or signal storms.

**Action:**
1. Acknowledge the alert.
2. Close open positions manually via your broker UI (paper: API call to
   `POST /positions/close`, planned).
3. Investigate cause before restarting any agent. The halt is a
   feature, not a bug — do not bypass it.
4. Restart the Risk and Execution agents in that order once the cause
   is resolved.

---

## 8. Transitioning paper → live

This is a **checklist, not an implementation guide**. Live trading is a
deliberately separate concern; do not flip the flag without going through
every step below.

- [ ] Exchange sandbox tests pass for at least 7 consecutive days
- [ ] Live API keys are stored in the secret manager, not `.env`
- [ ] `EXCHANGE_SANDBOX=false` and `TRADING_MODE=live` reviewed by a
      second human
- [ ] Capital cap configured — start with the smallest tradeable size
- [ ] Risk limits tightened for first-week live (e.g. 1% per position,
      2% daily loss, 5% max drawdown)
- [ ] Alerting reaches a phone, not just email (PagerDuty or equivalent)
- [ ] Runbook reviewed end-to-end with a stand-in operator who has not
      seen the system before
- [ ] Manual halt drill executed end-to-end on the live cluster
- [ ] Postgres backup verified by restoring into a scratch instance
- [ ] Decision log entry for the cutover, including who approved it

After cutover, repeat the daily checklist twice a day for the first week.

---

## 9. Backup and recovery

### Postgres

- **What to back up:** the `trading_db` database in full. It holds
  proposals, assessments, executions, positions, heartbeats, and alerts —
  everything needed for audit and post-trade analysis.
- **Cadence:** at least daily for paper, hourly for live.
- **How:** `pg_dump -Fc -d trading_db -f trading_db.dump`. Restore with
  `pg_restore -d trading_db trading_db.dump` on a fresh instance.
- **Verify:** monthly drill — restore the latest dump into a scratch DB
  and run `SELECT count(*) FROM executions WHERE created_at >= now() - interval '1 day';`
  to confirm it loaded.

### Redis

Redis is **ephemeral cache** in this system. Pub/sub messages are not
persisted, and the cached portfolio state is recomputed by the Risk and
Execution agents on startup. **Do not back up Redis.** If you lose the
Redis instance, restart the agents and the cache repopulates within a
heartbeat.

### Application state

The in-memory `PortfolioState` inside the Risk agent is reconstructed
from `paper_initial_balance_usd` on every start. After a clean shutdown
this is correct; after a crash mid-trade you may need to manually
reconcile open positions in the database against the broker UI before
restarting the Risk agent.
