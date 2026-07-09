import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import ArchitectureDiagram from "../components/ArchitectureDiagram.jsx";
import { api, fmtPct } from "../api.js";

const PHASES = [
  ["Phase 1", "Audit & fix", "Walked the tree, wrote an audit, fixed the known recurring bugs, verified no secrets."],
  ["Phase 2", "Pluggable data + live paper", "DataSource interface (keyless public exchange vs. local MT5); end-to-end live paper trading."],
  ["Phase 3", "Reproducibility", "Pinned deps, offline sample data + seeder, Makefile, CI, pre-commit, README quickstart."],
  ["Phase 4", "Honest evaluation", "Walk-forward backtest with fees + slippage, benchmarks, and a candid report — no lookahead."],
  ["Phase 5", "Retrain & measure", "Tried a walk-forward ML strategy; it didn't beat the rule baseline out-of-sample, so we kept rules."],
  ["Phase 6", "This dashboard", "React + FastAPI websockets: a showcase and a live view of the paper-trading loop."],
];

function MetricCard({ label, value, cls, sub }) {
  return (
    <div className="card stat">
      <div className="label">{label}</div>
      <div className={`value ${cls || ""}`}>{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

export default function Showcase() {
  const [evalData, setEvalData] = useState(null);
  const [overview, setOverview] = useState(null);

  useEffect(() => {
    api.evaluation().then(setEvalData).catch(() => {});
    api.overview().then(setOverview).catch(() => {});
  }, []);

  const baseline = evalData?.baseline;
  const strat = baseline?.strategy;
  const bh = baseline?.buy_and_hold;
  const rnd = baseline?.random_entry;
  const ml = baseline?.ml_evaluation;
  const improved = baseline?.improved_strategy;

  const returnsChart = strat
    ? [
        { name: "Strategy", value: strat.total_return_pct, fill: "#3b82f6" },
        { name: "Buy & Hold", value: bh?.total_return_pct ?? 0, fill: "#9ca3af" },
        { name: "Random", value: rnd?.mean_total_return_pct ?? 0, fill: "#8b5cf6" },
      ]
    : [];

  return (
    <>
      <section className="hero">
        {overview && (
          <span className={`pill ${overview.agents_healthy === overview.agents_total ? "green" : "amber"}`}>
            <span className="dot" /> {overview.agents_healthy}/{overview.agents_total} agents healthy · live {overview.exchange} data · {overview.trading_mode}
          </span>
        )}
        <h1>An autonomous multi-agent<br />trading system you can watch run.</h1>
        <p className="lead">
          Seven independent Python agents communicate <strong>only over Redis</strong> to turn live public
          market data into paper trades: candles → indicators → signal → decision → risk checks → simulated
          execution. It runs <strong>keyless</strong> and <strong>paper-only</strong> — no API keys, no real money.
        </p>
        <div className="cta">
          <Link className="btn primary" to="/live">▶ Open live dashboard</Link>
          <a className="btn ghost" href="https://github.com/Chinmay258/Multi-agent-trading-system" target="_blank" rel="noreferrer">View source</a>
        </div>
      </section>

      <section className="section">
        <h2>Architecture</h2>
        <p className="sub">No agent imports another. Every message is a typed Pydantic model on a named Redis channel — so any agent can be added, removed, or restarted independently.</p>
        <ArchitectureDiagram />
      </section>

      <section className="section">
        <h2>How it works</h2>
        <div className="grid cols-3">
          <div className="card"><h3>1 · Sense</h3><p className="muted" style={{ fontSize: 14 }}>The Market Data agent polls public OHLCV via CCXT (no key) and publishes candles. Technical Analysis maintains rolling buffers and computes RSI / MACD / Bollinger / EMA into a confidence-scored signal.</p></div>
          <div className="card"><h3>2 · Decide</h3><p className="muted" style={{ fontSize: 14 }}>The Decision agent aggregates signals and applies a trend filter to build a sized trade proposal. The Risk agent runs eight checks (drawdown, daily loss, exposure, staleness, sizing) before anything is approved.</p></div>
          <div className="card"><h3>3 · Execute (paper)</h3><p className="muted" style={{ fontSize: 14 }}>The Execution agent fills approved trades through a simulated PaperBroker (slippage + fees), tracks the portfolio in Redis, and the Monitoring agent watches every heartbeat.</p></div>
        </div>
      </section>

      <section className="section">
        <h2>Evaluation — honest results</h2>
        <p className="sub">A rigorous walk-forward backtest of the real pipeline (no lookahead, realistic fees + slippage), benchmarked against buy-and-hold and random entry. We report what we found, not what we wished for.</p>
        {!evalData?.available && <div className="card loading">Evaluation metrics not found. Run <span className="mono">make eval</span>.</div>}
        {strat && (
          <>
            <div className="grid cols-4" style={{ marginBottom: 16 }}>
              <MetricCard label="Strategy return" value={fmtPct(strat.total_return_pct)} cls={strat.total_return_pct >= 0 ? "pos" : "neg"} sub={`${baseline.config.timeframe} · ${baseline.config.since} → ${baseline.config.until}`} />
              <MetricCard label="Sharpe" value={strat.sharpe} sub={`max DD ${strat.max_drawdown_pct}%`} />
              <MetricCard label="Win rate" value={`${strat.win_rate_pct.toFixed(1)}%`} sub={`${strat.num_trades} trades · PF ${strat.profit_factor ?? "—"}`} />
              <MetricCard label="ML accuracy" value={ml?.available ? ml.accuracy : "—"} sub={`vs ${ml?.random_baseline_accuracy ?? "0.33"} random (3-class)`} />
            </div>
            <div className="grid cols-2">
              <div className="card">
                <h3>Total return vs. benchmarks</h3>
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={returnsChart}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#243049" />
                    <XAxis dataKey="name" tick={{ fill: "#8a97b0", fontSize: 12 }} />
                    <YAxis tick={{ fill: "#8a97b0", fontSize: 12 }} unit="%" />
                    <Tooltip contentStyle={{ background: "#0f1521", border: "1px solid #243049", borderRadius: 8 }} />
                    <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                      {returnsChart.map((e, i) => <Cell key={i} fill={e.fill} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <p className="muted" style={{ fontSize: 13 }}>Beating buy-and-hold here is mostly low exposure during a falling market — the strategy is statistically indistinguishable from random entry. No demonstrable edge.</p>
              </div>
              <div className="card">
                <h3>Phase 5 — before vs. after</h3>
                {improved ? (
                  <>
                    <table className="tbl">
                      <thead><tr><th>Variant</th><th>Return</th><th>Sharpe</th><th>Trades</th></tr></thead>
                      <tbody>
                        <tr><td>Baseline (rules) ✓ kept</td><td>{fmtPct(strat.total_return_pct)}</td><td>{strat.sharpe}</td><td>{strat.num_trades}</td></tr>
                        <tr><td className="muted">Walk-forward ML</td><td>{fmtPct(improved.metrics.total_return_pct)}</td><td>{improved.metrics.sharpe}</td><td>{improved.metrics.num_trades}</td></tr>
                      </tbody>
                    </table>
                    <p className="muted" style={{ fontSize: 13, marginTop: 12 }}>
                      We tried enabling the ML signal path with proper walk-forward retraining. It
                      <strong> {improved.kept ? "beat" : "did not beat"}</strong> the rule baseline out-of-sample
                      (Δ return {improved.vs_baseline.return_delta_pct} pp, Δ Sharpe {improved.vs_baseline.sharpe_delta}),
                      so the system keeps the simpler rule-based default. A documented negative result.
                    </p>
                  </>
                ) : <p className="muted">Run <span className="mono">make eval</span> to generate the before/after comparison.</p>}
              </div>
            </div>
          </>
        )}
      </section>

      <section className="section">
        <h2>The journey</h2>
        <p className="sub">This project was built in deliberate phases, each ending with tests and an honest summary.</p>
        <div className="card journey">
          {PHASES.map(([n, title, body]) => (
            <div className="step" key={n}>
              <div className="n">{n}</div>
              <div className="body"><strong>{title}</strong><span>{body}</span></div>
            </div>
          ))}
        </div>
      </section>
    </>
  );
}
