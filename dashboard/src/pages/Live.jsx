import { useEffect, useRef, useState } from "react";
import {
  Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api, fmtNum, fmtPct, fmtUsd, openStream } from "../api.js";

function Stat({ label, value, cls, sub }) {
  return (
    <div className="card stat">
      <div className="label">{label}</div>
      <div className={`value ${cls || ""}`}>{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

function dirClass(d) {
  if (!d) return "";
  const s = String(d).toLowerCase();
  if (s.includes("buy")) return "pos";
  if (s.includes("sell")) return "neg";
  return "muted";
}

export default function Live() {
  const [overview, setOverview] = useState(null);
  const [agents, setAgents] = useState([]);
  const [signals, setSignals] = useState([]);
  const [history, setHistory] = useState([]);
  const [feed, setFeed] = useState([]);
  const [wsStatus, setWsStatus] = useState("connecting");
  const [equity, setEquity] = useState([]);
  const seq = useRef(0);

  // Poll snapshots.
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const [ov, ag, sg, hi] = await Promise.all([
          api.overview(), api.agents(), api.signals(), api.history(),
        ]);
        if (!alive) return;
        setOverview(ov);
        setAgents(ag.agents || []);
        setSignals(sg.signals || []);
        setHistory(hi.results || []);
        setEquity((prev) => {
          const next = [...prev, { t: seq.current++, equity: ov.balance.total_equity_usd }];
          return next.slice(-120);
        });
      } catch { /* keep last good */ }
    };
    tick();
    const id = setInterval(tick, 4000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  // Live activity feed via websocket.
  useEffect(() => {
    const close = openStream(
      (env) => {
        if (env.type === "heartbeat") return; // heartbeats refresh via polling
        const row = { id: `${Date.now()}-${Math.random()}`, type: env.type, payload: env.payload };
        setFeed((f) => [row, ...f].slice(0, 40));
      },
      (s) => setWsStatus(s),
    );
    return close;
  }, []);

  const bal = overview?.balance;
  const ret = overview?.total_return_pct ?? 0;

  return (
    <>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", margin: "26px 0 18px", flexWrap: "wrap", gap: 10 }}>
        <h2 style={{ margin: 0 }}>Live paper-trading dashboard</h2>
        <div style={{ display: "flex", gap: 8 }}>
          <span className={`pill ${wsStatus === "connected" ? "green" : "amber"}`}><span className="dot" /> websocket {wsStatus}</span>
          {overview && <span className="pill">{overview.exchange} · {(overview.timeframes || []).join("/")}</span>}
          {overview && <span className={`pill ${overview.use_ml_signals ? "" : "green"}`}>signals: {overview.use_ml_signals ? "ML" : "rules"}</span>}
        </div>
      </div>

      <div className="grid cols-4" style={{ marginBottom: 16 }}>
        <Stat label="Equity" value={fmtUsd(bal?.total_equity_usd)} cls={ret >= 0 ? "pos" : "neg"} sub={`return ${fmtPct(ret)} on ${fmtUsd(overview?.initial_balance_usd)}`} />
        <Stat label="Free cash" value={fmtUsd(bal?.free_margin_usd)} sub={`in positions ${fmtUsd(bal?.used_margin_usd)}`} />
        <Stat label="Open positions" value={overview?.open_positions_count ?? "—"} sub={overview ? `${overview.symbols?.length} symbols tracked` : ""} />
        <Stat label="Agents healthy" value={overview ? `${overview.agents_healthy}/${overview.agents_total}` : "—"} cls={overview && overview.agents_healthy === overview.agents_total ? "pos" : "neg"} sub={`mode: ${overview?.trading_mode ?? "—"}`} />
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <h3>Equity (this session)</h3>
        {equity.length < 2 ? (
          <div className="loading"><span className="spinner" /> collecting live samples…</div>
        ) : (
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={equity}>
              <defs>
                <linearGradient id="eq" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.5} />
                  <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#243049" />
              <XAxis dataKey="t" hide />
              <YAxis domain={["auto", "auto"]} tick={{ fill: "#8a97b0", fontSize: 12 }} width={70} tickFormatter={(v) => `$${Math.round(v)}`} />
              <Tooltip contentStyle={{ background: "#0f1521", border: "1px solid #243049", borderRadius: 8 }} formatter={(v) => fmtUsd(v)} labelFormatter={() => ""} />
              <Area type="monotone" dataKey="equity" stroke="#3b82f6" strokeWidth={2} fill="url(#eq)" isAnimationActive={false} />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className="grid cols-2" style={{ marginBottom: 16 }}>
        <div className="card">
          <h3>Open positions</h3>
          {overview?.open_positions?.length ? (
            <table className="tbl">
              <thead><tr><th>Symbol</th><th>Side</th><th>Qty</th><th>Entry</th><th>Cost</th></tr></thead>
              <tbody>
                {overview.open_positions.map((p, i) => (
                  <tr key={i}>
                    <td>{p.symbol}</td>
                    <td className={dirClass(p.side)}>{String(p.side).toUpperCase()}</td>
                    <td>{fmtNum(p.quantity, 6)}</td>
                    <td>{fmtUsd(p.entry_price)}</td>
                    <td>{fmtUsd(p.cost_usd)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <p className="muted">No open positions.</p>}
        </div>

        <div className="card">
          <h3>Agent health</h3>
          <table className="tbl">
            <thead><tr><th>Agent</th><th>Status</th><th>Seen</th><th>Msgs</th><th>Errs</th></tr></thead>
            <tbody>
              {agents.length ? agents.map((a) => (
                <tr key={a.name}>
                  <td>{a.name.replace(/_agent$/, "")}</td>
                  <td><span className={`pill ${a.healthy ? "green" : "red"}`} style={{ fontSize: 11 }}><span className="dot" />{a.status}</span></td>
                  <td>{a.last_seen_seconds_ago}s</td>
                  <td>{a.messages_processed}</td>
                  <td className={a.errors_since_start ? "neg" : ""}>{a.errors_since_start}</td>
                </tr>
              )) : <tr><td colSpan="5" className="muted">waiting for heartbeats…</td></tr>}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid cols-2" style={{ marginBottom: 16 }}>
        <div className="card">
          <h3>Recent signals</h3>
          {signals.length ? (
            <table className="tbl">
              <thead><tr><th>Symbol</th><th>TF</th><th>Direction</th><th>Confidence</th><th>Price</th></tr></thead>
              <tbody>
                {signals.slice(0, 10).map((s, i) => (
                  <tr key={i}>
                    <td>{s.symbol}</td><td>{s.timeframe}</td>
                    <td className={dirClass(s.direction)}>{String(s.direction || "").toUpperCase()}</td>
                    <td>{s.confidence != null ? `${(s.confidence * 100).toFixed(0)}%` : "—"}</td>
                    <td>{fmtUsd(s.price)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <p className="muted">No signals yet. On higher timeframes (4h/1d) a quiet gap between candle closes is normal.</p>}
        </div>

        <div className="card">
          <h3>Live activity <span className="muted" style={{ fontWeight: 400, fontSize: 12 }}>(websocket)</span></h3>
          <div className="feed">
            {feed.length ? feed.map((r) => (
              <div className="feed-row" key={r.id}>
                <span className={`tag ${r.type}`}>{r.type}</span>
                <span className="mono" style={{ fontSize: 12 }}>
                  {r.payload.symbol || (r.payload.original_proposal && r.payload.original_proposal.symbol) || ""}
                  {" "}
                  <span className={dirClass(r.payload.side || r.payload.direction || r.payload.decision)}>
                    {String(r.payload.side || r.payload.direction || r.payload.decision || r.payload.status || "").toUpperCase()}
                  </span>
                </span>
              </div>
            )) : <div className="loading"><span className="spinner" /> waiting for live pipeline events…</div>}
          </div>
        </div>
      </div>

      <div className="card">
        <h3>Trade log <span className="muted" style={{ fontWeight: 400, fontSize: 12 }}>(recent fills)</span></h3>
        {history.length ? (
          <table className="tbl">
            <thead><tr><th>Symbol</th><th>Side</th><th>Status</th><th>Fill price</th><th>Qty</th></tr></thead>
            <tbody>
              {history.slice(0, 15).map((h, i) => (
                <tr key={i}>
                  <td>{h.symbol}</td>
                  <td className={dirClass(h.side)}>{String(h.side || "").toUpperCase()}</td>
                  <td className="muted">{h.status}</td>
                  <td>{fmtUsd(h.average_fill_price)}</td>
                  <td>{fmtNum(h.filled_quantity ?? h.quantity, 6)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <p className="muted">No fills recorded yet.</p>}
      </div>
    </>
  );
}
