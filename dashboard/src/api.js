// API + WebSocket helpers. Base URL is same-origin by default (served behind nginx
// in production); in dev, Vite proxies /api and /ws to the FastAPI control plane.
const BASE = import.meta.env.VITE_API_BASE || "";

async function getJSON(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

export const api = {
  overview: () => getJSON("/api/overview"),
  agents: () => getJSON("/api/agents"),
  signals: () => getJSON("/api/signals"),
  events: () => getJSON("/api/events"),
  evaluation: () => getJSON("/api/evaluation"),
  history: () => getJSON("/positions/history"),
};

// Open the live pipeline WebSocket. onMessage receives parsed envelopes:
//   { channel, type: "signal"|"proposal"|"assessment"|"fill"|"heartbeat", payload }
export function openStream(onMessage, onStatus) {
  let ws;
  let closed = false;
  let retry = 0;

  function connect() {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const host = BASE ? BASE.replace(/^https?:/, proto + ":") : `${proto}://${window.location.host}`;
    ws = new WebSocket(`${host}/ws/stream`);

    ws.onopen = () => { retry = 0; onStatus && onStatus("connected"); };
    ws.onclose = () => {
      onStatus && onStatus("disconnected");
      if (!closed) {
        retry = Math.min(retry + 1, 6);
        setTimeout(connect, 800 * retry);
      }
    };
    ws.onerror = () => ws.close();
    ws.onmessage = (ev) => {
      try { onMessage(JSON.parse(ev.data)); } catch { /* ignore malformed */ }
    };
  }

  connect();
  return () => { closed = true; ws && ws.close(); };
}

// Formatting helpers
export const fmtUsd = (n) =>
  n == null ? "—" : `$${Number(n).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
export const fmtPct = (n) => (n == null ? "—" : `${Number(n) >= 0 ? "+" : ""}${Number(n).toFixed(2)}%`);
export const fmtNum = (n, d = 2) => (n == null ? "—" : Number(n).toFixed(d));
