"""
api/routers/dashboard.py
------------------------
Read-only endpoints that power the showcase + live dashboard UI (``/dashboard``).

All endpoints are safe, keyless, and read from Redis cache / in-memory ring buffers /
the committed evaluation JSON. Nothing here places orders or mutates state.

    GET /api/overview     — one-shot snapshot (mode, balance, positions, agent health)
    GET /api/agents       — per-agent heartbeat status
    GET /api/signals      — recent technical signals + confidence (ring buffer)
    GET /api/events       — recent pipeline events (proposals/assessments/fills)
    GET /api/evaluation   — Phase 4/5 metrics (baseline + improved) for the charts
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request

from core.config import get_settings
from core.logging import get_logger
from core.models.system import AgentStatus

logger = get_logger("api.dashboard")

router = APIRouter(prefix="/api", tags=["dashboard"])

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR = _REPO_ROOT / "backtest" / "results"

_PAPER_CASH_KEY = "paper_portfolio:cash"
_PAPER_POSITIONS_KEY = "paper_portfolio:positions"


async def _read_balance_and_positions(bus: Any) -> tuple[dict, list]:
    """Read paper balance + positions straight from Redis (keyless paper demo)."""
    balance = {"total_equity_usd": 0.0, "free_margin_usd": 0.0, "used_margin_usd": 0.0}
    positions: list = []
    if not bus or bus._pool is None:  # noqa: SLF001
        return balance, positions
    try:
        cash_raw = await bus._pool.get(_PAPER_CASH_KEY)  # noqa: SLF001
        pos_raw = await bus._pool.get(_PAPER_POSITIONS_KEY)  # noqa: SLF001
        free_cash = float(cash_raw) if cash_raw else 0.0
        used = 0.0
        if pos_raw:
            data: dict = json.loads(pos_raw)
            for p in data.values():
                used += float(p.get("cost_usd", 0))
                positions.append(
                    {
                        "symbol": p["symbol"],
                        "side": p["side"],
                        "quantity": float(p["quantity"]),
                        "entry_price": float(p["entry_price"]),
                        "cost_usd": float(p.get("cost_usd", 0)),
                    }
                )
        balance = {
            "total_equity_usd": round(free_cash + used, 2),
            "free_margin_usd": round(free_cash, 2),
            "used_margin_usd": round(used, 2),
        }
    except Exception as exc:
        logger.error("overview_balance_read_failed", error=str(exc))
    return balance, positions


def _agents_snapshot(registry: dict) -> list[dict]:
    settings = get_settings()
    timeout = settings.monitoring.heartbeat_timeout_seconds
    now = datetime.now(UTC)
    out: list[dict] = []
    for name, hb in sorted(registry.items()):
        age = (now - hb.timestamp).total_seconds()
        stale = age > timeout
        out.append(
            {
                "name": name,
                "status": hb.status if isinstance(hb.status, str) else hb.status.value,
                "healthy": (hb.status == AgentStatus.RUNNING) and not stale,
                "last_seen_seconds_ago": round(age, 1),
                "uptime_seconds": round(hb.uptime_seconds, 1),
                "messages_processed": hb.messages_processed,
                "errors_since_start": hb.errors_since_start,
            }
        )
    return out


@router.get("/overview")
async def overview(request: Request) -> dict:
    """One-shot snapshot for the dashboard header + summary cards."""
    settings = get_settings()
    bus = getattr(request.app.state, "bus", None)
    registry: dict = getattr(request.app.state, "heartbeat_registry", {})

    balance, positions = await _read_balance_and_positions(bus)
    agents = _agents_snapshot(registry)
    healthy = sum(1 for a in agents if a["healthy"])
    initial = settings.paper_initial_balance_usd
    equity = balance["total_equity_usd"] or initial

    return {
        "trading_mode": settings.trading_mode.value,
        "execution_broker": settings.execution_broker,
        "data_source": getattr(settings, "data_source", "public"),
        "exchange": settings.exchange.name,
        "symbols": settings.market_data.symbols,
        "timeframes": settings.market_data.ohlcv_timeframes,
        "use_ml_signals": settings.technical_analysis.use_ml_signals,
        "initial_balance_usd": initial,
        "balance": balance,
        "total_return_pct": round((equity - initial) / initial * 100, 4) if initial else 0.0,
        "open_positions": positions,
        "open_positions_count": len(positions),
        "agents_total": len(agents),
        "agents_healthy": healthy,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.get("/agents")
async def agents(request: Request) -> dict:
    registry: dict = getattr(request.app.state, "heartbeat_registry", {})
    rows = _agents_snapshot(registry)
    return {"agents": rows, "count": len(rows), "healthy": sum(1 for a in rows if a["healthy"])}


@router.get("/signals")
async def signals(request: Request) -> dict:
    buf = getattr(request.app.state, "recent_signals", None)
    items = list(buf) if buf is not None else []
    out = []
    for s in items:
        out.append(
            {
                "symbol": s.get("symbol"),
                "timeframe": s.get("timeframe"),
                "direction": s.get("direction"),
                "confidence": s.get("confidence"),
                "price": s.get("price"),
                "timestamp": s.get("timestamp"),
            }
        )
    return {"signals": out, "count": len(out)}


@router.get("/events")
async def events(request: Request) -> dict:
    buf = getattr(request.app.state, "recent_events", None)
    items = list(buf) if buf is not None else []
    return {"events": items, "count": len(items)}


@router.get("/evaluation")
async def evaluation() -> dict:
    """Serve the committed Phase 4/5 evaluation metrics for the showcase charts."""

    def _load(name: str) -> dict | None:
        path = _RESULTS_DIR / name
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.error("evaluation_read_failed", file=name, error=str(exc))
        return None

    return {
        "baseline": _load("baseline_metrics.json"),
        "improved": _load("improved_metrics.json"),
        "available": (_RESULTS_DIR / "baseline_metrics.json").exists(),
    }
