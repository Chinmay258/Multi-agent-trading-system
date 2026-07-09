"""
api/routers/status.py
----------------------
Health and status endpoints for the trading system control plane.

Endpoints:
    GET /health     — system health summary with per-agent status

Since MonitoringAgent stores heartbeats in its own process memory (not Redis),
the API maintains its own lightweight in-memory heartbeat registry via a
background task. The registry is populated via app.state.heartbeat_registry
(a plain dict updated by the _heartbeat_watcher task in api/main.py).

Agent names are NOT hardcoded. The registry is populated dynamically from
whatever agents have sent heartbeats — same design as MonitoringAgent.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Request

from core.config import get_settings
from core.models.system import AgentStatus

router = APIRouter()


@router.get("/health")
async def get_health(request: Request) -> dict:
    """
    Return system health summary.

    Response shape:
        {
          "system_ok": bool,
          "trading_mode": "paper" | "live",
          "timestamp": "2024-01-01T00:00:00Z",
          "agents": {
            "<agent_name>": {
              "status": "running",
              "uptime_seconds": 123.4,
              "messages_processed": 42,
              "errors_since_start": 0,
              "last_seen_seconds_ago": 5.1
            }
          }
        }

    system_ok is True only when every known agent reports RUNNING status.
    If no agents have been seen yet the system is considered not-yet-healthy
    (system_ok=False) rather than healthy.
    """
    settings = get_settings()
    registry: dict = getattr(request.app.state, "heartbeat_registry", {})
    now = datetime.now(UTC)
    timeout_seconds: int = settings.monitoring.heartbeat_timeout_seconds

    agents_out: dict = {}
    all_running = bool(registry)  # False if no agents seen yet

    for name, hb in registry.items():
        age = (now - hb.timestamp).total_seconds()
        is_stale = age > timeout_seconds
        is_running = hb.status == AgentStatus.RUNNING and not is_stale
        if not is_running:
            all_running = False
        agents_out[name] = {
            "status": hb.status,
            "uptime_seconds": round(hb.uptime_seconds, 1),
            "messages_processed": hb.messages_processed,
            "errors_since_start": hb.errors_since_start,
            "last_seen_seconds_ago": round(age, 1),
            "is_stale": is_stale,
        }

    return {
        "system_ok": all_running,
        "trading_mode": settings.trading_mode.value,
        "timestamp": now.isoformat(),
        "agents": agents_out,
    }
