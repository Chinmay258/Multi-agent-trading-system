"""
api/main.py
-----------
FastAPI control plane for the trading system.

Provides:
    GET  /health                — system health + per-agent statuses
    POST /control/command       — send SystemCommandMessage to agents
    POST /control/halt          — trigger emergency RiskOverride
    GET  /positions             — current open positions (from Redis cache)
    GET  /positions/balance     — account balance (from Redis cache)
    GET  /positions/history     — last 50 execution results (from Redis list)
    WS   /ws/stream             — real-time execution results + heartbeats

Startup:
    - Connects a shared MessageBus to Redis
    - Starts a heartbeat watcher background task (populates heartbeat_registry)
    - Starts a WebSocket broadcast background task (fans out to WS clients)

Shutdown:
    - Cancels background tasks
    - Disconnects MessageBus

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import control, dashboard, positions, status
from api.websocket import router as ws_router
from api.websocket import start_broadcast_task
from core.messaging import Channels, MessageBus
from core.models.system import AgentHeartbeat


async def _heartbeat_watcher(bus: MessageBus, registry: dict[str, Any]) -> None:
    """
    Background task: subscribe to system.heartbeat and update the in-memory
    heartbeat registry so GET /health can report per-agent status.

    Uses a raw Redis pubsub connection to avoid sharing the single pubsub
    connection that MessageBus reserves for agent-style subscribe() calls.
    """
    if bus._pool is None:  # noqa: SLF001
        return

    pubsub = bus._pool.pubsub(ignore_subscribe_messages=True)  # noqa: SLF001
    await pubsub.subscribe(Channels.SYSTEM_HEARTBEAT)
    try:
        while True:
            try:
                raw = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if raw is None:
                    await asyncio.sleep(0)
                    continue
                data = raw.get("data", "")
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                if data:
                    try:
                        hb = AgentHeartbeat.model_validate_json(data)
                        registry[hb.agent_name] = hb
                    except Exception:
                        pass
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(1.0)
    finally:
        try:
            await pubsub.unsubscribe()
            await pubsub.aclose()
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage startup and shutdown of Redis connections and background tasks."""
    bus = MessageBus()
    await bus.connect()

    heartbeat_registry: dict[str, Any] = {}
    recent_signals: deque = deque(maxlen=50)
    recent_events: deque = deque(maxlen=100)
    app.state.bus = bus
    app.state.heartbeat_registry = heartbeat_registry
    app.state.recent_signals = recent_signals
    app.state.recent_events = recent_events

    # Background tasks: heartbeat watcher + WebSocket broadcaster (also fills buffers)
    hb_task = asyncio.create_task(
        _heartbeat_watcher(bus, heartbeat_registry),
        name="api_heartbeat_watcher",
    )
    ws_task = asyncio.create_task(
        start_broadcast_task(bus, recent_signals, recent_events),
        name="api_ws_broadcaster",
    )

    try:
        yield
    finally:
        hb_task.cancel()
        ws_task.cancel()
        # Bound the shutdown so a slow pubsub unwind can never hang uvicorn's
        # reload / container stop indefinitely (it would wedge the API otherwise).
        try:
            await asyncio.wait_for(
                asyncio.gather(hb_task, ws_task, return_exceptions=True),
                timeout=5.0,
            )
        except TimeoutError:
            pass
        await bus.disconnect()


app = FastAPI(
    title="TradingSystem Control Plane",
    description="HTTP and WebSocket control plane for the autonomous trading system.",
    version="0.4.0",
    lifespan=lifespan,
)

# CORS: this is a public, read-only demo dashboard. Allow any origin so the Vite
# dev server (and any deployment) can reach the API without per-origin config.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(status.router)
app.include_router(control.router)
app.include_router(positions.router)
app.include_router(dashboard.router)
app.include_router(ws_router)
