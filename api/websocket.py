"""
api/websocket.py
----------------
WebSocket endpoint and broadcast infrastructure for the trading control plane.

Endpoint:
    WS /ws/stream   — streams execution results and agent heartbeats in real time

Design:
    ConnectionManager holds all active WebSocket connections. A single background
    task (start_broadcast_task) subscribes to Redis pub/sub using a raw connection
    (bypassing MessageBus typed deserialization) so both execution.result and
    system.heartbeat can be forwarded as raw JSON to clients without needing a
    common Pydantic type.

    Each WebSocket client receives every message from both channels. The JSON
    payload is the model's canonical serialisation from the publishing agent —
    clients can discriminate between message types by inspecting the JSON fields
    (e.g. presence of "agent_name" for heartbeats vs "proposal_id" for results).
"""

from __future__ import annotations

import asyncio
import json
from collections import deque

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.messaging import Channels, MessageBus

router = APIRouter()


class ConnectionManager:
    """Tracks active WebSocket connections and broadcasts messages to all of them."""

    def __init__(self) -> None:
        self._clients: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._clients = [c for c in self._clients if c is not ws]

    async def broadcast(self, data: str) -> None:
        """Send data to all connected clients; silently remove disconnected ones."""
        dead: list[WebSocket] = []
        for ws in list(self._clients):
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    @property
    def client_count(self) -> int:
        return len(self._clients)


manager = ConnectionManager()


async def start_broadcast_task(
    bus: MessageBus,
    recent_signals: deque | None = None,
    recent_events: deque | None = None,
) -> None:
    """
    Background task: subscribe to the whole trade pipeline via raw Redis pub/sub and
    forward each message to WebSocket clients as a tagged JSON **envelope**:

        {"channel": "<redis-channel>", "type": "<kind>", "payload": {...}}

    Channels forwarded:
      - signal.technical.*  (kind "signal")     — TA signals + confidence
      - decision.proposal   (kind "proposal")
      - risk.assessment     (kind "assessment")
      - execution.result    (kind "fill")
      - system.heartbeat    (kind "heartbeat")  — per-agent health

    The same messages also populate the API's in-memory ring buffers (so a freshly
    loaded dashboard has recent history without waiting for new events).
    """
    if bus._pool is None:  # noqa: SLF001
        return

    pubsub = bus._pool.pubsub(ignore_subscribe_messages=True)  # noqa: SLF001
    await pubsub.psubscribe("signal.technical.*")
    await pubsub.subscribe(
        Channels.DECISION_PROPOSAL,
        Channels.RISK_ASSESSMENT,
        Channels.EXECUTION_RESULT,
        Channels.SYSTEM_HEARTBEAT,
    )

    def _kind(channel: str) -> str:
        if channel.startswith("signal.technical"):
            return "signal"
        if channel == Channels.DECISION_PROPOSAL:
            return "proposal"
        if channel == Channels.RISK_ASSESSMENT:
            return "assessment"
        if channel == Channels.EXECUTION_RESULT:
            return "fill"
        if channel == Channels.SYSTEM_HEARTBEAT:
            return "heartbeat"
        return "other"

    try:
        while True:
            try:
                raw = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if raw is None:
                    await asyncio.sleep(0)
                    continue
                channel = raw.get("channel", "")
                if isinstance(channel, bytes):
                    channel = channel.decode("utf-8")
                data = raw.get("data", "")
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                if not data:
                    continue

                try:
                    payload = json.loads(data)
                except Exception:
                    continue

                kind = _kind(channel)
                envelope = {"channel": channel, "type": kind, "payload": payload}

                # Populate ring buffers for REST snapshots.
                if kind == "signal" and recent_signals is not None:
                    recent_signals.appendleft(payload)
                elif kind in ("proposal", "assessment", "fill") and recent_events is not None:
                    recent_events.appendleft({"type": kind, **_summarise(kind, payload)})

                if manager.client_count > 0:
                    await manager.broadcast(json.dumps(envelope))
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(1.0)
    finally:
        try:
            await pubsub.punsubscribe()
            await pubsub.unsubscribe()
            await pubsub.aclose()
        except Exception:
            pass


def _summarise(kind: str, payload: dict) -> dict:
    """Flatten a pipeline message into a compact event row for the activity feed."""
    ts = payload.get("timestamp") or payload.get("created_at") or payload.get("executed_at")
    if kind == "proposal":
        return {
            "symbol": payload.get("symbol"),
            "side": payload.get("side"),
            "size_usd": payload.get("requested_size_usd"),
            "timestamp": ts,
        }
    if kind == "assessment":
        return {
            "symbol": (payload.get("original_proposal") or {}).get("symbol"),
            "decision": payload.get("decision"),
            "reason": payload.get("rejection_reason"),
            "timestamp": ts,
        }
    if kind == "fill":
        return {
            "symbol": payload.get("symbol"),
            "side": payload.get("side"),
            "status": payload.get("status"),
            "price": payload.get("average_fill_price"),
            "timestamp": ts,
        }
    return {"timestamp": ts}


@router.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket) -> None:
    """
    Real-time stream of execution results and agent heartbeats.

    Clients receive raw JSON payloads as published by agents on the Redis bus.
    The connection is kept alive with a 30-second sleep between keep-alive
    checks; actual data is pushed by the broadcast background task.
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep-alive loop: data is pushed by start_broadcast_task, not here.
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)
