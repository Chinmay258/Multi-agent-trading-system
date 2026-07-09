"""
api/routers/control.py
-----------------------
Control plane endpoints — send commands to agents or trigger emergency halts.

Endpoints:
    POST /control/command       — broadcast a SystemCommandMessage to the bus
    POST /control/halt          — trigger an emergency RiskOverride (trading halt)
    POST /control/test_order    — inject a minimal test order through MT5 (MT5 mode only)

These endpoints publish to Redis pub/sub. Agents that subscribe to
system.command or system.risk_override will act on the messages. There is no
acknowledgement — the publish is fire-and-forget.

/control/halt publishes a RiskOverride with requires_human_reset=True. This
causes every agent's inherited _risk_override_listener to set _trading_halted
and call stop() on itself. To resume trading, the agents must be restarted
manually (or via orchestration).

/control/test_order publishes a pre-approved RiskAssessment to risk.assessment.
The ExecutionAgent (in its own container) picks it up, calls MT5Bridge.place_order(),
and publishes an ExecutionResult. This endpoint waits up to 15 s for the result.
"""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.config import get_settings
from core.logging import get_logger
from core.messaging import Channels
from core.models.signals import AggregatedSignal, SignalDirection
from core.models.system import RiskOverride, SystemCommand, SystemCommandMessage
from core.models.trade import (
    ExecutionResult,
    OrderSide,
    OrderType,
    RiskAssessment,
    RiskDecision,
    TradeProposal,
)

logger = get_logger("api.control")

router = APIRouter(prefix="/control", tags=["control"])


class CommandRequest(BaseModel):
    """Body for POST /control/command."""

    command: SystemCommand
    target_agent: str = "all"
    reason: str | None = None


class HaltRequest(BaseModel):
    """Body for POST /control/halt."""

    reason: str = "Manual halt via API"


@router.post("/command")
async def post_command(body: CommandRequest, request: Request) -> dict:
    """
    Publish a SystemCommandMessage to the system.command channel.

    Agents that subscribe to this channel and match the target_agent (or
    target_agent='all') will act on the command. Agents that do not currently
    subscribe to system.command will not receive it.
    """
    bus = getattr(request.app.state, "bus", None)
    if bus is None:
        raise HTTPException(status_code=503, detail="Message bus not available")

    msg = SystemCommandMessage(
        command=body.command,
        target_agent=body.target_agent,
        issued_by="api",
        reason=body.reason,
    )
    try:
        count = await bus.publish(msg)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Publish failed: {exc}") from exc

    return {
        "ok": True,
        "command": body.command,
        "target_agent": body.target_agent,
        "command_id": str(msg.command_id),
        "subscribers_notified": count,
    }


@router.post("/halt")
async def post_halt(body: HaltRequest, request: Request) -> dict:
    """
    Trigger an emergency trading halt by publishing a RiskOverride.

    All agents inherit a _risk_override_listener that sets _trading_halted=True
    on receipt. When requires_human_reset=True (always the case here), agents
    also call their own stop() — requiring manual restart to resume.

    This endpoint is the operator's emergency stop button.
    """
    bus = getattr(request.app.state, "bus", None)
    if bus is None:
        raise HTTPException(status_code=503, detail="Message bus not available")

    override = RiskOverride(
        reason=body.reason,
        triggered_by="api:halt",
        requires_human_reset=True,
    )
    try:
        count = await bus.publish(override)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Publish failed: {exc}") from exc

    return {
        "ok": True,
        "override_id": str(override.override_id),
        "reason": body.reason,
        "requires_human_reset": True,
        "subscribers_notified": count,
    }


@router.post("/test_order")
async def post_test_order(request: Request) -> dict:
    """
    Inject a minimal BUY order through the full MT5 execution stack.

    Builds a pre-approved RiskAssessment and publishes it to risk.assessment.
    The ExecutionAgent (running in its own container) picks it up, calls
    MT5Bridge.place_order(), and publishes an ExecutionResult back to Redis.
    This endpoint waits up to 15 seconds for the matching result and returns it.

    Only available when EXECUTION_BROKER=mt5. Returns 400 otherwise.
    """
    settings = get_settings()
    if settings.execution_broker.lower() != "mt5":
        raise HTTPException(status_code=400, detail="test_order only available in MT5 mode")

    bus = getattr(request.app.state, "bus", None)
    if bus is None or bus._pool is None:  # noqa: SLF001
        raise HTTPException(status_code=503, detail="Message bus not available")

    proposal_id = uuid4()
    assessment_id = uuid4()
    symbol = settings.market_data.symbols[0]  # e.g. "BTC/USDT"

    # Minimal signal — no technical_signal means stops default to 0 (no SL/TP)
    signal = AggregatedSignal(
        symbol=symbol,
        direction=SignalDirection.BUY,
        confidence=1.0,
        composite_score=1.0,
    )
    proposal = TradeProposal(
        proposal_id=proposal_id,
        symbol=symbol,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        requested_size_usd=Decimal("10.0"),
        signal=signal,
        reasoning="API test_order — manual MT5 connectivity test",
    )
    # Set approved_quantity directly so MT5Bridge skips price-based volume calc
    assessment = RiskAssessment(
        assessment_id=assessment_id,
        proposal_id=proposal_id,
        decision=RiskDecision.APPROVED,
        approved_quantity=Decimal("0.01"),
        approved_stop_loss_pct=0.02,
        approved_take_profit_pct=0.04,
        original_proposal=proposal,
    )

    logger.info(
        "test_order_triggered",
        symbol=symbol,
        side="buy",
        quantity="0.01",
        proposal_id=str(proposal_id),
    )

    # Subscribe BEFORE publishing so we cannot miss the result
    pubsub = bus._pool.pubsub(ignore_subscribe_messages=True)  # noqa: SLF001
    await pubsub.subscribe(Channels.EXECUTION_RESULT)
    await asyncio.sleep(0.1)  # give Redis time to register the subscription

    try:
        await bus.publish(assessment)

        # Poll for the matching ExecutionResult (up to 20 s)
        loop = asyncio.get_event_loop()
        deadline = loop.time() + 20.0
        while loop.time() < deadline:
            raw = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if raw and raw.get("type") == "message":
                try:
                    result = ExecutionResult.model_validate_json(raw["data"])
                    if result.proposal_id == proposal_id:
                        logger.info(
                            "test_order_result_received",
                            status=result.status,
                            fill_price=float(result.average_fill_price or 0),
                            fill_qty=float(result.filled_quantity),
                            order_id=result.exchange_order_id,
                        )
                        return {
                            "ok": True,
                            "proposal_id": str(proposal_id),
                            "result": json.loads(result.to_json()),
                        }
                except Exception:
                    pass
            await asyncio.sleep(0)

        raise HTTPException(
            status_code=504,
            detail=f"Timeout: no ExecutionResult for {proposal_id} within 20 s",
        )
    finally:
        try:
            await pubsub.unsubscribe()
            await pubsub.aclose()
        except Exception:
            pass
