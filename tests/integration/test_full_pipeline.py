"""
tests/integration/test_full_pipeline.py
-----------------------------------------
Integration tests for Phase 4: ExecutionAgent + PaperBroker.

Requirements:
    - Running Redis instance (docker-compose up -d redis)
    - TRADING_MODE=paper (default — safe)

These tests use real MessageBus connections against Redis. Each test:
  - Starts an ExecutionAgent inside the same event loop
  - Publishes messages to risk.assessment or system.risk_override
  - Subscribes to execution.result and asserts the expected outcome

All tests are skipped automatically if Redis is unreachable.

Run:
    docker-compose up -d redis
    pytest tests/integration/test_full_pipeline.py -v
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from core.messaging import Channels, MessageBus
from core.models.signals import AggregatedSignal, SignalDirection, TechnicalSignal
from core.models.system import RiskOverride
from core.models.trade import (
    ExecutionResult,
    OrderSide,
    OrderStatus,
    OrderType,
    RiskAssessment,
    RiskDecision,
    TradeProposal,
)

# ---------------------------------------------------------------------------
# Redis availability check (runs once at collection time)
# ---------------------------------------------------------------------------


def _check_redis() -> bool:
    """Synchronous check — used by pytest.mark.skipif."""

    async def _ping() -> bool:
        try:
            bus = MessageBus()
            await bus.connect()
            ok = await bus.ping()
            await bus.disconnect()
            return ok
        except Exception:
            return False

    try:
        return asyncio.get_event_loop().run_until_complete(_ping())
    except RuntimeError:
        # No event loop in collection phase on some platforms
        return asyncio.run(_ping())


_REDIS_AVAILABLE = _check_redis()

pytestmark = pytest.mark.skipif(
    not _REDIS_AVAILABLE,
    reason="Redis not available — start with: docker-compose up -d redis",
)

# ---------------------------------------------------------------------------
# Test data builders
# ---------------------------------------------------------------------------

_FUTURE = datetime(2099, 1, 1, tzinfo=UTC)


def _make_assessment(
    side: OrderSide = OrderSide.BUY,
    approved_usd: float = 200.0,
    decision: RiskDecision = RiskDecision.APPROVED,
) -> RiskAssessment:
    tech = TechnicalSignal(
        symbol="BTC/USDT",
        timeframe="1m",
        expires_at=_FUTURE,
        direction=SignalDirection.BUY,
        confidence=0.8,
        price=42_000.0,
    )
    signal = AggregatedSignal(
        symbol="BTC/USDT",
        direction=SignalDirection.BUY,
        confidence=0.8,
        composite_score=0.5,
        technical_signal=tech,
        total_signals=1,
    )
    proposal = TradeProposal(
        symbol="BTC/USDT",
        side=side,
        order_type=OrderType.MARKET,
        requested_size_usd=Decimal(str(approved_usd)),
        signal=signal,
        reasoning="integration test",
    )
    approved = Decimal(str(approved_usd)) if decision != RiskDecision.REJECTED else None
    return RiskAssessment(
        proposal_id=proposal.proposal_id,
        decision=decision,
        approved_size_usd=approved,
        original_proposal=proposal,
    )


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


async def _start_agent():
    """Create and start an ExecutionAgent; return (agent, run_task)."""
    from agents.execution.agent import ExecutionAgent

    agent = ExecutionAgent()
    run_task = asyncio.create_task(agent.run(), name="execution_agent_test")
    await asyncio.sleep(0.5)  # allow agent to connect and subscribe
    return agent, run_task


async def _stop_agent(agent, run_task):
    """Signal the agent to stop and await the task."""
    await agent.stop()
    run_task.cancel()
    try:
        await asyncio.wait_for(run_task, timeout=2.0)
    except (TimeoutError, asyncio.CancelledError):
        pass


async def _wait_for_result(
    bus: MessageBus,
    proposal_id,
    timeout: float = 8.0,
) -> ExecutionResult | None:
    """
    Subscribe to execution.result and return the first result whose
    proposal_id matches, or None if timeout elapses first.
    """
    result: ExecutionResult | None = None

    async def _collect():
        nonlocal result
        async for msg in bus.subscribe(Channels.EXECUTION_RESULT, ExecutionResult):
            if msg.proposal_id == proposal_id:
                result = msg
                break

    try:
        await asyncio.wait_for(_collect(), timeout=timeout)
    except TimeoutError:
        pass
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def pub_bus():
    """Publisher bus for sending messages in tests."""
    bus = MessageBus()
    await bus.connect()
    yield bus
    await bus.disconnect()


@pytest.fixture
async def sub_bus():
    """Subscriber bus — separate connection to avoid shared pubsub state."""
    bus = MessageBus()
    await bus.connect()
    yield bus
    await bus.disconnect()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """
    End-to-end integration: RiskAssessment → ExecutionAgent → ExecutionResult.
    """

    async def test_approved_assessment_produces_execution_result(self, pub_bus, sub_bus):
        """
        An approved RiskAssessment published to risk.assessment must cause
        the ExecutionAgent to publish an ExecutionResult on execution.result.

        ExecutionResult must have:
        - is_paper = True
        - status ∈ {FILLED, PARTIALLY_FILLED}
        - proposal_id matching the original assessment
        """
        assessment = _make_assessment(side=OrderSide.BUY, approved_usd=200.0)
        agent, run_task = await _start_agent()

        try:
            # Start result collector concurrently
            collect_task = asyncio.create_task(
                _wait_for_result(sub_bus, assessment.proposal_id, timeout=8.0)
            )

            # Publish the approved assessment
            await pub_bus.publish(assessment)

            result: ExecutionResult | None = await collect_task
        finally:
            await _stop_agent(agent, run_task)

        assert result is not None, (
            "ExecutionResult not received within timeout. "
            "Check that ExecutionAgent is subscribed to risk.assessment."
        )
        assert result.is_paper is True
        assert result.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)
        assert result.proposal_id == assessment.proposal_id
        assert result.symbol == "BTC/USDT"

    async def test_rejected_assessment_produces_no_execution_result(self, pub_bus, sub_bus):
        """
        A REJECTED RiskAssessment must NOT trigger an ExecutionResult.
        The ExecutionAgent must silently skip rejected assessments.
        """
        assessment = _make_assessment(decision=RiskDecision.REJECTED)
        agent, run_task = await _start_agent()

        try:
            collect_task = asyncio.create_task(
                _wait_for_result(sub_bus, assessment.proposal_id, timeout=3.0)
            )
            await pub_bus.publish(assessment)
            result: ExecutionResult | None = await collect_task
        finally:
            await _stop_agent(agent, run_task)

        assert result is None, (
            "ExecutionResult must NOT be published for a rejected RiskAssessment."
        )

    async def test_risk_override_halts_execution(self, pub_bus, sub_bus):
        """
        After a RiskOverride is published, the ExecutionAgent must not execute
        any further approved assessments.

        requires_human_reset=False so the agent stays alive (we can observe it
        skipping assessments rather than shutting down mid-test).
        """
        assessment = _make_assessment(side=OrderSide.BUY, approved_usd=200.0)
        override = RiskOverride(
            reason="Integration test halt",
            triggered_by="test_suite",
            requires_human_reset=False,
        )

        agent, run_task = await _start_agent()

        try:
            # Publish override first; give the background listener time to act
            await pub_bus.publish(override)
            await asyncio.sleep(0.5)

            # Now publish an approved assessment — should be silently ignored
            collect_task = asyncio.create_task(
                _wait_for_result(sub_bus, assessment.proposal_id, timeout=3.0)
            )
            await pub_bus.publish(assessment)
            result: ExecutionResult | None = await collect_task
        finally:
            await _stop_agent(agent, run_task)

        assert result is None, "ExecutionResult must NOT be published after a RiskOverride halt."

    async def test_execution_result_cached_to_redis(self, pub_bus, sub_bus):
        """
        After a successful fill, ExecutionAgent must write balance and
        positions to the well-known Redis keys (execution:balance, execution:positions).
        """
        import json

        assessment = _make_assessment(side=OrderSide.BUY, approved_usd=200.0)
        agent, run_task = await _start_agent()

        try:
            collect_task = asyncio.create_task(
                _wait_for_result(sub_bus, assessment.proposal_id, timeout=8.0)
            )
            await pub_bus.publish(assessment)
            result = await collect_task
        finally:
            await _stop_agent(agent, run_task)

        assert result is not None, "Need a successful fill to test Redis caching"

        # Check Redis keys written by ExecutionAgent
        balance_raw = await pub_bus._pool.get("execution:balance")  # noqa: SLF001
        positions_raw = await pub_bus._pool.get("execution:positions")  # noqa: SLF001

        assert balance_raw is not None, "execution:balance key missing in Redis"
        assert positions_raw is not None, "execution:positions key missing in Redis"

        balance = json.loads(balance_raw)
        assert "total_equity_usd" in balance
        assert "free_margin_usd" in balance

        positions = json.loads(positions_raw)
        assert isinstance(positions, list)
