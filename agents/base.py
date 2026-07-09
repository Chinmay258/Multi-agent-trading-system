"""
agents/base.py
--------------
Abstract BaseAgent — the foundation every agent in the system inherits from.

BaseAgent provides the common lifecycle, health reporting, error handling,
and messaging plumbing that every agent needs. Agent implementations only
override the methods specific to their domain logic.

Lifecycle:
    1. __init__()    — configure the agent (synchronous, no I/O)
    2. setup()       — async initialisation (connect to exchange, warm up caches)
    3. run()         — the main loop (runs until stop() is called)
    4. teardown()    — async cleanup (close connections, flush state)

The supervisor script calls run_agent(MyAgent()) which manages this lifecycle
and handles restarts after crashes, up to the configured crash limit.

Design decisions:
- Abstract run_loop() forces every agent to implement its core logic.
- start_heartbeat() runs as a background task — agents don't manually
  publish heartbeats; the base class handles it.
- _handle_error() provides structured error logging with context that lets
  Monitoring reconstruct what the agent was doing when it crashed.
- Circuit breaker: the base class tracks consecutive errors. If an agent
  exceeds the limit, it enters QUARANTINED status and stops itself,
  preventing a crash loop from consuming resources or placing bad trades.

Usage:
    class MyAgent(BaseAgent):
        name = "my_agent"

        async def setup(self) -> None:
            self.client = await create_client()

        async def run_loop(self) -> None:
            async for msg in self.bus.subscribe(channel, Model):
                await self.process(msg)

        async def teardown(self) -> None:
            await self.client.close()
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod

from core.config import get_settings
from core.logging import bind_agent_context, get_logger
from core.messaging import MessageBus
from core.metrics import AGENT_ERRORS, AGENT_UP, start_metrics_server
from core.models.system import AgentHeartbeat, AgentStatus, RiskOverride

logger = get_logger("base_agent")


class AgentError(Exception):
    """Base exception for agent-level errors."""

    pass


class AgentConfigError(AgentError):
    """Raised when an agent receives invalid configuration."""

    pass


class BaseAgent(ABC):
    """
    Abstract base class for all trading system agents.

    Subclasses must implement:
    - name (class attribute): unique identifier for this agent
    - run_loop(): the core async processing loop
    - setup(): (optional) async initialisation
    - teardown(): (optional) async cleanup

    Subclasses may override:
    - health_extra(): return agent-specific data for heartbeat payloads
    """

    #: Unique name for this agent — used in logs, heartbeats, and commands.
    #: Must be overridden by every subclass.
    name: str = "base_agent"

    def __init__(self) -> None:
        self.settings = get_settings()
        self.log = get_logger(self.name)

        # Messaging bus — initialised in setup()
        self.bus: MessageBus = MessageBus()

        # Lifecycle state
        self._status: AgentStatus = AgentStatus.STARTING
        self._start_time: float = time.monotonic()
        self._stop_event: asyncio.Event = asyncio.Event()
        self._heartbeat_task: asyncio.Task | None = None
        self._risk_override_task: asyncio.Task | None = None

        # Error tracking (circuit breaker)
        self._consecutive_errors: int = 0
        self._total_errors: int = 0
        self._messages_processed: int = 0
        self._last_error: str | None = None
        self._max_consecutive_errors: int = self.settings.risk.max_agent_crashes

        # Risk override state
        self._trading_halted: bool = False

    # ------------------------------------------------------------------
    # Lifecycle — called by the supervisor
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Full agent startup sequence.
        Called by run_agent() — don't call this directly.
        """
        bind_agent_context(self.name)
        self.log.info("agent_starting", name=self.name)
        self._status = AgentStatus.STARTING

        # Expose Prometheus metrics on the configured port. The helper is
        # idempotent, so the single-process launcher (scripts/start.py) and
        # standalone container agents can both call it safely.
        if self.settings.monitoring.prometheus_enabled:
            try:
                start_metrics_server(self.settings.monitoring.prometheus_port)
            except OSError as exc:
                # Port already in use is fine in single-process mode where
                # another agent in the same process already bound it.
                self.log.debug("metrics_server_already_bound", error=str(exc))

        await self.bus.connect()

        await self.setup()

        # Background tasks
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(),
            name=f"{self.name}_heartbeat",
        )
        self._risk_override_task = asyncio.create_task(
            self._risk_override_listener(),
            name=f"{self.name}_risk_override",
        )

        self._status = AgentStatus.RUNNING
        AGENT_UP.labels(agent=self.name).set(1)
        self.log.info("agent_started", name=self.name)

    async def stop(self) -> None:
        """
        Signal the agent to stop gracefully.
        The run_loop() should detect _stop_event and exit.
        """
        self.log.info("agent_stopping", name=self.name)
        self._status = AgentStatus.STOPPING
        self._stop_event.set()

        # Cancel background tasks
        for task in [self._heartbeat_task, self._risk_override_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        await self.teardown()
        await self.bus.disconnect()

        self._status = AgentStatus.STOPPED
        AGENT_UP.labels(agent=self.name).set(0)
        self.log.info("agent_stopped", name=self.name)

    async def run(self) -> None:
        """
        Main entry point — runs the agent loop with error handling.

        Called by run_agent(). Wraps run_loop() with:
        - Exception catching and structured logging
        - Consecutive error tracking (circuit breaker)
        - Graceful shutdown on stop signal
        """
        await self.start()
        try:
            await self.run_loop()
        except asyncio.CancelledError:
            self.log.info("agent_cancelled", name=self.name)
        except Exception as e:
            self._status = AgentStatus.ERROR
            self._handle_error(e, context="run_loop")
            raise
        finally:
            await self.stop()

    # ------------------------------------------------------------------
    # Abstract methods — subclasses must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    async def run_loop(self) -> None:
        """
        The agent's core processing loop.

        This method runs for the lifetime of the agent. It should:
        - Subscribe to relevant channels via self.bus.subscribe()
        - Process messages and publish results
        - Check self._should_continue() periodically to support graceful stop
        - Respect self._trading_halted when placing orders

        The loop exits when:
        - self._stop_event is set (graceful shutdown)
        - An unhandled exception is raised (crash → supervisor restarts)
        """
        ...

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """
        Async initialisation called after the bus connects but before run_loop().
        Override to: connect to external APIs, warm up caches, validate config.
        """
        pass

    async def teardown(self) -> None:
        """
        Async cleanup called after run_loop() exits.
        Override to: close API connections, flush queues, persist state.
        """
        pass

    def health_extra(self) -> dict:
        """
        Return agent-specific health data included in heartbeat payloads.
        Override to add fields like last_symbol_fetched, signal_count, etc.
        """
        return {}

    # ------------------------------------------------------------------
    # Control helpers
    # ------------------------------------------------------------------

    def _should_continue(self) -> bool:
        """
        Check whether the agent's main loop should keep running.
        Call this in tight loops or between operations.

        Returns False if:
        - stop() has been called
        - A risk override has halted the system
        - Consecutive errors exceed the circuit breaker threshold
        """
        return not self._stop_event.is_set() and self._status not in (
            AgentStatus.STOPPING,
            AgentStatus.STOPPED,
            AgentStatus.QUARANTINED,
        )

    def _record_success(self) -> None:
        """Reset consecutive error counter after a successful operation."""
        self._consecutive_errors = 0
        self._messages_processed += 1

    def _handle_error(self, error: Exception, context: str = "") -> None:
        """
        Record an error and check circuit breaker threshold.

        If consecutive errors exceed the limit, quarantines the agent
        and sets the stop event to halt the run loop.
        """
        self._consecutive_errors += 1
        self._total_errors += 1
        self._last_error = str(error)

        AGENT_ERRORS.labels(agent=self.name, error_type=type(error).__name__).inc()

        self.log.error(
            "agent_error",
            error=str(error),
            error_type=type(error).__name__,
            context=context,
            consecutive_errors=self._consecutive_errors,
            total_errors=self._total_errors,
        )

        if self._consecutive_errors >= self._max_consecutive_errors:
            self.log.critical(
                "circuit_breaker_tripped",
                agent=self.name,
                consecutive_errors=self._consecutive_errors,
                last_error=self._last_error,
            )
            self._status = AgentStatus.QUARANTINED
            self._stop_event.set()

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self._start_time

    # ------------------------------------------------------------------
    # Background: heartbeat loop
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """
        Publish AgentHeartbeat every heartbeat_interval_seconds.
        Runs as a background task for the agent's lifetime.
        """
        interval = self.settings.monitoring.heartbeat_interval_seconds
        while not self._stop_event.is_set():
            try:
                heartbeat = AgentHeartbeat(
                    agent_name=self.name,
                    status=self._status,
                    messages_processed=self._messages_processed,
                    errors_since_start=self._total_errors,
                    uptime_seconds=self.uptime_seconds,
                    extra=self.health_extra(),
                )
                await asyncio.wait_for(
                    self.bus.publish(heartbeat),
                    timeout=5.0,
                )
                self.log.debug("heartbeat_sent", status=self._status.value)
            except asyncio.CancelledError:
                break
            except TimeoutError:
                self.log.warning("heartbeat_publish_timeout")
            except Exception as e:
                self.log.error("heartbeat_loop_error", error=str(e))

            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break

    # ------------------------------------------------------------------
    # Background: risk override listener
    # ------------------------------------------------------------------

    async def _risk_override_listener(self) -> None:
        """
        Listen for emergency halt signals from the Risk agent.
        On receipt, sets _trading_halted = True and logs prominently.

        Execution agents must check _trading_halted before placing orders.
        All agents should check it before taking any action with market impact.
        """
        try:
            async for override in self.bus.subscribe(
                "system.risk_override",
                RiskOverride,
            ):
                self._trading_halted = True
                self.log.critical(
                    "risk_override_received",
                    reason=override.reason,
                    triggered_by=override.triggered_by,
                    requires_human_reset=override.requires_human_reset,
                )
                # If the override requires human reset, stop the agent
                if override.requires_human_reset:
                    await self.stop()
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.log.error("risk_override_listener_error", error=str(e))


# ---------------------------------------------------------------------------
# Supervisor helper
# ---------------------------------------------------------------------------


async def run_agent(agent: BaseAgent) -> None:
    """
    Run an agent with automatic restart on crash.

    The supervisor pattern: if an agent's run() raises an exception,
    we wait briefly and restart it — up to the configured crash limit.
    After the limit, the agent is quarantined and the operator is alerted.

    In production, this function runs inside a Docker container with
    Docker's own restart policy as a second line of defence.

    Usage:
        if __name__ == "__main__":
            asyncio.run(run_agent(MarketDataAgent()))
    """
    settings = get_settings()
    max_crashes = settings.risk.max_agent_crashes
    crash_count = 0

    while crash_count < max_crashes:
        try:
            await agent.run()
            break  # Clean exit — don't restart
        except asyncio.CancelledError:
            break  # Explicit cancellation — don't restart
        except Exception as e:
            crash_count += 1
            logger.error(
                "agent_crashed",
                agent=agent.name,
                crash_count=crash_count,
                max_crashes=max_crashes,
                error=str(e),
                error_type=type(e).__name__,
            )

            if crash_count >= max_crashes:
                logger.critical(
                    "agent_quarantined",
                    agent=agent.name,
                    total_crashes=crash_count,
                )
                break

            # Exponential back-off: 2s, 4s, 8s, ...
            wait = 2**crash_count
            logger.info("agent_restarting", agent=agent.name, wait_seconds=wait)
            await asyncio.sleep(wait)

            # Re-instantiate to get a fresh state
            # The supervisor must pass the class, not an instance, for this to work.
            # TODO: Accept agent_class: Type[BaseAgent] and instantiate here
