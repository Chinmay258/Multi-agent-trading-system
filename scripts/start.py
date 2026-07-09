"""
scripts/start.py
----------------
Single-process orchestrator that boots every agent + the FastAPI control plane
in one asyncio event loop.

Use this for local development, smoke tests, and lightweight deployments where
running each agent in its own container is overkill. In production we still
recommend ``docker-compose up`` because each container then has its own restart
policy, log stream, and resource limits.

Dependency order (matches the data flow)
----------------------------------------
1. MonitoringAgent          — independent; comes up first so it can observe
                              every subsequent agent's heartbeat.
2. MarketDataAgent          — needs Redis + Postgres reachable.
3. TechnicalAnalysisAgent   — subscribes to candles published by (2).
4. SentimentAgent           — only started when ``SENTIMENT_ENABLED=true``.
5. DecisionAgent            — consumes (3) and optionally (4).
6. RiskAgent                — gates (5).
7. ExecutionAgent           — acts on Risk-approved assessments.
8. FastAPI server           — exposes /health, /control, /positions, /ws.

Shutdown
--------
SIGINT / SIGTERM (or any agent task crashing) set a shared stop event. We then
cancel running tasks in **reverse start order** so the closest-to-the-money
component (ExecutionAgent) is quieted before its upstream stops sending fresh
proposals.

On Windows ``loop.add_signal_handler`` is not implemented; we fall back to
``signal.signal`` and post the stop event via ``loop.call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from types import FrameType

import structlog
import uvicorn

from agents.base import BaseAgent, run_agent
from agents.decision.agent import DecisionAgent
from agents.execution.agent import ExecutionAgent
from agents.market_data.agent import MarketDataAgent
from agents.monitoring.agent import MonitoringAgent
from agents.risk.agent import RiskAgent
from agents.technical_analysis.agent import TechnicalAnalysisAgent
from core.config import Settings, get_settings
from core.logging import configure_logging, get_logger
from core.metrics import start_metrics_server


def _print_banner(settings: Settings) -> None:
    """Write a one-shot human-readable startup summary to stdout."""
    line = "=" * 64
    print(line, flush=True)
    print(" TRADING SYSTEM — Phase 6 single-process launcher", flush=True)
    print(line, flush=True)
    print(f"  trading_mode       : {settings.trading_mode.value}", flush=True)
    print(f"  environment        : {settings.environment.value}", flush=True)
    print(f"  symbols            : {', '.join(settings.market_data.symbols)}", flush=True)
    print(f"  timeframes         : {', '.join(settings.market_data.ohlcv_timeframes)}", flush=True)
    print(f"  paper balance USD  : {settings.paper_initial_balance_usd}", flush=True)
    print(f"  sentiment enabled  : {settings.sentiment.enabled}", flush=True)
    print(f"  api listen         : http://{settings.api_host}:{settings.api_port}", flush=True)
    if settings.monitoring.prometheus_enabled:
        prom = f"http://0.0.0.0:{settings.monitoring.prometheus_port}"
    else:
        prom = "disabled"
    print(f"  prometheus listen  : {prom}", flush=True)
    print(line, flush=True)


def _build_agents(settings: Settings, log: structlog.stdlib.BoundLogger) -> list[BaseAgent]:
    """Instantiate the agent list in start order, honouring feature flags."""
    agents: list[BaseAgent] = [
        MonitoringAgent(),
        MarketDataAgent(),
        TechnicalAnalysisAgent(),
    ]

    if settings.sentiment.enabled:
        # SentimentAgent is a Phase 7+ stub today; import on demand so the
        # absence of an implementation doesn't crash the launcher when the
        # feature flag is off (the common case in MVP deployments).
        try:
            from agents.sentiment.agent import SentimentAgent  # type: ignore

            agents.append(SentimentAgent())
            log.info("sentiment_agent_enabled")
        except ImportError as exc:
            log.warning(
                "sentiment_agent_unavailable",
                detail="SENTIMENT_ENABLED=true but agents.sentiment.agent is not implemented",
                error=str(exc),
            )

    agents.extend(
        [
            DecisionAgent(),
            RiskAgent(),
            ExecutionAgent(),
        ]
    )
    return agents


async def _run_api_server(settings: Settings) -> None:
    """Run uvicorn inside this event loop so it cooperates with cancellation."""
    config = uvicorn.Config(
        app="api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_config=None,  # let our structlog handler own root logging
        access_log=False,  # access logs are noisy in dev; FastAPI logs handle the rest
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    # Server.serve() returns when server.should_exit becomes True, which we
    # trigger by cancelling the task wrapping this coroutine.
    await server.serve()


def _install_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    stop_event: asyncio.Event,
    log: structlog.stdlib.BoundLogger,
) -> None:
    """Wire SIGINT / SIGTERM to the shared stop event."""

    def _trigger() -> None:
        if not stop_event.is_set():
            log.info("shutdown_signal_received")
            stop_event.set()

    if sys.platform == "win32":
        # Windows asyncio does not support add_signal_handler; fall back to
        # the synchronous signal module and post into the loop thread-safely.
        def _win_handler(signum: int, frame: FrameType | None) -> None:
            loop.call_soon_threadsafe(_trigger)

        signal.signal(signal.SIGINT, _win_handler)
        try:
            signal.signal(signal.SIGTERM, _win_handler)
        except (AttributeError, ValueError):
            # SIGTERM is not always present on Windows builds; SIGINT is enough.
            pass
    else:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _trigger)


async def _shutdown(
    tasks: list[asyncio.Task],
    log: structlog.stdlib.BoundLogger,
) -> None:
    """Cancel running tasks in reverse start order and await them."""
    log.info("shutting_down", tasks=len(tasks))
    for task in reversed(tasks):
        if not task.done():
            task.cancel()
    # gather with return_exceptions so a single misbehaving task can't block
    # the rest from being awaited (and prevents an "unawaited coroutine" warning).
    await asyncio.gather(*tasks, return_exceptions=True)
    log.info("shutdown_complete")


async def _watch_for_crash(
    tasks: list[asyncio.Task],
    stop_event: asyncio.Event,
    log: structlog.stdlib.BoundLogger,
) -> None:
    """
    If any supervised task exits before stop_event is set, treat it as a crash:
    log critically and trigger the shared stop. The supervisor in run_agent()
    already restarts crashes within its limit; reaching this point means we've
    exhausted those restarts and the system can't make forward progress.
    """
    while not stop_event.is_set():
        for task in tasks:
            if task.done():
                exc: BaseException | None = task.exception() if not task.cancelled() else None
                log.critical(
                    "agent_task_crashed",
                    agent=task.get_name(),
                    error=str(exc) if exc else "task completed unexpectedly",
                )
                stop_event.set()
                return
        await asyncio.sleep(0.5)


async def main() -> None:
    """Entry point — orchestrates the entire system lifecycle."""
    configure_logging()
    log = get_logger("start")

    settings = get_settings()
    _print_banner(settings)

    if settings.monitoring.prometheus_enabled:
        start_metrics_server(settings.monitoring.prometheus_port)
        log.info("metrics_server_started", port=settings.monitoring.prometheus_port)

    agents = _build_agents(settings, log)
    log.info("agents_built", count=len(agents), names=[a.name for a in agents])

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    _install_signal_handlers(loop, stop_event, log)

    tasks: list[asyncio.Task] = []
    for agent in agents:
        task = asyncio.create_task(run_agent(agent), name=agent.name)
        tasks.append(task)

    api_task = asyncio.create_task(_run_api_server(settings), name="api_server")
    tasks.append(api_task)

    crash_watcher = asyncio.create_task(
        _watch_for_crash(tasks, stop_event, log),
        name="crash_watcher",
    )

    try:
        await stop_event.wait()
    finally:
        crash_watcher.cancel()
        await asyncio.gather(crash_watcher, return_exceptions=True)
        await _shutdown(tasks, log)


if __name__ == "__main__":
    asyncio.run(main())
