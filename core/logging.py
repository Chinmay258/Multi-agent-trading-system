"""
core/logging.py
---------------
Structured JSON logging for the trading system.

Uses structlog for structured, context-rich logs. Every log entry is a JSON
object with consistent fields, making it trivially parseable by Loki, ELK,
Datadog, or any log aggregation system.

Design decisions:
- structlog over standard logging: structured context binding (agent name,
  symbol, correlation IDs) travels with the log record without polluting
  every log call with extra kwargs.
- JSON in production, human-readable console in development: same code path,
  different renderer — controlled by environment.
- get_logger() is the single entry point. Every module calls it with a name.
- Async-safe: structlog's async-compatible processors are used where needed.
- No global state mutation after setup: configure_logging() is called once
  at process startup, then each agent calls get_logger() with its context.

Usage:
    from core.logging import get_logger, configure_logging
    configure_logging()  # once at startup

    logger = get_logger("market_data_agent")
    logger.info("candle_received", symbol="BTC/USDT", close=42000.0, volume=123.4)
    logger.error("fetch_failed", symbol="BTC/USDT", error=str(e), retry=3)

    # Bind persistent context for a session
    log = logger.bind(symbol="BTC/USDT", agent="market_data")
    log.info("stream_started")
    log.warning("gap_detected", missing_candles=3)
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger

from core.config import Environment, get_settings

# ---------------------------------------------------------------------------
# Custom processors
# ---------------------------------------------------------------------------


def add_trading_context(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """
    Inject system-level context into every log entry.
    Keeps the core identity fields consistent across all agents.
    """
    settings = get_settings()
    event_dict.setdefault("app", settings.app_name)
    event_dict.setdefault("env", settings.environment.value)
    event_dict.setdefault("trading_mode", settings.trading_mode.value)
    return event_dict


def drop_color_message_key(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """
    Remove uvicorn's 'color_message' key — it's ANSI noise in JSON logs.
    Safe no-op if the key doesn't exist.
    """
    event_dict.pop("color_message", None)
    return event_dict


def sanitise_secrets(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """
    Scrub known secret field names from log entries.
    Defense-in-depth: secrets should never reach log calls, but just in case.
    """
    SECRET_KEYS = {"api_key", "api_secret", "password", "token", "secret", "passphrase"}
    for key in list(event_dict.keys()):
        if any(s in key.lower() for s in SECRET_KEYS):
            event_dict[key] = "***REDACTED***"
    return event_dict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def configure_logging() -> None:
    """
    Configure structlog and the stdlib logging root logger.

    Call this ONCE at process startup (before any agents are created).
    Subsequent calls are idempotent — structlog ignores re-configuration.

    Output format:
    - Development: coloured, human-readable console output
    - Staging/Production: JSON, one object per line, stdout
    """
    settings = get_settings()
    is_dev = settings.environment == Environment.DEVELOPMENT
    log_level = settings.log_level.value

    # Shared processors run on every log entry regardless of renderer
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,  # thread-local context
        structlog.stdlib.add_logger_name,  # logger name field
        structlog.stdlib.add_log_level,  # level field
        structlog.stdlib.ExtraAdder(),  # extra= kwargs from stdlib
        structlog.processors.TimeStamper(fmt="iso"),  # ISO8601 timestamp
        drop_color_message_key,
        sanitise_secrets,
        add_trading_context,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,  # exception tracebacks
    ]

    if is_dev:
        # Pretty console output for local development
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        # JSON output for log aggregation systems
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Wire structlog into stdlib logging (for third-party libraries like CCXT)
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Quieten noisy third-party loggers
    for noisy in ("ccxt", "aiohttp", "asyncio", "urllib3", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------


def get_logger(name: str, **initial_context: Any) -> structlog.stdlib.BoundLogger:
    """
    Return a structlog BoundLogger for the given name.

    The name should identify the component — typically the agent name or
    module path. Any keyword args become persistent context on every log
    entry from this logger instance.

    Args:
        name: Component identifier (e.g. "market_data_agent", "risk_agent")
        **initial_context: Key-value pairs bound to every log entry

    Returns:
        A BoundLogger ready to call .info(), .warning(), .error() etc.

    Example:
        logger = get_logger("execution_agent", symbol="BTC/USDT", mode="paper")
        logger.info("order_placed", order_id="abc123", size=0.01)
        # → {"event": "order_placed", "logger": "execution_agent",
        #    "symbol": "BTC/USDT", "mode": "paper", "order_id": "abc123",
        #    "size": 0.01, "level": "info", "timestamp": "..."}
    """
    return structlog.get_logger(name).bind(**initial_context)


# ---------------------------------------------------------------------------
# Correlation ID helpers
# ---------------------------------------------------------------------------


def bind_request_id(request_id: str) -> None:
    """
    Bind a correlation/request ID to the current async context.
    All logs within the same asyncio task will carry this ID.
    Useful for tracing a trade proposal through all agents.
    """
    structlog.contextvars.bind_contextvars(request_id=request_id)


def clear_request_id() -> None:
    """Clear the correlation ID from the current async context."""
    structlog.contextvars.unbind_contextvars("request_id")


def bind_agent_context(agent_name: str, **kwargs: Any) -> None:
    """
    Bind agent-level context to all logs in the current async task.
    Call once per agent startup to avoid repeating agent= on every log call.
    """
    structlog.contextvars.bind_contextvars(agent=agent_name, **kwargs)
