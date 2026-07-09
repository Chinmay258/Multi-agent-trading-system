"""
core/metrics.py
---------------
Prometheus metrics registry — the single source of truth for all
system-level instrumentation.

Every metric the system exposes is declared here. Agents import the
relevant constants (e.g. ``AGENT_UP``, ``ORDERS_PLACED``) and call
``.labels(...).inc()`` / ``.set()`` / ``.observe()`` on them. No agent
constructs its own Counter/Gauge/Histogram — that keeps the metric
namespace coherent and prevents accidental duplicates that would raise
``ValueError: Duplicated timeseries`` at import time.

Why these specific metrics
--------------------------
- **Counters** answer "how many?" questions and only ever go up.
  They drive rate panels (``rate(metric[5m])``) and alert on spikes.
- **Gauges** answer "what is it right now?" questions and can move
  in either direction. They drive value panels and threshold alerts.
- **Histograms** answer "how long / how big?" questions and capture
  the distribution. They drive p95/p99 latency panels.

Design notes
------------
- We use the default global REGISTRY so ``start_http_server`` picks up
  every metric automatically. Tests get a handle via ``get_registry()``.
- ``start_metrics_server`` is idempotent: calling it twice is a no-op,
  so unit tests and lifecycle helpers can call it freely.
- ``time_histogram`` is a small context-manager helper around
  ``Histogram.labels(...).time()``; it makes call sites read naturally
  and avoids the easy-to-forget ``.labels(...)`` step.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

# ---------------------------------------------------------------------------
# Counters — monotonic, always increasing
# ---------------------------------------------------------------------------

MESSAGES_PUBLISHED: Counter = Counter(
    "trading_messages_published_total",
    "Messages published to the Redis pub/sub bus by an agent.",
    labelnames=("agent", "channel"),
)

MESSAGES_CONSUMED: Counter = Counter(
    "trading_messages_consumed_total",
    "Messages successfully consumed from the Redis pub/sub bus by an agent.",
    labelnames=("agent", "channel"),
)

AGENT_ERRORS: Counter = Counter(
    "trading_agent_errors_total",
    "Unhandled errors raised inside an agent loop, labelled by exception type.",
    labelnames=("agent", "error_type"),
)

ORDERS_PLACED: Counter = Counter(
    "trading_orders_placed_total",
    "Orders sent to the broker (paper or live), broken down by symbol and side.",
    labelnames=("symbol", "side", "mode"),
)

ORDERS_REJECTED: Counter = Counter(
    "trading_orders_rejected_total",
    "Trade proposals rejected by the Risk agent, labelled by rejection reason.",
    labelnames=("symbol", "reason"),
)


# ---------------------------------------------------------------------------
# Gauges — point-in-time values, can go up or down
# ---------------------------------------------------------------------------

AGENT_UP: Gauge = Gauge(
    "trading_agent_up",
    "1 if the agent process is running, 0 if it has stopped or quarantined.",
    labelnames=("agent",),
)

PORTFOLIO_VALUE: Gauge = Gauge(
    "trading_portfolio_value_usd",
    "Current paper portfolio value in USD (cash + open position cost basis).",
)

DAILY_PNL: Gauge = Gauge(
    "trading_portfolio_daily_pnl_usd",
    "Realised daily profit and loss in USD, reset at UTC midnight.",
)

OPEN_POSITIONS: Gauge = Gauge(
    "trading_open_positions_count",
    "Number of paper positions currently open.",
)

SIGNAL_CONFIDENCE: Gauge = Gauge(
    "trading_signal_confidence",
    "Confidence score [0,1] of the most recent technical signal per symbol.",
    labelnames=("symbol",),
)

DATA_AGE_SECONDS: Gauge = Gauge(
    "trading_data_age_seconds",
    "Seconds since the most recent OHLCV candle was received for this symbol.",
    labelnames=("symbol",),
)


# ---------------------------------------------------------------------------
# Histograms — distributions of timing / size
# ---------------------------------------------------------------------------

# Latency buckets chosen for sub-second indicator computation; aligned
# with the default Prometheus buckets but with finer resolution under 100ms.
_LATENCY_BUCKETS = (
    0.005,
    0.010,
    0.025,
    0.050,
    0.100,
    0.250,
    0.500,
    1.0,
    2.5,
    5.0,
    10.0,
)

SIGNAL_GENERATION_SECONDS: Histogram = Histogram(
    "trading_signal_generation_seconds",
    "Time taken to generate a TechnicalSignal from a warm candle buffer.",
    labelnames=("symbol",),
    buckets=_LATENCY_BUCKETS,
)

ORDER_FILL_LATENCY_SECONDS: Histogram = Histogram(
    "trading_order_fill_latency_seconds",
    "End-to-end latency of an order fill, measured at the broker layer.",
    labelnames=("mode",),
    buckets=_LATENCY_BUCKETS,
)

INDICATOR_COMPUTATION_SECONDS: Histogram = Histogram(
    "trading_indicator_computation_seconds",
    "Time taken to compute a single technical indicator (RSI/MACD/etc).",
    buckets=_LATENCY_BUCKETS,
)


# ---------------------------------------------------------------------------
# HTTP exporter
# ---------------------------------------------------------------------------

_server_started: bool = False


def start_metrics_server(port: int) -> None:
    """
    Start the Prometheus scrape endpoint on ``port`` if it is not running.

    Idempotent — calling multiple times is safe. The first call binds the
    HTTP server to ``0.0.0.0:<port>``; subsequent calls are no-ops, even
    when called from a different module or test.
    """
    global _server_started
    if _server_started:
        return
    start_http_server(port)
    _server_started = True


def get_registry() -> CollectorRegistry:
    """
    Return the default Prometheus registry used by every metric in this module.

    Exposed so tests can introspect metric values via
    ``registry.get_sample_value("name", labels)`` without reaching into
    private prometheus_client internals.
    """
    return REGISTRY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def time_histogram(histogram: Histogram, **labels: str) -> Iterator[None]:
    """
    Context manager that records the wall-clock duration of the enclosed block
    into ``histogram``. When ``labels`` are provided they are applied via
    ``histogram.labels(**labels)`` first; otherwise the unlabelled histogram is
    observed.

    Example:
        with time_histogram(SIGNAL_GENERATION_SECONDS, symbol="BTC/USDT"):
            signal = generator.generate(buffer)
    """
    target = histogram.labels(**labels) if labels else histogram
    start = time.perf_counter()
    try:
        yield
    finally:
        target.observe(time.perf_counter() - start)
