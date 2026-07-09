"""
core/exceptions.py
------------------
Domain exception hierarchy for the trading system.

Typed exceptions make error handling intentional rather than accidental.
They also make logs more searchable — "ExchangeConnectionError" is far more
useful than "Exception: connection refused" when debugging at 3am.

Convention:
- TradingSystemError: base for all domain exceptions
- Each subsystem has its own branch (MarketData, Signal, Trade, Risk, Execution)
- Agents catch specific exceptions and let others propagate to the supervisor

Usage:
    from core.exceptions import ExchangeConnectionError, RiskVetoError

    try:
        await exchange.fetch_ohlcv(symbol, timeframe)
    except ExchangeConnectionError as e:
        logger.error("exchange_unreachable", error=str(e))
        # retry logic...
    except ExchangeRateLimitError:
        await asyncio.sleep(backoff)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class TradingSystemError(Exception):
    """Base for all domain exceptions in this system."""

    pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigError(TradingSystemError):
    """Invalid or missing configuration."""

    pass


class LiveTradingNotEnabled(ConfigError):
    """Attempted live trading without explicit opt-in."""

    pass


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------


class MarketDataError(TradingSystemError):
    """Base for market data errors."""

    pass


class ExchangeConnectionError(MarketDataError):
    """Cannot connect to the exchange API."""

    pass


class ExchangeRateLimitError(MarketDataError):
    """Hit the exchange rate limit — back off."""

    pass


class ExchangeAuthError(MarketDataError):
    """Invalid API credentials."""

    pass


class StaleDataError(MarketDataError):
    """Market data is too old to be trusted."""

    def __init__(self, symbol: str, age_seconds: float, max_age_seconds: int):
        self.symbol = symbol
        self.age_seconds = age_seconds
        self.max_age_seconds = max_age_seconds
        super().__init__(f"Data for {symbol} is {age_seconds:.1f}s old (max: {max_age_seconds}s)")


class InsufficientDataError(MarketDataError):
    """Not enough data points to compute an indicator."""

    def __init__(self, required: int, available: int, indicator: str = ""):
        self.required = required
        self.available = available
        self.indicator = indicator
        super().__init__(f"Insufficient data for {indicator}: need {required}, have {available}")


# ---------------------------------------------------------------------------
# Signal / analysis
# ---------------------------------------------------------------------------


class SignalError(TradingSystemError):
    """Base for signal generation errors."""

    pass


class SignalExpiredError(SignalError):
    """Signal TTL has elapsed — discard it."""

    pass


class LowConfidenceSignalError(SignalError):
    """Signal confidence is below the required threshold."""

    def __init__(self, confidence: float, threshold: float):
        self.confidence = confidence
        self.threshold = threshold
        super().__init__(f"Signal confidence {confidence:.2f} below threshold {threshold:.2f}")


# ---------------------------------------------------------------------------
# Risk management
# ---------------------------------------------------------------------------


class RiskError(TradingSystemError):
    """Base for risk management errors."""

    pass


class RiskVetoError(RiskError):
    """
    The Risk agent has vetoed a trade proposal.
    Contains the reason for structured logging.
    """

    def __init__(self, reason: str, proposal_id: str | None = None):
        self.reason = reason
        self.proposal_id = proposal_id
        super().__init__(f"Trade vetoed: {reason}")


class DailyLossLimitError(RiskError):
    """Daily loss limit has been reached — halt trading."""

    def __init__(self, current_loss_pct: float, limit_pct: float):
        self.current_loss_pct = current_loss_pct
        self.limit_pct = limit_pct
        super().__init__(f"Daily loss {current_loss_pct:.2%} exceeds limit {limit_pct:.2%}")


class DrawdownLimitError(RiskError):
    """Total drawdown limit has been reached — emergency halt."""

    def __init__(self, current_drawdown_pct: float, limit_pct: float):
        self.current_drawdown_pct = current_drawdown_pct
        self.limit_pct = limit_pct
        super().__init__(f"Drawdown {current_drawdown_pct:.2%} exceeds limit {limit_pct:.2%}")


class CircuitBreakerError(RiskError):
    """Circuit breaker is active — no trades allowed until human reset."""

    pass


class PositionSizeError(RiskError):
    """Requested position size violates risk limits."""

    pass


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class ExecutionError(TradingSystemError):
    """Base for order execution errors."""

    pass


class OrderRejectedError(ExecutionError):
    """Exchange rejected the order."""

    def __init__(self, reason: str, order_id: str | None = None):
        self.reason = reason
        self.order_id = order_id
        super().__init__(f"Order rejected by exchange: {reason}")


class OrderTimeoutError(ExecutionError):
    """Order did not fill within the expected time window."""

    pass


class InsufficientBalanceError(ExecutionError):
    """Insufficient funds in the exchange account."""

    def __init__(self, required: float, available: float, currency: str = "USD"):
        self.required = required
        self.available = available
        self.currency = currency
        super().__init__(f"Insufficient {currency}: need {required:.2f}, have {available:.2f}")


# ---------------------------------------------------------------------------
# Messaging / infrastructure
# ---------------------------------------------------------------------------


class MessagingError(TradingSystemError):
    """Redis pub/sub or messaging layer error."""

    pass


class DatabaseError(TradingSystemError):
    """Database operation failed."""

    pass
