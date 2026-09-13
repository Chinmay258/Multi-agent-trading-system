"""
core/models/__init__.py
-----------------------
Public re-exports for all shared data models.

Import from here rather than individual modules to insulate callers from
internal file reorganisation. The public API of the models package is
whatever is exported in this file.

Example:
    from core.models import OHLCVCandle, TradeProposal, TechnicalSignal
"""

from core.models.market import (
    BaseMarketModel,
    MarketEventType,
    OHLCVCandle,
    OrderBook,
    OrderBookLevel,
    Ticker,
    Timeframe,
)
from core.models.signals import (
    AggregatedSignal,
    IndicatorName,
    IndicatorReading,
    SentimentSignal,
    SignalDirection,
    SignalSource,
    TechnicalSignal,
)
from core.models.system import (
    AgentHeartbeat,
    AgentStatus,
    AlertSeverity,
    AlertType,
    RiskOverride,
    SystemAlert,
    SystemCommand,
    SystemCommandMessage,
)
from core.models.trade import (
    ExecutionResult,
    OrderSide,
    OrderStatus,
    OrderType,
    RejectionReason,
    RiskAssessment,
    RiskDecision,
    TradeProposal,
)

__all__ = [
    # Market
    "BaseMarketModel",
    "MarketEventType",
    "OHLCVCandle",
    "OrderBook",
    "OrderBookLevel",
    "Ticker",
    "Timeframe",
    # Signals
    "AggregatedSignal",
    "IndicatorName",
    "IndicatorReading",
    "SentimentSignal",
    "SignalDirection",
    "SignalSource",
    "TechnicalSignal",
    # Trade lifecycle
    "ExecutionResult",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "RejectionReason",
    "RiskAssessment",
    "RiskDecision",
    "TradeProposal",
    # System
    "AgentHeartbeat",
    "AgentStatus",
    "AlertSeverity",
    "AlertType",
    "RiskOverride",
    "SystemAlert",
    "SystemCommand",
    "SystemCommandMessage",
]
