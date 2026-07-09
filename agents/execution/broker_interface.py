"""
agents/execution/broker_interface.py
-------------------------------------
Abstract ExecutionBroker interface.

This is the single most important abstraction for MT5 readiness.
ExecutionAgent ONLY talks to this interface — never to a concrete adapter.

All execution adapters implement this contract:
  - PaperBroker   (agents/execution/paper_broker.py)   ← current
  - LiveBroker    (agents/execution/live_broker.py)    ← Phase 5
  - MT5Bridge     (agents/execution/mt5_bridge.py)     ← Phase 6

Swapping from paper → live → MT5 is a config change, not a code change.

Design decisions:
- All methods are async. MT5Bridge will use async ZMQ sockets.
  Blocking adapters (e.g. synchronous CCXT) must wrap in asyncio.to_thread().
- All methods accept/return types from core/models/trade.py — no
  exchange-specific types leak through this interface.
- BrokerCapabilities allows the ExecutionAgent to query what the
  connected broker supports (e.g. does it have native stop losses?).
  This avoids runtime surprises when switching brokers.

Usage:
    # Dependency injection in ExecutionAgent.__init__():
    broker: ExecutionBroker = get_broker_from_config()

    # ExecutionAgent never does this:
    from agents.execution.paper_broker import PaperBroker  # ← wrong
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal

from core.models.trade import ExecutionResult, RiskAssessment


@dataclass(frozen=True)
class BrokerCapabilities:
    """
    Describes what features a broker adapter supports.
    ExecutionAgent reads this at startup to adjust its behaviour.

    Example: if supports_native_stops=False, ExecutionAgent manages
    stop losses itself (polling position price) rather than placing
    exchange-native stop orders.
    """

    broker_name: str
    supports_market_orders: bool = True
    supports_limit_orders: bool = True
    supports_native_stops: bool = True  # Exchange-side stop loss orders
    supports_native_take_profit: bool = True
    supports_partial_fills: bool = True
    supports_position_query: bool = True  # Can query open positions
    supports_balance_query: bool = True  # Can query account balance
    is_paper: bool = False  # True only for PaperBroker
    requires_symbol_mapping: bool = False  # True for MT5 (BTC/USDT → BTCUSD)


@dataclass
class BrokerBalance:
    """
    Normalised account balance — broker-agnostic.
    MT5 returns balance in account currency; CCXT in per-asset dicts.
    We normalise to a simple equity/free/used structure.
    """

    total_equity_usd: Decimal
    free_margin_usd: Decimal
    used_margin_usd: Decimal
    currency: str = "USD"
    raw: dict | None = None  # Original broker response for debugging


@dataclass
class BrokerPosition:
    """Normalised open position from the broker."""

    symbol: str
    side: str  # "buy" | "sell"
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealised_pnl_usd: Decimal
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    position_id: str | None = None


class ExecutionBroker(ABC):
    """
    Abstract base for all execution adapters.

    Implement this interface to add a new broker/execution backend.
    ExecutionAgent is programmed entirely against this ABC.

    Method contract:
    - All methods are async.
    - All methods raise exceptions from core/exceptions.py, not broker-native errors.
      Adapters are responsible for translating broker errors to our domain exceptions.
    - place_order() must be idempotent when given the same proposal_id.
      This protects against duplicate orders on retry.
    """

    @property
    @abstractmethod
    def capabilities(self) -> BrokerCapabilities:
        """
        Return the capabilities of this broker adapter.
        Called once by ExecutionAgent at startup.
        """
        ...

    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to the broker.
        Raise ExchangeConnectionError if connection fails.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Gracefully close the broker connection.
        Must be safe to call even if connect() was never called.
        """
        ...

    @abstractmethod
    async def place_order(self, assessment: RiskAssessment) -> ExecutionResult:
        """
        Place an order based on an approved RiskAssessment.

        The assessment contains the full approved parameters:
        - assessment.original_proposal: original trade intent
        - assessment.approved_size_usd: Risk-approved position size
        - assessment.approved_stop_loss_pct: Risk-approved stop loss %
        - assessment.approved_take_profit_pct: Risk-approved take profit %

        Returns:
            ExecutionResult with fill details (price, quantity, fees).
            For paper brokers: simulated fill.
            For MT5: actual MT5 execution report.

        Raises:
            OrderRejectedError: Broker rejected the order.
            InsufficientBalanceError: Not enough funds.
            ExchangeConnectionError: Lost connection during execution.
        """
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an open order by ID.

        Returns:
            True if successfully cancelled, False if order was already filled/cancelled.

        Raises:
            ExchangeConnectionError: Broker unreachable.
        """
        ...

    @abstractmethod
    async def get_positions(self) -> list[BrokerPosition]:
        """
        Return all currently open positions.

        Used by RiskAgent to monitor exposure.
        Returns empty list if no positions open.
        """
        ...

    @abstractmethod
    async def get_balance(self) -> BrokerBalance:
        """
        Return current account balance.

        Used by RiskAgent for position sizing and drawdown calculation.
        """
        ...

    @abstractmethod
    async def ping(self) -> bool:
        """
        Health check — returns True if broker is reachable.
        Used by MonitoringAgent.
        """
        ...

    # ------------------------------------------------------------------
    # Optional overrides (have sensible defaults, adapters may override)
    # ------------------------------------------------------------------

    async def close_position(self, symbol: str) -> ExecutionResult | None:
        """
        Close an open position for a symbol at market price.

        Default: places a market order in the opposite direction.
        MT5 adapter should override with native position close call.
        """
        # Default implementation — adapters that support it should override
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement close_position(). "
            "Place a market order in the opposite direction instead."
        )

    async def modify_stops(
        self,
        order_id: str,
        symbol: str,
        stop_loss: Decimal | None = None,
        take_profit: Decimal | None = None,
    ) -> bool:
        """
        Modify stop loss / take profit for an existing order.

        MT5 supports this natively. For other brokers, this requires
        cancelling and re-placing stop orders.
        Default: raise NotImplementedError to force explicit handling.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement modify_stops(). "
            "Cancel and re-place stop orders instead."
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(broker={self.capabilities.broker_name})"
