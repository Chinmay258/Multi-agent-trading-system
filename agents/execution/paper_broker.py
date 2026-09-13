"""
agents/execution/paper_broker.py
----------------------------------
PaperBroker — simulated execution adapter for paper trading.

Implements ExecutionBroker exactly. Fills orders at the signal's reference price ±
slippage, simulates partial fills, and tracks a paper portfolio of cash + open positions.

Design invariant: this class must NEVER place a real order.
assert_paper_mode() is called at connect() and place_order() to enforce this.
If TRADING_MODE is switched to "live", both calls raise RuntimeError immediately.

Paper fill mechanics:
- Reference price comes from assessment.original_proposal.signal.technical_signal.price
- BUY fills at ref_price * (1 + slippage); SELL fills at ref_price * (1 - slippage)
- Slippage 0.05% (DEFAULT_SLIPPAGE_PCT); taker fee 0.1% of fill notional
- Partial fills: 20% probability, 60–99% of requested quantity (opening orders only)
- Simulated latency: uniform [20ms, 150ms] via asyncio.sleep
- One position per symbol:
    * an order on the same side as an open position is rejected (no stacking)
    * an order on the opposite side closes the open position at the reference price
- Opening a position (long or short) reserves notional + fee from cash. Closing returns
  the cost basis plus gross PnL minus the closing fee; realized_pnl_usd is the cash
  returned minus the cost basis, so it is net of both fees.
- Idempotency: place_order with the same proposal_id returns the cached result
  (the most recent 1,000 orders are remembered).
"""

from __future__ import annotations

import asyncio
import json
import random
import time as _time
from collections import OrderedDict
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from uuid import UUID, uuid4

from agents.execution.broker_interface import (
    BrokerBalance,
    BrokerCapabilities,
    BrokerPosition,
    ExecutionBroker,
)
from core.config import Settings
from core.exceptions import InsufficientBalanceError, OrderRejectedError
from core.logging import get_logger
from core.messaging import MessageBus
from core.metrics import ORDER_FILL_LATENCY_SECONDS, ORDERS_PLACED
from core.models.trade import (
    ExecutionResult,
    OrderSide,
    OrderStatus,
    OrderType,
    RiskAssessment,
)

#: Default slippage fraction (0.05% each side)
DEFAULT_SLIPPAGE_PCT: float = 0.0005

#: Taker fee fraction (0.1%)
_TAKER_FEE_PCT: Decimal = Decimal("0.001")

#: Precision for quantity values (8 decimal places — crypto standard)
_QTY_PLACES: Decimal = Decimal("0.00000001")

#: Probability that a fill is partial rather than full
_PARTIAL_FILL_PROBABILITY: float = 0.20

#: Range for partial fill quantity as fraction of requested
_PARTIAL_FILL_RANGE: tuple[float, float] = (0.60, 0.99)

#: Simulated round-trip latency range in seconds
_LATENCY_RANGE: tuple[float, float] = (0.020, 0.150)

#: How many filled orders to remember for idempotency
_IDEMPOTENCY_CACHE_SIZE = 1000

#: Redis keys for persisted portfolio state (no TTL — these survive restarts)
_PORTFOLIO_CASH_KEY = "paper_portfolio:cash"
_PORTFOLIO_POSITIONS_KEY = "paper_portfolio:positions"


def _side_str(side: OrderSide | str) -> str:
    return side.value if isinstance(side, OrderSide) else str(side)


@dataclass
class _PaperPosition:
    """Internal in-memory representation of an open paper position."""

    symbol: str
    side: str  # "buy" | "sell"
    quantity: Decimal
    entry_price: Decimal
    cost_usd: Decimal  # cash reserved for this position (fill notional + fee)


class PaperBroker(ExecutionBroker):
    """
    Simulated broker adapter for paper trading.

    Tracks a paper portfolio: a cash balance and a dict of open positions
    keyed by symbol. All fills are simulated — no real orders are ever placed.

    Thread safety: not designed for concurrent access. ExecutionAgent calls
    methods sequentially within a single asyncio event loop.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cash_balance: Decimal = Decimal(str(settings.paper_initial_balance_usd))
        self._positions: dict[str, _PaperPosition] = {}
        # proposal_id (str) → ExecutionResult for idempotency (bounded, oldest evicted)
        self._filled_orders: OrderedDict[str, ExecutionResult] = OrderedDict()
        self._log = get_logger("paper_broker")
        # Injected by ExecutionAgent.setup() before connect() is called
        self._bus: MessageBus | None = None

    # ------------------------------------------------------------------
    # ExecutionBroker — capabilities
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> BrokerCapabilities:
        return BrokerCapabilities(
            broker_name="paper",
            is_paper=True,
            supports_native_stops=False,
            supports_native_take_profit=False,
            supports_partial_fills=True,
        )

    # ------------------------------------------------------------------
    # Bus injection
    # ------------------------------------------------------------------

    def set_bus(self, bus: MessageBus) -> None:
        """Inject the shared MessageBus before connect() is called.

        ExecutionAgent calls this in setup() so PaperBroker can persist
        portfolio state through the already-connected Redis pool rather
        than opening a second connection.
        """
        self._bus = bus

    # ------------------------------------------------------------------
    # ExecutionBroker — lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Verify paper mode, restore portfolio state from Redis, and log startup.

        Restore is best-effort via the injected MessageBus. If the bus is not set
        (unit tests) or Redis has no prior state, the broker initialises fresh from
        paper_initial_balance_usd.
        """
        self._settings.assert_paper_mode()

        if self._bus is not None and self._bus.connected:
            try:
                cash_raw = await self._bus.kv_get(_PORTFOLIO_CASH_KEY)
                if cash_raw is not None:
                    self._cash_balance = Decimal(cash_raw)
                    positions_raw = await self._bus.kv_get(_PORTFOLIO_POSITIONS_KEY)
                    if positions_raw is not None:
                        positions_data = json.loads(positions_raw)
                        self._positions = {
                            symbol: _PaperPosition(
                                symbol=p["symbol"],
                                side=p["side"],
                                quantity=Decimal(p["quantity"]),
                                entry_price=Decimal(p["entry_price"]),
                                cost_usd=Decimal(p["cost_usd"]),
                            )
                            for symbol, p in positions_data.items()
                        }
                    self._log.info(
                        "portfolio_restored",
                        cash=float(self._cash_balance),
                        open_positions=len(self._positions),
                    )
                else:
                    self._log.info("portfolio_initialised", balance=float(self._cash_balance))
            except Exception as exc:
                self._log.warning("portfolio_restore_failed", error=str(exc))
                self._log.info("portfolio_initialised", balance=float(self._cash_balance))
        else:
            self._log.info("portfolio_initialised", balance=float(self._cash_balance))

        # Seed the keys immediately so they exist before the first fill.
        await self._save_portfolio_state()

        self._log.info(
            "paper_broker_connected",
            cash_balance_usd=float(self._cash_balance),
            open_positions=len(self._positions),
        )

    async def disconnect(self) -> None:
        await self._save_portfolio_state()
        self._log.info(
            "paper_broker_disconnected",
            cash_balance_usd=float(self._cash_balance),
            open_positions=len(self._positions),
        )

    # ------------------------------------------------------------------
    # Portfolio persistence
    # ------------------------------------------------------------------

    async def _save_portfolio_state(self) -> None:
        """Persist cash balance and open positions via the MessageBus.

        No-op when the bus is not injected or not connected (unit tests). Called after
        every portfolio-mutating operation so state survives an unexpected restart.
        Keys have no TTL — they are portfolio state, not cache.
        """
        if self._bus is None or not self._bus.connected:
            return
        try:
            await self._bus.kv_set(_PORTFOLIO_CASH_KEY, str(self._cash_balance))
            positions_data = {
                symbol: {
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "quantity": str(pos.quantity),
                    "entry_price": str(pos.entry_price),
                    "cost_usd": str(pos.cost_usd),
                }
                for symbol, pos in self._positions.items()
            }
            await self._bus.kv_set(_PORTFOLIO_POSITIONS_KEY, json.dumps(positions_data))
            self._log.info(
                "portfolio_state_saved",
                cash=float(self._cash_balance),
                positions=len(self._positions),
            )
        except Exception as exc:
            self._log.warning("portfolio_save_failed", error=str(exc))

    def _remember(self, proposal_id: str, result: ExecutionResult) -> None:
        self._filled_orders[proposal_id] = result
        if len(self._filled_orders) > _IDEMPOTENCY_CACHE_SIZE:
            self._filled_orders.popitem(last=False)

    # ------------------------------------------------------------------
    # ExecutionBroker — order operations
    # ------------------------------------------------------------------

    async def place_order(self, assessment: RiskAssessment) -> ExecutionResult:
        """
        Simulate order placement and return a paper ExecutionResult.

        Steps:
          1. Assert paper mode (hard guard).
          2. Return cached result if this proposal was already filled (idempotency).
          3. Extract reference price from technical signal.
          4. Sleep to simulate network/exchange latency.
          5. If the symbol has an open position: reject a same-side order, or close the
             position when the order is on the opposite side.
          6. Otherwise compute the fill price with slippage and a possible partial fill,
             check cash, open the position, and return the result.
        """
        self._settings.assert_paper_mode()

        # --- Idempotency ---
        pid = str(assessment.proposal_id)
        cached = self._filled_orders.get(pid)
        if cached is not None:
            self._log.info("paper_order_duplicate_skipped", proposal_id=pid)
            return cached

        # --- Reference price ---
        tech = assessment.original_proposal.signal.technical_signal
        if tech is None:
            raise OrderRejectedError(
                "no reference price: technical_signal is None in proposal signal"
            )
        ref_price = Decimal(str(tech.price))
        if ref_price <= Decimal("0"):
            raise OrderRejectedError(f"reference price must be positive, got {ref_price}")

        # --- Latency simulation ---
        _fill_start = _time.perf_counter()
        await asyncio.sleep(random.uniform(*_LATENCY_RANGE))

        symbol = assessment.original_proposal.symbol
        side = assessment.original_proposal.side
        side_str = _side_str(side)

        # --- Existing position: no stacking; opposite side closes it ---
        existing = self._positions.get(symbol)
        if existing is not None:
            if existing.side == side_str:
                raise OrderRejectedError(
                    f"{symbol} already has an open {existing.side} position; "
                    "stacking is not supported"
                )
            close_result = await self._close(
                symbol,
                ref_price,
                proposal_id=assessment.proposal_id,
                assessment_id=assessment.assessment_id,
            )
            if close_result is None:  # pragma: no cover - guarded by the lookup above
                raise OrderRejectedError(f"no open position to close for {symbol}")
            self._remember(pid, close_result)
            ORDERS_PLACED.labels(symbol=symbol, side=side_str, mode="paper").inc()
            ORDER_FILL_LATENCY_SECONDS.labels(mode="paper").observe(
                _time.perf_counter() - _fill_start
            )
            return close_result

        # --- Fill price with slippage ---
        slippage = Decimal(str(DEFAULT_SLIPPAGE_PCT))
        if side_str == OrderSide.BUY.value:
            fill_price = ref_price * (Decimal("1") + slippage)
        else:
            fill_price = ref_price * (Decimal("1") - slippage)
        fill_price = fill_price.quantize(_QTY_PLACES, rounding=ROUND_DOWN)

        # --- Requested quantity from approved size ---
        effective_size = assessment.effective_size_usd
        if effective_size is None or effective_size <= Decimal("0"):
            raise OrderRejectedError("assessment has no effective_size_usd")
        requested_qty = (effective_size / fill_price).quantize(_QTY_PLACES, rounding=ROUND_DOWN)
        if requested_qty <= Decimal("0"):
            raise OrderRejectedError(
                f"computed quantity is zero for size={effective_size} price={fill_price}"
            )

        # --- Partial fill simulation ---
        if random.random() < _PARTIAL_FILL_PROBABILITY:
            fill_ratio = Decimal(str(random.uniform(*_PARTIAL_FILL_RANGE)))
            fill_qty = (requested_qty * fill_ratio).quantize(_QTY_PLACES, rounding=ROUND_DOWN)
            status = OrderStatus.PARTIALLY_FILLED
        else:
            fill_qty = requested_qty
            status = OrderStatus.FILLED

        if fill_qty <= Decimal("0"):
            fill_qty = requested_qty
            status = OrderStatus.FILLED

        # --- Fee and total cost ---
        notional = fill_qty * fill_price
        fee = (notional * _TAKER_FEE_PCT).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        total_cost = notional + fee

        # --- Balance check ---
        if total_cost > self._cash_balance:
            raise InsufficientBalanceError(
                required=float(total_cost),
                available=float(self._cash_balance),
            )

        # --- Update paper portfolio ---
        self._cash_balance -= total_cost
        self._positions[symbol] = _PaperPosition(
            symbol=symbol,
            side=side_str,
            quantity=fill_qty,
            entry_price=fill_price,
            cost_usd=total_cost,
        )

        # --- Build result ---
        result = ExecutionResult(
            proposal_id=assessment.proposal_id,
            assessment_id=assessment.assessment_id,
            symbol=symbol,
            side=side,
            order_type=assessment.original_proposal.order_type,
            status=status,
            requested_quantity=requested_qty,
            filled_quantity=fill_qty,
            average_fill_price=fill_price,
            total_cost_usd=notional,
            fee_usd=fee,
            fee_currency="USDT",
            is_paper=True,
        )
        self._remember(pid, result)

        ORDERS_PLACED.labels(symbol=symbol, side=side_str, mode="paper").inc()
        ORDER_FILL_LATENCY_SECONDS.labels(mode="paper").observe(_time.perf_counter() - _fill_start)

        await self._save_portfolio_state()

        self._log.info(
            "paper_order_filled",
            symbol=symbol,
            side=side_str,
            status=status,
            fill_price=float(fill_price),
            fill_qty=float(fill_qty),
            fee_usd=float(fee),
            cash_remaining=float(self._cash_balance),
        )
        return result

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Paper cancel always succeeds — no real exchange involved."""
        self._log.info("paper_cancel_order", order_id=order_id, symbol=symbol)
        return True

    # ------------------------------------------------------------------
    # ExecutionBroker — portfolio queries
    # ------------------------------------------------------------------

    async def get_positions(self) -> list[BrokerPosition]:
        """Return all open paper positions as BrokerPosition instances."""
        return [
            BrokerPosition(
                symbol=pos.symbol,
                side=pos.side,
                quantity=pos.quantity,
                entry_price=pos.entry_price,
                # No live price feed inside the broker — use entry as current price.
                current_price=pos.entry_price,
                unrealised_pnl_usd=Decimal("0"),
            )
            for pos in self._positions.values()
        ]

    async def get_balance(self) -> BrokerBalance:
        """
        Return paper portfolio balance.

        total_equity  = free cash + value locked in open positions
        free_margin   = cash available for new positions
        used_margin   = cost basis of all open positions
        """
        used = sum((pos.cost_usd for pos in self._positions.values()), Decimal("0"))
        return BrokerBalance(
            total_equity_usd=self._cash_balance + used,
            free_margin_usd=self._cash_balance,
            used_margin_usd=used,
        )

    async def ping(self) -> bool:
        """PaperBroker is always reachable."""
        return True

    # ------------------------------------------------------------------
    # ExecutionBroker — optional overrides
    # ------------------------------------------------------------------

    async def close_position(
        self, symbol: str, price: Decimal | None = None
    ) -> ExecutionResult | None:
        """
        Close the open position for ``symbol`` at ``price`` (the current market price).

        Returns None if no position is open. Without a price the position is closed at its
        entry price, which realises only fees and slippage. Uses a synthetic
        proposal_id/assessment_id since this close is not tied to a RiskAssessment.
        """
        pos = self._positions.get(symbol)
        if pos is None:
            self._log.info("paper_close_position_no_op", symbol=symbol)
            return None
        ref_price = price if price is not None and price > Decimal("0") else pos.entry_price
        return await self._close(symbol, ref_price, proposal_id=uuid4(), assessment_id=uuid4())

    async def _close(
        self,
        symbol: str,
        ref_price: Decimal,
        proposal_id: UUID,
        assessment_id: UUID,
    ) -> ExecutionResult | None:
        """Close a position at ``ref_price`` ± slippage and realise its PnL."""
        pos = self._positions.pop(symbol, None)
        if pos is None:
            return None

        slippage = Decimal(str(DEFAULT_SLIPPAGE_PCT))
        if pos.side == OrderSide.BUY.value:
            # Closing a long = selling slightly below the reference price.
            close_side = OrderSide.SELL
            exit_price = (ref_price * (Decimal("1") - slippage)).quantize(
                _QTY_PLACES, rounding=ROUND_DOWN
            )
            notional = pos.quantity * exit_price
            fee = (notional * _TAKER_FEE_PCT).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
            cash_returned = notional - fee
        else:
            # Closing a short = buying back slightly above the reference price.
            close_side = OrderSide.BUY
            exit_price = (ref_price * (Decimal("1") + slippage)).quantize(
                _QTY_PLACES, rounding=ROUND_DOWN
            )
            notional = pos.quantity * exit_price
            fee = (notional * _TAKER_FEE_PCT).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
            gross_pnl = (pos.entry_price - exit_price) * pos.quantity
            cash_returned = pos.cost_usd + gross_pnl - fee

        realized = (cash_returned - pos.cost_usd).quantize(Decimal("0.01"))
        self._cash_balance += cash_returned

        await self._save_portfolio_state()

        result = ExecutionResult(
            proposal_id=proposal_id,
            assessment_id=assessment_id,
            symbol=symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            requested_quantity=pos.quantity,
            filled_quantity=pos.quantity,
            average_fill_price=exit_price,
            total_cost_usd=notional,
            fee_usd=fee,
            fee_currency="USDT",
            is_paper=True,
            realized_pnl_usd=realized,
        )
        self._log.info(
            "paper_position_closed",
            symbol=symbol,
            close_price=float(exit_price),
            realized_pnl_usd=float(realized),
            cash_balance_usd=float(self._cash_balance),
        )
        return result
