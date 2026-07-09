"""
agents/execution/mt5_bridge.py
--------------------------------
MT5Bridge — aiohttp HTTP server that implements ExecutionBroker for MetaTrader 5.

Architecture (reversed from a typical TCP client):
  The MT5 EA (TradingSystemEA.mq5) polls this HTTP server every second using
  MT5's built-in WebRequest() function.  Python never initiates a connection —
  it only serves HTTP requests from the EA.

Endpoints:
  POST /mt5/state    — EA sends heartbeat: account balance + open positions
  GET  /mt5/command  — EA polls for the next pending command (returns
                       {"status":"empty"} if nothing is queued)
  POST /mt5/result   — EA sends the execution result for a previously issued command

Command lifecycle:
  1. place_order() builds a command dict and calls _enqueue_and_wait().
  2. _enqueue_and_wait() creates an asyncio.Future, stores it in _result_futures,
     puts the command on _command_queue, then awaits the future with a timeout.
  3. On the next GET /mt5/command, the EA dequeues the command and executes it.
  4. The EA POSTs the result to /mt5/result; _handle_result() resolves the future.
  5. _enqueue_and_wait() returns the result dict to the caller.

get_balance() and get_positions() return cached data from the most-recent heartbeat
— they are always instant (no round-trip to the EA).

All config is read from get_settings().mt5.
No external libraries beyond aiohttp (already in pyproject.toml).
"""

from __future__ import annotations

import asyncio
import json
import time
from decimal import Decimal
from typing import Any
from uuid import uuid4

from aiohttp import web

from agents.execution.broker_interface import (
    BrokerBalance,
    BrokerCapabilities,
    BrokerPosition,
    ExecutionBroker,
)
from agents.execution.symbol_mapper import SymbolMapper
from core.config import get_settings
from core.exceptions import ExchangeConnectionError, OrderRejectedError
from core.logging import get_logger
from core.models.trade import (
    ExecutionResult,
    OrderSide,
    OrderStatus,
    OrderType,
    RiskAssessment,
)

_MAGIC_NUMBER = 20250101
_QTY_PLACES = Decimal("0.00000001")
# The EA posts a heartbeat (POST /mt5/state) roughly once per second. A gap of
# 5s means ~5 missed polls — treat the EA as disconnected past that. This matches
# ping()'s docstring ("within the last 5 seconds"); a larger value would let a
# dead EA be reported as healthy and mask execution failures.
_HEARTBEAT_STALE_SECONDS = 5.0


class MT5Bridge(ExecutionBroker):
    """
    aiohttp HTTP server that bridges Python to a MetaTrader 5 terminal.

    The EA polls this server every second via WebRequest().  Python queues
    commands and resolves asyncio.Futures when the EA posts results back.

    Set EXECUTION_BROKER=mt5 in .env to activate this adapter.
    """

    def __init__(self) -> None:
        settings = get_settings()
        mt5_cfg = settings.mt5
        self._listen_port = mt5_cfg.listen_port
        self._timeout_ms = mt5_cfg.request_timeout_ms
        self._mapper = SymbolMapper()
        self._log = get_logger("mt5_bridge")

        # Lot-size constraints read from config (broker/symbol-specific).
        self._min_volume: float = mt5_cfg.min_volume
        self._max_volume: float = mt5_cfg.max_volume
        self._volume_step: float = mt5_cfg.volume_step

        # HTTP server (populated by connect(), torn down by disconnect())
        self._runner: web.AppRunner | None = None

        # Command queue: place_order() enqueues; EA dequeues via GET /mt5/command
        self._command_queue: asyncio.Queue[dict] = asyncio.Queue()

        # Pending futures keyed by command_id; resolved when EA POSTs /mt5/result
        self._result_futures: dict[str, asyncio.Future[dict]] = {}

        # Cached account state from the most-recent EA heartbeat
        self._cached_state: dict[str, Any] = {}
        self._last_heartbeat_at: float | None = None

        # Set by ExecutionAgent.setup() via set_bus(); used to write Redis cache
        self._bus: Any = None

    # ------------------------------------------------------------------
    # ExecutionBroker — capabilities
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> BrokerCapabilities:
        return BrokerCapabilities(
            broker_name="mt5",
            supports_native_stops=True,
            supports_native_take_profit=True,
            supports_partial_fills=False,
            requires_symbol_mapping=True,
            is_paper=False,
        )

    # ------------------------------------------------------------------
    # ExecutionBroker — lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Start the aiohttp HTTP server that the MT5 EA will poll."""
        app = web.Application()
        app.router.add_post("/mt5/state", self._handle_state)
        app.router.add_get("/mt5/command", self._handle_command)
        app.router.add_post("/mt5/result", self._handle_result)

        self._runner = web.AppRunner(app, access_log=None, keepalive_timeout=30)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self._listen_port)
        await site.start()

        self._log.info("mt5_http_server_started", port=self._listen_port)

    async def disconnect(self) -> None:
        """Stop the HTTP server and clean up resources."""
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        self._log.info("mt5_bridge_disconnected")

    def set_bus(self, bus: Any) -> None:
        """Inject the MessageBus so heartbeat state can be written to Redis."""
        self._bus = bus

    # ------------------------------------------------------------------
    # ExecutionBroker — order operations
    # ------------------------------------------------------------------

    async def place_order(self, assessment: RiskAssessment) -> ExecutionResult:
        """Queue a PLACE_ORDER command and wait for the EA to execute it."""
        proposal = assessment.original_proposal
        symbol = proposal.symbol
        side = proposal.side

        mt5_symbol = self._mapper.to_mt5(symbol)
        volume = self._resolve_volume(assessment)

        # Send percentage values so the EA computes absolute SL/TP from the
        # live ask/bid at execution time — more accurate than using the
        # signal price (which may be several seconds old by fill time).
        sl_pct = float(assessment.approved_stop_loss_pct or 0.01)
        tp_pct = float(assessment.approved_take_profit_pct or 0.015)

        command: dict[str, Any] = {
            "action": "PLACE_ORDER",
            "proposal_id": str(assessment.proposal_id),
            "symbol": mt5_symbol,
            "side": side.value if isinstance(side, OrderSide) else str(side),
            "order_type": "market",
            "volume": float(volume),
            "stop_loss_pct": sl_pct,
            "take_profit_pct": tp_pct,
            "magic": _MAGIC_NUMBER,
            "comment": f"ts_{str(assessment.proposal_id)[:8]}",
        }

        self._log.info(
            "mt5_place_order",
            symbol=mt5_symbol,
            side=command["side"],
            volume=command["volume"],
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
        )

        response = await self._enqueue_and_wait(command)

        if response.get("status") == "error":
            raise OrderRejectedError(
                reason=response.get("error_message", "MT5 rejected the order"),
                order_id=str(response.get("order_id", "")),
            )

        fill_qty = Decimal(str(response.get("fill_quantity", volume)))
        fill_price = Decimal(str(response.get("fill_price", 0)))
        status = OrderStatus.FILLED if fill_qty >= volume else OrderStatus.PARTIALLY_FILLED

        return ExecutionResult(
            proposal_id=assessment.proposal_id,
            assessment_id=assessment.assessment_id,
            exchange_order_id=str(response.get("order_id", "")),
            symbol=symbol,
            side=side,
            order_type=proposal.order_type,
            status=status,
            requested_quantity=volume,
            filled_quantity=fill_qty,
            average_fill_price=fill_price,
            total_cost_usd=fill_qty * fill_price if fill_price else None,
            is_paper=False,
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Queue a CANCEL_ORDER command and wait for the EA to execute it."""
        mt5_symbol = self._mapper.to_mt5(symbol)
        response = await self._enqueue_and_wait(
            {"action": "CANCEL_ORDER", "order_id": order_id, "symbol": mt5_symbol}
        )
        return response.get("status") == "ok"

    async def close_position(self, symbol: str) -> ExecutionResult | None:
        """Queue a CLOSE_POSITION command and wait for the EA to market-close it."""
        mt5_symbol = self._mapper.to_mt5(symbol)
        try:
            response = await self._enqueue_and_wait(
                {
                    "action": "CLOSE_POSITION",
                    "symbol": mt5_symbol,
                    "magic": _MAGIC_NUMBER,
                }
            )
        except ExchangeConnectionError as exc:
            self._log.warning("mt5_close_position_timeout", symbol=symbol, error=str(exc))
            return None

        if response.get("status") == "error":
            self._log.warning(
                "mt5_close_position_rejected",
                symbol=symbol,
                error=response.get("error_message"),
            )
            return None

        fill_qty = Decimal(str(response.get("fill_quantity", 0)))
        fill_price = Decimal(str(response.get("fill_price", 0)))
        raw_side = response.get("side", "sell")
        side = OrderSide.BUY if raw_side == "buy" else OrderSide.SELL

        return ExecutionResult(
            proposal_id=uuid4(),
            assessment_id=uuid4(),
            exchange_order_id=str(response.get("order_id", "")),
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED if fill_qty > 0 else OrderStatus.REJECTED,
            requested_quantity=fill_qty,
            filled_quantity=fill_qty,
            average_fill_price=fill_price if fill_price > 0 else None,
            total_cost_usd=fill_qty * fill_price if fill_price > 0 else None,
            is_paper=False,
        )

    async def get_positions(self) -> list[BrokerPosition]:
        """Return cached positions from the most-recent EA heartbeat."""
        raw_positions: list[dict] = self._cached_state.get("positions", [])
        positions: list[BrokerPosition] = []
        for pos in raw_positions:
            mt5_sym = pos.get("symbol", "")
            try:
                python_sym = self._mapper.to_python(mt5_sym)
            except ValueError:
                self._log.warning("unknown_mt5_symbol_in_positions", symbol=mt5_sym)
                python_sym = mt5_sym

            sl = pos.get("stop_loss")
            tp = pos.get("take_profit")
            positions.append(
                BrokerPosition(
                    symbol=python_sym,
                    side=pos.get("side", "buy"),
                    quantity=Decimal(str(pos.get("quantity", 0))),
                    entry_price=Decimal(str(pos.get("entry_price", 0))),
                    current_price=Decimal(str(pos.get("current_price", 0))),
                    unrealised_pnl_usd=Decimal(str(pos.get("unrealised_pnl", 0))),
                    stop_loss=Decimal(str(sl)) if sl else None,
                    take_profit=Decimal(str(tp)) if tp else None,
                    position_id=str(pos.get("ticket", "")),
                )
            )
        return positions

    async def get_balance(self) -> BrokerBalance:
        """Return cached balance from the most-recent EA heartbeat."""
        balance_data = self._cached_state.get("balance")
        if balance_data is None:
            raise ExchangeConnectionError(
                "No balance data available — no heartbeat received from MT5 EA yet"
            )
        return BrokerBalance(
            total_equity_usd=Decimal(str(balance_data.get("total_equity", 0))),
            free_margin_usd=Decimal(str(balance_data.get("free_margin", 0))),
            used_margin_usd=Decimal(str(balance_data.get("used_margin", 0))),
            currency=balance_data.get("currency", "USD"),
            raw=balance_data,
        )

    async def ping(self) -> bool:
        """Return True if a heartbeat was received within the last 5 seconds."""
        if self._last_heartbeat_at is None:
            return False
        return (time.monotonic() - self._last_heartbeat_at) < _HEARTBEAT_STALE_SECONDS

    # ------------------------------------------------------------------
    # ExecutionBroker — optional overrides
    # ------------------------------------------------------------------

    async def modify_stops(
        self,
        order_id: str,
        symbol: str,
        stop_loss: Decimal | None = None,
        take_profit: Decimal | None = None,
    ) -> bool:
        """Queue a MODIFY_STOPS command and wait for the EA to execute it."""
        mt5_symbol = self._mapper.to_mt5(symbol)
        response = await self._enqueue_and_wait(
            {
                "action": "MODIFY_STOPS",
                "order_id": order_id,
                "symbol": mt5_symbol,
                "stop_loss": float(stop_loss) if stop_loss is not None else 0.0,
                "take_profit": float(take_profit) if take_profit is not None else 0.0,
            }
        )
        return response.get("status") == "ok"

    # ------------------------------------------------------------------
    # HTTP handlers
    # ------------------------------------------------------------------

    async def _handle_state(self, request: web.Request) -> web.Response:
        """POST /mt5/state — EA sends heartbeat with balance and open positions."""
        try:
            body: dict = await asyncio.wait_for(request.json(), timeout=3.0)
        except (TimeoutError, Exception) as exc:
            self._log.warning("mt5_state_handler_error", error=str(exc))
            return web.json_response({"status": "error", "message": "invalid JSON"}, status=400)

        self._cached_state = body
        self._last_heartbeat_at = time.monotonic()
        self._log.debug("mt5_heartbeat_received")
        asyncio.create_task(self._save_mt5_state(body))
        return web.json_response({"status": "ok"})

    async def _handle_command(self, _request: web.Request) -> web.Response:
        """GET /mt5/command — EA polls for the next pending command.

        Uses compact JSON (no spaces) so the EA's ParseJSONString can match
        "key":"value" without needing to handle whitespace variants.
        """
        try:
            command = self._command_queue.get_nowait()
        except asyncio.QueueEmpty:
            # Compact JSON: EA's ParseJSONString looks for "status":"…" (no space)
            return web.Response(
                text='{"status":"empty"}',
                content_type="application/json",
            )
        except Exception as exc:
            self._log.warning("mt5_command_handler_error", error=str(exc))
            return web.json_response({"status": "error"}, status=500)

        self._log.debug("mt5_command_dispatched", action=command.get("action"))
        payload = json.dumps({"status": "ok", **command}, separators=(",", ":"))
        return web.Response(text=payload, content_type="application/json")

    async def _handle_result(self, request: web.Request) -> web.Response:
        """POST /mt5/result — EA sends execution result for a pending command."""
        try:
            body: dict = await asyncio.wait_for(request.json(), timeout=3.0)
        except (TimeoutError, Exception) as exc:
            self._log.warning("mt5_result_handler_error", error=str(exc))
            return web.json_response({"status": "error", "message": "invalid JSON"}, status=400)

        command_id: str = body.get("command_id", "")
        future = self._result_futures.pop(command_id, None)
        if future is not None and not future.done():
            future.set_result(body)
            self._log.debug("mt5_result_resolved", command_id=command_id)
        else:
            self._log.warning("mt5_result_unknown_command_id", command_id=command_id)

        return web.json_response({"status": "ok"})

    # ------------------------------------------------------------------
    # Internal: command/result round-trip
    # ------------------------------------------------------------------

    async def _enqueue_and_wait(self, command: dict) -> dict:
        """
        Put a command on the queue and wait for the EA to post the result.

        Raises ExchangeConnectionError if no result arrives within _timeout_ms.
        """
        command_id = str(uuid4())
        command = {**command, "command_id": command_id}

        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict] = loop.create_future()
        self._result_futures[command_id] = future

        await self._command_queue.put(command)

        try:
            return await asyncio.wait_for(future, timeout=self._timeout_ms / 1000)
        except TimeoutError as exc:
            self._result_futures.pop(command_id, None)
            raise ExchangeConnectionError(
                f"MT5 EA did not respond to {command.get('action')} within {self._timeout_ms}ms"
            ) from exc

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    async def _save_mt5_state(self, state: dict) -> None:
        """Persist heartbeat balance and positions to Redis (TTL 30 s)."""
        if self._bus is None or self._bus._pool is None:  # noqa: SLF001
            return
        try:
            bal = state.get("balance", {})
            balance_json = json.dumps(
                {
                    "total_equity_usd": float(bal.get("total_equity", 0)),
                    "free_margin_usd": float(bal.get("free_margin", 0)),
                    "used_margin_usd": float(bal.get("used_margin", 0)),
                    "currency": bal.get("currency", "USD"),
                }
            )

            positions_out: list[dict] = []
            for pos in state.get("positions", []):
                mt5_sym = pos.get("symbol", "")
                try:
                    python_sym = self._mapper.to_python(mt5_sym)
                except ValueError:
                    python_sym = mt5_sym
                sl = pos.get("stop_loss")
                tp = pos.get("take_profit")
                positions_out.append(
                    {
                        "symbol": python_sym,
                        "side": pos.get("side", "buy"),
                        "quantity": float(pos.get("quantity", 0)),
                        "entry_price": float(pos.get("entry_price", 0)),
                        "current_price": float(pos.get("current_price", 0)),
                        "unrealised_pnl_usd": float(pos.get("unrealised_pnl", 0)),
                        "stop_loss": float(sl) if sl else None,
                        "take_profit": float(tp) if tp else None,
                    }
                )
            positions_json = json.dumps(positions_out)

            await self._bus._pool.setex("mt5:balance", 30, balance_json)  # noqa: SLF001
            await self._bus._pool.setex("mt5:positions", 30, positions_json)  # noqa: SLF001
        except Exception as exc:
            self._log.warning("mt5_redis_state_write_failed", error=str(exc))

    def _resolve_volume(self, assessment: RiskAssessment) -> Decimal:
        """Determine the trade volume in base-currency units, clamped to broker limits."""
        if assessment.approved_quantity is not None:
            raw = float(assessment.approved_quantity)
        else:
            effective_size = assessment.effective_size_usd
            if not effective_size or effective_size <= Decimal("0"):
                raise OrderRejectedError("assessment has no effective_size_usd")

            ref_price = self._get_ref_price(assessment)
            if ref_price <= Decimal("0"):
                raise OrderRejectedError("cannot compute volume: no reference price")

            raw = float((effective_size / ref_price).quantize(_QTY_PLACES))

        adjusted = self._clamp_volume(raw)
        return Decimal(str(adjusted))

    def _clamp_volume(self, volume: float) -> float:
        """
        Round volume to the nearest lot step, then enforce broker min/max.

        Prevents MT5 error 10014 (invalid volume) on brokers with strict lot
        size constraints (e.g. Tickmill BTCUSD: min=0.01, step=0.01, max=10.0).
        """
        original = volume
        step = self._volume_step

        # Round to nearest step
        volume = round(round(volume / step) * step, 2)
        # Enforce minimum
        volume = max(volume, self._min_volume)
        # Enforce maximum
        volume = min(volume, self._max_volume)

        self._log.info(
            "volume_adjusted",
            original=original,
            adjusted=volume,
            min=self._min_volume,
        )
        return volume

    def _get_ref_price(self, assessment: RiskAssessment) -> Decimal:
        """Extract reference price from the proposal's technical signal."""
        tech = assessment.original_proposal.signal.technical_signal
        if tech is None:
            return Decimal("0")
        return Decimal(str(tech.price))

    def _compute_stops(
        self,
        ref_price: Decimal,
        side: OrderSide | str,
        assessment: RiskAssessment,
    ) -> tuple[float, float]:
        """Compute absolute stop-loss and take-profit prices."""
        if ref_price <= Decimal("0"):
            return 0.0, 0.0

        stop_pct = Decimal(str(assessment.approved_stop_loss_pct or 0.02))
        tp_pct = Decimal(str(assessment.approved_take_profit_pct or 0.04))
        side_val = side.value if isinstance(side, OrderSide) else str(side)

        if side_val == "buy":
            stop_price = float(ref_price * (Decimal("1") - stop_pct))
            tp_price = float(ref_price * (Decimal("1") + tp_pct))
        else:
            stop_price = float(ref_price * (Decimal("1") + stop_pct))
            tp_price = float(ref_price * (Decimal("1") - tp_pct))

        return stop_price, tp_price
