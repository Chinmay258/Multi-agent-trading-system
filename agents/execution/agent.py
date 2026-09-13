"""
agents/execution/agent.py
--------------------------
ExecutionAgent — the system's order execution layer.

Subscribes to:
    risk.assessment         — RiskAssessment from RiskAgent
    market.ticker.{symbol}  — last traded price, used by the stop-loss / take-profit monitor

Publishes:
    execution.result        — ExecutionResult after each fill attempt. Failed orders are
                              published too (status=rejected) so Risk releases the size it
                              reserved.

Architecture rules:
    - Only imports from core/ and own package (agents.execution.*)
    - The broker is chosen from config; the run loop talks only to the ExecutionBroker
      interface
    - All orders are gated by assessment.is_approved AND not self._trading_halted
    - Portfolio state is cached to Redis after every fill so the API can serve it without
      talking to the broker.

Redis keys written by this agent:
    execution:balance       — JSON dict of BrokerBalance fields (TTL 300s)
    execution:positions     — JSON array of BrokerPosition fields (TTL 300s)
    execution:history       — list of ExecutionResult JSON (last 100, trimmed atomically)

Stop-loss / take-profit:
    Paper mode has no exchange-side stops, so a monitor checks open positions every 30 s
    against the latest ticker price and closes breaches at that price. The close is
    published on execution.result like any other fill, so Risk's ledger stays in sync.
    MT5 mode relies on the terminal's native SL/TP and runs the monitor as a 5 s safety net.
"""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal

from agents.base import BaseAgent, run_agent
from agents.execution.broker_interface import BrokerPosition, ExecutionBroker
from core.messaging import Channels
from core.models.market import Ticker
from core.models.trade import ExecutionResult, OrderStatus, RiskAssessment

_BALANCE_KEY = "execution:balance"
_POSITIONS_KEY = "execution:positions"
_HISTORY_KEY = "execution:history"
_HISTORY_MAX = 100
_CACHE_TTL = 300  # seconds
_PAPER_MONITOR_INTERVAL_SECONDS = 30
_DB_WRITE_TIMEOUT_S = 2.0


class ExecutionAgent(BaseAgent):
    """
    Consumes approved RiskAssessments and routes them to the active broker.

    Run loop:
        1. Subscribe to risk.assessment.
        2. For each assessment:
           a. Check _should_continue() → exit if stopping.
           b. Check _trading_halted → skip if a halt is active.
           c. Skip rejected assessments (Risk agent already logged them).
           d. Place order via broker → publish ExecutionResult → cache portfolio → audit row.
    """

    name = "execution_agent"

    def __init__(self) -> None:
        super().__init__()
        # Broker is selected and instantiated in setup() so the import only happens
        # when the specific adapter is actually needed.
        self._broker: ExecutionBroker | None = None
        self._last_result: ExecutionResult | None = None
        # Latest traded price per symbol, from market.ticker.* messages.
        self._last_price: dict[str, Decimal] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Select broker from config, connect, and seed the Redis portfolio cache."""
        broker_type = self.settings.execution_broker.lower()
        if broker_type == "mt5":
            from agents.execution.mt5_bridge import MT5Bridge  # deferred import

            self._broker = MT5Bridge()
        else:
            from agents.execution.paper_broker import PaperBroker  # deferred import

            self._broker = PaperBroker(self.settings)

        if hasattr(self._broker, "set_bus"):
            self._broker.set_bus(self.bus)
        await self._broker.connect()
        caps = self._broker.capabilities
        self.log.info(
            "execution_agent_setup",
            broker=caps.broker_name,
            is_paper=caps.is_paper,
            supports_partial_fills=caps.supports_partial_fills,
        )
        # Seed cache so the API has an immediate response before any trade
        await self._cache_portfolio_state()

    async def teardown(self) -> None:
        """Disconnect broker on shutdown."""
        if self._broker is not None:
            await self._broker.disconnect()

    # ------------------------------------------------------------------
    # Core processing loop
    # ------------------------------------------------------------------

    async def run_loop(self) -> None:
        """
        Subscribe to risk.assessment and execute approved orders.

        Background tasks (a ticker listener and the stop-loss / take-profit monitor) are
        owned by this loop and cancelled when it exits.
        """
        assert self._broker is not None, "setup() must complete before run_loop()"

        background: list[asyncio.Task] = [
            asyncio.create_task(self._consume_tickers(), name="execution_ticker_listener")
        ]
        if self.settings.execution_broker.lower() == "mt5":
            background.append(
                asyncio.create_task(self._position_monitor_loop(), name="execution_mt5_monitor")
            )
        elif self._broker.capabilities.is_paper:
            background.append(
                asyncio.create_task(self._paper_position_monitor(), name="execution_paper_monitor")
            )

        try:
            async for assessment in self.bus.subscribe(
                Channels.RISK_ASSESSMENT,
                RiskAssessment,
            ):
                if not self._should_continue():
                    break

                if self._trading_halted:
                    self.log.warning(
                        "skipping_assessment_trading_halted",
                        proposal_id=str(assessment.proposal_id),
                        symbol=assessment.original_proposal.symbol,
                    )
                    continue

                if not assessment.is_approved:
                    self.log.info(
                        "assessment_rejected_skipped",
                        proposal_id=str(assessment.proposal_id),
                        symbol=assessment.original_proposal.symbol,
                        decision=assessment.decision,
                        reason=assessment.rejection_reason,
                    )
                    self._record_success()
                    continue

                try:
                    result = await self._broker.place_order(assessment)
                except Exception as exc:
                    self._handle_error(
                        exc,
                        context=f"place_order:{assessment.original_proposal.symbol}",
                    )
                    await self._publish_failure(assessment, exc)
                    continue

                try:
                    await self._record_fill(result, persist=True)
                    self._record_success()
                except Exception as exc:
                    self._handle_error(exc, context=f"record_fill:{result.symbol}")
                    continue

                self.log.info(
                    "order_executed",
                    proposal_id=str(assessment.proposal_id),
                    symbol=assessment.original_proposal.symbol,
                    side=str(result.side),
                    status=result.status,
                    fill_price=float(result.average_fill_price or 0),
                    fill_qty=float(result.filled_quantity),
                    realized_pnl_usd=(
                        float(result.realized_pnl_usd)
                        if result.realized_pnl_usd is not None
                        else None
                    ),
                    is_paper=result.is_paper,
                )
        finally:
            for task in background:
                task.cancel()
            await asyncio.gather(*background, return_exceptions=True)

    async def _record_fill(self, result: ExecutionResult, persist: bool) -> None:
        """Publish a fill, refresh the portfolio cache, append history, write the audit row."""
        await self.bus.publish(result)
        await self._cache_portfolio_state()
        await self._append_history(result)
        self._last_result = result
        if persist:
            await self._persist_execution(result)

    async def _publish_failure(self, assessment: RiskAssessment, exc: Exception) -> None:
        """Publish a rejected ExecutionResult so Risk releases the size it reserved."""
        proposal = assessment.original_proposal
        result = ExecutionResult(
            proposal_id=assessment.proposal_id,
            assessment_id=assessment.assessment_id,
            symbol=proposal.symbol,
            side=proposal.side,
            order_type=proposal.order_type,
            status=OrderStatus.REJECTED,
            requested_quantity=Decimal("0"),
            is_paper=self._broker.capabilities.is_paper if self._broker else True,
            error_message=(str(exc) or type(exc).__name__)[:500],
        )
        try:
            await self.bus.publish(result)
            await self._append_history(result)
        except Exception as publish_exc:
            self.log.error("execution_failure_publish_failed", error=str(publish_exc))

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------

    async def _consume_tickers(self) -> None:
        """Track the last traded price per symbol from market.ticker.* messages."""
        channels = [Channels.ticker(symbol) for symbol in self.settings.market_data.symbols]
        async for _channel, ticker in self.bus.subscribe_many(channels, Ticker):
            if not self._should_continue():
                break
            if ticker.last > Decimal("0"):
                self._last_price[ticker.symbol] = ticker.last

    async def _current_price(self, symbol: str) -> Decimal | None:
        """Latest ticker price; falls back to the most recent signal's price."""
        price = self._last_price.get(symbol)
        if price is not None:
            return price
        try:
            raw = await self.bus.kv_get(f"signal:technical:{symbol.replace('/', '-')}:latest")
        except Exception:
            return None
        if raw is None:
            return None
        try:
            value = Decimal(str(json.loads(raw)["price"]))
        except Exception:
            return None
        return value if value > Decimal("0") else None

    # ------------------------------------------------------------------
    # Stop-loss / take-profit monitors
    # ------------------------------------------------------------------

    async def _paper_position_monitor(self) -> None:
        """Paper mode: close positions whose price crosses the SL/TP thresholds."""
        assert self._broker is not None
        sl_pct = self.settings.mt5.stop_loss_pct
        tp_pct = self.settings.mt5.take_profit_pct

        while self._should_continue():
            await asyncio.sleep(_PAPER_MONITOR_INTERVAL_SECONDS)
            try:
                for pos in await self._broker.get_positions():
                    price = await self._current_price(pos.symbol)
                    if price is None:
                        continue  # no price yet — check again next cycle
                    await self._check_and_maybe_close(pos, sl_pct, tp_pct, price)
            except Exception as exc:
                self.log.error("paper_monitor_error", error=str(exc))

    async def _position_monitor_loop(self) -> None:
        """
        MT5 mode: safety-net monitor behind the terminal's native SL/TP.

        Current price comes from the MT5 heartbeat (pos.current_price, 1 Hz), falling back
        to the ticker feed.
        """
        assert self._broker is not None
        sl_pct = self.settings.mt5.stop_loss_pct
        tp_pct = self.settings.mt5.take_profit_pct
        interval = self.settings.mt5.position_monitor_interval_seconds
        self.log.info(
            "position_monitor_started",
            interval_seconds=interval,
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
        )

        while self._should_continue():
            try:
                await asyncio.sleep(interval)
                if not self._should_continue():
                    break
                if self._trading_halted:
                    continue

                positions = await asyncio.wait_for(self._broker.get_positions(), timeout=10.0)
                for pos in positions:
                    price = pos.current_price
                    if price <= Decimal("0"):
                        fallback = await self._current_price(pos.symbol)
                        if fallback is None:
                            continue
                        price = fallback
                    await asyncio.wait_for(
                        self._check_and_maybe_close(pos, sl_pct, tp_pct, price),
                        timeout=15.0,
                    )
            except TimeoutError:
                self.log.warning("position_monitor_timeout_skipping_cycle")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.log.error("position_monitor_error", error=str(exc))

    async def _check_and_maybe_close(
        self,
        pos: BrokerPosition,
        sl_pct: float,
        tp_pct: float,
        price: Decimal,
    ) -> None:
        """Evaluate SL/TP for one position and close it at ``price`` if breached."""
        entry = float(pos.entry_price)
        current = float(price)
        if entry <= 0 or current <= 0:
            return

        if pos.side == "buy":
            pnl_pct = (current - entry) / entry
        else:
            pnl_pct = (entry - current) / entry

        if pnl_pct <= -sl_pct:
            reason = "stop_loss_triggered"
        elif pnl_pct >= tp_pct:
            reason = "take_profit_triggered"
        else:
            return

        self.log.info(
            reason,
            symbol=pos.symbol,
            side=pos.side,
            entry_price=entry,
            current_price=current,
            pnl_pct=round(pnl_pct * 100, 3),
        )

        assert self._broker is not None
        result = await self._broker.close_position(pos.symbol, price=price)
        if result is not None:
            # Synthetic ids: there is no proposal row to reference, so no audit row.
            await self._record_fill(result, persist=False)
            self.log.info(
                "position_closed",
                symbol=pos.symbol,
                reason=reason,
                fill_price=float(result.average_fill_price or 0),
                fill_qty=float(result.filled_quantity),
            )

    # ------------------------------------------------------------------
    # Portfolio caching and audit
    # ------------------------------------------------------------------

    async def _cache_portfolio_state(self) -> None:
        """Write current balance and positions to Redis so the API can serve them."""
        if self._broker is None or not self.bus.connected:
            return
        try:
            balance = await self._broker.get_balance()
            positions = await self._broker.get_positions()

            balance_json = json.dumps(
                {
                    "total_equity_usd": float(balance.total_equity_usd),
                    "free_margin_usd": float(balance.free_margin_usd),
                    "used_margin_usd": float(balance.used_margin_usd),
                    "currency": balance.currency,
                }
            )
            positions_json = json.dumps(
                [
                    {
                        "symbol": p.symbol,
                        "side": p.side,
                        "quantity": float(p.quantity),
                        "entry_price": float(p.entry_price),
                        "current_price": float(p.current_price),
                        "unrealised_pnl_usd": float(p.unrealised_pnl_usd),
                        "stop_loss": float(p.stop_loss) if p.stop_loss else None,
                        "take_profit": float(p.take_profit) if p.take_profit else None,
                    }
                    for p in positions
                ]
            )

            await self.bus.kv_set(_BALANCE_KEY, balance_json, ttl_seconds=_CACHE_TTL)
            await self.bus.kv_set(_POSITIONS_KEY, positions_json, ttl_seconds=_CACHE_TTL)
        except Exception as exc:
            self.log.warning("portfolio_cache_write_failed", error=str(exc))

    async def _append_history(self, result: ExecutionResult) -> None:
        """Prepend the result to the Redis history list, capped at 100, in one transaction."""
        if not self.bus.connected:
            return
        try:
            await self.bus.list_push_capped(_HISTORY_KEY, result.to_json(), _HISTORY_MAX)
        except Exception as exc:
            self.log.warning("history_append_failed", error=str(exc))

    async def _persist_execution(self, result: ExecutionResult) -> None:
        """Best-effort audit write of a risk-approved fill (bounded by a 2 s timeout)."""
        try:
            await asyncio.wait_for(self._write_execution(result), timeout=_DB_WRITE_TIMEOUT_S)
        except Exception as exc:
            self.log.warning(
                "execution_persist_failed",
                result_id=str(result.result_id),
                error=str(exc) or type(exc).__name__,
            )

    async def _write_execution(self, result: ExecutionResult) -> None:
        from core.db.connection import get_session
        from core.db.repositories.trade_repo import TradeRepository

        async with get_session() as session:
            await TradeRepository(session).save_execution(result)

    # ------------------------------------------------------------------
    # Health reporting
    # ------------------------------------------------------------------

    def health_extra(self) -> dict:
        """Expose broker identity, halt state, and last fill for heartbeat monitoring."""
        if self._broker is None:
            return {"trading_halted": self._trading_halted}
        extra: dict = {
            "broker": self._broker.capabilities.broker_name,
            "is_paper": self._broker.capabilities.is_paper,
            "trading_halted": self._trading_halted,
            "prices_tracked": len(self._last_price),
        }
        if self._last_result:
            extra["last_fill"] = {
                "symbol": self._last_result.symbol,
                "side": self._last_result.side,
                "status": self._last_result.status,
                "fill_price": float(self._last_result.average_fill_price or 0),
                "fill_qty": float(self._last_result.filled_quantity),
            }
        return extra


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(run_agent(ExecutionAgent(), install_signal_handlers=True))
