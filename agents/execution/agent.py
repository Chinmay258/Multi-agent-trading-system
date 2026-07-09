"""
agents/execution/agent.py
--------------------------
ExecutionAgent — the system's order execution layer.

Subscribes to:
    risk.assessment     — RiskAssessment from RiskAgent

Publishes:
    execution.result    — ExecutionResult after each fill attempt

Architecture rules:
    - Only imports from core/ and own package (agents.execution.*)
    - Broker is selected from config — ExecutionAgent never names PaperBroker directly
      in its logic; it talks only to the ExecutionBroker interface
    - All orders are gated by assessment.is_approved AND not self._trading_halted
    - RiskOverride handling: inherited from BaseAgent. Sets _trading_halted = True
      and (when requires_human_reset=True) calls stop(). The run_loop checks
      _trading_halted at the top of every iteration to halt immediately.
    - Portfolio state is cached to Redis after every fill so the FastAPI
      control plane can read it without talking to the broker directly.

Redis cache keys written by this agent:
    execution:balance       — JSON dict of BrokerBalance fields (TTL 300s)
    execution:positions     — JSON array of BrokerPosition fields (TTL 300s)
    execution:history       — Redis list of ExecutionResult JSON (last 100 fills)
"""

from __future__ import annotations

import asyncio
import json

from agents.base import BaseAgent, run_agent
from agents.execution.broker_interface import BrokerPosition, ExecutionBroker
from core.messaging import Channels
from core.models.trade import ExecutionResult, RiskAssessment

_BALANCE_KEY = "execution:balance"
_POSITIONS_KEY = "execution:positions"
_HISTORY_KEY = "execution:history"
_HISTORY_MAX = 100
_CACHE_TTL = 300  # seconds
_MONITOR_INTERVAL_SECONDS = 5  # default; overridden by MT5_POSITION_MONITOR_INTERVAL_SECONDS


class ExecutionAgent(BaseAgent):
    """
    Consumes approved RiskAssessments and routes them to the active broker.

    The broker implementation (paper vs live vs MT5) is selected at construction
    time based on TRADING_MODE config. The agent itself is broker-agnostic.

    Run loop:
        1. Subscribe to risk.assessment.
        2. For each assessment:
           a. Check _should_continue() → exit if stopping.
           b. Check _trading_halted → skip if risk override received.
           c. Skip rejected assessments silently (Risk agent already logged them).
           d. Place order via broker → publish ExecutionResult → cache portfolio.
    """

    name = "execution_agent"

    def __init__(self) -> None:
        super().__init__()
        # Broker is selected and instantiated in setup() — not here — so that
        # the import only happens when the specific adapter is actually needed.
        self._broker: ExecutionBroker | None = None
        self._last_result: ExecutionResult | None = None

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

        In MT5 mode also starts a background position monitor that checks
        open positions every MT5_POSITION_MONITOR_INTERVAL_SECONDS (default 5 s)
        as a safety net backup to MT5's native SL/TP.

        _trading_halted is checked at the top of each iteration — not only
        before place_order — so the agent stops processing new assessments
        immediately on RiskOverride, even if a message arrived concurrently.
        """
        assert self._broker is not None, "setup() must complete before run_loop()"

        # Start position monitor in MT5 mode
        monitor_task: asyncio.Task | None = None
        if self.settings.execution_broker.lower() == "mt5":
            monitor_task = asyncio.create_task(
                self._position_monitor_loop(),
                name="execution_position_monitor",
            )

        # Start paper SL/TP monitor when using PaperBroker
        from agents.execution.paper_broker import PaperBroker  # noqa: PLC0415

        if isinstance(self._broker, PaperBroker):
            asyncio.create_task(self._paper_position_monitor())

        try:
            async for assessment in self.bus.subscribe(
                Channels.RISK_ASSESSMENT,
                RiskAssessment,
            ):
                if not self._should_continue():
                    break

                # Immediate gate on risk override — do not even log the assessment
                if self._trading_halted:
                    self.log.warning(
                        "skipping_assessment_trading_halted",
                        proposal_id=str(assessment.proposal_id),
                        symbol=assessment.original_proposal.symbol,
                    )
                    continue

                # Skip rejected assessments — Risk agent already explained why
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

                # Execute approved order
                try:
                    result = await self._broker.place_order(assessment)
                    await self.bus.publish(result)
                    await self._cache_portfolio_state()
                    await self._append_history(result)
                    self._last_result = result
                    self._record_success()

                    self.log.info(
                        "order_executed",
                        proposal_id=str(assessment.proposal_id),
                        symbol=assessment.original_proposal.symbol,
                        side=str(assessment.original_proposal.side),
                        status=result.status,
                        fill_price=float(result.average_fill_price or 0),
                        fill_qty=float(result.filled_quantity),
                        is_paper=result.is_paper,
                    )
                except Exception as exc:
                    self._handle_error(
                        exc,
                        context=f"place_order:{assessment.original_proposal.symbol}",
                    )
        finally:
            if monitor_task is not None:
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass

    # ------------------------------------------------------------------
    # Position monitor (MT5 mode)
    # ------------------------------------------------------------------

    async def _position_monitor_loop(self) -> None:
        """
        Background task: safety-net monitor that closes positions on SL/TP breach.

        MT5's native SL/TP (set on the order at placement time) is the primary
        mechanism and fires in milliseconds. This loop is a Python-side backup:
        it runs every position_monitor_interval_seconds (default 5 s) and issues
        a CLOSE_POSITION command if a breach is detected.

        Current price comes from the MT5 heartbeat (pos.current_price, 1 Hz),
        with the Redis signal cache as a fallback.
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
                await asyncio.sleep(self.settings.mt5.position_monitor_interval_seconds)

                if not self._should_continue():
                    break
                if self._trading_halted:
                    continue

                positions = await asyncio.wait_for(self._broker.get_positions(), timeout=10.0)

                for pos in positions:
                    await asyncio.wait_for(
                        self._check_and_maybe_close(pos, sl_pct, tp_pct),
                        timeout=15.0,
                    )

            except TimeoutError:
                self.log.warning("position_monitor_timeout_skipping_cycle")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.log.error("position_monitor_error", error=str(exc))

    async def _paper_position_monitor(self) -> None:
        """
        Background task for paper mode: closes positions when SL or TP is breached.

        Current price is read from Redis key
        signal:technical:{symbol.replace("/","-")}:latest
        (published by TechnicalAnalysisAgent). If that key is absent or expired
        the position is skipped for this cycle — price will be checked again after
        the next interval.
        """
        assert self._broker is not None
        SL = self.settings.mt5.stop_loss_pct
        TP = self.settings.mt5.take_profit_pct
        interval = 30  # seconds

        while self._should_continue():
            await asyncio.sleep(interval)
            try:
                positions = await self._broker.get_positions()
                for pos in positions:
                    entry = float(pos.entry_price)
                    if entry == 0:
                        continue

                    # Read current price from the TA agent's Redis signal cache
                    current = 0.0
                    if self.bus._pool is not None:  # noqa: SLF001
                        cache_key = f"signal:technical:{pos.symbol.replace('/', '-')}:latest"
                        raw = await self.bus._pool.get(cache_key)  # noqa: SLF001
                        if raw is not None:
                            try:
                                signal_data = json.loads(raw)
                                current = float(signal_data["price"])
                            except Exception:
                                pass

                    if current == 0:
                        continue  # no price available — skip this cycle

                    if pos.side == "buy":
                        pnl_pct = (current - entry) / entry
                    else:
                        pnl_pct = (entry - current) / entry

                    if pnl_pct <= -SL:
                        self.log.info(
                            "paper_stop_loss_triggered",
                            symbol=pos.symbol,
                            pnl_pct=pnl_pct,
                        )
                        await self._broker.close_position(pos.symbol)

                    elif pnl_pct >= TP:
                        self.log.info(
                            "paper_take_profit_triggered",
                            symbol=pos.symbol,
                            pnl_pct=pnl_pct,
                        )
                        await self._broker.close_position(pos.symbol)

            except Exception as e:
                self.log.error("paper_monitor_error", error=str(e))

    async def _check_and_maybe_close(
        self,
        pos: BrokerPosition,
        sl_pct: float,
        tp_pct: float,
    ) -> None:
        """Evaluate SL/TP for one position and close it if a threshold is breached."""
        # Primary: use current_price from MT5 heartbeat (populated by get_positions()
        # from _cached_state — the EA posts this at 1 Hz so it is always fresh).
        current_price = float(pos.current_price)

        # Fallback: Redis signal cache if the heartbeat price is missing.
        if current_price <= 0 and self.bus._pool is not None:  # noqa: SLF001
            cache_key = f"signal_cache.technical.{pos.symbol.replace('/', '-')}"
            raw = await self.bus._pool.get(cache_key)  # noqa: SLF001
            if raw is not None:
                try:
                    signal_data = json.loads(raw)
                    current_price = float(signal_data["price"])
                except Exception:
                    pass

        if current_price <= 0:
            self.log.debug("position_monitor_no_price", symbol=pos.symbol)
            return

        self.log.debug(
            "position_monitor_check",
            symbol=pos.symbol,
            current_price=current_price,
        )

        entry = float(pos.entry_price)
        if entry <= 0 or current_price <= 0:
            return

        if pos.side == "buy":
            pnl_pct = (current_price - entry) / entry
        else:
            pnl_pct = (entry - current_price) / entry

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
            current_price=current_price,
            pnl_pct=round(pnl_pct * 100, 3),
        )

        result = await self._broker.close_position(pos.symbol)
        if result is not None:
            await self.bus.publish(result)
            await self._cache_portfolio_state()
            await self._append_history(result)
            self.log.info(
                "position_closed",
                symbol=pos.symbol,
                reason=reason,
                fill_price=float(result.average_fill_price or 0),
                fill_qty=float(result.filled_quantity),
            )

    # ------------------------------------------------------------------
    # Portfolio caching
    # ------------------------------------------------------------------

    async def _cache_portfolio_state(self) -> None:
        """
        Write current balance and positions to Redis so the API can serve
        them without calling the broker directly.

        Uses _pool directly (same pattern as RiskAgent) since BrokerBalance
        and BrokerPosition are dataclasses, not BaseMarketModel subclasses,
        so bus.cache_set() cannot be used here.
        """
        if self.bus._pool is None:  # noqa: SLF001
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

            await self.bus._pool.setex(_BALANCE_KEY, _CACHE_TTL, balance_json)  # noqa: SLF001
            await self.bus._pool.setex(_POSITIONS_KEY, _CACHE_TTL, positions_json)  # noqa: SLF001
        except Exception as exc:
            self.log.warning("portfolio_cache_write_failed", error=str(exc))

    async def _append_history(self, result: ExecutionResult) -> None:
        """Prepend the latest ExecutionResult to the Redis history list (cap at 100)."""
        if self.bus._pool is None:  # noqa: SLF001
            return
        try:
            await self.bus._pool.lpush(_HISTORY_KEY, result.to_json())  # noqa: SLF001
            await self.bus._pool.ltrim(_HISTORY_KEY, 0, _HISTORY_MAX - 1)  # noqa: SLF001
        except Exception as exc:
            self.log.warning("history_append_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Health reporting
    # ------------------------------------------------------------------

    def health_extra(self) -> dict:
        """Expose broker balance and last fill for heartbeat monitoring."""
        if self._broker is None:
            return {"trading_halted": self._trading_halted}
        extra: dict = {
            "broker": self._broker.capabilities.broker_name,
            "is_paper": self._broker.capabilities.is_paper,
            "trading_halted": self._trading_halted,
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
    asyncio.run(run_agent(ExecutionAgent()))
