"""
agents/risk/agent.py
---------------------
Risk Management Agent — the last line of defence before any order reaches
the exchange.

Subscribes to:
    decision.proposal       — TradeProposal from DecisionAgent
    execution.result        — fills, used to keep the portfolio ledger in sync

Publishes:
    risk.assessment         — RiskAssessment (APPROVED / MODIFIED / REJECTED)
    system.risk_override    — Emergency halt when a limit is breached

Checks, in order (limits come from RiskSettings):
    1. Circuit breaker tripped → REJECT
    2. Trading halted (RiskOverride or persisted halt latch) → REJECT
    3. Daily loss ≥ max_daily_loss_pct → REJECT + emergency halt
    4. Total drawdown ≥ max_total_drawdown_pct → REJECT + emergency halt
    5. Underlying signal older than max_data_staleness_seconds → REJECT
    6. Symbol already has an open position:
         same side     → REJECT (no stacking)
         opposite side → APPROVE as a close, sized to the open position
    7. Open positions ≥ max_open_positions → REJECT
    8. Position sizing (fixed-fraction, clamped to limits)
    9. Approved size < min_order_size_usd → REJECT
   10. APPROVED (size = requested) or MODIFIED (size reduced)

Ledger: approval reserves the approved size; the matching execution result reconciles it
to the actual filled cost (see DrawdownMonitor). Results are applied idempotently by
result_id, so a redelivered result never double-counts.

Audit: every proposal and verdict is written to trade_proposals / risk_assessments. The
write is best-effort and bounded by a 2 s timeout so a slow database never blocks trading.

Architecture rules:
    - Imports only from core/ and own package (agents.risk.*)
    - Channel names from Channels class only
    - Config via self.settings only
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

from agents.base import BaseAgent
from agents.risk.circuit_breaker import CircuitBreaker
from agents.risk.drawdown_monitor import DrawdownMonitor
from agents.risk.position_sizer import PositionSizer
from core.messaging import HALT_KEY, Channels
from core.metrics import DAILY_PNL, OPEN_POSITIONS, ORDERS_REJECTED, PORTFOLIO_VALUE
from core.models.system import RiskOverride
from core.models.trade import (
    ExecutionResult,
    OrderSide,
    OrderStatus,
    RejectionReason,
    RiskAssessment,
    RiskDecision,
    TradeProposal,
)

# Redis key where the Risk agent publishes the current portfolio value.
_PORTFOLIO_VALUE_KEY = "portfolio:paper:value_usd"

# How many recent execution result ids to remember for idempotent application.
_SEEN_RESULTS_MAX = 1000

# Upper bound on the audit write so the database can never stall the risk path.
_DB_WRITE_TIMEOUT_S = 2.0

_FILLED_STATUSES = {OrderStatus.FILLED.value, OrderStatus.PARTIALLY_FILLED.value}


def _side_value(side: OrderSide | str) -> str:
    return side.value if isinstance(side, OrderSide) else str(side)


def _status_value(status: OrderStatus | str) -> str:
    return status.value if isinstance(status, OrderStatus) else str(status)


class RiskAgent(BaseAgent):
    """
    Enforces hard risk limits and approves/modifies/rejects trade proposals.

    State is held in DrawdownMonitor (in-memory). The portfolio value is mirrored to
    Redis after each change.
    """

    name = "risk_agent"

    def __init__(self) -> None:
        super().__init__()
        risk = self.settings.risk

        self._drawdown = DrawdownMonitor(
            initial_balance_usd=self.settings.paper_initial_balance_usd,
            max_daily_loss_pct=risk.max_daily_loss_pct,
            max_total_drawdown_pct=risk.max_total_drawdown_pct,
        )
        self._sizer = PositionSizer(
            max_position_pct=risk.max_position_pct,
            min_order_usd=risk.min_order_size_usd,
            max_order_usd=risk.max_order_size_usd,
        )
        self._breaker = CircuitBreaker()
        # result_id → None, insertion-ordered; oldest entries are evicted past the cap.
        self._seen_results: dict[str, None] = {}

    async def setup(self) -> None:
        """Persist initial portfolio value to Redis so other components see it."""
        await self._persist_portfolio_value(self._drawdown.state.portfolio_value_usd)
        self.log.info(
            "risk_agent_setup",
            initial_balance_usd=self.settings.paper_initial_balance_usd,
            max_position_pct=self.settings.risk.max_position_pct,
            max_open_positions=self.settings.risk.max_open_positions,
            max_daily_loss_pct=self.settings.risk.max_daily_loss_pct,
            max_total_drawdown_pct=self.settings.risk.max_total_drawdown_pct,
        )

    async def run_loop(self) -> None:
        """Subscribe to decision.proposal and evaluate each proposal in order."""
        result_task = asyncio.create_task(
            self._track_execution_results(), name="risk_execution_result_tracker"
        )
        try:
            async for proposal in self.bus.subscribe(
                Channels.DECISION_PROPOSAL,
                TradeProposal,
            ):
                if not self._should_continue():
                    break

                try:
                    assessment = await self._evaluate(proposal)
                    # Write the audit rows before publishing so the execution row that
                    # follows can reference them.
                    await self._persist_decision(proposal, assessment)
                    await self.bus.publish(assessment)
                    self._record_success()

                    self.log.info(
                        "assessment_published",
                        proposal_id=str(proposal.proposal_id),
                        symbol=proposal.symbol,
                        decision=assessment.decision,
                        rejection_reason=assessment.rejection_reason,
                    )
                except Exception as exc:
                    self._handle_error(exc, context=f"evaluate:{proposal.symbol}")
        finally:
            result_task.cancel()
            await asyncio.gather(result_task, return_exceptions=True)

    # ------------------------------------------------------------------
    # Execution result tracking
    # ------------------------------------------------------------------

    async def _track_execution_results(self) -> None:
        """Subscribe to execution.result and keep the ledger in sync with actual fills."""
        async for result in self.bus.subscribe(Channels.EXECUTION_RESULT, ExecutionResult):
            if not self._should_continue():
                break
            try:
                action = self._apply_execution_result(result)
                self.log.debug(
                    "execution_result_applied",
                    result_id=str(result.result_id),
                    symbol=result.symbol,
                    action=action,
                )
                await self._persist_portfolio_value(self._drawdown.state.portfolio_value_usd)
                self._export_state_metrics()
            except Exception as exc:
                self.log.error("execution_result_tracking_error", error=str(exc))

    def _apply_execution_result(self, result: ExecutionResult) -> str:
        """
        Apply one execution result to the ledger. Returns what happened, for logs/tests.

        - duplicate:  result_id already applied → ignored
        - released:   an opening order failed → its reservation is returned to cash
        - ignored:    a failed order that held no reservation (e.g. a failed close)
        - closed:     a fill on the opposite side of an open position → realise PnL
        - reconciled: a fill for a reserved position → reservation replaced by actual cost
        - adopted:    a fill the ledger never reserved (e.g. after a restart) → tracked
        """
        result_id = str(result.result_id)
        if result_id in self._seen_results:
            return "duplicate"
        self._seen_results[result_id] = None
        if len(self._seen_results) > _SEEN_RESULTS_MAX:
            del self._seen_results[next(iter(self._seen_results))]

        symbol = result.symbol
        side = _side_value(result.side)
        held_side = self._drawdown.position_side(symbol)
        filled = _status_value(
            result.status
        ) in _FILLED_STATUSES and result.filled_quantity > Decimal("0")

        if not filled:
            if held_side == side:
                self._drawdown.close_position(symbol, Decimal("0"))
                return "released"
            return "ignored"

        if held_side is not None and held_side != side:
            pnl = result.realized_pnl_usd if result.realized_pnl_usd is not None else Decimal("0")
            self._drawdown.close_position(symbol, pnl)
            return "closed"

        actual_cost = (result.total_cost_usd or Decimal("0")) + (result.fee_usd or Decimal("0"))
        if held_side == side:
            self._drawdown.adjust_position_cost(symbol, actual_cost)
            return "reconciled"

        self._drawdown.open_position(symbol, actual_cost, side=side)
        return "adopted"

    # ------------------------------------------------------------------
    # Evaluation pipeline
    # ------------------------------------------------------------------

    async def _evaluate(self, proposal: TradeProposal) -> RiskAssessment:
        """
        Run all risk checks in priority order and return a RiskAssessment.

        Checks are ordered by severity: system-level halts first, then data freshness and
        position state, then portfolio limits, then sizing.
        """
        state = self._drawdown.state
        portfolio_value = state.portfolio_value_usd
        daily_loss_pct = state.daily_loss_pct
        open_count = state.open_positions_count

        # 1. Circuit breaker
        if self._breaker.is_tripped:
            return self._reject(
                proposal,
                RejectionReason.CIRCUIT_BREAKER_ACTIVE,
                f"Circuit breaker tripped: {self._breaker.reason}",
                portfolio_value,
                daily_loss_pct,
                open_count,
            )

        # 2. Trading halted (override message or persisted halt latch)
        if self._trading_halted:
            return self._reject(
                proposal,
                RejectionReason.CIRCUIT_BREAKER_ACTIVE,
                "Trading halted by risk override signal",
                portfolio_value,
                daily_loss_pct,
                open_count,
            )

        # 3. Daily loss limit
        if self._drawdown.daily_loss_limit_breached():
            await self._emergency_halt(
                f"Daily loss limit breached: {state.daily_loss_pct:.2%} "
                f">= {self.settings.risk.max_daily_loss_pct:.2%}",
                daily_loss_pct=state.daily_loss_pct,
            )
            return self._reject(
                proposal,
                RejectionReason.DAILY_LOSS_LIMIT,
                f"Daily loss {state.daily_loss_pct:.2%} exceeds limit",
                portfolio_value,
                daily_loss_pct,
                open_count,
            )

        # 4. Total drawdown limit
        if self._drawdown.total_drawdown_limit_breached():
            await self._emergency_halt(
                f"Total drawdown limit breached: {state.total_drawdown_pct:.2%} "
                f">= {self.settings.risk.max_total_drawdown_pct:.2%}",
                total_drawdown_pct=state.total_drawdown_pct,
            )
            return self._reject(
                proposal,
                RejectionReason.TOTAL_DRAWDOWN_LIMIT,
                f"Total drawdown {state.total_drawdown_pct:.2%} exceeds limit",
                portfolio_value,
                daily_loss_pct,
                open_count,
            )

        # 5. Signal / data staleness
        signal_age = self._signal_age_seconds(proposal)
        if signal_age > self.settings.risk.max_data_staleness_seconds:
            return self._reject(
                proposal,
                RejectionReason.STALE_MARKET_DATA,
                f"Underlying signal is {signal_age:.0f}s old "
                f"(limit {self.settings.risk.max_data_staleness_seconds}s)",
                portfolio_value,
                daily_loss_pct,
                open_count,
            )

        # 6. Existing position on this symbol: no stacking; opposite side closes it.
        proposal_side = _side_value(proposal.side)
        held_side = self._drawdown.position_side(proposal.symbol)
        if held_side is not None:
            if held_side == proposal_side:
                return self._reject(
                    proposal,
                    RejectionReason.DUPLICATE_SIGNAL,
                    f"{proposal.symbol} already has an open {held_side} position (no stacking)",
                    portfolio_value,
                    daily_loss_pct,
                    open_count,
                )
            return RiskAssessment(
                proposal_id=proposal.proposal_id,
                decision=RiskDecision.APPROVED,
                approved_size_usd=state.open_positions[proposal.symbol],
                portfolio_value_usd=portfolio_value,
                current_daily_loss_pct=daily_loss_pct,
                open_positions_count=open_count,
                original_proposal=proposal,
            )

        # 7. Max open positions
        if open_count >= self.settings.risk.max_open_positions:
            return self._reject(
                proposal,
                RejectionReason.MAX_OPEN_POSITIONS,
                f"Max open positions reached ({open_count} / "
                f"{self.settings.risk.max_open_positions})",
                portfolio_value,
                daily_loss_pct,
                open_count,
            )

        # 8. Position sizing
        stop_pct = self._sizer.get_stop_loss_pct(proposal)
        approved_size = self._sizer.calculate(portfolio_value, stop_pct)

        # 9. Minimum order check
        if approved_size < Decimal(str(self.settings.risk.min_order_size_usd)):
            return self._reject(
                proposal,
                RejectionReason.ORDER_SIZE_TOO_SMALL,
                f"Approved size ${approved_size:.2f} < minimum "
                f"${self.settings.risk.min_order_size_usd}",
                portfolio_value,
                daily_loss_pct,
                open_count,
            )

        # 10. Decide APPROVED or MODIFIED
        requested = proposal.requested_size_usd
        if approved_size >= requested:
            decision = RiskDecision.APPROVED
            approved_size = requested  # never increase beyond what was requested
        else:
            decision = RiskDecision.MODIFIED

        # Reserve the approved size; the fill reconciles it to the actual cost.
        self._drawdown.open_position(proposal.symbol, approved_size, side=proposal_side)
        await self._persist_portfolio_value(self._drawdown.state.portfolio_value_usd)
        self._export_state_metrics()

        return RiskAssessment(
            proposal_id=proposal.proposal_id,
            decision=decision,
            approved_size_usd=approved_size,
            approved_stop_loss_pct=stop_pct,
            approved_take_profit_pct=proposal.suggested_take_profit_pct,
            portfolio_value_usd=portfolio_value,
            current_daily_loss_pct=daily_loss_pct,
            open_positions_count=open_count + 1,  # includes the new position
            original_proposal=proposal,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reject(
        self,
        proposal: TradeProposal,
        reason: RejectionReason,
        detail: str,
        portfolio_value: Decimal,
        daily_loss_pct: float,
        open_count: int,
    ) -> RiskAssessment:
        ORDERS_REJECTED.labels(symbol=proposal.symbol, reason=reason.value).inc()
        return RiskAssessment(
            proposal_id=proposal.proposal_id,
            decision=RiskDecision.REJECTED,
            rejection_reason=reason,
            rejection_detail=detail,
            portfolio_value_usd=portfolio_value,
            current_daily_loss_pct=daily_loss_pct,
            open_positions_count=open_count,
            original_proposal=proposal,
        )

    async def _emergency_halt(
        self,
        reason: str,
        daily_loss_pct: float | None = None,
        total_drawdown_pct: float | None = None,
    ) -> None:
        """Trip the circuit breaker, persist the halt latch, and broadcast a RiskOverride."""
        self._breaker.trip(reason)
        override = RiskOverride(
            reason=reason,
            triggered_by=self.name,
            daily_loss_pct=daily_loss_pct,
            total_drawdown_pct=total_drawdown_pct,
            requires_human_reset=True,
        )
        try:
            # Persist first: agents restarted by Docker read this latch on startup.
            if self.bus.connected:
                await self.bus.kv_set(HALT_KEY, reason)
            await self.bus.publish(override)
            self.log.critical("emergency_halt_triggered", reason=reason)
        except Exception as exc:
            self.log.error("emergency_halt_publish_failed", error=str(exc))

    @staticmethod
    def _signal_age_seconds(proposal: TradeProposal) -> float:
        """
        Return the age of the underlying technical signal in seconds.

        Uses the TechnicalSignal timestamp as a proxy for market data freshness. Returns
        0.0 if no technical signal is available.
        """
        tech = proposal.signal.technical_signal
        if tech is None:
            return 0.0
        now = datetime.now(UTC)
        age = (now - tech.timestamp).total_seconds()
        return max(0.0, age)

    def _export_state_metrics(self) -> None:
        state = self._drawdown.state
        PORTFOLIO_VALUE.set(float(state.portfolio_value_usd))
        OPEN_POSITIONS.set(state.open_positions_count)
        DAILY_PNL.set(float(state.daily_realized_pnl))

    async def _persist_portfolio_value(self, value: Decimal) -> None:
        """Mirror the current portfolio value to Redis (skipped when not connected)."""
        if not self.bus.connected:
            return
        try:
            await self.bus.kv_set(_PORTFOLIO_VALUE_KEY, str(value))
        except Exception as exc:
            self.log.warning("portfolio_cache_write_failed", error=str(exc))

    async def _persist_decision(self, proposal: TradeProposal, assessment: RiskAssessment) -> None:
        """Best-effort audit write of the proposal and verdict."""
        try:
            await asyncio.wait_for(
                self._write_decision(proposal, assessment), timeout=_DB_WRITE_TIMEOUT_S
            )
        except Exception as exc:
            self.log.warning(
                "decision_persist_failed",
                proposal_id=str(proposal.proposal_id),
                error=str(exc) or type(exc).__name__,
            )

    async def _write_decision(self, proposal: TradeProposal, assessment: RiskAssessment) -> None:
        from core.db.connection import get_session
        from core.db.repositories.trade_repo import TradeRepository

        async with get_session() as session:
            repo = TradeRepository(session)
            await repo.save_proposal(proposal)
            await repo.save_assessment(assessment)

    def health_extra(self) -> dict:
        state = self._drawdown.state
        return {
            "portfolio_value_usd": float(state.portfolio_value_usd),
            "open_positions_count": state.open_positions_count,
            "open_positions": {k: float(v) for k, v in state.open_positions.items()},
            "position_sides": dict(state.position_sides),
            "daily_loss_pct": round(state.daily_loss_pct, 4),
            "total_drawdown_pct": round(state.total_drawdown_pct, 4),
            "circuit_breaker_tripped": self._breaker.is_tripped,
            "circuit_breaker_reason": self._breaker.reason,
        }


if __name__ == "__main__":
    import asyncio

    from agents.base import run_agent

    asyncio.run(run_agent(RiskAgent(), install_signal_handlers=True))
