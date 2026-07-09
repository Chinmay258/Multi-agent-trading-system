"""
agents/risk/agent.py
---------------------
Risk Management Agent — the last line of defence before any order reaches
the exchange.

Subscribes to:
    decision.proposal       — TradeProposal from DecisionAgent

Publishes:
    risk.assessment         — RiskAssessment (APPROVED / MODIFIED / REJECTED)
    system.risk_override    — Emergency halt when a limit is breached

Enforcement rules (all sourced from RiskSettings):
    1. Circuit breaker active → REJECT
    2. Trading halted via RiskOverride → REJECT
    3. Daily loss ≥ max_daily_loss_pct → REJECT + emergency halt
    4. Total drawdown ≥ max_total_drawdown_pct → REJECT + emergency halt
    5. Open positions ≥ max_open_positions → REJECT
    6. Signal data older than max_data_staleness_seconds → REJECT
    7. Approved size < min_order_size_usd → REJECT
    8. Approved size > requested size → APPROVED (no upward modification)
    9. Approved size < requested size → MODIFIED

Portfolio state is kept in-memory (PortfolioState) and the authoritative
portfolio value is written to Redis on each change so the Decision agent
can read it for sizing proposals.

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
from core.messaging import Channels
from core.metrics import OPEN_POSITIONS, ORDERS_REJECTED, PORTFOLIO_VALUE
from core.models.system import RiskOverride
from core.models.trade import (
    ExecutionResult,
    OrderSide,
    RejectionReason,
    RiskAssessment,
    RiskDecision,
    TradeProposal,
)

# Redis key where the Risk agent publishes the current portfolio value so
# the Decision agent can read it for proposal sizing.
_PORTFOLIO_VALUE_KEY = "portfolio:paper:value_usd"


class RiskAgent(BaseAgent):
    """
    Enforces hard risk limits and approves/modifies/rejects trade proposals.

    State is held in DrawdownMonitor (in-memory). The authoritative portfolio
    value is mirrored to Redis after each change.
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

    async def setup(self) -> None:
        """Persist initial portfolio value to Redis so the Decision agent sees it."""
        await self._persist_portfolio_value(self._drawdown.state.portfolio_value_usd)
        self.log.info(
            "risk_agent_setup",
            initial_balance_usd=self.settings.paper_initial_balance_usd,
            max_position_pct=self.settings.risk.max_position_pct,
            max_open_positions=self.settings.risk.max_open_positions,
            max_daily_loss_pct=self.settings.risk.max_daily_loss_pct,
            max_total_drawdown_pct=self.settings.risk.max_total_drawdown_pct,
        )
        self.log.info(
            "risk_agent_subscribing",
            channel=Channels.DECISION_PROPOSAL,
        )

    async def run_loop(self) -> None:
        """Subscribe to decision.proposal and evaluate each proposal in order."""
        result_task = asyncio.create_task(self._track_execution_results())
        try:
            async for proposal in self.bus.subscribe(
                Channels.DECISION_PROPOSAL,
                TradeProposal,
            ):
                if not self._should_continue():
                    break

                try:
                    assessment = await self._evaluate(proposal)
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
            try:
                await result_task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------
    # Execution result tracker
    # ------------------------------------------------------------------

    async def _track_execution_results(self) -> None:
        """
        Subscribe to execution.result and keep DrawdownMonitor in sync.

        Opening fills (BUY) are already registered by _evaluate() when a
        proposal is approved. This task handles the missing half: closing
        fills (SELL) that remove positions from the portfolio tracker,
        allowing the open-positions count to decrease and new proposals to
        be approved.
        """
        async for result in self.bus.subscribe(Channels.EXECUTION_RESULT, ExecutionResult):
            if not self._should_continue():
                break
            try:
                if result.side == OrderSide.SELL:
                    self._drawdown.close_position(
                        result.symbol,
                        float(result.total_cost_usd or 0),
                    )
                else:
                    self._drawdown.open_position(
                        result.symbol,
                        float(result.total_cost_usd or 0),
                    )
                await self._persist_portfolio_value(self._drawdown.state.portfolio_value_usd)
                PORTFOLIO_VALUE.set(float(self._drawdown.state.portfolio_value_usd))
                OPEN_POSITIONS.set(self._drawdown.state.open_positions_count)
            except Exception as exc:
                self.log.error("execution_result_tracking_error", error=str(exc))

    # ------------------------------------------------------------------
    # Evaluation pipeline
    # ------------------------------------------------------------------

    async def _evaluate(self, proposal: TradeProposal) -> RiskAssessment:
        """
        Run all risk checks in priority order and return a RiskAssessment.

        Checks are ordered by severity: system-level halts first, then
        portfolio-level limits, then sizing.
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

        # 2. Trading halted from base class risk override listener
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

        # 5. Max open positions
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

        # 6. Signal / data staleness
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

        # 7. Position sizing
        stop_pct = self._sizer.get_stop_loss_pct(proposal)
        approved_size = self._sizer.calculate(portfolio_value, stop_pct)

        # 8. Minimum order check
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

        # 9. Decide APPROVED or MODIFIED
        requested = proposal.requested_size_usd
        if approved_size >= requested:
            decision = RiskDecision.APPROVED
            approved_size = requested  # never increase beyond what was requested
        else:
            decision = RiskDecision.MODIFIED

        # Register position in portfolio tracker and persist value
        self._drawdown.open_position(proposal.symbol, approved_size)
        await self._persist_portfolio_value(self._drawdown.state.portfolio_value_usd)
        PORTFOLIO_VALUE.set(float(self._drawdown.state.portfolio_value_usd))
        OPEN_POSITIONS.set(self._drawdown.state.open_positions_count)

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
        """Trip the circuit breaker and broadcast a RiskOverride."""
        self._breaker.trip(reason)
        override = RiskOverride(
            reason=reason,
            triggered_by=self.name,
            daily_loss_pct=daily_loss_pct,
            total_drawdown_pct=total_drawdown_pct,
            requires_human_reset=True,
        )
        try:
            await self.bus.publish(override)
            self.log.critical("emergency_halt_triggered", reason=reason)
        except Exception as exc:
            self.log.error("emergency_halt_publish_failed", error=str(exc))

    @staticmethod
    def _signal_age_seconds(proposal: TradeProposal) -> float:
        """
        Return the age of the underlying technical signal in seconds.

        Uses the TechnicalSignal timestamp as a proxy for market data
        freshness. Returns 0.0 if no technical signal is available
        (gives the benefit of the doubt for non-technical-signal proposals).
        """
        tech = proposal.signal.technical_signal
        if tech is None:
            return 0.0
        now = datetime.now(UTC)
        # tech.timestamp may be stored as a string in some serialization paths;
        # Pydantic always restores it as datetime when the model is constructed.
        age = (now - tech.timestamp).total_seconds()
        return max(0.0, age)

    async def _persist_portfolio_value(self, value: Decimal) -> None:
        """
        Write current portfolio value to Redis so the Decision agent can
        read it for proposal sizing.

        Falls back silently on Redis failure — the Decision agent falls back
        to the initial balance if the key is absent.
        """
        try:
            if self.bus._pool is not None:  # noqa: SLF001  (private attr, same service)
                await self.bus._pool.set(_PORTFOLIO_VALUE_KEY, str(value))  # noqa: SLF001
        except Exception as exc:
            self.log.warning("portfolio_cache_write_failed", error=str(exc))

    def health_extra(self) -> dict:
        state = self._drawdown.state
        return {
            "portfolio_value_usd": float(state.portfolio_value_usd),
            "open_positions_count": state.open_positions_count,
            "open_positions": {k: float(v) for k, v in state.open_positions.items()},
            "daily_loss_pct": round(state.daily_loss_pct, 4),
            "total_drawdown_pct": round(state.total_drawdown_pct, 4),
            "circuit_breaker_tripped": self._breaker.is_tripped,
            "circuit_breaker_reason": self._breaker.reason,
        }


if __name__ == "__main__":
    import asyncio

    from agents.base import run_agent

    asyncio.run(run_agent(RiskAgent()))
