"""
core/db/repositories/trade_repo.py
------------------------------------
Async repository for the trade lifecycle tables:
  trade_proposals → risk_assessments → executions

All methods accept Pydantic models and persist the subset of fields that exist
in the database schema.  Fields that exist only on the Pydantic model but not
in the DB (e.g. approved_take_profit_pct, stop_loss_order_id) are silently
dropped — see init.sql for the authoritative schema.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import UUID

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.db.models import ExecutionRow, RiskAssessmentRow, TradeProposalRow
from core.models.trade import (
    ExecutionResult,
    OrderSide,
    OrderStatus,
    RiskAssessment,
    TradeProposal,
)


def _row_to_execution(row: ExecutionRow) -> ExecutionResult:
    """Convert an ExecutionRow ORM row to an ExecutionResult Pydantic model."""
    return ExecutionResult(
        result_id=UUID(str(row.result_id)),
        proposal_id=UUID(str(row.proposal_id)),
        assessment_id=UUID(str(row.assessment_id)),
        exchange_order_id=row.exchange_order_id,
        timestamp=row.created_at,
        symbol=row.symbol,
        side=row.side,
        order_type=row.order_type,
        status=row.status,
        requested_quantity=Decimal(str(row.requested_quantity)),
        filled_quantity=Decimal(str(row.filled_quantity)),
        average_fill_price=(
            Decimal(str(row.average_fill_price)) if row.average_fill_price is not None else None
        ),
        total_cost_usd=(
            Decimal(str(row.total_cost_usd)) if row.total_cost_usd is not None else None
        ),
        fee_usd=Decimal(str(row.fee_usd)) if row.fee_usd is not None else None,
        fee_currency=row.fee_currency,
        is_paper=row.is_paper,
        error_message=row.error_message,
        retry_count=row.retry_count,
    )


class TradeRepository:
    """
    Thin async wrapper over trade_proposals, risk_assessments, and executions.

    Usage:
        async with get_session() as session:
            repo = TradeRepository(session)
            await repo.save_proposal(proposal)
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def save_proposal(self, proposal: TradeProposal) -> None:
        """Persist a TradeProposal to trade_proposals."""
        # use_enum_values=True on BaseMarketModel means enum fields are already strings.
        row = TradeProposalRow(
            proposal_id=proposal.proposal_id,
            symbol=proposal.symbol,
            side=str(proposal.side),
            order_type=str(proposal.order_type),
            requested_size_usd=proposal.requested_size_usd,
            suggested_stop_loss_pct=(
                Decimal(str(proposal.suggested_stop_loss_pct))
                if proposal.suggested_stop_loss_pct is not None
                else None
            ),
            suggested_take_profit_pct=(
                Decimal(str(proposal.suggested_take_profit_pct))
                if proposal.suggested_take_profit_pct is not None
                else None
            ),
            signal_direction=str(proposal.signal.direction),
            signal_confidence=Decimal(str(proposal.signal.confidence)),
            reasoning=proposal.reasoning,
            created_at=proposal.timestamp,
        )
        self.session.add(row)
        await self.session.commit()

    async def save_assessment(self, assessment: RiskAssessment) -> None:
        """Persist a RiskAssessment to risk_assessments.

        Note: approved_take_profit_pct is not stored (not in the DB schema).
        """
        row = RiskAssessmentRow(
            assessment_id=assessment.assessment_id,
            proposal_id=assessment.proposal_id,
            decision=str(assessment.decision),
            rejection_reason=(
                str(assessment.rejection_reason)
                if assessment.rejection_reason is not None
                else None
            ),
            rejection_detail=assessment.rejection_detail,
            approved_size_usd=assessment.approved_size_usd,
            approved_stop_loss_pct=(
                Decimal(str(assessment.approved_stop_loss_pct))
                if assessment.approved_stop_loss_pct is not None
                else None
            ),
            portfolio_value_usd=assessment.portfolio_value_usd,
            current_daily_loss_pct=(
                Decimal(str(assessment.current_daily_loss_pct))
                if assessment.current_daily_loss_pct is not None
                else None
            ),
            open_positions_count=assessment.open_positions_count,
            created_at=assessment.timestamp,
        )
        self.session.add(row)
        await self.session.commit()

    async def save_execution(self, result: ExecutionResult) -> None:
        """Persist an ExecutionResult to executions.

        Note: stop_loss_order_id and take_profit_order_id are not stored
        (not in the DB schema).
        """
        row = ExecutionRow(
            result_id=result.result_id,
            proposal_id=result.proposal_id,
            assessment_id=result.assessment_id,
            exchange_order_id=result.exchange_order_id,
            symbol=result.symbol,
            side=str(result.side),
            order_type=str(result.order_type),
            status=str(result.status),
            requested_quantity=result.requested_quantity,
            filled_quantity=result.filled_quantity,
            average_fill_price=result.average_fill_price,
            total_cost_usd=result.total_cost_usd,
            fee_usd=result.fee_usd,
            fee_currency=result.fee_currency,
            is_paper=result.is_paper,
            error_message=result.error_message,
            retry_count=result.retry_count,
            created_at=result.timestamp,
        )
        self.session.add(row)
        await self.session.commit()

    async def get_executions(
        self,
        symbol: str,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[ExecutionResult]:
        """Return executions for a symbol in ascending timestamp order."""
        stmt = (
            select(ExecutionRow)
            .where(ExecutionRow.symbol == symbol)
            .order_by(ExecutionRow.created_at.asc())
        )
        if since is not None:
            stmt = stmt.where(ExecutionRow.created_at >= since)
        stmt = stmt.limit(limit)

        rows = (await self.session.execute(stmt)).scalars().all()
        return [_row_to_execution(r) for r in rows]

    async def get_daily_pnl(self, target_date: date) -> float:
        """
        Compute realized PnL for a UTC calendar day.

        PnL = SUM(sell proceeds) - SUM(buy costs) - SUM(fees)
        Only FILLED and PARTIALLY_FILLED executions are included.

        Returns 0.0 when there are no qualifying executions.
        """
        day_start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=UTC)
        day_end = day_start + timedelta(days=1)

        filled_statuses = [
            OrderStatus.FILLED.value,
            OrderStatus.PARTIALLY_FILLED.value,
        ]

        # Signed cost: positive for sells, negative for buys.
        signed_cost = case(
            (ExecutionRow.side == OrderSide.SELL.value, ExecutionRow.total_cost_usd),
            else_=-ExecutionRow.total_cost_usd,
        )
        stmt = select(
            func.coalesce(func.sum(signed_cost), Decimal("0"))
            - func.coalesce(func.sum(ExecutionRow.fee_usd), Decimal("0"))
        ).where(
            ExecutionRow.created_at >= day_start,
            ExecutionRow.created_at < day_end,
            ExecutionRow.status.in_(filled_statuses),
        )
        result = await self.session.execute(stmt)
        value = result.scalar()
        return float(value) if value is not None else 0.0
