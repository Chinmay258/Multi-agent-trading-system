"""
core/db/repositories/position_repo.py
---------------------------------------
Async repository for the positions table.

The positions table has a UNIQUE constraint on symbol (one open position per
symbol at a time). upsert_position() does a SELECT-then-update-or-insert so it
works identically on PostgreSQL and SQLite (the test database).

BrokerPosition field name differences vs. DB column names:
  BrokerPosition.stop_loss   → positions.stop_loss_price
  BrokerPosition.take_profit → positions.take_profit_price
  BrokerPosition.position_id → positions.position_id (Optional[str] vs. UUID)
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from agents.execution.broker_interface import BrokerPosition
from core.config import get_settings
from core.db.models import PositionRow


def _row_to_position(row: PositionRow) -> BrokerPosition:
    """Convert an ORM PositionRow to a BrokerPosition dataclass."""
    return BrokerPosition(
        symbol=row.symbol,
        side=row.side,
        quantity=Decimal(str(row.quantity)),
        entry_price=Decimal(str(row.entry_price)),
        current_price=Decimal(str(row.current_price))
        if row.current_price is not None
        else Decimal("0"),
        unrealised_pnl_usd=(
            Decimal(str(row.unrealised_pnl_usd))
            if row.unrealised_pnl_usd is not None
            else Decimal("0")
        ),
        stop_loss=(Decimal(str(row.stop_loss_price)) if row.stop_loss_price is not None else None),
        take_profit=(
            Decimal(str(row.take_profit_price)) if row.take_profit_price is not None else None
        ),
        position_id=str(row.position_id),
    )


class PositionRepository:
    """
    Thin async wrapper over the positions table.

    Usage:
        async with get_session() as session:
            repo = PositionRepository(session)
            await repo.upsert_position(broker_position)
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert_position(self, position: BrokerPosition) -> None:
        """
        Insert or update the position for a given symbol.

        Looks up by symbol (UNIQUE column). If a row exists, updates it.
        If not, creates a new row with a fresh UUID.
        """
        now = datetime.now(UTC)
        is_paper = get_settings().is_paper_trading

        stmt = select(PositionRow).where(PositionRow.symbol == position.symbol)
        existing = (await self.session.execute(stmt)).scalars().first()

        if existing is not None:
            existing.side = position.side
            existing.quantity = position.quantity
            existing.entry_price = position.entry_price
            existing.current_price = position.current_price
            existing.unrealised_pnl_usd = position.unrealised_pnl_usd
            existing.stop_loss_price = position.stop_loss
            existing.take_profit_price = position.take_profit
            existing.is_paper = is_paper
            existing.updated_at = now
        else:
            row = PositionRow(
                position_id=uuid4(),
                symbol=position.symbol,
                side=position.side,
                quantity=position.quantity,
                entry_price=position.entry_price,
                current_price=position.current_price,
                unrealised_pnl_usd=position.unrealised_pnl_usd,
                stop_loss_price=position.stop_loss,
                take_profit_price=position.take_profit,
                is_paper=is_paper,
                opened_at=now,
                updated_at=now,
            )
            self.session.add(row)

        await self.session.commit()

    async def get_open_positions(self) -> list[BrokerPosition]:
        """Return all open positions as BrokerPosition dataclasses."""
        stmt = select(PositionRow)
        rows = (await self.session.execute(stmt)).scalars().all()
        return [_row_to_position(r) for r in rows]

    async def close_position(self, symbol: str) -> None:
        """Delete the position row for a symbol (mark as closed)."""
        await self.session.execute(delete(PositionRow).where(PositionRow.symbol == symbol))
        await self.session.commit()

    async def get_position(self, symbol: str) -> BrokerPosition | None:
        """Return the open position for a symbol, or None if not found."""
        stmt = select(PositionRow).where(PositionRow.symbol == symbol)
        row = (await self.session.execute(stmt)).scalars().first()
        return _row_to_position(row) if row is not None else None
