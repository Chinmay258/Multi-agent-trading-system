"""
core/db/repositories/candle_repo.py
-------------------------------------
Async repository for OHLCV candle persistence.

All methods accept / return core Pydantic models (OHLCVCandle) so callers
never touch ORM rows directly.  The session is injected at construction time
— there is no class-level state.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.db.models import OHLCVCandleRow
from core.models.market import OHLCVCandle


def _row_to_candle(row: OHLCVCandleRow) -> OHLCVCandle:
    """Convert an ORM row to an OHLCVCandle Pydantic model."""
    return OHLCVCandle(
        symbol=row.symbol,
        timeframe=row.timeframe,
        timestamp=row.timestamp,
        open=Decimal(str(row.open)),
        high=Decimal(str(row.high)),
        low=Decimal(str(row.low)),
        close=Decimal(str(row.close)),
        volume=Decimal(str(row.volume)),
        quote_volume=Decimal(str(row.quote_volume)) if row.quote_volume is not None else None,
        received_at=row.received_at,
    )


class CandleRepository:
    """
    Thin async wrapper over the ohlcv_candles table.

    Usage:
        async with get_session() as session:
            repo = CandleRepository(session)
            await repo.upsert_candle(candle)
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert_candle(self, candle: OHLCVCandle) -> None:
        """
        Insert or update a candle by its composite PK (symbol, timeframe, timestamp).

        Idempotent: calling twice with the same candle leaves exactly one row.
        """
        row = OHLCVCandleRow(
            symbol=candle.symbol,
            timeframe=candle.timeframe,
            timestamp=candle.timestamp,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
            quote_volume=candle.quote_volume,
            received_at=candle.received_at,
        )
        await self.session.merge(row)
        await self.session.commit()

    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> list[OHLCVCandle]:
        """
        Return candles for a symbol/timeframe in ascending timestamp order.

        Args:
            symbol:    Trading pair, e.g. "BTC/USDT".
            timeframe: Candle interval, e.g. "1h".
            since:     Inclusive lower bound (UTC).
            until:     Exclusive upper bound (UTC).
            limit:     Maximum number of rows returned. None means no limit.
        """
        stmt = (
            select(OHLCVCandleRow)
            .where(
                OHLCVCandleRow.symbol == symbol,
                OHLCVCandleRow.timeframe == timeframe,
            )
            .order_by(OHLCVCandleRow.timestamp.asc())
        )
        if since is not None:
            stmt = stmt.where(OHLCVCandleRow.timestamp >= since)
        if until is not None:
            stmt = stmt.where(OHLCVCandleRow.timestamp < until)
        if limit is not None:
            stmt = stmt.limit(limit)

        rows = (await self.session.execute(stmt)).scalars().all()
        return [_row_to_candle(r) for r in rows]

    async def get_latest_candle(self, symbol: str, timeframe: str) -> OHLCVCandle | None:
        """Return the most recent candle for a symbol/timeframe, or None."""
        stmt = (
            select(OHLCVCandleRow)
            .where(
                OHLCVCandleRow.symbol == symbol,
                OHLCVCandleRow.timeframe == timeframe,
            )
            .order_by(OHLCVCandleRow.timestamp.desc())
            .limit(1)
        )
        row = (await self.session.execute(stmt)).scalars().first()
        return _row_to_candle(row) if row is not None else None
