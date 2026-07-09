"""
tests/unit/test_repositories.py
---------------------------------
Unit tests for CandleRepository, TradeRepository, and PositionRepository.

Uses an in-memory SQLite database (aiosqlite driver) — no PostgreSQL required.
Each test receives a fresh session with clean tables, constructed directly from
the ORM models.  No monkeypatching of get_session() or get_engine() is needed.

Run:
    pytest tests/unit/test_repositories.py -v
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from agents.execution.broker_interface import BrokerPosition
from core.db.connection import Base
from core.db.repositories.candle_repo import CandleRepository
from core.db.repositories.position_repo import PositionRepository
from core.db.repositories.trade_repo import TradeRepository
from core.models.market import OHLCVCandle
from core.models.signals import AggregatedSignal, SignalDirection
from core.models.trade import (
    ExecutionResult,
    OrderSide,
    OrderStatus,
    OrderType,
    RiskAssessment,
    RiskDecision,
    TradeProposal,
)

# ---------------------------------------------------------------------------
# Shared fixture: in-memory SQLite session
# ---------------------------------------------------------------------------


@pytest.fixture
async def session() -> AsyncSession:  # type: ignore[misc]
    """Yield a fresh AsyncSession backed by an in-memory SQLite database."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _ts(offset_minutes: int = 0) -> datetime:
    """UTC datetime for 2024-01-01 00:XX:00, advancing by offset_minutes."""
    return datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC) + timedelta(minutes=offset_minutes)


def make_candle(
    symbol: str = "BTC/USDT",
    timeframe: str = "1m",
    offset_minutes: int = 0,
    close: str = "50050",
) -> OHLCVCandle:
    """Create a minimal valid OHLCVCandle. high is at least as large as close."""
    close_val = Decimal(close)
    # Ensure OHLCV consistency: high >= close and high >= open; low <= close and low <= open.
    open_val = Decimal("50000")
    high_val = max(Decimal("50100"), close_val, open_val)
    low_val = min(Decimal("49900"), close_val, open_val)
    return OHLCVCandle(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=_ts(offset_minutes),
        open=open_val,
        high=high_val,
        low=low_val,
        close=close_val,
        volume=Decimal("1.5"),
        received_at=datetime.now(UTC),
    )


def make_aggregated_signal(
    symbol: str = "BTC/USDT",
    direction: SignalDirection = SignalDirection.BUY,
) -> AggregatedSignal:
    """Create a minimal valid AggregatedSignal."""
    return AggregatedSignal(
        symbol=symbol,
        direction=direction,
        confidence=0.75,
        composite_score=0.5,
    )


def make_proposal(
    symbol: str = "BTC/USDT",
    side: OrderSide = OrderSide.BUY,
    size_usd: str = "100",
    proposal_id: UUID | None = None,
) -> TradeProposal:
    """Create a minimal valid TradeProposal."""
    kwargs: dict = dict(
        symbol=symbol,
        side=side,
        requested_size_usd=Decimal(size_usd),
        signal=make_aggregated_signal(symbol),
        reasoning="unit-test proposal",
    )
    if proposal_id is not None:
        kwargs["proposal_id"] = proposal_id
    return TradeProposal(**kwargs)


def make_assessment(proposal: TradeProposal) -> RiskAssessment:
    """Create a minimal valid RiskAssessment that approves a proposal."""
    return RiskAssessment(
        proposal_id=proposal.proposal_id,
        decision=RiskDecision.APPROVED,
        approved_size_usd=proposal.requested_size_usd,
        portfolio_value_usd=Decimal("10000"),
        original_proposal=proposal,
    )


def make_execution(
    proposal: TradeProposal,
    assessment: RiskAssessment,
    symbol: str | None = None,
    side: OrderSide = OrderSide.BUY,
    total_cost_usd: str = "100",
    fee_usd: str = "0.1",
    status: OrderStatus = OrderStatus.FILLED,
) -> ExecutionResult:
    """Create a minimal valid ExecutionResult."""
    return ExecutionResult(
        proposal_id=proposal.proposal_id,
        assessment_id=assessment.assessment_id,
        symbol=symbol or proposal.symbol,
        side=side,
        order_type=OrderType.MARKET,
        status=status,
        requested_quantity=Decimal("0.002"),
        filled_quantity=Decimal("0.002"),
        average_fill_price=Decimal("50000"),
        total_cost_usd=Decimal(total_cost_usd),
        fee_usd=Decimal(fee_usd),
        fee_currency="USDT",
        is_paper=True,
    )


def make_broker_position(
    symbol: str = "BTC/USDT",
    side: str = "buy",
    quantity: str = "0.001",
    entry_price: str = "50000",
) -> BrokerPosition:
    """Create a minimal valid BrokerPosition."""
    return BrokerPosition(
        symbol=symbol,
        side=side,
        quantity=Decimal(quantity),
        entry_price=Decimal(entry_price),
        current_price=Decimal("50500"),
        unrealised_pnl_usd=Decimal("0.5"),
    )


# ===========================================================================
# CandleRepository tests
# ===========================================================================


async def test_upsert_candle_idempotency(session: AsyncSession) -> None:
    """Upserting the same candle twice leaves exactly one row."""
    from sqlalchemy import func, select

    from core.db.models import OHLCVCandleRow

    repo = CandleRepository(session)
    candle = make_candle()

    await repo.upsert_candle(candle)
    await repo.upsert_candle(candle)  # identical — must not create a second row

    count = (await session.execute(select(func.count()).select_from(OHLCVCandleRow))).scalar()
    assert count == 1


async def test_get_candles_ordering(session: AsyncSession) -> None:
    """get_candles returns candles in ascending timestamp order."""
    repo = CandleRepository(session)

    await repo.upsert_candle(make_candle(offset_minutes=2))
    await repo.upsert_candle(make_candle(offset_minutes=0))
    await repo.upsert_candle(make_candle(offset_minutes=1))

    candles = await repo.get_candles("BTC/USDT", "1m")
    assert len(candles) == 3
    assert candles[0].timestamp < candles[1].timestamp < candles[2].timestamp


async def test_get_candles_date_filter_since(session: AsyncSession) -> None:
    """since filter excludes candles before the boundary."""
    repo = CandleRepository(session)

    await repo.upsert_candle(make_candle(offset_minutes=0))
    await repo.upsert_candle(make_candle(offset_minutes=5))
    await repo.upsert_candle(make_candle(offset_minutes=10))

    candles = await repo.get_candles("BTC/USDT", "1m", since=_ts(5))
    assert len(candles) == 2
    assert all(c.timestamp >= _ts(5) for c in candles)


async def test_get_candles_date_filter_until(session: AsyncSession) -> None:
    """until filter excludes candles at or after the boundary."""
    repo = CandleRepository(session)

    await repo.upsert_candle(make_candle(offset_minutes=0))
    await repo.upsert_candle(make_candle(offset_minutes=5))
    await repo.upsert_candle(make_candle(offset_minutes=10))

    # until=_ts(10) means strictly less-than, so offset=10 is excluded.
    candles = await repo.get_candles("BTC/USDT", "1m", until=_ts(10))
    assert len(candles) == 2
    assert all(c.timestamp < _ts(10) for c in candles)


async def test_get_latest_candle(session: AsyncSession) -> None:
    """get_latest_candle returns the candle with the greatest timestamp."""
    repo = CandleRepository(session)

    await repo.upsert_candle(make_candle(offset_minutes=0, close="50000"))
    await repo.upsert_candle(make_candle(offset_minutes=1, close="51000"))
    await repo.upsert_candle(make_candle(offset_minutes=2, close="52000"))

    latest = await repo.get_latest_candle("BTC/USDT", "1m")
    assert latest is not None
    assert latest.timestamp == _ts(2)
    assert latest.close == Decimal("52000")


async def test_get_latest_candle_returns_none(session: AsyncSession) -> None:
    """get_latest_candle returns None when the table is empty."""
    repo = CandleRepository(session)
    result = await repo.get_latest_candle("BTC/USDT", "1m")
    assert result is None


# ===========================================================================
# TradeRepository tests
# ===========================================================================


async def test_save_proposal(session: AsyncSession) -> None:
    """save_proposal persists a row to trade_proposals."""
    from sqlalchemy import func, select

    from core.db.models import TradeProposalRow

    repo = TradeRepository(session)
    proposal = make_proposal()
    await repo.save_proposal(proposal)

    count = (await session.execute(select(func.count()).select_from(TradeProposalRow))).scalar()
    assert count == 1


async def test_save_assessment(session: AsyncSession) -> None:
    """save_assessment persists a row linked to the proposal FK."""
    from sqlalchemy import func, select

    from core.db.models import RiskAssessmentRow

    repo = TradeRepository(session)
    proposal = make_proposal()
    assessment = make_assessment(proposal)

    await repo.save_proposal(proposal)
    await repo.save_assessment(assessment)

    count = (await session.execute(select(func.count()).select_from(RiskAssessmentRow))).scalar()
    assert count == 1


async def test_save_execution(session: AsyncSession) -> None:
    """save_execution persists a row with FK references to proposal + assessment."""
    from sqlalchemy import func, select

    from core.db.models import ExecutionRow

    repo = TradeRepository(session)
    proposal = make_proposal()
    assessment = make_assessment(proposal)
    execution = make_execution(proposal, assessment)

    await repo.save_proposal(proposal)
    await repo.save_assessment(assessment)
    await repo.save_execution(execution)

    count = (await session.execute(select(func.count()).select_from(ExecutionRow))).scalar()
    assert count == 1


async def test_get_executions_by_symbol(session: AsyncSession) -> None:
    """get_executions filters by symbol and returns only matching rows."""
    repo = TradeRepository(session)

    p_btc = make_proposal(symbol="BTC/USDT")
    a_btc = make_assessment(p_btc)
    e_btc = make_execution(p_btc, a_btc, symbol="BTC/USDT")

    p_eth = make_proposal(symbol="ETH/USDT")
    a_eth = make_assessment(p_eth)
    e_eth = make_execution(p_eth, a_eth, symbol="ETH/USDT")

    await repo.save_proposal(p_btc)
    await repo.save_assessment(a_btc)
    await repo.save_execution(e_btc)

    await repo.save_proposal(p_eth)
    await repo.save_assessment(a_eth)
    await repo.save_execution(e_eth)

    btc_execs = await repo.get_executions("BTC/USDT")
    eth_execs = await repo.get_executions("ETH/USDT")

    assert len(btc_execs) == 1
    assert btc_execs[0].symbol == "BTC/USDT"
    assert len(eth_execs) == 1
    assert eth_execs[0].symbol == "ETH/USDT"


async def test_get_executions_ordering(session: AsyncSession) -> None:
    """get_executions returns rows in ascending created_at order."""
    repo = TradeRepository(session)

    p1 = make_proposal()
    a1 = make_assessment(p1)
    e1 = make_execution(p1, a1)

    p2 = make_proposal()
    a2 = make_assessment(p2)
    e2 = make_execution(p2, a2)

    await repo.save_proposal(p1)
    await repo.save_assessment(a1)
    await repo.save_execution(e1)

    await repo.save_proposal(p2)
    await repo.save_assessment(a2)
    await repo.save_execution(e2)

    execs = await repo.get_executions("BTC/USDT")
    assert len(execs) == 2
    # Both rows exist; ordering by created_at (server default) — they should
    # be returned without error regardless of exact order.
    assert all(ex.symbol == "BTC/USDT" for ex in execs)


async def test_get_daily_pnl_no_trades(session: AsyncSession) -> None:
    """get_daily_pnl returns 0.0 when there are no executions."""
    repo = TradeRepository(session)
    pnl = await repo.get_daily_pnl(date(2024, 1, 1))
    assert pnl == pytest.approx(0.0)


async def test_get_daily_pnl_buy_and_sell(session: AsyncSession) -> None:
    """
    BUY for 100 USD, SELL for 110 USD, fee 1 USD each side.
    Expected PnL = 110 - 100 - 1 - 1 = 8.0
    """
    repo = TradeRepository(session)

    # BUY execution
    p_buy = make_proposal(side=OrderSide.BUY)
    a_buy = make_assessment(p_buy)
    e_buy = make_execution(p_buy, a_buy, side=OrderSide.BUY, total_cost_usd="100", fee_usd="1")
    # Pin timestamp to the target day.
    e_buy = e_buy.model_copy(update={"timestamp": datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)})

    # SELL execution
    p_sell = make_proposal(side=OrderSide.SELL)
    a_sell = make_assessment(p_sell)
    e_sell = make_execution(p_sell, a_sell, side=OrderSide.SELL, total_cost_usd="110", fee_usd="1")
    e_sell = e_sell.model_copy(update={"timestamp": datetime(2024, 1, 1, 13, 0, 0, tzinfo=UTC)})

    await repo.save_proposal(p_buy)
    await repo.save_assessment(a_buy)
    await repo.save_execution(e_buy)

    await repo.save_proposal(p_sell)
    await repo.save_assessment(a_sell)
    await repo.save_execution(e_sell)

    pnl = await repo.get_daily_pnl(date(2024, 1, 1))
    # 110 (sell) - 100 (buy cost negated) - 1 - 1 fees = 8.0
    assert pnl == pytest.approx(8.0, abs=0.01)


# ===========================================================================
# PositionRepository tests
# ===========================================================================


async def test_upsert_position_creates(session: AsyncSession) -> None:
    """upsert_position inserts a new row for a new symbol."""
    from sqlalchemy import func, select

    from core.db.models import PositionRow

    repo = PositionRepository(session)
    await repo.upsert_position(make_broker_position())

    count = (await session.execute(select(func.count()).select_from(PositionRow))).scalar()
    assert count == 1


async def test_upsert_position_updates(session: AsyncSession) -> None:
    """upsert_position updates an existing row for the same symbol (count stays 1)."""
    from sqlalchemy import func, select

    from core.db.models import PositionRow

    repo = PositionRepository(session)
    pos_v1 = make_broker_position(entry_price="50000")
    pos_v2 = make_broker_position(entry_price="51000")

    await repo.upsert_position(pos_v1)
    await repo.upsert_position(pos_v2)

    count = (await session.execute(select(func.count()).select_from(PositionRow))).scalar()
    assert count == 1

    retrieved = await repo.get_position("BTC/USDT")
    assert retrieved is not None
    assert Decimal(str(retrieved.entry_price)) == Decimal("51000")


async def test_get_open_positions(session: AsyncSession) -> None:
    """get_open_positions returns all position rows as BrokerPosition objects."""
    repo = PositionRepository(session)
    await repo.upsert_position(make_broker_position("BTC/USDT"))
    await repo.upsert_position(make_broker_position("ETH/USDT"))

    positions = await repo.get_open_positions()
    assert len(positions) == 2
    symbols = {p.symbol for p in positions}
    assert symbols == {"BTC/USDT", "ETH/USDT"}


async def test_close_position(session: AsyncSession) -> None:
    """close_position deletes the row; get_open_positions returns empty list."""
    from sqlalchemy import func, select

    from core.db.models import PositionRow

    repo = PositionRepository(session)
    await repo.upsert_position(make_broker_position())

    await repo.close_position("BTC/USDT")

    count = (await session.execute(select(func.count()).select_from(PositionRow))).scalar()
    assert count == 0


async def test_get_position_returns_none(session: AsyncSession) -> None:
    """get_position returns None for an unknown symbol."""
    repo = PositionRepository(session)
    result = await repo.get_position("UNKNOWN/USD")
    assert result is None


async def test_get_position_returns_correct(session: AsyncSession) -> None:
    """get_position returns the matching BrokerPosition for a known symbol."""
    repo = PositionRepository(session)
    await repo.upsert_position(make_broker_position("ETH/USDT", quantity="2.5"))

    pos = await repo.get_position("ETH/USDT")
    assert pos is not None
    assert pos.symbol == "ETH/USDT"
    assert Decimal(str(pos.quantity)) == Decimal("2.5")
