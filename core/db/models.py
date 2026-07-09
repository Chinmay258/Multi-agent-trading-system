"""
core/db/models.py
-----------------
SQLAlchemy 2.0 ORM models mirroring infrastructure/postgres/init.sql exactly.

Column names, types, and constraints match the SQL schema verbatim — do NOT
rename anything here without also updating the migration and the SQL file.

All models inherit from Base (core.db.connection). Import this module wherever
you need to reference ORM rows; the repository layer is the only place that
should do so outside of tests.

Design note on defaults:
  Server-side defaults (uuid_generate_v4(), NOW()) live in the Alembic
  migration only.  The ORM uses Python-side defaults so that the models work
  identically against PostgreSQL (prod) and SQLite (unit tests).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    FetchedValue,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from core.db.connection import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# ohlcv_candles — TimescaleDB hypertable, composite PK
# ---------------------------------------------------------------------------


class OHLCVCandleRow(Base):
    """
    One OHLCV candle persisted to the database.

    Primary key is composite (symbol, timeframe, timestamp) — NOT the auto-
    generated id column.  The id column is a server-side BIGSERIAL used by
    TimescaleDB internally; SQLAlchemy never writes to it.
    """

    __tablename__ = "ohlcv_candles"
    __table_args__ = (
        Index(
            "idx_ohlcv_symbol_timeframe",
            "symbol",
            "timeframe",
            "timestamp",
        ),
    )

    # Server-generated BIGSERIAL — excluded from INSERT via FetchedValue.
    # SQLite tests will leave this NULL (never queried).
    id: Mapped[int | None] = mapped_column(BigInteger, server_default=FetchedValue())

    # Composite primary key
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True, nullable=False)
    timeframe: Mapped[str] = mapped_column(String(5), primary_key=True, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, nullable=False
    )

    open: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    volume: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False)
    quote_volume: Mapped[Decimal | None] = mapped_column(Numeric(30, 8), nullable=True)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    def __repr__(self) -> str:
        return (
            f"OHLCVCandleRow(symbol={self.symbol!r}, timeframe={self.timeframe!r}, "
            f"timestamp={self.timestamp!r}, close={self.close})"
        )


# ---------------------------------------------------------------------------
# trade_proposals
# ---------------------------------------------------------------------------


class TradeProposalRow(Base):
    """One row in trade_proposals — the Decision agent's intent record."""

    __tablename__ = "trade_proposals"

    proposal_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    order_type: Mapped[str] = mapped_column(String(20), nullable=False)
    requested_size_usd: Mapped[Decimal] = mapped_column(Numeric(20, 2), nullable=False)
    suggested_stop_loss_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    suggested_take_profit_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    signal_direction: Mapped[str] = mapped_column(String(20), nullable=False)
    signal_confidence: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    def __repr__(self) -> str:
        return (
            f"TradeProposalRow(proposal_id={self.proposal_id!r}, symbol={self.symbol!r}, "
            f"side={self.side!r}, requested_size_usd={self.requested_size_usd})"
        )


# ---------------------------------------------------------------------------
# risk_assessments
# ---------------------------------------------------------------------------


class RiskAssessmentRow(Base):
    """Risk agent's verdict on a TradeProposal."""

    __tablename__ = "risk_assessments"

    assessment_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    proposal_id: Mapped[UUID] = mapped_column(
        ForeignKey("trade_proposals.proposal_id"), nullable=False
    )
    decision: Mapped[str] = mapped_column(String(20), nullable=False)
    rejection_reason: Mapped[str | None] = mapped_column(String(50), nullable=True)
    rejection_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    approved_size_usd: Mapped[Decimal | None] = mapped_column(Numeric(20, 2), nullable=True)
    approved_stop_loss_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    portfolio_value_usd: Mapped[Decimal | None] = mapped_column(Numeric(20, 2), nullable=True)
    current_daily_loss_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    open_positions_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    def __repr__(self) -> str:
        return (
            f"RiskAssessmentRow(assessment_id={self.assessment_id!r}, "
            f"proposal_id={self.proposal_id!r}, decision={self.decision!r})"
        )


# ---------------------------------------------------------------------------
# executions
# ---------------------------------------------------------------------------


class ExecutionRow(Base):
    """Execution agent's fill record — primary audit trail for every order."""

    __tablename__ = "executions"

    result_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    proposal_id: Mapped[UUID] = mapped_column(
        ForeignKey("trade_proposals.proposal_id"), nullable=False
    )
    assessment_id: Mapped[UUID] = mapped_column(
        ForeignKey("risk_assessments.assessment_id"), nullable=False
    )
    exchange_order_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    order_type: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    requested_quantity: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False)
    filled_quantity: Mapped[Decimal] = mapped_column(
        Numeric(30, 8), nullable=False, default=Decimal("0")
    )
    average_fill_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    total_cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(20, 2), nullable=True)
    fee_usd: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    fee_currency: Mapped[str | None] = mapped_column(String(10), nullable=True)
    is_paper: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    def __repr__(self) -> str:
        return (
            f"ExecutionRow(result_id={self.result_id!r}, symbol={self.symbol!r}, "
            f"side={self.side!r}, status={self.status!r})"
        )


# ---------------------------------------------------------------------------
# positions
# ---------------------------------------------------------------------------


class PositionRow(Base):
    """Currently open positions — one row per symbol (UNIQUE constraint on symbol)."""

    __tablename__ = "positions"

    position_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, unique=True)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    current_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    unrealised_pnl_usd: Mapped[Decimal | None] = mapped_column(Numeric(20, 2), nullable=True)
    stop_loss_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    take_profit_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    is_paper: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    def __repr__(self) -> str:
        return (
            f"PositionRow(symbol={self.symbol!r}, side={self.side!r}, "
            f"quantity={self.quantity}, entry_price={self.entry_price})"
        )


# ---------------------------------------------------------------------------
# agent_heartbeats — TimescaleDB hypertable
# ---------------------------------------------------------------------------


class AgentHeartbeatRow(Base):
    """Heartbeat records from every agent — used by MonitoringAgent dashboards."""

    __tablename__ = "agent_heartbeats"
    __table_args__ = (Index("idx_heartbeats_agent", "agent_name", "recorded_at"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    agent_name: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    messages_processed: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    errors_since_start: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    uptime_seconds: Mapped[Decimal | None] = mapped_column(Numeric(15, 2), nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    def __repr__(self) -> str:
        return (
            f"AgentHeartbeatRow(agent_name={self.agent_name!r}, "
            f"status={self.status!r}, recorded_at={self.recorded_at!r})"
        )


# ---------------------------------------------------------------------------
# system_alerts
# ---------------------------------------------------------------------------


class SystemAlertRow(Base):
    """System-wide alerts raised by any agent."""

    __tablename__ = "system_alerts"

    alert_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    alert_type: Mapped[str] = mapped_column(String(50), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    agent_name: Mapped[str | None] = mapped_column(String(50), nullable=True)
    resolved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    def __repr__(self) -> str:
        return (
            f"SystemAlertRow(alert_id={self.alert_id!r}, alert_type={self.alert_type!r}, "
            f"severity={self.severity!r}, resolved={self.resolved})"
        )
