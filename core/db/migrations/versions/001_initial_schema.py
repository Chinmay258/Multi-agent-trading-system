"""Initial schema — all tables, indexes, TimescaleDB hypertables.

Revision ID: 001
Revises:
Create Date: 2026-05-21

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect as sa_inspect

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # Extensions (idempotent)
    # ------------------------------------------------------------------
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # ------------------------------------------------------------------
    # ohlcv_candles — may already exist when init.sql pre-created it
    # ------------------------------------------------------------------
    existing_tables = sa_inspect(op.get_bind()).get_table_names()
    if "ohlcv_candles" not in existing_tables:
        op.create_table(
            "ohlcv_candles",
            sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
            sa.Column("symbol", sa.String(20), nullable=False),
            sa.Column("timeframe", sa.String(5), nullable=False),
            sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
            sa.Column("open", sa.Numeric(20, 8), nullable=False),
            sa.Column("high", sa.Numeric(20, 8), nullable=False),
            sa.Column("low", sa.Numeric(20, 8), nullable=False),
            sa.Column("close", sa.Numeric(20, 8), nullable=False),
            sa.Column("volume", sa.Numeric(30, 8), nullable=False),
            sa.Column("quote_volume", sa.Numeric(30, 8), nullable=True),
            sa.Column(
                "received_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("NOW()"),
                nullable=False,
            ),
            sa.PrimaryKeyConstraint("symbol", "timeframe", "timestamp"),
        )
    op.create_index(
        "idx_ohlcv_symbol_timeframe",
        "ohlcv_candles",
        ["symbol", "timeframe", "timestamp"],
        if_not_exists=True,
    )

    # ------------------------------------------------------------------
    # trade_proposals
    # ------------------------------------------------------------------
    op.create_table(
        "trade_proposals",
        sa.Column(
            "proposal_id",
            sa.dialects.postgresql.UUID(as_uuid=False),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("side", sa.String(10), nullable=False),
        sa.Column("order_type", sa.String(20), nullable=False),
        sa.Column("requested_size_usd", sa.Numeric(20, 2), nullable=False),
        sa.Column("suggested_stop_loss_pct", sa.Numeric(10, 6), nullable=True),
        sa.Column("suggested_take_profit_pct", sa.Numeric(10, 6), nullable=True),
        sa.Column("signal_direction", sa.String(20), nullable=False),
        sa.Column("signal_confidence", sa.Numeric(5, 4), nullable=False),
        sa.Column("reasoning", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("proposal_id"),
    )

    # ------------------------------------------------------------------
    # risk_assessments
    # ------------------------------------------------------------------
    op.create_table(
        "risk_assessments",
        sa.Column(
            "assessment_id",
            sa.dialects.postgresql.UUID(as_uuid=False),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("proposal_id", sa.dialects.postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("decision", sa.String(20), nullable=False),
        sa.Column("rejection_reason", sa.String(50), nullable=True),
        sa.Column("rejection_detail", sa.Text(), nullable=True),
        sa.Column("approved_size_usd", sa.Numeric(20, 2), nullable=True),
        sa.Column("approved_stop_loss_pct", sa.Numeric(10, 6), nullable=True),
        sa.Column("portfolio_value_usd", sa.Numeric(20, 2), nullable=True),
        sa.Column("current_daily_loss_pct", sa.Numeric(10, 6), nullable=True),
        sa.Column("open_positions_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["proposal_id"], ["trade_proposals.proposal_id"]),
        sa.PrimaryKeyConstraint("assessment_id"),
    )

    # ------------------------------------------------------------------
    # executions
    # ------------------------------------------------------------------
    op.create_table(
        "executions",
        sa.Column(
            "result_id",
            sa.dialects.postgresql.UUID(as_uuid=False),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("proposal_id", sa.dialects.postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("assessment_id", sa.dialects.postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("exchange_order_id", sa.String(100), nullable=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("side", sa.String(10), nullable=False),
        sa.Column("order_type", sa.String(20), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("requested_quantity", sa.Numeric(30, 8), nullable=False),
        sa.Column(
            "filled_quantity",
            sa.Numeric(30, 8),
            server_default="0",
            nullable=False,
        ),
        sa.Column("average_fill_price", sa.Numeric(20, 8), nullable=True),
        sa.Column("total_cost_usd", sa.Numeric(20, 2), nullable=True),
        sa.Column("fee_usd", sa.Numeric(20, 8), nullable=True),
        sa.Column("fee_currency", sa.String(10), nullable=True),
        sa.Column("is_paper", sa.Boolean(), server_default="TRUE", nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["assessment_id"], ["risk_assessments.assessment_id"]),
        sa.ForeignKeyConstraint(["proposal_id"], ["trade_proposals.proposal_id"]),
        sa.PrimaryKeyConstraint("result_id"),
    )

    # ------------------------------------------------------------------
    # positions
    # ------------------------------------------------------------------
    op.create_table(
        "positions",
        sa.Column(
            "position_id",
            sa.dialects.postgresql.UUID(as_uuid=False),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("side", sa.String(10), nullable=False),
        sa.Column("quantity", sa.Numeric(30, 8), nullable=False),
        sa.Column("entry_price", sa.Numeric(20, 8), nullable=False),
        sa.Column("current_price", sa.Numeric(20, 8), nullable=True),
        sa.Column("unrealised_pnl_usd", sa.Numeric(20, 2), nullable=True),
        sa.Column("stop_loss_price", sa.Numeric(20, 8), nullable=True),
        sa.Column("take_profit_price", sa.Numeric(20, 8), nullable=True),
        sa.Column("is_paper", sa.Boolean(), server_default="TRUE", nullable=False),
        sa.Column(
            "opened_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("position_id"),
        sa.UniqueConstraint("symbol"),
    )

    # ------------------------------------------------------------------
    # agent_heartbeats
    # ------------------------------------------------------------------
    op.create_table(
        "agent_heartbeats",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("agent_name", sa.String(50), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("messages_processed", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column("errors_since_start", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column("uptime_seconds", sa.Numeric(15, 2), nullable=True),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_heartbeats_agent",
        "agent_heartbeats",
        ["agent_name", "recorded_at"],
    )

    # ------------------------------------------------------------------
    # system_alerts
    # ------------------------------------------------------------------
    op.create_table(
        "system_alerts",
        sa.Column(
            "alert_id",
            sa.dialects.postgresql.UUID(as_uuid=False),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("alert_type", sa.String(50), nullable=False),
        sa.Column("severity", sa.String(20), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("agent_name", sa.String(50), nullable=True),
        sa.Column("resolved", sa.Boolean(), server_default="FALSE", nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("alert_id"),
    )

    # ------------------------------------------------------------------
    # TimescaleDB hypertables (PostgreSQL + TimescaleDB only)
    # ------------------------------------------------------------------
    op.execute(
        "SELECT create_hypertable('ohlcv_candles', 'timestamp', "
        "if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day')"
    )
    # agent_heartbeats PK is (id) only — TimescaleDB requires the partition
    # column to be in the PK, so hypertable conversion may fail on existing
    # schemas. Skip silently; it works fine as a regular table.
    op.execute(
        """
        DO $$
        BEGIN
            PERFORM create_hypertable(
                'agent_heartbeats', 'recorded_at',
                if_not_exists => TRUE,
                chunk_time_interval => INTERVAL '1 day'
            );
        EXCEPTION WHEN others THEN
            NULL;
        END
        $$;
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
            PERFORM add_compression_policy(
                'ohlcv_candles', INTERVAL '7 days',
                if_not_exists => TRUE
            );
        EXCEPTION WHEN others THEN
            NULL;
        END
        $$;
        """
    )


def downgrade() -> None:
    op.drop_index("idx_heartbeats_agent", table_name="agent_heartbeats")
    op.drop_index("idx_ohlcv_symbol_timeframe", table_name="ohlcv_candles")
    op.drop_table("system_alerts")
    op.drop_table("agent_heartbeats")
    op.drop_table("positions")
    op.drop_table("executions")
    op.drop_table("risk_assessments")
    op.drop_table("trade_proposals")
    op.drop_table("ohlcv_candles")
