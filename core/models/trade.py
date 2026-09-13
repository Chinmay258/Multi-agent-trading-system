"""
core/models/trade.py
--------------------
Trade lifecycle models — from proposal through risk review to execution.

The trade lifecycle has three stages:
1. TradeProposal   — Decision agent's intent ("I want to buy X")
2. RiskAssessment  — Risk agent's verdict ("approved/rejected, here's sizing")
3. ExecutionResult — Execution agent's outcome ("filled at price P, fee F")

Each stage is a separate immutable model. This creates a clean, auditable
record of every decision in the trade lifecycle. The Execution agent only
ever sees RiskAssessment-approved proposals — it has no business logic.

Design decisions:
- TradeProposal.proposal_id is the correlation ID that links all three
  stages together in the database.
- RiskDecision is explicit (APPROVED/REJECTED/MODIFIED) rather than a bool,
  because "modified" (e.g. reduced position size) is meaningfully different
  from "approved as-is".
- ExecutionResult stores both requested and actual fill price/size to capture
  slippage — critical for strategy analysis.
- OrderStatus mirrors exchange order states for clean state machine logic.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from uuid import UUID, uuid4

from pydantic import Field, field_validator

from core.models.market import BaseMarketModel
from core.models.signals import AggregatedSignal

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderStatus(str, Enum):
    """Mirrors standard exchange order states."""

    PENDING = "pending"  # Created, not yet submitted to exchange
    OPEN = "open"  # Submitted, awaiting fill
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"  # Fully executed
    CANCELLED = "cancelled"
    REJECTED = "rejected"  # Rejected by exchange
    EXPIRED = "expired"


class RiskDecision(str, Enum):
    APPROVED = "approved"  # Proposal accepted as-is
    MODIFIED = "modified"  # Accepted with adjusted size/stops
    REJECTED = "rejected"  # Proposal vetoed entirely


class RejectionReason(str, Enum):
    """Why the Risk agent rejected a proposal — for audit and analysis."""

    MAX_POSITION_SIZE = "max_position_size"
    MAX_OPEN_POSITIONS = "max_open_positions"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    TOTAL_DRAWDOWN_LIMIT = "total_drawdown_limit"
    STALE_MARKET_DATA = "stale_market_data"
    LOW_CONFIDENCE = "low_confidence"
    SYMBOL_NOT_WHITELISTED = "symbol_not_whitelisted"
    ORDER_SIZE_TOO_SMALL = "order_size_too_small"
    ORDER_SIZE_TOO_LARGE = "order_size_too_large"
    CIRCUIT_BREAKER_ACTIVE = "circuit_breaker_active"
    INSUFFICIENT_BALANCE = "insufficient_balance"
    DUPLICATE_SIGNAL = "duplicate_signal"


# ---------------------------------------------------------------------------
# Trade proposal (Decision agent → Risk agent)
# ---------------------------------------------------------------------------


class TradeProposal(BaseMarketModel):
    """
    A structured trade proposal from the Decision agent.
    Published on: decision.proposal

    The proposal represents intent — it does NOT mean the trade will execute.
    The Risk agent reviews it and either approves, modifies, or rejects it.
    """

    proposal_id: UUID = Field(
        default_factory=uuid4, description="Correlation ID for full lifecycle"
    )
    symbol: str
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Requested sizes (Risk agent may reduce these)
    requested_size_usd: Decimal = Field(description="Requested position size in USD")
    requested_quantity: Decimal | None = Field(
        default=None, description="Requested quantity in base asset (computed from price if None)"
    )
    limit_price: Decimal | None = Field(default=None, description="Limit price for LIMIT orders")

    # Risk parameters proposed by Decision agent (Risk agent may tighten)
    suggested_stop_loss_pct: float | None = Field(
        default=0.02, description="Suggested stop loss as fraction below entry (2% default)"
    )
    suggested_take_profit_pct: float | None = Field(
        default=0.04, description="Suggested take profit as fraction above entry (4% default)"
    )

    # Signal context (why this trade is proposed)
    signal: AggregatedSignal = Field(
        description="The aggregated signal that triggered this proposal"
    )
    reasoning: str = Field(description="Human-readable explanation of the trade rationale")

    @field_validator("requested_size_usd")
    @classmethod
    def validate_positive_size(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError("Position size must be positive")
        return v

    @property
    def channel_key(self) -> str:
        return "decision.proposal"

    @property
    def is_long(self) -> bool:
        return self.side == OrderSide.BUY

    @property
    def is_short(self) -> bool:
        return self.side == OrderSide.SELL


# ---------------------------------------------------------------------------
# Risk assessment (Risk agent → Execution agent)
# ---------------------------------------------------------------------------


class RiskAssessment(BaseMarketModel):
    """
    The Risk agent's verdict on a TradeProposal.
    Published on: risk.assessment

    If decision == REJECTED, the Execution agent takes no action.
    If decision == MODIFIED, the approved_* fields override the proposal's requested_* fields.
    """

    assessment_id: UUID = Field(default_factory=uuid4)
    proposal_id: UUID = Field(description="Links back to TradeProposal.proposal_id")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    decision: RiskDecision
    rejection_reason: RejectionReason | None = None
    rejection_detail: str | None = Field(
        default=None, description="Human-readable explanation for rejection (logged, not traded)"
    )

    # Approved parameters (may differ from proposal if MODIFIED)
    approved_size_usd: Decimal | None = None
    approved_quantity: Decimal | None = None
    approved_stop_loss_pct: float | None = None
    approved_take_profit_pct: float | None = None

    # Portfolio state at decision time (for audit)
    portfolio_value_usd: Decimal | None = None
    current_daily_loss_pct: float | None = None
    open_positions_count: int = Field(default=0)

    # Original proposal (carried forward for full audit trail)
    original_proposal: TradeProposal

    @property
    def channel_key(self) -> str:
        return "risk.assessment"

    @property
    def is_approved(self) -> bool:
        return self.decision in (RiskDecision.APPROVED, RiskDecision.MODIFIED)

    @property
    def is_rejected(self) -> bool:
        return self.decision == RiskDecision.REJECTED

    @property
    def effective_size_usd(self) -> Decimal | None:
        """The size the Execution agent should use."""
        return self.approved_size_usd or self.original_proposal.requested_size_usd


# ---------------------------------------------------------------------------
# Execution result (Execution agent → all subscribers)
# ---------------------------------------------------------------------------


class ExecutionResult(BaseMarketModel):
    """
    The outcome of an execution attempt.
    Published on: execution.result

    Stored verbatim in the database — this is the primary audit record.
    In paper trading mode, fill_price is simulated; the field is identical.
    """

    result_id: UUID = Field(default_factory=uuid4)
    proposal_id: UUID = Field(description="Links back to TradeProposal.proposal_id")
    assessment_id: UUID = Field(description="Links back to RiskAssessment.assessment_id")
    exchange_order_id: str | None = Field(
        default=None, description="Exchange-assigned order ID (None for paper trades)"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    symbol: str
    side: OrderSide
    order_type: OrderType
    status: OrderStatus

    # Requested vs actual (for slippage tracking)
    requested_quantity: Decimal
    filled_quantity: Decimal = Field(default=Decimal("0"))
    average_fill_price: Decimal | None = None
    total_cost_usd: Decimal | None = None

    # Fees
    fee_usd: Decimal | None = Field(default=None, description="Total fee paid in USD")
    fee_currency: str | None = None

    # Realised PnL, net of entry and exit fees (closing fills only; None when opening)
    realized_pnl_usd: Decimal | None = Field(
        default=None, description="Net realised PnL in USD for a closing fill"
    )

    # Stop loss and take profit order IDs (if placed)
    stop_loss_order_id: str | None = None
    take_profit_order_id: str | None = None

    # Mode
    is_paper: bool = Field(description="True if this was a paper trade")

    # Error info (for failed executions)
    error_message: str | None = None
    retry_count: int = Field(default=0)

    @property
    def channel_key(self) -> str:
        return "execution.result"

    @property
    def slippage_pct(self) -> float | None:
        """
        Percentage slippage between expected and actual fill.
        Negative = filled better than expected (good). Positive = worse (bad).
        Only meaningful for MARKET orders.
        """
        if not self.average_fill_price or self.order_type != OrderType.MARKET:
            return None
        # TODO: requires original limit/expected price from proposal for proper calc
        return None

    @property
    def is_successful(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)
