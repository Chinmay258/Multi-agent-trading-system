"""
agents/decision/proposal_builder.py
-------------------------------------
Converts an AggregatedSignal into a TradeProposal.

Responsible for:
- Translating aggregated signal direction into an order side (BUY/SELL)
- Computing the requested_size_usd from the portfolio fraction
- Suppressing NEUTRAL aggregated signals (no proposal emitted)
- Writing a human-readable reasoning string for the audit trail

Design decisions:
- The Decision agent requests max_position_pct of the portfolio. This is
  an upper bound — the Risk agent will reduce it if needed.
- NEUTRAL signals produce no proposal. The caller should check for None.
- requested_size_usd is capped at the absolute max_order_size_usd config
  value as a sanity guard, even before the Risk agent applies its limits.
"""

from __future__ import annotations

from decimal import Decimal

from core.models.signals import AggregatedSignal, SignalDirection, _to_direction
from core.models.trade import OrderSide, OrderType, TradeProposal


class ProposalBuilder:
    """
    Builds TradeProposals from AggregatedSignals.

    The requested size equals max_position_pct * portfolio_value — the
    Risk agent applies further constraints before approving execution.
    """

    def __init__(
        self,
        max_position_pct: float = 0.02,
        default_stop_loss_pct: float = 0.02,
        default_take_profit_pct: float = 0.04,
        max_order_size_usd: float = 1000.0,
    ) -> None:
        self.max_position_pct = max_position_pct
        self.default_stop_loss_pct = default_stop_loss_pct
        self.default_take_profit_pct = default_take_profit_pct
        self.max_order_size_usd = max_order_size_usd

    def build(
        self,
        signal: AggregatedSignal,
        portfolio_value_usd: Decimal,
    ) -> TradeProposal | None:
        """
        Build a TradeProposal from a validated AggregatedSignal.

        Args:
            signal: Aggregated signal (already filtered by SignalAggregator)
            portfolio_value_usd: Current paper portfolio value for sizing

        Returns:
            TradeProposal, or None if signal direction is NEUTRAL.
        """
        direction = _to_direction(signal.direction)

        if direction == SignalDirection.NEUTRAL:
            return None

        if direction in (SignalDirection.BUY, SignalDirection.STRONG_BUY):
            side = OrderSide.BUY
        elif direction in (SignalDirection.SELL, SignalDirection.STRONG_SELL):
            side = OrderSide.SELL
        else:
            return None

        requested_size_usd = Decimal(str(self.max_position_pct)) * portfolio_value_usd
        # Sanity cap (Risk agent also enforces this, but belt-and-suspenders)
        abs_max = Decimal(str(self.max_order_size_usd))
        if requested_size_usd > abs_max:
            requested_size_usd = abs_max

        reasoning = self._build_reasoning(signal, side, requested_size_usd)

        return TradeProposal(
            symbol=signal.symbol,
            side=side,
            order_type=OrderType.MARKET,
            requested_size_usd=requested_size_usd,
            suggested_stop_loss_pct=self.default_stop_loss_pct,
            suggested_take_profit_pct=self.default_take_profit_pct,
            signal=signal,
            reasoning=reasoning,
        )

    def _build_reasoning(
        self,
        signal: AggregatedSignal,
        side: OrderSide,
        size_usd: Decimal,
    ) -> str:
        """Human-readable explanation stored in proposal for audit trail."""
        side_str = side.value if hasattr(side, "value") else str(side)
        direction_str = _to_direction(signal.direction).value
        sources = []
        if signal.technical_signal is not None:
            sources.append(f"technical({signal.technical_weight:.0%})")
        if signal.sentiment_signal is not None:
            sources.append(f"sentiment({signal.sentiment_weight:.0%})")

        return (
            f"{side_str.upper()} {signal.symbol}: "
            f"direction={direction_str} "
            f"confidence={signal.confidence:.2f} "
            f"score={signal.composite_score:+.3f} "
            f"size=${float(size_usd):.2f} "
            f"sources=[{', '.join(sources)}]"
        )
