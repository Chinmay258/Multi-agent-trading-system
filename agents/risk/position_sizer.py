"""
agents/risk/position_sizer.py
------------------------------
Fixed-fraction position sizing for the Risk agent.

Formula:
    raw_size = (risk_per_trade_pct × portfolio_value) / stop_loss_pct
    final_size = clamp(raw_size, min_order_usd, min(max_position_usd, max_order_usd))

The fixed-fraction approach risks a constant fraction of the portfolio on
every trade regardless of signal strength. The stop loss percentage controls
position size: a tighter stop (lower pct) allows a larger position for the
same dollar risk; a wider stop forces a smaller position.

ATR-based stop extraction:
    The TA agent embeds ATR data in TechnicalSignal.metadata. If present,
    we use atr_pct_of_price × ATR_STOP_MULTIPLIER as the stop distance.
    This adapts stops to current market volatility — wider during high-vol
    regimes, tighter during low-vol.
"""

from __future__ import annotations

from decimal import Decimal

from core.models.trade import TradeProposal

# 1.5 × ATR gives the stop-loss distance when ATR metadata is available.
# This provides a volatility-adaptive stop wider than the average daily range.
ATR_STOP_MULTIPLIER = 1.5

# Absolute floor for any computed stop (avoids division by near-zero)
MIN_STOP_LOSS_PCT = 0.005  # 0.5%


class PositionSizer:
    """
    Computes approved position sizes using fixed-fraction risk management.

    risk_per_trade_pct represents the fraction of portfolio the trader is
    willing to lose on a single trade (if stopped out). The resulting
    position size equals that dollar risk divided by the stop distance.

    Example with defaults:
        portfolio = $10,000, risk_per_trade = 1%, stop = 2%
        raw_size = ($100) / 0.02 = $5,000 → capped at max_position = $200
    """

    def __init__(
        self,
        max_position_pct: float,
        min_order_usd: float,
        max_order_usd: float,
        risk_per_trade_pct: float = 0.01,
    ) -> None:
        self.max_position_pct = max_position_pct
        self.min_order_usd = min_order_usd
        self.max_order_usd = max_order_usd
        self.risk_per_trade_pct = risk_per_trade_pct

    def calculate(
        self,
        portfolio_value_usd: Decimal,
        stop_loss_pct: float,
    ) -> Decimal:
        """
        Compute the approved position size for a given portfolio and stop.

        Args:
            portfolio_value_usd: Current portfolio value in USD
            stop_loss_pct: Stop-loss distance as a fraction (e.g. 0.02 = 2%)

        Returns:
            Approved size in USD, clamped to configured limits.
        """
        # Guard against pathological stops
        effective_stop = max(stop_loss_pct, MIN_STOP_LOSS_PCT)

        risk_amount = Decimal(str(self.risk_per_trade_pct)) * portfolio_value_usd
        raw_size = risk_amount / Decimal(str(effective_stop))

        max_from_pct = Decimal(str(self.max_position_pct)) * portfolio_value_usd
        abs_max = Decimal(str(self.max_order_usd))
        abs_min = Decimal(str(self.min_order_usd))

        # Apply upper caps first, then lower floor
        capped = min(raw_size, max_from_pct, abs_max)
        return max(capped, abs_min)

    def get_stop_loss_pct(self, proposal: TradeProposal) -> float:
        """
        Determine the effective stop-loss percentage for a proposal.

        Priority:
        1. ATR-based stop from technical signal metadata (volatility-adaptive)
        2. suggested_stop_loss_pct from the proposal itself
        3. Hard-coded 2% fallback

        Args:
            proposal: The incoming TradeProposal to extract stop data from

        Returns:
            Stop loss as a fraction (e.g. 0.02 = 2% below entry).
        """
        # 1. ATR-based stop
        atr_pct = self._extract_atr_pct(proposal)
        if atr_pct is not None and atr_pct > 0:
            return atr_pct * ATR_STOP_MULTIPLIER

        # 2. Proposal's suggested stop
        suggested = proposal.suggested_stop_loss_pct
        if suggested is not None and suggested > 0:
            return float(suggested)

        # 3. Absolute default
        return 0.02

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_atr_pct(self, proposal: TradeProposal) -> float | None:
        """
        Extract ATR percentage from the TechnicalSignal metadata, if present.

        The TA agent stores ATR as 'atr_pct_of_price' in the signal metadata
        dict (see agents/technical_analysis/signal_generator.py).
        """
        tech_signal = proposal.signal.technical_signal
        if tech_signal is None:
            return None
        atr_raw = tech_signal.metadata.get("atr_pct_of_price")
        if atr_raw is None:
            return None
        try:
            return float(atr_raw)
        except (ValueError, TypeError):
            return None
