"""
agents/risk/drawdown_monitor.py
---------------------------------
Portfolio state tracking and drawdown monitoring for the Risk agent.

Tracks:
    - Paper portfolio cash (in-memory)
    - Open positions (symbol → cost basis in USD) and which side each one is on
    - Daily realised PnL (resets each calendar day UTC)
    - Total drawdown from initial balance

Design decisions:
- State is in-memory. While the agent runs it is kept in sync from execution results;
  after a restart it starts from the initial balance and adopts positions as fills arrive.
- Daily reset is checked lazily on each limit evaluation (no background task). The first
  evaluation after midnight resets the daily counter.
- portfolio_value_usd is cash plus the cost basis of open positions (not mark-to-market).
- Reservation model: open_position() is called when a proposal is APPROVED and reserves
  the approved size. When the fill arrives, adjust_position_cost() replaces that reservation
  with the actual filled cost, so each position is counted exactly once. A failed order
  releases its reservation with close_position(symbol, pnl_usd=0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal


@dataclass
class PortfolioState:
    """
    Live in-memory snapshot of the paper portfolio.

    Attributes:
        initial_balance_usd: Starting balance (never changes — used for total drawdown).
        current_balance_usd: Cash balance after reservations and realised PnL.
        daily_start_balance_usd: Portfolio value at start of current UTC day.
        open_positions: symbol → cost basis in USD (not mark-to-market).
        position_sides: symbol → "buy" | "sell" for each open position.
        daily_realized_pnl: Running sum of closed-trade PnL for today.
        last_reset_date: UTC date of last daily reset.
    """

    initial_balance_usd: Decimal
    current_balance_usd: Decimal
    daily_start_balance_usd: Decimal
    open_positions: dict[str, Decimal] = field(default_factory=dict)
    position_sides: dict[str, str] = field(default_factory=dict)
    daily_realized_pnl: Decimal = field(default=Decimal("0"))
    last_reset_date: date = field(default_factory=lambda: datetime.now(UTC).date())

    @property
    def portfolio_value_usd(self) -> Decimal:
        """Cash + sum of open position cost basis."""
        position_value = sum(self.open_positions.values(), Decimal("0"))
        return self.current_balance_usd + position_value

    @property
    def open_positions_count(self) -> int:
        return len(self.open_positions)

    @property
    def daily_loss_pct(self) -> float:
        """
        Fraction of daily-start portfolio lost today (positive = loss).
        Returns 0.0 if daily_start_balance_usd is zero.
        """
        start = self.daily_start_balance_usd
        if start == Decimal("0"):
            return 0.0
        current = self.portfolio_value_usd
        return float((start - current) / start)

    @property
    def total_drawdown_pct(self) -> float:
        """
        Fraction of initial balance lost (positive = drawdown).
        Returns 0.0 if initial_balance_usd is zero.
        """
        initial = self.initial_balance_usd
        if initial == Decimal("0"):
            return 0.0
        current = self.portfolio_value_usd
        return float((initial - current) / initial)


class DrawdownMonitor:
    """
    Evaluates daily loss and total drawdown limits against live portfolio state.

    Usage:
        monitor = DrawdownMonitor(initial_balance_usd=10_000.0, ...)
        if monitor.daily_loss_limit_breached():
            halt_trading()
        monitor.open_position("BTC/USDT", Decimal("200"), side="buy")   # on approval
        monitor.adjust_position_cost("BTC/USDT", Decimal("199.90"))     # on fill
        monitor.close_position("BTC/USDT", pnl_usd=Decimal("15"))       # on closing fill
    """

    def __init__(
        self,
        initial_balance_usd: float,
        max_daily_loss_pct: float,
        max_total_drawdown_pct: float,
    ) -> None:
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_total_drawdown_pct = max_total_drawdown_pct

        initial = Decimal(str(initial_balance_usd))
        self.state = PortfolioState(
            initial_balance_usd=initial,
            current_balance_usd=initial,
            daily_start_balance_usd=initial,
        )

    # ------------------------------------------------------------------
    # Limit checks
    # ------------------------------------------------------------------

    def daily_loss_limit_breached(self) -> bool:
        """True if today's loss fraction meets or exceeds the configured limit."""
        self._maybe_reset_daily()
        return self.state.daily_loss_pct >= self.max_daily_loss_pct

    def total_drawdown_limit_breached(self) -> bool:
        """True if total drawdown fraction meets or exceeds the configured limit."""
        return self.state.total_drawdown_pct >= self.max_total_drawdown_pct

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def position_side(self, symbol: str) -> str | None:
        """Return "buy" / "sell" for an open position, or None if the symbol is flat."""
        if symbol not in self.state.open_positions:
            return None
        return self.state.position_sides.get(symbol, "buy")

    def open_position(self, symbol: str, size_usd: Decimal, side: str | None = None) -> None:
        """
        Register (reserve) a position.

        Deducts size from cash and records it in open_positions. Called when the Risk
        agent approves a proposal, and to adopt a fill the ledger never reserved.
        """
        self.state.current_balance_usd -= size_usd
        self.state.open_positions[symbol] = size_usd
        if side is not None:
            self.state.position_sides[symbol] = side

    def adjust_position_cost(self, symbol: str, actual_cost_usd: Decimal) -> None:
        """
        Replace a reserved cost with the actual filled cost.

        The fill can differ from the approved size (partial fills, fees). Cash absorbs the
        difference so the position is counted once. No-op if the symbol has no position.
        """
        reserved = self.state.open_positions.get(symbol)
        if reserved is None:
            return
        self.state.current_balance_usd += reserved - actual_cost_usd
        self.state.open_positions[symbol] = actual_cost_usd

    def close_position(self, symbol: str, pnl_usd: Decimal) -> None:
        """
        Record a closed position and update balances.

        Args:
            symbol: Trading pair being closed.
            pnl_usd: Realised profit (positive) or loss (negative), net of fees.
        """
        entry_cost = self.state.open_positions.pop(symbol, Decimal("0"))
        self.state.position_sides.pop(symbol, None)
        # Return entry cost + realised PnL to cash balance
        self.state.current_balance_usd += entry_cost + pnl_usd
        self.state.daily_realized_pnl += pnl_usd

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_reset_daily(self) -> None:
        """Reset daily tracking if we've crossed into a new UTC calendar day."""
        today = datetime.now(UTC).date()
        if self.state.last_reset_date != today:
            self.state.daily_start_balance_usd = self.state.portfolio_value_usd
            self.state.daily_realized_pnl = Decimal("0")
            self.state.last_reset_date = today
