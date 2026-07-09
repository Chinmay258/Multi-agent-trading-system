"""
agents/risk/drawdown_monitor.py
---------------------------------
Portfolio state tracking and drawdown monitoring for the Risk agent.

Tracks:
    - Paper portfolio balance (in-memory, reset-proof across daily boundaries)
    - Open positions (symbol → size_usd mapping)
    - Daily realised PnL (resets each calendar day UTC)
    - Total drawdown from initial balance

Design decisions:
- State is in-memory only for Phase 3. Phase 4 will persist to PostgreSQL
  via the Execution agent's fill notifications.
- Daily reset is checked lazily on each limit evaluation (no background task).
  This means the first evaluation after midnight resets the daily counter —
  acceptable since it slightly delays enforcement, not skips it.
- portfolio_value_usd is NOT just cash balance — it includes the notional
  value of open positions (their entry cost, not mark-to-market). Phase 4
  will upgrade this to use real-time mark-to-market from execution fills.
- open_position() is called when a trade is APPROVED, not when filled.
  In paper trading this is fine; in live trading Phase 4 must reconcile
  against actual fills.
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
        current_balance_usd: Cash balance after realised PnL.
        daily_start_balance_usd: Portfolio value at start of current UTC day.
        open_positions: symbol → entry cost in USD (not mark-to-market).
        daily_realized_pnl: Running sum of closed-trade PnL for today.
        last_reset_date: UTC date of last daily reset.
    """

    initial_balance_usd: Decimal
    current_balance_usd: Decimal
    daily_start_balance_usd: Decimal
    open_positions: dict[str, Decimal] = field(default_factory=dict)
    daily_realized_pnl: Decimal = field(default=Decimal("0"))
    last_reset_date: date = field(default_factory=lambda: datetime.now(UTC).date())

    @property
    def portfolio_value_usd(self) -> Decimal:
        """Cash + sum of open position entry costs."""
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
        Fraction of initial balance lost from peak (positive = drawdown).
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
        monitor.open_position("BTC/USDT", Decimal("200"))
        monitor.close_position("BTC/USDT", pnl_usd=Decimal("15"))
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

    def open_position(self, symbol: str, size_usd: Decimal) -> None:
        """
        Register an approved position.

        Deducts size from cash balance and records in open_positions.
        Called when the Risk agent approves a proposal.
        """
        self.state.current_balance_usd -= size_usd
        self.state.open_positions[symbol] = size_usd

    def close_position(self, symbol: str, pnl_usd: Decimal) -> None:
        """
        Record a closed position and update balances.

        Args:
            symbol: Trading pair being closed.
            pnl_usd: Realised profit (positive) or loss (negative).
        """
        entry_cost = self.state.open_positions.pop(symbol, Decimal("0"))
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
