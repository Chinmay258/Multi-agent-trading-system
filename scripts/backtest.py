"""
scripts/backtest.py
--------------------
Offline backtesting harness.

Loads historical OHLCV from the database, replays candles through the
TA → Decision → Risk pipeline, simulates fills at the candle close price,
and prints a performance summary.

No Redis, no live message bus — fully self-contained.

Usage:
    python scripts/backtest.py \\
        --symbol BTC/USDT \\
        --timeframe 1h \\
        --since 2024-01-01 \\
        --until 2024-03-01 \\
        --initial-balance 10000
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

# Ensure project root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.decision.proposal_builder import ProposalBuilder
from agents.decision.signal_aggregator import SignalAggregator
from agents.risk.drawdown_monitor import DrawdownMonitor
from agents.risk.position_sizer import PositionSizer
from agents.technical_analysis.candle_buffer import CandleBufferRegistry
from agents.technical_analysis.signal_generator import SignalGenerator
from core.config import get_settings
from core.db.connection import get_session
from core.db.repositories.candle_repo import CandleRepository

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class TradeRecord:
    """A single simulated round-trip trade."""

    symbol: str
    side: str
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    pnl: Decimal
    entry_time: datetime
    exit_time: datetime


@dataclass
class BacktestResult:
    """Aggregate performance metrics for a completed backtest run."""

    symbol: str
    timeframe: str
    since: datetime
    until: datetime
    initial_balance: float
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    trades: list[TradeRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sharpe ratio helper
# ---------------------------------------------------------------------------


def _compute_sharpe(trades: list[TradeRecord], initial_balance: float) -> float:
    """
    Approximate annualised Sharpe ratio from trade-level PnL.

    Groups by calendar date, computes daily return (pnl / initial_balance),
    then returns mean / std * sqrt(252).  Returns 0.0 if there is insufficient
    data (fewer than 2 distinct days).
    """
    if len(trades) < 2:
        return 0.0

    daily: dict[str, float] = {}
    for t in trades:
        day_key = t.exit_time.strftime("%Y-%m-%d")
        daily[day_key] = daily.get(day_key, 0.0) + float(t.pnl)

    returns = [v / initial_balance for v in daily.values()]
    if len(returns) < 2:
        return 0.0

    mean_r = sum(returns) / len(returns)
    variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    std_r = math.sqrt(variance)
    if std_r == 0:
        return 0.0
    return (mean_r / std_r) * math.sqrt(252)


# ---------------------------------------------------------------------------
# Core replay loop
# ---------------------------------------------------------------------------


async def run_backtest(
    symbol: str,
    timeframe: str,
    since: datetime,
    until: datetime,
    initial_balance: float,
) -> BacktestResult:
    """
    Replay historical candles through TA → Decision → Risk and simulate fills.

    Fill price: candle close (no look-ahead — signal is generated on the same
    candle, fill on close is optimistic but standard for close-bar strategies).
    """
    settings = get_settings()

    # -- Pipeline components (no Redis dependency) --
    buffer_registry = CandleBufferRegistry()
    signal_gen = SignalGenerator()
    aggregator = SignalAggregator(
        min_confidence=settings.technical_analysis.min_signal_confidence,
    )
    builder = ProposalBuilder(
        max_position_pct=settings.risk.max_position_pct,
        max_order_size_usd=settings.risk.max_order_size_usd,
    )
    drawdown_monitor = DrawdownMonitor(
        initial_balance_usd=initial_balance,
        max_daily_loss_pct=settings.risk.max_daily_loss_pct,
        max_total_drawdown_pct=settings.risk.max_total_drawdown_pct,
    )
    sizer = PositionSizer(
        max_position_pct=settings.risk.max_position_pct,
        min_order_usd=settings.risk.min_order_size_usd,
        max_order_usd=settings.risk.max_order_size_usd,
    )

    # -- Load candles from DB --
    print(f"Loading candles for {symbol} {timeframe} {since.date()} → {until.date()} …")
    async with get_session() as session:
        repo = CandleRepository(session)
        candles = await repo.get_candles(symbol, timeframe, since, until, limit=100_000)

    if not candles:
        print("No candles found in database. Run seed_historical.py first.")
        return BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            since=since,
            until=until,
            initial_balance=initial_balance,
            final_balance=initial_balance,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            total_pnl=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
        )

    print(f"Loaded {len(candles)} candles. Replaying …")

    # Track open positions: symbol → (entry_price, quantity, side, entry_time)
    open_positions: dict[str, tuple[Decimal, Decimal, str, datetime]] = {}
    trades: list[TradeRecord] = []
    peak_portfolio = Decimal(str(initial_balance))
    max_drawdown = 0.0

    for candle in candles:
        buf = buffer_registry.add(candle)
        if not buf.is_warm:
            continue

        tech_signal = signal_gen.generate(buf)
        if tech_signal is None:
            continue

        agg_signal = aggregator.aggregate(candle.symbol, tech_signal, sentiment=None)
        if agg_signal is None:
            continue

        # Check hard risk limits before every trade.
        if (
            drawdown_monitor.daily_loss_limit_breached()
            or drawdown_monitor.total_drawdown_limit_breached()
        ):
            continue

        portfolio_value = Decimal(str(drawdown_monitor.state.portfolio_value_usd))
        proposal = builder.build(agg_signal, portfolio_value)
        if proposal is None:
            continue

        sym = candle.symbol
        new_side = str(proposal.side)  # "buy" or "sell"

        # Close any existing opposite position.
        if sym in open_positions:
            entry_price, qty, pos_side, entry_time = open_positions[sym]
            if pos_side != new_side:
                if pos_side == "buy":
                    pnl = (candle.close - entry_price) * qty
                else:
                    pnl = (entry_price - candle.close) * qty
                drawdown_monitor.close_position(sym, pnl)
                trades.append(
                    TradeRecord(
                        symbol=sym,
                        side=pos_side,
                        entry_price=entry_price,
                        exit_price=candle.close,
                        quantity=qty,
                        pnl=pnl,
                        entry_time=entry_time,
                        exit_time=candle.timestamp,
                    )
                )
                del open_positions[sym]

                # Update drawdown tracking.
                current_portfolio = Decimal(str(drawdown_monitor.state.portfolio_value_usd))
                if current_portfolio > peak_portfolio:
                    peak_portfolio = current_portfolio
                elif peak_portfolio > 0:
                    dd = float((peak_portfolio - current_portfolio) / peak_portfolio)
                    max_drawdown = max(max_drawdown, dd)

        # Don't stack same-direction positions.
        if sym in open_positions:
            continue

        # Open new position.
        stop_pct = sizer.get_stop_loss_pct(proposal)
        size_usd = sizer.calculate(portfolio_value, stop_pct)
        if size_usd <= 0:
            continue

        qty = size_usd / candle.close
        drawdown_monitor.open_position(sym, size_usd)
        open_positions[sym] = (candle.close, qty, new_side, candle.timestamp)

    # Close any remaining positions at the last candle's close.
    if candles:
        last_candle = candles[-1]
        for sym, (entry_price, qty, pos_side, entry_time) in list(open_positions.items()):
            if pos_side == "buy":
                pnl = (last_candle.close - entry_price) * qty
            else:
                pnl = (entry_price - last_candle.close) * qty
            drawdown_monitor.close_position(sym, pnl)
            trades.append(
                TradeRecord(
                    symbol=sym,
                    side=pos_side,
                    entry_price=entry_price,
                    exit_price=last_candle.close,
                    quantity=qty,
                    pnl=pnl,
                    entry_time=entry_time,
                    exit_time=last_candle.timestamp,
                )
            )

    final_balance = float(drawdown_monitor.state.portfolio_value_usd)
    total_pnl = final_balance - initial_balance
    winning = [t for t in trades if t.pnl > 0]
    losing = [t for t in trades if t.pnl <= 0]
    sharpe = _compute_sharpe(trades, initial_balance)

    return BacktestResult(
        symbol=symbol,
        timeframe=timeframe,
        since=since,
        until=until,
        initial_balance=initial_balance,
        final_balance=final_balance,
        total_trades=len(trades),
        winning_trades=len(winning),
        losing_trades=len(losing),
        total_pnl=total_pnl,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe,
        trades=trades,
    )


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_summary(result: BacktestResult) -> None:
    """Print a human-readable performance summary."""
    win_rate = result.winning_trades / result.total_trades * 100 if result.total_trades > 0 else 0.0
    pnl_sign = "+" if result.total_pnl >= 0 else ""
    print(
        f"\n{'=' * 60}\n"
        f"Backtest: {result.symbol} {result.timeframe}  "
        f"{result.since.date()} → {result.until.date()}\n"
        f"{'=' * 60}\n"
        f"  Initial balance : ${result.initial_balance:,.2f}\n"
        f"  Final balance   : ${result.final_balance:,.2f}\n"
        f"  Total PnL       : {pnl_sign}${result.total_pnl:,.2f}\n"
        f"  Total trades    : {result.total_trades}\n"
        f"  Win rate        : {win_rate:.1f}%  "
        f"({result.winning_trades}W / {result.losing_trades}L)\n"
        f"  Max drawdown    : -{result.max_drawdown * 100:.1f}%\n"
        f"  Sharpe ratio    : {result.sharpe_ratio:.2f}\n"
        f"{'=' * 60}\n"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline backtesting harness — loads candles from DB, no Redis."
    )
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair (default: BTC/USDT)")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe (default: 1h)")
    parser.add_argument(
        "--since",
        default="2024-01-01",
        help="Start date inclusive, YYYY-MM-DD (default: 2024-01-01)",
    )
    parser.add_argument(
        "--until",
        default="2024-04-01",
        help="End date exclusive, YYYY-MM-DD (default: 2024-04-01)",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=10_000.0,
        help="Starting portfolio value in USD (default: 10000)",
    )
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=UTC)
    until = datetime.strptime(args.until, "%Y-%m-%d").replace(tzinfo=UTC)

    result = await run_backtest(
        symbol=args.symbol,
        timeframe=args.timeframe,
        since=since,
        until=until,
        initial_balance=args.initial_balance,
    )
    print_summary(result)


if __name__ == "__main__":
    asyncio.run(_main())
