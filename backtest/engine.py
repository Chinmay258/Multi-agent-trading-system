"""
backtest/engine.py
------------------
Event-driven backtest engine. The one rule that matters most: **no lookahead.**

- A signal is generated at bar ``i`` using only candles ``[0..i]`` (it reads close[i]).
- The resulting order is filled at bar ``i+1``'s **open** — never the same bar's close.
- SL/TP brackets are checked intrabar against each subsequent bar's high/low.
- Fees (0.1%/side) and slippage (0.05%/side) are applied on entry and exit, matched to
  the live PaperBroker.

It reuses the *real* ``SignalGenerator`` (rule-based) so the backtest exercises the same
indicator/scoring logic the live agents run. ML-strategy backtesting (which needs
walk-forward retraining to avoid in-sample bias) is Phase 5; ``signal_source="ml"`` is
intentionally not enabled here.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from backtest.config import BacktestConfig
from backtest.types import EquityPoint, Trade
from core.models.market import OHLCVCandle
from core.models.signals import SignalDirection, _to_direction


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class _Position:
    __slots__ = ("side", "entry_fill", "qty", "entry_time", "entry_index", "size_usd")

    def __init__(
        self,
        side: str,
        entry_fill: float,
        qty: float,
        entry_time: datetime,
        entry_index: int,
        size_usd: float,
    ) -> None:
        self.side = side
        self.entry_fill = entry_fill
        self.qty = qty
        self.entry_time = entry_time
        self.entry_index = entry_index
        self.size_usd = size_usd


class BacktestRun:
    """Container for a completed run's outputs."""

    def __init__(self, config: BacktestConfig, curve: list[EquityPoint], trades: list[Trade]):
        self.config = config
        self.curve = curve
        self.trades = trades


def _make_signal_fn(config: BacktestConfig) -> Callable[[Any], str | None]:
    """Return a callable(buffer) -> 'buy'|'sell'|None for the configured source."""
    if config.signal_source == "rules":
        from agents.technical_analysis.signal_generator import SignalGenerator

        gen = SignalGenerator()

        def signal_fn(buf: Any) -> str | None:
            sig = gen.generate(buf)
            if sig is None or sig.confidence < config.min_confidence:
                return None
            d = _to_direction(sig.direction)
            if d in (SignalDirection.BUY, SignalDirection.STRONG_BUY):
                return "buy"
            if d in (SignalDirection.SELL, SignalDirection.STRONG_SELL):
                return "sell"
            return None

        return signal_fn

    raise NotImplementedError(
        "signal_source='ml' backtesting requires walk-forward retraining to avoid "
        "in-sample bias; that is Phase 5. Use signal_source='rules' for the baseline."
    )


def run_with_decider(
    candles: list[OHLCVCandle],
    config: BacktestConfig,
    decide: Callable[[Any, int], str | None],
) -> BacktestRun:
    """
    Core no-lookahead replay loop, parameterised by a ``decide(buffer, i) -> side|None``
    callback. Reused by both the real strategy and the benchmark simulations so they
    share identical fee/slippage/SL-TP/equity accounting.
    """
    from agents.technical_analysis.candle_buffer import CandleBufferRegistry

    registry = CandleBufferRegistry()

    fee = config.fee_pct
    slip = config.slippage_pct

    realized_equity = config.initial_balance
    position: _Position | None = None
    pending: str | None = None  # decision made last bar, to fill at this bar's open

    curve: list[EquityPoint] = []
    trades: list[Trade] = []

    def _close(
        pos: _Position, exit_price_raw: float, exit_time: datetime, exit_index: int, reason: str
    ) -> float:
        """Close a position at a raw price; apply slippage+fees; return net pnl."""
        # Slippage is adverse to the closing side.
        if pos.side == "buy":  # closing a long = sell → fill slightly lower
            exit_fill = exit_price_raw * (1 - slip)
            gross = (exit_fill - pos.entry_fill) * pos.qty
        else:  # closing a short = buy → fill slightly higher
            exit_fill = exit_price_raw * (1 + slip)
            gross = (pos.entry_fill - exit_fill) * pos.qty
        fees = fee * (pos.entry_fill * pos.qty + exit_fill * pos.qty)
        net = gross - fees
        trades.append(
            Trade(
                symbol=config.symbol,
                side=pos.side,
                entry_time=pos.entry_time,
                exit_time=exit_time,
                entry_price=pos.entry_fill,
                exit_price=exit_fill,
                quantity=pos.qty,
                notional_usd=pos.size_usd,
                pnl_usd=net,
                return_pct=(net / pos.size_usd) if pos.size_usd > 0 else 0.0,
                bars_held=exit_index - pos.entry_index,
                exit_reason=reason,
            )
        )
        return net

    def _open(side: str, raw_price: float, ts: datetime, index: int) -> _Position:
        size_usd = _clamp(
            config.max_position_pct * realized_equity, config.min_order_usd, config.max_order_usd
        )
        # Slippage is adverse to the opening side.
        entry_fill = raw_price * (1 + slip) if side == "buy" else raw_price * (1 - slip)
        qty = size_usd / entry_fill if entry_fill > 0 else 0.0
        return _Position(side, entry_fill, qty, ts, index, size_usd)

    n = len(candles)
    for i in range(n):
        c = candles[i]
        buf = registry.add(c)
        o, h, low_, close = float(c.open), float(c.high), float(c.low), float(c.close)

        # (1) Fill any pending order at THIS bar's open (no lookahead).
        if pending is not None:
            if position is None:
                position = _open(pending, o, c.timestamp, i)
            elif position.side != pending:
                realized_equity += _close(position, o, c.timestamp, i, "signal")
                position = _open(pending, o, c.timestamp, i)
            # same-direction signal → no stacking
            pending = None

        # (2) Intrabar SL/TP on the open position (adverse level checked first).
        if position is not None:
            ep = position.entry_fill
            if position.side == "buy":
                sl_price = ep * (1 - config.stop_loss_pct)
                tp_price = ep * (1 + config.take_profit_pct)
                if low_ <= sl_price:
                    realized_equity += _close(position, sl_price, c.timestamp, i, "stop_loss")
                    position = None
                elif h >= tp_price:
                    realized_equity += _close(position, tp_price, c.timestamp, i, "take_profit")
                    position = None
            else:  # short
                sl_price = ep * (1 + config.stop_loss_pct)
                tp_price = ep * (1 - config.take_profit_pct)
                if h >= sl_price:
                    realized_equity += _close(position, sl_price, c.timestamp, i, "stop_loss")
                    position = None
                elif low_ <= tp_price:
                    realized_equity += _close(position, tp_price, c.timestamp, i, "take_profit")
                    position = None

        # (3) Mark-to-market equity at this bar's close.
        if position is not None:
            if position.side == "buy":
                unrealized = (close - position.entry_fill) * position.qty
            else:
                unrealized = (position.entry_fill - close) * position.qty
            equity = realized_equity + unrealized
        else:
            equity = realized_equity
        curve.append(
            EquityPoint(timestamp=c.timestamp, equity=equity, in_market=position is not None)
        )

        # (4) Decide the action for the NEXT bar's open (only past+present data).
        if i < n - 1:
            pending = decide(buf, i)

    # Close any residual position at the final close (end-of-data).
    if position is not None and curve:
        last = candles[-1]
        realized_equity += _close(position, float(last.close), last.timestamp, n - 1, "eod")
        # Reflect realised close in the final equity point.
        curve[-1] = EquityPoint(timestamp=last.timestamp, equity=realized_equity, in_market=False)

    return BacktestRun(config=config, curve=curve, trades=trades)


def run_strategy(candles: list[OHLCVCandle], config: BacktestConfig) -> BacktestRun:
    """Replay candles through the real signal logic with no-lookahead execution."""
    signal_fn = _make_signal_fn(config)
    return run_with_decider(candles, config, lambda buf, i: signal_fn(buf))
