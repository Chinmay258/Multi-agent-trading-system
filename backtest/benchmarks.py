"""
backtest/benchmarks.py
----------------------
Two honest benchmarks every strategy result is compared against:

1. **Buy & hold** — buy the asset on the first bar and hold to the end (one round-trip
   of fees + slippage). The bar a directional strategy must beat to justify its
   complexity and trading costs.
2. **Random entry** — the same execution model (next-open fills, SL/TP, fees) driven by
   random entries at a frequency matched to the strategy, averaged over many seeds. If
   the strategy can't beat coin-flips, it has no edge.
"""

from __future__ import annotations

import random
from typing import Any

from backtest.config import BacktestConfig
from backtest.engine import BacktestRun, run_with_decider
from backtest.metrics import compute_metrics
from backtest.types import EquityPoint
from core.models.market import OHLCVCandle


def buy_and_hold(
    candles: list[OHLCVCandle], config: BacktestConfig
) -> tuple[list[EquityPoint], dict]:
    """Buy on the first bar's close, hold to the end. One round-trip of costs."""
    if not candles:
        return [], {}
    fee, slip = config.fee_pct, config.slippage_pct
    entry_fill = float(candles[0].close) * (1 + slip)
    # Deploy the full balance, less the entry fee.
    invested = config.initial_balance * (1 - fee)
    qty = invested / entry_fill if entry_fill > 0 else 0.0

    curve: list[EquityPoint] = []
    for c in candles:
        equity = qty * float(c.close)
        curve.append(EquityPoint(timestamp=c.timestamp, equity=equity, in_market=True))
    # Apply the exit cost at the end (sell the whole position once).
    if curve:
        last = candles[-1]
        exit_fill = float(last.close) * (1 - slip)
        final_equity = qty * exit_fill * (1 - fee)
        curve[-1] = EquityPoint(timestamp=last.timestamp, equity=final_equity, in_market=False)

    metrics = compute_metrics(curve, [], config.periods_per_year, config.initial_balance)
    return curve, metrics


def random_entry(
    candles: list[OHLCVCandle],
    config: BacktestConfig,
    target_trades: int,
) -> dict:
    """
    Average metrics over ``config.random_runs`` random-entry simulations whose entry
    frequency is matched to ``target_trades`` (the strategy's trade count). Same fills,
    SL/TP, and costs as the real strategy.
    """
    n = len(candles)
    if n < 2:
        return {}
    # Probability of entering on any flat bar so expected entries ≈ target_trades.
    warm_bars = max(n - 60, 1)
    entry_prob = min(1.0, max(target_trades, 1) / warm_bars)

    final_returns: list[float] = []
    sharpes: list[float] = []
    win_rates: list[float] = []
    sample_run: BacktestRun | None = None

    for run_idx in range(config.random_runs):
        rng = random.Random(config.random_seed + run_idx)

        def decide(buf: Any, i: int, _rng: random.Random = rng) -> str | None:
            if buf.size < 60:
                return None
            if _rng.random() < entry_prob:
                return "buy" if _rng.random() < 0.5 else "sell"
            return None

        run = run_with_decider(candles, config, decide)
        m = compute_metrics(run.curve, run.trades, config.periods_per_year, config.initial_balance)
        final_returns.append(m["total_return_pct"])
        sharpes.append(m["sharpe"])
        win_rates.append(m["win_rate_pct"])
        if sample_run is None:
            sample_run = run

    k = len(final_returns)
    mean = lambda xs: round(sum(xs) / k, 4) if k else 0.0  # noqa: E731
    sorted_ret = sorted(final_returns)
    return {
        "runs": k,
        "entry_prob": round(entry_prob, 5),
        "mean_total_return_pct": mean(final_returns),
        "median_total_return_pct": round(sorted_ret[k // 2], 4) if k else 0.0,
        "best_total_return_pct": round(max(final_returns), 4) if k else 0.0,
        "worst_total_return_pct": round(min(final_returns), 4) if k else 0.0,
        "mean_sharpe": mean(sharpes),
        "mean_win_rate_pct": mean(win_rates),
    }
