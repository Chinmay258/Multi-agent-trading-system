"""
backtest/metrics.py
-------------------
Performance metrics computed from an equity curve + trade list. Everything here is
pure and unit-tested. Annualisation uses the timeframe's bars-per-year so Sharpe /
Sortino / CAGR are comparable across timeframes.

No metric is "smoothed" or cherry-picked. Drawdown is peak-to-trough on the realised
equity curve; Sharpe/Sortino use per-bar returns; profit factor / expectancy come
straight from realised trade PnL.
"""

from __future__ import annotations

import math
from collections import defaultdict

from backtest.types import EquityPoint, Trade


def _per_bar_returns(equity: list[float]) -> list[float]:
    out: list[float] = []
    for i in range(1, len(equity)):
        prev = equity[i - 1]
        out.append((equity[i] - prev) / prev if prev > 0 else 0.0)
    return out


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float], ddof: int = 1) -> float:
    n = len(xs)
    if n <= ddof:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - ddof))


def max_drawdown(equity: list[float]) -> tuple[float, int]:
    """
    Return (max_drawdown_fraction, longest_drawdown_length_in_bars).

    max_drawdown_fraction is a positive number (e.g. 0.25 == a 25% peak-to-trough drop).
    """
    if not equity:
        return 0.0, 0
    peak = equity[0]
    max_dd = 0.0
    cur_len = 0
    max_len = 0
    for v in equity:
        if v >= peak:
            peak = v
            cur_len = 0
        else:
            cur_len += 1
            max_len = max(max_len, cur_len)
            dd = (peak - v) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
    return max_dd, max_len


def monthly_returns(curve: list[EquityPoint]) -> dict[str, float]:
    """Calendar-month returns keyed 'YYYY-MM', from first/last equity in each month."""
    by_month: dict[str, list[EquityPoint]] = defaultdict(list)
    for p in curve:
        by_month[p.timestamp.strftime("%Y-%m")].append(p)
    out: dict[str, float] = {}
    for month, pts in sorted(by_month.items()):
        start, end = pts[0].equity, pts[-1].equity
        out[month] = (end - start) / start if start > 0 else 0.0
    return out


def compute_metrics(
    curve: list[EquityPoint],
    trades: list[Trade],
    periods_per_year: float,
    initial_balance: float,
) -> dict:
    """Compute the full metric set from an equity curve + trades."""
    equity = [p.equity for p in curve]
    final = equity[-1] if equity else initial_balance
    total_return = (final - initial_balance) / initial_balance if initial_balance > 0 else 0.0

    # CAGR from elapsed wall-clock time.
    if len(curve) >= 2:
        years = (curve[-1].timestamp - curve[0].timestamp).total_seconds() / (365.25 * 86400)
    else:
        years = 0.0
    cagr = ((final / initial_balance) ** (1 / years) - 1) if years > 0 and final > 0 else 0.0

    rets = _per_bar_returns(equity)
    mean_r = _mean(rets)
    std_r = _std(rets)
    downside = [r for r in rets if r < 0]
    dstd = _std(downside) if len(downside) > 1 else 0.0
    ann = math.sqrt(periods_per_year)
    sharpe = (mean_r / std_r * ann) if std_r > 0 else 0.0
    sortino = (mean_r / dstd * ann) if dstd > 0 else 0.0

    mdd, mdd_len = max_drawdown(equity)
    calmar = (cagr / mdd) if mdd > 0 else 0.0

    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    gross_win = sum(t.pnl_usd for t in wins)
    gross_loss = -sum(t.pnl_usd for t in losses)  # positive magnitude
    win_rate = len(wins) / len(trades) if trades else 0.0
    profit_factor = (
        (gross_win / gross_loss) if gross_loss > 0 else (math.inf if gross_win > 0 else 0.0)
    )
    avg_win = _mean([t.pnl_usd for t in wins])
    avg_loss = _mean([t.pnl_usd for t in losses])
    expectancy = _mean([t.pnl_usd for t in trades])

    exposure = (sum(1 for p in curve if p.in_market) / len(curve)) if curve else 0.0

    return {
        "initial_balance": round(initial_balance, 2),
        "final_balance": round(final, 2),
        "total_return_pct": round(total_return * 100, 4),
        "cagr_pct": round(cagr * 100, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "calmar": round(calmar, 4),
        "max_drawdown_pct": round(mdd * 100, 4),
        "max_drawdown_duration_bars": mdd_len,
        "num_trades": len(trades),
        "win_rate_pct": round(win_rate * 100, 4),
        "profit_factor": round(profit_factor, 4) if math.isfinite(profit_factor) else None,
        "expectancy_usd": round(expectancy, 4),
        "avg_win_usd": round(avg_win, 4),
        "avg_loss_usd": round(avg_loss, 4),
        "exposure_pct": round(exposure * 100, 4),
        "monthly_returns": {k: round(v * 100, 4) for k, v in monthly_returns(curve).items()},
    }
