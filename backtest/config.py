"""
backtest/config.py
------------------
Configuration for a backtest run. Defaults are deliberately conservative and
matched to the live system so the backtest reflects reality, not a fantasy.
"""

from __future__ import annotations

from dataclasses import dataclass

# Bars per year for annualising metrics, per timeframe.
PERIODS_PER_YEAR: dict[str, float] = {
    "1m": 525_600.0,
    "5m": 105_120.0,
    "15m": 35_040.0,
    "1h": 8_760.0,
    "4h": 2_190.0,
    "1d": 365.0,
}


@dataclass(frozen=True)
class BacktestConfig:
    """Parameters for one backtest run."""

    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    initial_balance: float = 10_000.0

    # Costs — matched to agents/execution/paper_broker.py exactly.
    fee_pct: float = 0.001  # 0.1% taker fee per side
    slippage_pct: float = 0.0005  # 0.05% slippage per side

    # Exit model. The live system's MT5 path uses these defaults; paper mode has
    # no native SL/TP, so the backtest applies an explicit, documented exit model:
    # SL/TP brackets + exit-on-opposite-signal.
    stop_loss_pct: float = 0.03  # exit if price moves 3% against entry
    take_profit_pct: float = 0.08  # exit if price moves 8% in favour

    # Sizing — matched to RiskSettings defaults.
    max_position_pct: float = 0.02  # 2% of equity per trade
    min_order_usd: float = 10.0
    max_order_usd: float = 1_000.0

    # Signal source. "rules" is the deterministic, lookahead-free baseline.
    # ML strategy backtesting (with walk-forward retraining) is Phase 5.
    signal_source: str = "rules"  # "rules" | "ml"
    min_confidence: float = 0.50

    # Reproducibility for the random-entry benchmark.
    random_seed: int = 42
    random_runs: int = 200

    @property
    def periods_per_year(self) -> float:
        return PERIODS_PER_YEAR.get(self.timeframe, 365.0)

    # Round-trip cost as a fraction (both sides of fee + slippage).
    @property
    def round_trip_cost_pct(self) -> float:
        return 2.0 * (self.fee_pct + self.slippage_pct)
