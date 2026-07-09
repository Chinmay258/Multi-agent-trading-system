"""
backtest/types.py
-----------------
Shared result types for the backtest harness (kept separate to avoid import cycles).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Trade:
    """A single completed round-trip trade."""

    symbol: str
    side: str  # "buy" | "sell"
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float  # base-asset units
    notional_usd: float  # entry notional
    pnl_usd: float  # net of fees + slippage
    return_pct: float  # net return on the position notional
    bars_held: int
    exit_reason: str  # "take_profit" | "stop_loss" | "signal" | "eod"


@dataclass
class EquityPoint:
    """One mark-to-market point on the equity curve."""

    timestamp: datetime
    equity: float
    in_market: bool
