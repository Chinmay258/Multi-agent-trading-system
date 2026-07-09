"""
backtest/
---------
Rigorous, honest evaluation harness for the trading system.

Replays historical OHLCV through the *real* signal pipeline with **no lookahead**
(signals computed at bar i, orders filled at bar i+1's open), realistic **fees +
slippage** (matched to the live PaperBroker), and the system's **SL/TP + position
sizing**. Produces:

- ``backtest/results/baseline_metrics.json`` — full performance + benchmark metrics.
- ``backtest/results/EVALUATION_REPORT.html`` (+ ``.pdf``) — charts + candid limitations.

Run it with ``make eval`` (i.e. ``python -m backtest.run``).

Design principle: if a number looks too good, assume a bug (lookahead / fees /
survivorship) until proven otherwise. A credible mediocre result beats a fake good one.
"""

from backtest.config import BacktestConfig

__all__ = ["BacktestConfig"]
