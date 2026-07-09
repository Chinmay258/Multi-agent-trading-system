"""
tests/unit/test_backtest.py
---------------------------
Unit tests for the evaluation harness. The most important one is the **no-lookahead**
test: a decision made on bar i must fill at bar i+1's open, never bar i's close.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from backtest.benchmarks import buy_and_hold
from backtest.config import BacktestConfig
from backtest.engine import run_with_decider
from backtest.metrics import compute_metrics, max_drawdown
from backtest.types import EquityPoint
from core.models.market import OHLCVCandle

_BASE = datetime(2025, 1, 1, tzinfo=UTC)


def _candle(i: int, o: float, h: float, low: float, c: float, v: float = 100.0) -> OHLCVCandle:
    return OHLCVCandle(
        symbol="BTC/USDT",
        timeframe="1d",
        timestamp=_BASE + timedelta(days=i),
        open=Decimal(str(o)),
        high=Decimal(str(h)),
        low=Decimal(str(low)),
        close=Decimal(str(c)),
        volume=Decimal(str(v)),
    )


def _flat_series(n: int, price: float = 100.0) -> list[OHLCVCandle]:
    return [_candle(i, price, price + 0.5, price - 0.5, price) for i in range(n)]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_max_drawdown_simple(self) -> None:
        # 100 -> 120 (peak) -> 90 -> 110. Max DD = (120-90)/120 = 25%.
        # Both 90 and 110 sit below the 120 peak (never recovered), so DD runs 2 bars.
        dd, length = max_drawdown([100, 120, 90, 110])
        assert round(dd, 4) == 0.25
        assert length == 2

    def test_max_drawdown_monotonic_up_is_zero(self) -> None:
        dd, length = max_drawdown([100, 101, 102, 103])
        assert dd == 0.0
        assert length == 0

    def test_compute_metrics_keys_and_basic_values(self) -> None:
        curve = [
            EquityPoint(_BASE + timedelta(days=i), 10000 + i * 100, in_market=False)
            for i in range(10)
        ]
        m = compute_metrics(curve, [], periods_per_year=365.0, initial_balance=10000.0)
        for key in (
            "total_return_pct",
            "cagr_pct",
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown_pct",
            "win_rate_pct",
            "profit_factor",
            "expectancy_usd",
            "exposure_pct",
            "monthly_returns",
            "num_trades",
        ):
            assert key in m
        assert m["total_return_pct"] > 0  # equity rose
        assert m["max_drawdown_pct"] == 0.0  # monotonic up
        assert m["exposure_pct"] == 0.0  # never in market


# ---------------------------------------------------------------------------
# engine — the no-lookahead guarantee
# ---------------------------------------------------------------------------


class TestNoLookahead:
    def test_entry_fills_at_next_bar_open_not_signal_bar_close(self) -> None:
        # Distinct open vs close so we can tell which bar/price the fill used.
        candles = [
            _candle(0, 100, 100, 100, 100),
            _candle(1, 100, 100, 100, 100),
            # bar 2: decision made here (close=100). Must NOT fill at this close.
            _candle(2, 100, 105, 99, 105),
            # bar 3: open=200 (deliberately far) — the fill must land on THIS open.
            _candle(3, 200, 205, 195, 200),
            _candle(4, 200, 205, 195, 200),
        ]
        cfg = BacktestConfig(
            timeframe="1d", slippage_pct=0.0, fee_pct=0.0, stop_loss_pct=10.0, take_profit_pct=10.0
        )

        def decide(buf, i):
            return "buy" if i == 2 else None

        run = run_with_decider(candles, cfg, decide)
        assert run.trades, "a trade should have been opened and closed at EOD"
        t = run.trades[0]
        # Entry must be bar 3's open (200), not bar 2's close (105).
        assert t.entry_price == 200.0
        assert t.entry_time == candles[3].timestamp

    def test_stop_loss_exit_recorded(self) -> None:
        candles = [
            _candle(0, 100, 100, 100, 100),
            _candle(1, 100, 100, 100, 100),
            _candle(2, 100, 100, 100, 100),  # decide buy here
            _candle(3, 100, 101, 100, 100),  # fill at open 100
            _candle(4, 100, 100, 90, 92),  # low 90 < 100*(1-0.03)=97 → stop hit
        ]
        cfg = BacktestConfig(
            timeframe="1d", slippage_pct=0.0, fee_pct=0.0, stop_loss_pct=0.03, take_profit_pct=0.50
        )

        def decide(buf, i):
            return "buy" if i == 2 else None

        run = run_with_decider(candles, cfg, decide)
        assert run.trades[0].exit_reason == "stop_loss"
        assert run.trades[0].pnl_usd < 0


# ---------------------------------------------------------------------------
# benchmarks
# ---------------------------------------------------------------------------


class TestBuyAndHold:
    def test_buy_and_hold_tracks_price(self) -> None:
        # Price doubles 100 -> 200 over the window → ~+100% before costs.
        candles = [_candle(i, 100 + i, 100 + i + 0.5, 100 + i - 0.5, 100 + i) for i in range(101)]
        cfg = BacktestConfig(timeframe="1d")
        curve, m = buy_and_hold(candles, cfg)
        assert curve
        assert m["total_return_pct"] > 90  # ~100% minus small costs
        assert m["exposure_pct"] > 99  # in market the whole time


# ---------------------------------------------------------------------------
# walk-forward ML (Phase 5)
# ---------------------------------------------------------------------------


def _noisy_walk(n: int, seed: int = 7) -> list[OHLCVCandle]:
    """Deterministic noisy random walk so labels span buy/hold/sell classes."""
    import random as _r

    rng = _r.Random(seed)
    out: list[OHLCVCandle] = []
    price = 100.0
    for i in range(n):
        prev = price
        price = max(1.0, price * (1 + rng.uniform(-0.03, 0.03)))
        hi = max(prev, price) * 1.005
        lo = min(prev, price) * 0.995
        out.append(_candle(i, prev, hi, lo, price, v=100.0 + rng.random()))
    return out


class TestWalkForward:
    def test_runs_and_respects_warmup(self) -> None:
        from backtest.walkforward import WalkForwardParams, run_walkforward_ml

        candles = _noisy_walk(280)
        cfg = BacktestConfig(timeframe="1d")
        wf = WalkForwardParams(
            horizon=5,
            threshold=0.01,
            initial_train=100,
            retrain_every=50,
            min_train_samples=40,
            calibrate=False,
        )
        run, diag = run_walkforward_ml(candles, cfg, wf)

        # Equity curve has one point per candle and the model retrained at least once.
        assert len(run.curve) == len(candles)
        assert diag["retrains"] >= 1
        # No trade may be entered before the initial training window is available.
        for t in run.trades:
            assert t.entry_time >= candles[wf.initial_train].timestamp
