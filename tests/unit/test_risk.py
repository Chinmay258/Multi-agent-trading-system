"""
tests/unit/test_risk.py
------------------------
Unit tests for Risk agent components.

Tests PositionSizer, DrawdownMonitor, and CircuitBreaker in isolation — no
Redis, no network, no agent lifecycle. All test scenarios build Pydantic
models directly and invoke the pure-logic methods.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from agents.risk.circuit_breaker import CircuitBreaker
from agents.risk.drawdown_monitor import DrawdownMonitor, PortfolioState
from agents.risk.position_sizer import ATR_STOP_MULTIPLIER, PositionSizer
from core.models.signals import (
    AggregatedSignal,
    SignalDirection,
    TechnicalSignal,
)
from core.models.trade import OrderSide, OrderType, TradeProposal

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_FUTURE = datetime(2099, 1, 1, tzinfo=UTC)


def _make_technical(
    symbol: str = "BTC/USDT",
    atr_pct: float | None = None,
    ts: datetime | None = None,
) -> TechnicalSignal:
    meta: dict = {}
    if atr_pct is not None:
        meta["atr_pct_of_price"] = atr_pct
    return TechnicalSignal(
        symbol=symbol,
        timeframe="1m",
        expires_at=_FUTURE,
        direction=SignalDirection.BUY,
        confidence=0.8,
        price=42000.0,
        metadata=meta,
        timestamp=ts or datetime.now(UTC),
    )


def _make_aggregated(
    symbol: str = "BTC/USDT",
    technical: TechnicalSignal | None = None,
) -> AggregatedSignal:
    return AggregatedSignal(
        symbol=symbol,
        direction=SignalDirection.BUY,
        confidence=0.8,
        composite_score=0.5,
        technical_signal=technical,
        total_signals=1,
    )


def _make_proposal(
    symbol: str = "BTC/USDT",
    requested_usd: float = 200.0,
    stop_loss_pct: float = 0.02,
    technical: TechnicalSignal | None = None,
) -> TradeProposal:
    agg = _make_aggregated(symbol=symbol, technical=technical)
    return TradeProposal(
        symbol=symbol,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        requested_size_usd=Decimal(str(requested_usd)),
        suggested_stop_loss_pct=stop_loss_pct,
        suggested_take_profit_pct=0.04,
        signal=agg,
        reasoning="test proposal",
    )


# ---------------------------------------------------------------------------
# PositionSizer tests
# ---------------------------------------------------------------------------


class TestPositionSizer:
    def setup_method(self):
        self.sizer = PositionSizer(
            max_position_pct=0.02,
            min_order_usd=10.0,
            max_order_usd=1000.0,
            risk_per_trade_pct=0.01,
        )
        self.portfolio = Decimal("10000")

    # --- calculate() ---

    def test_basic_calculation(self):
        # risk = 1% * 10000 = 100, stop = 2%, size = 100/0.02 = 5000
        # capped at max_position = 2% * 10000 = 200
        size = self.sizer.calculate(self.portfolio, 0.02)
        assert size == Decimal("200")

    def test_capped_at_max_position_pct(self):
        # Tight stop → very large raw size, but capped at max_position_pct
        size = self.sizer.calculate(self.portfolio, 0.001)  # 0.1% stop → raw = 100/0.001 = 100,000
        assert size <= Decimal("200")  # 2% of 10,000

    def test_capped_at_max_order_usd(self):
        huge_portfolio = Decimal("1_000_000")
        # 2% of 1M = 20,000 > max_order_usd=1000 → capped at 1000
        size = self.sizer.calculate(huge_portfolio, 0.02)
        assert size == Decimal("1000")

    def test_floored_at_min_order_usd(self):
        tiny_portfolio = Decimal("100")
        # risk = 1% * 100 = 1, stop = 2%, raw = 50; max_pos = 2; min_order = 10
        size = self.sizer.calculate(tiny_portfolio, 0.02)
        assert size == Decimal("10")

    def test_zero_stop_uses_minimum_stop_floor(self):
        # Should not raise ZeroDivisionError
        size = self.sizer.calculate(self.portfolio, 0.0)
        assert size > 0

    def test_negative_stop_uses_minimum_stop_floor(self):
        size = self.sizer.calculate(self.portfolio, -0.05)
        assert size > 0

    def test_larger_portfolio_gives_larger_size(self):
        size_small = self.sizer.calculate(Decimal("5000"), 0.02)
        size_large = self.sizer.calculate(Decimal("50000"), 0.02)
        assert size_large > size_small

    # --- get_stop_loss_pct() ---

    def test_uses_atr_when_available(self):
        tech = _make_technical(atr_pct=0.01)  # 1% ATR
        proposal = _make_proposal(technical=tech)
        stop = self.sizer.get_stop_loss_pct(proposal)
        assert stop == pytest.approx(0.01 * ATR_STOP_MULTIPLIER)

    def test_falls_back_to_suggested_when_no_atr(self):
        tech = _make_technical(atr_pct=None)  # no ATR metadata
        proposal = _make_proposal(stop_loss_pct=0.03, technical=tech)
        stop = self.sizer.get_stop_loss_pct(proposal)
        assert stop == pytest.approx(0.03)

    def test_falls_back_to_default_when_no_technical_signal(self):
        proposal = _make_proposal(stop_loss_pct=0.025, technical=None)
        stop = self.sizer.get_stop_loss_pct(proposal)
        assert stop == pytest.approx(0.025)

    def test_falls_back_to_002_when_suggested_is_none(self):
        # Create proposal with suggested_stop_loss_pct=None
        agg = _make_aggregated()
        proposal = TradeProposal(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            requested_size_usd=Decimal("200"),
            suggested_stop_loss_pct=None,
            suggested_take_profit_pct=0.04,
            signal=agg,
            reasoning="test",
        )
        stop = self.sizer.get_stop_loss_pct(proposal)
        assert stop == pytest.approx(0.02)

    def test_atr_takes_priority_over_suggested_stop(self):
        tech = _make_technical(atr_pct=0.005)  # 0.5% ATR → stop = 0.5% * 1.5 = 0.75%
        proposal = _make_proposal(stop_loss_pct=0.03, technical=tech)
        stop = self.sizer.get_stop_loss_pct(proposal)
        assert stop == pytest.approx(0.005 * ATR_STOP_MULTIPLIER)

    def test_zero_atr_falls_back_to_suggested(self):
        tech = _make_technical(atr_pct=0.0)  # zero ATR → skip, use suggested
        proposal = _make_proposal(stop_loss_pct=0.025, technical=tech)
        stop = self.sizer.get_stop_loss_pct(proposal)
        assert stop == pytest.approx(0.025)


# ---------------------------------------------------------------------------
# DrawdownMonitor tests
# ---------------------------------------------------------------------------


class TestPortfolioState:
    def _make_state(self, balance: float = 10000.0) -> PortfolioState:
        b = Decimal(str(balance))
        return PortfolioState(
            initial_balance_usd=b,
            current_balance_usd=b,
            daily_start_balance_usd=b,
        )

    def test_initial_portfolio_value_equals_balance(self):
        state = self._make_state(10000)
        assert state.portfolio_value_usd == Decimal("10000")

    def test_portfolio_value_includes_open_positions(self):
        state = self._make_state(9800)
        state.open_positions["BTC/USDT"] = Decimal("200")
        assert state.portfolio_value_usd == Decimal("10000")

    def test_open_positions_count(self):
        state = self._make_state()
        state.open_positions["BTC/USDT"] = Decimal("200")
        state.open_positions["ETH/USDT"] = Decimal("100")
        assert state.open_positions_count == 2

    def test_daily_loss_pct_when_no_loss(self):
        state = self._make_state(10000)
        assert state.daily_loss_pct == pytest.approx(0.0)

    def test_daily_loss_pct_with_loss(self):
        state = self._make_state()
        state.current_balance_usd = Decimal("9500")
        # daily_start = 10000, current = 9500, loss = 500/10000 = 5%
        assert state.daily_loss_pct == pytest.approx(0.05)

    def test_daily_loss_pct_with_gain(self):
        state = self._make_state()
        state.current_balance_usd = Decimal("10500")
        # gain → negative loss_pct
        assert state.daily_loss_pct < 0.0

    def test_total_drawdown_pct_when_no_loss(self):
        state = self._make_state(10000)
        assert state.total_drawdown_pct == pytest.approx(0.0)

    def test_total_drawdown_pct_with_loss(self):
        state = self._make_state()
        state.current_balance_usd = Decimal("8500")
        # drawdown = 1500/10000 = 15%
        assert state.total_drawdown_pct == pytest.approx(0.15)


class TestDrawdownMonitor:
    def setup_method(self):
        self.monitor = DrawdownMonitor(
            initial_balance_usd=10_000.0,
            max_daily_loss_pct=0.05,
            max_total_drawdown_pct=0.15,
        )

    def test_no_breach_initially(self):
        assert not self.monitor.daily_loss_limit_breached()
        assert not self.monitor.total_drawdown_limit_breached()

    def test_daily_loss_limit_triggered(self):
        # Simulate a 5% daily loss: reduce current_balance_usd by 5% of 10,000 = 500
        self.monitor.state.current_balance_usd = Decimal("9500")
        assert self.monitor.daily_loss_limit_breached()

    def test_daily_loss_below_limit_not_triggered(self):
        # 4% loss → below 5% limit
        self.monitor.state.current_balance_usd = Decimal("9600")
        assert not self.monitor.daily_loss_limit_breached()

    def test_total_drawdown_limit_triggered(self):
        # 15% total drawdown from 10,000 = 1,500 loss → 8,500 remaining
        self.monitor.state.current_balance_usd = Decimal("8500")
        assert self.monitor.total_drawdown_limit_breached()

    def test_total_drawdown_below_limit_not_triggered(self):
        # 14% drawdown → below 15% limit
        self.monitor.state.current_balance_usd = Decimal("8600")
        assert not self.monitor.total_drawdown_limit_breached()

    def test_open_position_reduces_cash(self):
        self.monitor.open_position("BTC/USDT", Decimal("200"))
        assert self.monitor.state.current_balance_usd == Decimal("9800")
        assert "BTC/USDT" in self.monitor.state.open_positions

    def test_open_position_preserved_in_portfolio_value(self):
        self.monitor.open_position("BTC/USDT", Decimal("200"))
        # portfolio_value = cash(9800) + position(200) = 10,000
        assert self.monitor.state.portfolio_value_usd == Decimal("10000")

    def test_close_position_with_profit_increases_balance(self):
        self.monitor.open_position("BTC/USDT", Decimal("200"))
        self.monitor.close_position("BTC/USDT", pnl_usd=Decimal("20"))
        # cash = 9800 + 200 (entry) + 20 (pnl) = 10,020
        assert self.monitor.state.current_balance_usd == Decimal("10020")
        assert "BTC/USDT" not in self.monitor.state.open_positions

    def test_close_position_with_loss_decreases_balance(self):
        self.monitor.open_position("BTC/USDT", Decimal("200"))
        self.monitor.close_position("BTC/USDT", pnl_usd=Decimal("-15"))
        # cash = 9800 + 200 - 15 = 9,985
        assert self.monitor.state.current_balance_usd == Decimal("9985")

    def test_close_unknown_position_is_noop(self):
        original = self.monitor.state.current_balance_usd
        self.monitor.close_position("XRP/USDT", pnl_usd=Decimal("10"))
        # entry cost for unknown symbol = 0, so balance += 0 + 10 = +10
        # This is intentional — the position entry cost is 0 if not tracked
        assert self.monitor.state.current_balance_usd == original + Decimal("10")

    def test_daily_pnl_tracked(self):
        self.monitor.open_position("BTC/USDT", Decimal("200"))
        self.monitor.close_position("BTC/USDT", pnl_usd=Decimal("30"))
        assert self.monitor.state.daily_realized_pnl == Decimal("30")

    def test_multiple_positions(self):
        self.monitor.open_position("BTC/USDT", Decimal("200"))
        self.monitor.open_position("ETH/USDT", Decimal("100"))
        assert self.monitor.state.open_positions_count == 2
        assert self.monitor.state.portfolio_value_usd == Decimal("10000")


# ---------------------------------------------------------------------------
# CircuitBreaker tests
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_initially_not_tripped(self):
        cb = CircuitBreaker()
        assert not cb.is_tripped
        assert cb.reason is None
        assert cb.tripped_at is None

    def test_trip_sets_is_tripped(self):
        cb = CircuitBreaker()
        cb.trip("daily loss limit")
        assert cb.is_tripped

    def test_trip_stores_reason(self):
        cb = CircuitBreaker()
        cb.trip("daily loss limit")
        assert cb.reason == "daily loss limit"

    def test_trip_records_timestamp(self):
        cb = CircuitBreaker()
        before = datetime.now(UTC)
        cb.trip("test")
        after = datetime.now(UTC)
        assert cb.tripped_at is not None
        assert before <= cb.tripped_at <= after

    def test_trip_is_idempotent(self):
        cb = CircuitBreaker()
        cb.trip("first reason")
        cb.trip("second reason")  # should not overwrite
        assert cb.reason == "first reason"

    def test_reset_clears_state(self):
        cb = CircuitBreaker()
        cb.trip("daily loss limit")
        cb.reset()
        assert not cb.is_tripped
        assert cb.reason is None
        assert cb.tripped_at is None

    def test_reset_then_trip_again(self):
        cb = CircuitBreaker()
        cb.trip("first trip")
        cb.reset()
        cb.trip("second trip")
        assert cb.is_tripped
        assert cb.reason == "second trip"

    def test_repr_tripped(self):
        cb = CircuitBreaker()
        cb.trip("some reason")
        assert "TRIPPED" in repr(cb)
        assert "some reason" in repr(cb)

    def test_repr_closed(self):
        cb = CircuitBreaker()
        assert "CLOSED" in repr(cb)


# ---------------------------------------------------------------------------
# Integration: PositionSizer + DrawdownMonitor together
# ---------------------------------------------------------------------------


class TestRiskPipeline:
    """
    Verify that PositionSizer and DrawdownMonitor work correctly together
    in a simple simulated trade cycle.
    """

    def setup_method(self):
        self.monitor = DrawdownMonitor(
            initial_balance_usd=10_000.0,
            max_daily_loss_pct=0.05,
            max_total_drawdown_pct=0.15,
        )
        self.sizer = PositionSizer(
            max_position_pct=0.02,
            min_order_usd=10.0,
            max_order_usd=1000.0,
            risk_per_trade_pct=0.01,
        )

    def test_approve_then_check_position_count(self):
        proposal = _make_proposal(technical=None)
        portfolio_val = self.monitor.state.portfolio_value_usd
        stop = self.sizer.get_stop_loss_pct(proposal)
        size = self.sizer.calculate(portfolio_val, stop)

        self.monitor.open_position(proposal.symbol, size)
        assert self.monitor.state.open_positions_count == 1

    def test_large_loss_triggers_daily_limit(self):
        # Simulate a 5% portfolio loss
        loss = Decimal("500")
        self.monitor.open_position("BTC/USDT", Decimal("200"))
        self.monitor.close_position("BTC/USDT", pnl_usd=-loss)
        # Cash: 10000 - 200 (open) + 200 (close) - 500 (loss) = 9500
        assert self.monitor.daily_loss_limit_breached()

    def test_total_drawdown_accumulates_across_trades(self):
        # Multiple losing trades push total drawdown to 15%
        self.monitor.open_position("BTC/USDT", Decimal("1000"))
        self.monitor.close_position("BTC/USDT", pnl_usd=Decimal("-1500"))
        # Cash: 10000 - 1000 + 1000 - 1500 = 8500 → 15% drawdown
        assert self.monitor.total_drawdown_limit_breached()
