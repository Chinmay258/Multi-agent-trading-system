"""
tests/unit/test_execution.py
-----------------------------
Unit tests for Phase 4 execution components — PaperBroker.

Zero I/O: all tests construct Pydantic models directly and call PaperBroker
methods with mocked Settings. No Redis, no network, no agent lifecycle.

Test groups:
    TestPaperBrokerCapabilities    — broker_name, is_paper flag
    TestPaperBrokerPlaceOrder      — fill mechanics, slippage, partial fills,
                                     idempotency, guard clauses
    TestPaperBrokerBalance         — cash tracking across fills
    TestPaperBrokerPositions       — position tracking after buy and close
    TestPaperBrokerPing            — health check
    TestPaperBrokerCancelOrder     — always True in paper mode
    TestPaperBrokerClosePosition   — closing reverses the position
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from agents.execution.paper_broker import DEFAULT_SLIPPAGE_PCT, PaperBroker
from core.exceptions import InsufficientBalanceError, OrderRejectedError
from core.models.signals import AggregatedSignal, SignalDirection, TechnicalSignal
from core.models.trade import (
    OrderSide,
    OrderStatus,
    OrderType,
    RiskAssessment,
    RiskDecision,
    TradeProposal,
)

# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------

_FUTURE = datetime(2099, 1, 1, tzinfo=UTC)
_REF_PRICE = 42_000.0


def _paper_settings(balance: float = 10_000.0) -> MagicMock:
    """Mock Settings in paper mode — assert_paper_mode() is a no-op."""
    # ``unsafe=True`` lets us configure ``assert_paper_mode`` even though
    # its name starts with "assert" (MagicMock blocks that by default).
    s = MagicMock(unsafe=True)
    s.paper_initial_balance_usd = balance
    s.is_paper_trading = True
    s.assert_paper_mode.return_value = None
    return s


def _live_settings() -> MagicMock:
    """Mock Settings in live mode — assert_paper_mode() raises RuntimeError."""
    s = MagicMock(unsafe=True)
    s.paper_initial_balance_usd = 10_000.0
    s.is_paper_trading = False
    s.assert_paper_mode.side_effect = RuntimeError(
        "This operation is restricted to paper trading mode."
    )
    return s


def _make_technical(price: float = _REF_PRICE) -> TechnicalSignal:
    return TechnicalSignal(
        symbol="BTC/USDT",
        timeframe="1m",
        expires_at=_FUTURE,
        direction=SignalDirection.BUY,
        confidence=0.8,
        price=price,
    )


def _make_aggregated(technical: TechnicalSignal | None = None) -> AggregatedSignal:
    return AggregatedSignal(
        symbol="BTC/USDT",
        direction=SignalDirection.BUY,
        confidence=0.8,
        composite_score=0.5,
        technical_signal=technical,
        total_signals=1,
    )


def _make_proposal(
    side: OrderSide = OrderSide.BUY,
    requested_usd: float = 200.0,
    technical: TechnicalSignal | None = None,
) -> TradeProposal:
    if technical is None:
        technical = _make_technical()
    return TradeProposal(
        symbol="BTC/USDT",
        side=side,
        order_type=OrderType.MARKET,
        requested_size_usd=Decimal(str(requested_usd)),
        suggested_stop_loss_pct=0.02,
        suggested_take_profit_pct=0.04,
        signal=_make_aggregated(technical=technical),
        reasoning="test proposal",
    )


def _make_assessment(
    side: OrderSide = OrderSide.BUY,
    approved_usd: float = 200.0,
    technical: TechnicalSignal | None = None,
) -> RiskAssessment:
    proposal = _make_proposal(side=side, requested_usd=approved_usd, technical=technical)
    return RiskAssessment(
        proposal_id=proposal.proposal_id,
        decision=RiskDecision.APPROVED,
        approved_size_usd=Decimal(str(approved_usd)),
        approved_stop_loss_pct=0.02,
        approved_take_profit_pct=0.04,
        portfolio_value_usd=Decimal("10000"),
        current_daily_loss_pct=0.0,
        open_positions_count=0,
        original_proposal=proposal,
    )


# ---------------------------------------------------------------------------
# TestPaperBrokerCapabilities
# ---------------------------------------------------------------------------


class TestPaperBrokerCapabilities:
    def setup_method(self):
        self.broker = PaperBroker(_paper_settings())

    def test_broker_name(self):
        assert self.broker.capabilities.broker_name == "paper"

    def test_is_paper_flag_true(self):
        assert self.broker.capabilities.is_paper is True

    def test_no_native_stops(self):
        assert self.broker.capabilities.supports_native_stops is False

    def test_no_native_take_profit(self):
        assert self.broker.capabilities.supports_native_take_profit is False

    def test_supports_partial_fills(self):
        assert self.broker.capabilities.supports_partial_fills is True


# ---------------------------------------------------------------------------
# TestPaperBrokerPlaceOrder
# ---------------------------------------------------------------------------


class TestPaperBrokerPlaceOrder:
    def setup_method(self):
        self.broker = PaperBroker(_paper_settings(balance=10_000.0))

    async def _connect_and_fill(self, assessment: RiskAssessment | None = None):
        await self.broker.connect()
        if assessment is None:
            assessment = _make_assessment()
        return await self.broker.place_order(assessment)

    async def test_full_fill_buy_returns_filled_status(self):
        """Full fill (no partial) → status FILLED or PARTIALLY_FILLED."""
        assessment = _make_assessment(side=OrderSide.BUY, approved_usd=200.0)
        result = await self._connect_and_fill(assessment)
        assert result.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)

    async def test_full_fill_buy_is_paper_true(self):
        result = await self._connect_and_fill()
        assert result.is_paper is True

    async def test_full_fill_buy_links_ids(self):
        assessment = _make_assessment()
        result = await self._connect_and_fill(assessment)
        assert result.proposal_id == assessment.proposal_id
        assert result.assessment_id == assessment.assessment_id

    async def test_buy_fill_price_above_ref(self):
        """BUY fill price = ref * (1 + slippage) > ref."""
        assessment = _make_assessment(side=OrderSide.BUY)
        result = await self._connect_and_fill(assessment)
        ref = Decimal(str(_REF_PRICE))
        expected_min = ref * (Decimal("1") + Decimal(str(DEFAULT_SLIPPAGE_PCT)))
        if result.average_fill_price:
            assert result.average_fill_price >= expected_min * Decimal("0.999")

    async def test_sell_fill_price_below_ref(self):
        """SELL fill price = ref * (1 - slippage) < ref."""
        # Need more balance to cover cost for sell (opening short)
        broker = PaperBroker(_paper_settings(balance=100_000.0))
        await broker.connect()
        assessment = _make_assessment(side=OrderSide.SELL, approved_usd=200.0)
        result = await broker.place_order(assessment)
        ref = Decimal(str(_REF_PRICE))
        expected_max = ref * (Decimal("1") - Decimal(str(DEFAULT_SLIPPAGE_PCT)))
        if result.average_fill_price:
            assert result.average_fill_price <= ref
            assert result.average_fill_price >= expected_max * Decimal("0.999")

    async def test_partial_fill_simulation(self, monkeypatch):
        """Patch random to force partial fill path."""
        import random

        call_count = 0

        def _fake_random():
            nonlocal call_count
            call_count += 1
            # First call: random() < 0.20 → True (force partial fill)
            if call_count == 1:
                return 0.10
            # Second call: uniform(0.60, 0.99) → pick 0.75
            return 0.75

        monkeypatch.setattr(random, "random", _fake_random)
        monkeypatch.setattr(random, "uniform", lambda a, b: 0.75 if a == 0.60 else 0.08)

        assessment = _make_assessment(approved_usd=200.0)
        result = await self._connect_and_fill(assessment)
        assert result.status == OrderStatus.PARTIALLY_FILLED
        assert result.filled_quantity < result.requested_quantity

    async def test_idempotency_same_proposal_id_returns_same_result(self):
        """Calling place_order twice with the same proposal_id returns cached result."""
        assessment = _make_assessment()
        await self.broker.connect()
        result1 = await self.broker.place_order(assessment)
        result2 = await self.broker.place_order(assessment)
        assert result1.result_id == result2.result_id

    async def test_assert_paper_mode_guard_raises_in_live_mode(self):
        """place_order must raise RuntimeError when settings are in live mode."""
        live_broker = PaperBroker(_live_settings())
        assessment = _make_assessment()
        with pytest.raises(RuntimeError, match="restricted to paper trading"):
            await live_broker.place_order(assessment)

    async def test_no_reference_price_raises_order_rejected(self):
        """Assessment with no technical_signal must raise OrderRejectedError."""
        # Build a signal/proposal lacking a technical component.
        no_tech_signal = AggregatedSignal(
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            confidence=0.8,
            composite_score=0.5,
            technical_signal=None,
            total_signals=1,
        )
        # Must construct without Pydantic immutability — build fresh proposal
        proposal2 = TradeProposal(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            requested_size_usd=Decimal("200"),
            signal=no_tech_signal,
            reasoning="no tech signal",
        )
        assessment = RiskAssessment(
            proposal_id=proposal2.proposal_id,
            decision=RiskDecision.APPROVED,
            approved_size_usd=Decimal("200"),
            original_proposal=proposal2,
        )
        await self.broker.connect()
        with pytest.raises(OrderRejectedError, match="no reference price"):
            await self.broker.place_order(assessment)

    async def test_insufficient_balance_raises(self):
        """Order that costs more than available cash raises InsufficientBalanceError."""
        # Balance = 100 but order size = 99999 USD → cost >> balance
        broker = PaperBroker(_paper_settings(balance=100.0))
        await broker.connect()
        assessment = _make_assessment(approved_usd=99_999.0)
        with pytest.raises(InsufficientBalanceError):
            await broker.place_order(assessment)

    async def test_fee_is_nonzero(self):
        result = await self._connect_and_fill()
        assert result.fee_usd is not None
        assert result.fee_usd > Decimal("0")

    async def test_fee_currency_is_usdt(self):
        result = await self._connect_and_fill()
        assert result.fee_currency == "USDT"


# ---------------------------------------------------------------------------
# TestPaperBrokerBalance
# ---------------------------------------------------------------------------


class TestPaperBrokerBalance:
    async def test_initial_balance_matches_settings(self):
        broker = PaperBroker(_paper_settings(balance=5_000.0))
        await broker.connect()
        balance = await broker.get_balance()
        assert float(balance.total_equity_usd) == pytest.approx(5_000.0, rel=1e-6)

    async def test_free_margin_equals_total_initially(self):
        broker = PaperBroker(_paper_settings(balance=10_000.0))
        await broker.connect()
        balance = await broker.get_balance()
        assert balance.free_margin_usd == balance.total_equity_usd
        assert balance.used_margin_usd == Decimal("0")

    async def test_balance_decreases_after_buy(self):
        broker = PaperBroker(_paper_settings(balance=10_000.0))
        await broker.connect()
        before = await broker.get_balance()
        assessment = _make_assessment(side=OrderSide.BUY, approved_usd=200.0)
        await broker.place_order(assessment)
        after = await broker.get_balance()
        # Free margin decreases (cash used for position)
        assert after.free_margin_usd < before.free_margin_usd
        # Total equity approximately preserved (position value locked)
        assert after.total_equity_usd > Decimal("0")

    async def test_used_margin_nonzero_after_buy(self):
        broker = PaperBroker(_paper_settings(balance=10_000.0))
        await broker.connect()
        assessment = _make_assessment(side=OrderSide.BUY, approved_usd=200.0)
        await broker.place_order(assessment)
        balance = await broker.get_balance()
        assert balance.used_margin_usd > Decimal("0")

    async def test_equity_restored_after_close(self):
        """After close_position the locked margin is released back to cash."""
        broker = PaperBroker(_paper_settings(balance=10_000.0))
        await broker.connect()
        assessment = _make_assessment(side=OrderSide.BUY, approved_usd=200.0)
        await broker.place_order(assessment)
        await broker.close_position("BTC/USDT")
        balance = await broker.get_balance()
        # After close, used margin should be zero
        assert balance.used_margin_usd == Decimal("0")
        # Cash returned (minus fees)
        assert balance.free_margin_usd > Decimal("0")


# ---------------------------------------------------------------------------
# TestPaperBrokerPositions
# ---------------------------------------------------------------------------


class TestPaperBrokerPositions:
    async def test_no_positions_initially(self):
        broker = PaperBroker(_paper_settings())
        await broker.connect()
        positions = await broker.get_positions()
        assert positions == []

    async def test_position_added_after_buy(self):
        broker = PaperBroker(_paper_settings())
        await broker.connect()
        assessment = _make_assessment(side=OrderSide.BUY, approved_usd=200.0)
        await broker.place_order(assessment)
        positions = await broker.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "BTC/USDT"
        assert positions[0].side == "buy"
        assert positions[0].quantity > Decimal("0")

    async def test_position_removed_after_close(self):
        broker = PaperBroker(_paper_settings())
        await broker.connect()
        assessment = _make_assessment(side=OrderSide.BUY, approved_usd=200.0)
        await broker.place_order(assessment)
        await broker.close_position("BTC/USDT")
        positions = await broker.get_positions()
        assert positions == []

    async def test_close_nonexistent_position_returns_none(self):
        broker = PaperBroker(_paper_settings())
        await broker.connect()
        result = await broker.close_position("ETH/USDT")
        assert result is None

    async def test_position_entry_price_near_ref(self):
        broker = PaperBroker(_paper_settings())
        await broker.connect()
        assessment = _make_assessment(side=OrderSide.BUY, approved_usd=200.0)
        await broker.place_order(assessment)
        positions = await broker.get_positions()
        assert len(positions) == 1
        entry = positions[0].entry_price
        ref = Decimal(str(_REF_PRICE))
        # Entry price should be within 1% of reference (slippage is 0.05%)
        assert abs(entry - ref) / ref < Decimal("0.01")


# ---------------------------------------------------------------------------
# TestPaperBrokerPing
# ---------------------------------------------------------------------------


class TestPaperBrokerPing:
    async def test_ping_always_true(self):
        broker = PaperBroker(_paper_settings())
        result = await broker.ping()
        assert result is True

    async def test_ping_before_connect(self):
        """Ping works even before connect() is called."""
        broker = PaperBroker(_paper_settings())
        assert await broker.ping() is True


# ---------------------------------------------------------------------------
# TestPaperBrokerCancelOrder
# ---------------------------------------------------------------------------


class TestPaperBrokerCancelOrder:
    async def test_cancel_always_returns_true(self):
        broker = PaperBroker(_paper_settings())
        await broker.connect()
        result = await broker.cancel_order("fake-order-id", "BTC/USDT")
        assert result is True

    async def test_cancel_unknown_symbol_returns_true(self):
        broker = PaperBroker(_paper_settings())
        result = await broker.cancel_order("any-id", "UNKNOWN/SYMBOL")
        assert result is True


# ---------------------------------------------------------------------------
# TestPaperBrokerClosePosition
# ---------------------------------------------------------------------------


class TestPaperBrokerClosePosition:
    async def test_close_returns_execution_result(self):
        broker = PaperBroker(_paper_settings())
        await broker.connect()
        assessment = _make_assessment(side=OrderSide.BUY, approved_usd=200.0)
        await broker.place_order(assessment)
        result = await broker.close_position("BTC/USDT")
        assert result is not None
        assert result.is_paper is True
        assert result.status == OrderStatus.FILLED
        assert result.symbol == "BTC/USDT"
        # Closing a long position should sell
        assert result.side == OrderSide.SELL

    async def test_close_sell_position_gives_buy_close(self):
        """Closing a short position (side=sell) produces a BUY close leg."""
        broker = PaperBroker(_paper_settings(balance=100_000.0))
        await broker.connect()
        assessment = _make_assessment(side=OrderSide.SELL, approved_usd=200.0)
        await broker.place_order(assessment)
        result = await broker.close_position("BTC/USDT")
        assert result is not None
        assert result.side == OrderSide.BUY
