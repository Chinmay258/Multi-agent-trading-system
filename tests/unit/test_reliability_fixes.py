"""
tests/unit/test_reliability_fixes.py
------------------------------------
Regression tests for accounting, data-path, and control-plane fixes:

- The Risk ledger counts a filled position once, realises PnL on close, releases a failed
  order's reservation, and ignores a redelivered result.
- Risk rejects a same-side proposal for a symbol that already has a position, and approves
  an opposite-side proposal as a close even when the position limit is reached.
- PaperBroker closes at the market price (long and short), closes on an opposite-side order,
  and rejects stacking.
- Only closed candles are published.
- Exchange fetches are retried on transient network errors.
- The supervisor restarts a crashed agent with a fresh instance.
- A persisted halt latch is honoured at startup.
- The execution history list is capped atomically.
- Control endpoints require an API key.
- The random-entry benchmark reports an empirical p-value.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from agents.base import BaseAgent, run_agent
from agents.execution.paper_broker import PaperBroker
from agents.market_data.normalizer import closed_candles
from agents.risk.agent import RiskAgent
from core.config import get_settings
from core.exceptions import ExchangeConnectionError, OrderRejectedError
from core.messaging import HALT_KEY, MessageBus
from core.models.market import OHLCVCandle
from core.models.signals import AggregatedSignal, SignalDirection, TechnicalSignal
from core.models.trade import (
    ExecutionResult,
    OrderSide,
    OrderStatus,
    OrderType,
    RejectionReason,
    RiskAssessment,
    RiskDecision,
    TradeProposal,
)

_FUTURE = datetime(2099, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _proposal(
    symbol: str = "BTC/USDT",
    side: OrderSide = OrderSide.BUY,
    size: float = 200.0,
    price: float = 42_000.0,
) -> TradeProposal:
    direction = SignalDirection.BUY if side == OrderSide.BUY else SignalDirection.SELL
    technical = TechnicalSignal(
        symbol=symbol,
        timeframe="1m",
        timestamp=datetime.now(UTC),
        expires_at=_FUTURE,
        direction=direction,
        confidence=0.8,
        price=price,
    )
    aggregated = AggregatedSignal(
        symbol=symbol,
        direction=direction,
        confidence=0.8,
        composite_score=0.5 if side == OrderSide.BUY else -0.5,
        technical_signal=technical,
        total_signals=1,
    )
    return TradeProposal(
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        requested_size_usd=Decimal(str(size)),
        suggested_stop_loss_pct=0.02,
        signal=aggregated,
        reasoning="regression test",
    )


def _assessment(
    symbol: str = "BTC/USDT",
    side: OrderSide = OrderSide.BUY,
    size: float = 200.0,
    price: float = 42_000.0,
) -> RiskAssessment:
    proposal = _proposal(symbol=symbol, side=side, size=size, price=price)
    return RiskAssessment(
        proposal_id=proposal.proposal_id,
        decision=RiskDecision.APPROVED,
        approved_size_usd=Decimal(str(size)),
        original_proposal=proposal,
    )


def _fill(
    symbol: str,
    side: OrderSide,
    cost: str,
    fee: str,
    status: OrderStatus = OrderStatus.FILLED,
    realized: str | None = None,
) -> ExecutionResult:
    filled = status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)
    return ExecutionResult(
        proposal_id=uuid4(),
        assessment_id=uuid4(),
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        status=status,
        requested_quantity=Decimal("0.005"),
        filled_quantity=Decimal("0.005") if filled else Decimal("0"),
        average_fill_price=Decimal("42000"),
        total_cost_usd=Decimal(cost),
        fee_usd=Decimal(fee),
        is_paper=True,
        realized_pnl_usd=Decimal(realized) if realized is not None else None,
    )


def _paper_settings(balance: float = 10_000.0) -> MagicMock:
    settings = MagicMock(unsafe=True)
    settings.paper_initial_balance_usd = balance
    settings.assert_paper_mode.return_value = None
    return settings


def _candle(timeframe: str, ts: datetime, price: float = 100.0) -> OHLCVCandle:
    p = Decimal(str(price))
    return OHLCVCandle(
        symbol="BTC/USDT",
        timeframe=timeframe,
        timestamp=ts,
        open=p,
        high=p + 1,
        low=p - 1,
        close=p,
        volume=Decimal("10"),
    )


# ---------------------------------------------------------------------------
# Risk ledger
# ---------------------------------------------------------------------------


class TestRiskLedger:
    async def test_buy_fill_is_counted_once(self) -> None:
        risk = RiskAgent()
        assessment = await risk._evaluate(_proposal())
        assert assessment.is_approved

        action = risk._apply_execution_result(_fill("BTC/USDT", OrderSide.BUY, "199.80", "0.19"))

        state = risk._drawdown.state
        assert action == "reconciled"
        assert state.open_positions_count == 1
        assert state.open_positions["BTC/USDT"] == Decimal("199.99")
        # The position is counted once: no phantom loss from double-deducted cash.
        assert state.portfolio_value_usd == Decimal("10000")

    async def test_flat_round_trips_do_not_create_phantom_drawdown(self) -> None:
        risk = RiskAgent()
        for _ in range(8):
            opened = await risk._evaluate(_proposal(side=OrderSide.BUY))
            assert opened.is_approved
            buy = _fill("BTC/USDT", OrderSide.BUY, "199.80", "0.19")
            assert risk._apply_execution_result(buy) == "reconciled"

            closing = await risk._evaluate(_proposal(side=OrderSide.SELL))
            assert closing.is_approved
            sell = _fill("BTC/USDT", OrderSide.SELL, "199.60", "0.19", realized="-0.39")
            assert risk._apply_execution_result(sell) == "closed"

        state = risk._drawdown.state
        assert state.open_positions_count == 0
        # Only fees are lost: 8 round trips x $0.39.
        assert state.portfolio_value_usd == Decimal("10000") - Decimal("0.39") * 8
        assert not risk._drawdown.total_drawdown_limit_breached()

    async def test_failed_order_releases_reservation(self) -> None:
        risk = RiskAgent()
        await risk._evaluate(_proposal())
        assert risk._drawdown.state.open_positions_count == 1

        rejected = _fill("BTC/USDT", OrderSide.BUY, "0", "0", status=OrderStatus.REJECTED)
        assert risk._apply_execution_result(rejected) == "released"
        assert risk._drawdown.state.open_positions_count == 0
        assert risk._drawdown.state.portfolio_value_usd == Decimal("10000")

    async def test_redelivered_result_is_applied_once(self) -> None:
        risk = RiskAgent()
        await risk._evaluate(_proposal())
        fill = _fill("BTC/USDT", OrderSide.BUY, "199.80", "0.19")

        assert risk._apply_execution_result(fill) == "reconciled"
        assert risk._apply_execution_result(fill) == "duplicate"
        assert risk._drawdown.state.portfolio_value_usd == Decimal("10000")

    async def test_same_side_proposal_is_rejected(self) -> None:
        risk = RiskAgent()
        await risk._evaluate(_proposal())
        second = await risk._evaluate(_proposal())

        assert second.decision == RiskDecision.REJECTED
        assert second.rejection_reason == RejectionReason.DUPLICATE_SIGNAL

    async def test_close_is_allowed_at_the_position_limit(self) -> None:
        risk = RiskAgent()
        limit = risk.settings.risk.max_open_positions
        symbols = [f"C{i}/USDT" for i in range(limit)]
        for symbol in symbols:
            assert (await risk._evaluate(_proposal(symbol=symbol))).is_approved

        blocked = await risk._evaluate(_proposal(symbol="NEW/USDT"))
        assert blocked.rejection_reason == RejectionReason.MAX_OPEN_POSITIONS

        close = await risk._evaluate(_proposal(symbol=symbols[0], side=OrderSide.SELL))
        assert close.is_approved
        assert close.approved_size_usd == risk._drawdown.state.open_positions[symbols[0]]


# ---------------------------------------------------------------------------
# Paper broker
# ---------------------------------------------------------------------------


class TestPaperBrokerCloses:
    async def test_long_closes_at_market_price(self) -> None:
        broker = PaperBroker(_paper_settings())
        await broker.connect()
        with patch("random.random", return_value=0.99):  # no partial fill
            await broker.place_order(_assessment(side=OrderSide.BUY, price=40_000.0))

        close = await broker.close_position("BTC/USDT", price=Decimal("44000"))

        assert close is not None
        assert close.side == OrderSide.SELL
        assert close.average_fill_price == Decimal("43978")  # 44,000 less 0.05% slippage
        assert close.realized_pnl_usd is not None and close.realized_pnl_usd > Decimal("15")
        balance = await broker.get_balance()
        assert balance.used_margin_usd == Decimal("0")
        assert float(balance.total_equity_usd) == pytest.approx(
            10_000 + float(close.realized_pnl_usd), abs=0.01
        )

    async def test_short_profits_when_price_falls(self) -> None:
        broker = PaperBroker(_paper_settings())
        await broker.connect()
        with patch("random.random", return_value=0.99):
            await broker.place_order(_assessment(side=OrderSide.SELL, price=40_000.0))

        close = await broker.close_position("BTC/USDT", price=Decimal("36000"))

        assert close is not None
        assert close.side == OrderSide.BUY
        assert close.realized_pnl_usd is not None and close.realized_pnl_usd > Decimal("15")
        balance = await broker.get_balance()
        assert float(balance.total_equity_usd) == pytest.approx(
            10_000 + float(close.realized_pnl_usd), abs=0.01
        )

    async def test_opposite_side_order_closes_position(self) -> None:
        broker = PaperBroker(_paper_settings())
        await broker.connect()
        with patch("random.random", return_value=0.99):
            await broker.place_order(_assessment(side=OrderSide.BUY, price=40_000.0))
            result = await broker.place_order(_assessment(side=OrderSide.SELL, price=42_000.0))

        assert result.side == OrderSide.SELL
        assert result.realized_pnl_usd is not None and result.realized_pnl_usd > Decimal("0")
        assert await broker.get_positions() == []

    async def test_same_side_order_is_not_stacked(self) -> None:
        broker = PaperBroker(_paper_settings())
        await broker.connect()
        with patch("random.random", return_value=0.99):
            await broker.place_order(_assessment(side=OrderSide.BUY))
            with pytest.raises(OrderRejectedError):
                await broker.place_order(_assessment(side=OrderSide.BUY))
        assert len(await broker.get_positions()) == 1


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------


class TestClosedCandles:
    def test_forming_candle_is_dropped(self) -> None:
        now = datetime(2026, 1, 1, 12, 30, tzinfo=UTC)
        closed = _candle("4h", datetime(2026, 1, 1, 8, 0, tzinfo=UTC))  # closed at 12:00
        forming = _candle("4h", datetime(2026, 1, 1, 12, 0, tzinfo=UTC))  # closes at 16:00

        assert closed_candles([closed, forming], now=now) == [closed]

    def test_candle_closing_exactly_now_is_kept(self) -> None:
        start = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        candle = _candle("1h", start)

        assert closed_candles([candle], now=start + timedelta(hours=1)) == [candle]


class TestExchangeRetries:
    async def test_transient_network_errors_are_retried(self, monkeypatch) -> None:
        import ccxt.async_support as ccxt
        from tenacity import wait_none

        from agents.market_data.fetcher import ExchangeFetcher

        monkeypatch.setattr(ExchangeFetcher._raw_fetch_ohlcv.retry, "wait", wait_none())
        calls = {"n": 0}
        row = [1_735_689_600_000, 100.0, 101.0, 99.0, 100.5, 10.0]

        class FlakyExchange:
            async def fetch_ohlcv(self, *args: object, **kwargs: object) -> list:
                calls["n"] += 1
                if calls["n"] < 3:
                    raise ccxt.NetworkError("transient")
                return [row]

        fetcher = ExchangeFetcher()
        fetcher._exchange = FlakyExchange()
        fetcher._connected = True

        candles = await fetcher.fetch_ohlcv("BTC/USDT", "1h", limit=1)

        assert calls["n"] == 3
        assert len(candles) == 1

    async def test_persistent_errors_surface_after_retries(self, monkeypatch) -> None:
        import ccxt.async_support as ccxt
        from tenacity import wait_none

        from agents.market_data.fetcher import ExchangeFetcher

        monkeypatch.setattr(ExchangeFetcher._raw_fetch_ohlcv.retry, "wait", wait_none())
        calls = {"n": 0}

        class DownExchange:
            async def fetch_ohlcv(self, *args: object, **kwargs: object) -> list:
                calls["n"] += 1
                raise ccxt.NetworkError("down")

        fetcher = ExchangeFetcher()
        fetcher._exchange = DownExchange()
        fetcher._connected = True

        with pytest.raises(ExchangeConnectionError):
            await fetcher.fetch_ohlcv("BTC/USDT", "1h", limit=1)
        assert calls["n"] == 5


# ---------------------------------------------------------------------------
# Agent lifecycle
# ---------------------------------------------------------------------------


class _FlakyAgent(BaseAgent):
    """Crashes on its first instance only."""

    name = "flaky_agent"
    instances: list[_FlakyAgent] = []

    def __init__(self) -> None:
        super().__init__()
        _FlakyAgent.instances.append(self)

    async def run(self) -> None:
        if len(_FlakyAgent.instances) == 1:
            raise RuntimeError("boom")

    async def run_loop(self) -> None:
        return None


class _IdleAgent(BaseAgent):
    name = "idle_agent"

    async def run_loop(self) -> None:
        return None


def _fake_bus(bus: MessageBus) -> MessageBus:
    import fakeredis.aioredis

    bus._pool = fakeredis.aioredis.FakeRedis(decode_responses=True)
    bus._connected = True
    return bus


class TestAgentLifecycle:
    async def test_crash_restarts_with_a_fresh_instance(self) -> None:
        _FlakyAgent.instances.clear()

        await run_agent(_FlakyAgent())

        assert len(_FlakyAgent.instances) == 2
        assert _FlakyAgent.instances[0] is not _FlakyAgent.instances[1]

    async def test_persisted_halt_is_honoured_at_startup(self) -> None:
        agent = _IdleAgent()
        _fake_bus(agent.bus)
        await agent.bus.kv_set(HALT_KEY, "manual test halt")

        await agent._load_halt_latch()

        assert agent._trading_halted is True

    async def test_no_latch_means_not_halted(self) -> None:
        agent = _IdleAgent()
        _fake_bus(agent.bus)

        await agent._load_halt_latch()

        assert agent._trading_halted is False

    async def test_history_list_is_capped(self) -> None:
        bus = _fake_bus(MessageBus())
        for i in range(5):
            await bus.list_push_capped("history", str(i), max_len=3)

        assert await bus._pool.lrange("history", 0, -1) == ["4", "3", "2"]


# ---------------------------------------------------------------------------
# Control plane
# ---------------------------------------------------------------------------


class TestControlAuth:
    @pytest.fixture(autouse=True)
    def _fresh_settings(self):
        get_settings.cache_clear()
        yield
        get_settings.cache_clear()

    def _client(self):
        from fastapi.testclient import TestClient

        from api.main import app

        return TestClient(app)  # no lifespan: no Redis connection is attempted

    def test_disabled_without_configured_key(self, monkeypatch) -> None:
        monkeypatch.setenv("CONTROL_API_KEY", "")
        response = self._client().post("/control/halt", json={})
        assert response.status_code == 403

    def test_wrong_key_is_rejected(self, monkeypatch) -> None:
        monkeypatch.setenv("CONTROL_API_KEY", "correct-key")
        response = self._client().post("/control/halt", json={}, headers={"X-API-Key": "wrong"})
        assert response.status_code == 401

    def test_missing_key_is_rejected(self, monkeypatch) -> None:
        monkeypatch.setenv("CONTROL_API_KEY", "correct-key")
        response = self._client().post("/control/halt", json={})
        assert response.status_code == 401

    def test_correct_key_passes_authentication(self, monkeypatch) -> None:
        monkeypatch.setenv("CONTROL_API_KEY", "correct-key")
        response = self._client().post(
            "/control/halt", json={}, headers={"X-API-Key": "correct-key"}
        )
        # Authenticated; without the app lifespan there is no bus, so 503.
        assert response.status_code == 503


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class TestRandomEntrySignificance:
    def test_p_value_is_reported(self) -> None:
        from backtest.benchmarks import random_entry
        from backtest.config import BacktestConfig

        start = datetime(2024, 1, 1, tzinfo=UTC)
        candles = [_candle("1d", start + timedelta(days=i), 100.0 + (i % 7)) for i in range(120)]
        cfg = BacktestConfig(timeframe="1d", random_runs=20)

        out = random_entry(candles, cfg, target_trades=5, strategy_return_pct=0.0)

        beats = out["strategy_beats_runs"]
        assert 0 <= beats <= 20
        assert out["p_value_vs_random"] == round((20 - beats + 1) / 21, 4)
        assert 0 < out["p_value_vs_random"] <= 1
