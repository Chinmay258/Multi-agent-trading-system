"""
tests/unit/test_mt5_bridge.py
------------------------------
Unit tests for Phase 7 MT5 integration — WebRequest (aiohttp HTTP server) architecture.

Zero I/O — no real MT5 terminal, no network.
The HTTP server is never started; individual handler methods are called directly.
Place/cancel/modify commands are tested by mocking _enqueue_and_wait().

Test groups:
    TestSymbolMapper             — to_mt5, to_python, unknown symbol raises
    TestMT5BridgeCapabilities    — broker_name, is_paper, requires_symbol_mapping
    TestMT5BridgePing            — no heartbeat → False, recent → True, stale → False
    TestMT5BridgePlaceOrder      — fill → FILLED, error → OrderRejectedError, is_paper
    TestMT5BridgeGetBalance      — maps cached balance fields; raises when no cache
    TestMT5BridgeGetPositions    — empty list, MT5 symbol mapped to Python format
    TestMT5BridgeCancelOrder     — ok → True, not_found → False
    TestMT5BridgeHTTPHandlers    — _handle_state, _handle_command, _handle_result
    TestMT5BridgeEnqueueAndWait  — timeout raises, success resolves
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.execution.mt5_bridge import MT5Bridge
from agents.execution.symbol_mapper import SymbolMapper
from core.exceptions import ExchangeConnectionError, OrderRejectedError
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
# Shared helpers
# ---------------------------------------------------------------------------

_FUTURE = datetime(2099, 1, 1, tzinfo=UTC)
_REF_PRICE = 42_000.0


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
    approved_quantity: Decimal | None = Decimal("0.01"),
    technical: TechnicalSignal | None = None,
) -> RiskAssessment:
    proposal = _make_proposal(side=side, requested_usd=approved_usd, technical=technical)
    return RiskAssessment(
        proposal_id=proposal.proposal_id,
        decision=RiskDecision.APPROVED,
        approved_size_usd=Decimal(str(approved_usd)),
        approved_quantity=approved_quantity,
        approved_stop_loss_pct=0.02,
        approved_take_profit_pct=0.04,
        portfolio_value_usd=Decimal("10000"),
        current_daily_loss_pct=0.0,
        open_positions_count=0,
        original_proposal=proposal,
    )


def _make_bridge() -> MT5Bridge:
    """Construct MT5Bridge without starting the HTTP server."""
    return MT5Bridge()


def _make_request(body: dict) -> AsyncMock:
    """Return a mock aiohttp Request whose json() coroutine returns body."""
    request = AsyncMock()
    request.json = AsyncMock(return_value=body)
    return request


# ---------------------------------------------------------------------------
# TestSymbolMapper
# ---------------------------------------------------------------------------


class TestSymbolMapper:
    def setup_method(self) -> None:
        self.mapper = SymbolMapper()

    def test_symbol_mapper_to_mt5(self) -> None:
        assert self.mapper.to_mt5("BTC/USDT") == "BTCUSD"

    def test_symbol_mapper_to_python(self) -> None:
        assert self.mapper.to_python("BTCUSD") == "BTC/USDT"

    def test_symbol_mapper_eth(self) -> None:
        assert self.mapper.to_mt5("ETH/USDT") == "ETHUSD"
        assert self.mapper.to_python("ETHUSD") == "ETH/USDT"

    def test_symbol_mapper_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown symbol"):
            self.mapper.to_mt5("UNKNOWN/PAIR")

    def test_symbol_mapper_unknown_mt5_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown MT5 symbol"):
            self.mapper.to_python("UNKNOWNXYZ")

    def test_symbol_mapper_extra_pairs(self) -> None:
        mapper = SymbolMapper(extra={"DOGE/USDT": "DOGEUSD"})
        assert mapper.to_mt5("DOGE/USDT") == "DOGEUSD"
        assert mapper.to_python("DOGEUSD") == "DOGE/USDT"

    def test_symbol_mapper_extra_does_not_remove_defaults(self) -> None:
        mapper = SymbolMapper(extra={"DOGE/USDT": "DOGEUSD"})
        assert mapper.to_mt5("BTC/USDT") == "BTCUSD"


# ---------------------------------------------------------------------------
# TestMT5BridgeCapabilities
# ---------------------------------------------------------------------------


class TestMT5BridgeCapabilities:
    def test_capabilities(self) -> None:
        bridge = MT5Bridge()
        caps = bridge.capabilities
        assert caps.broker_name == "mt5"
        assert caps.is_paper is False
        assert caps.requires_symbol_mapping is True
        assert caps.supports_native_stops is True
        assert caps.supports_native_take_profit is True
        assert caps.supports_partial_fills is False


# ---------------------------------------------------------------------------
# TestMT5BridgePing
# ---------------------------------------------------------------------------


class TestMT5BridgePing:
    async def test_ping_no_heartbeat(self) -> None:
        bridge = _make_bridge()
        assert await bridge.ping() is False

    async def test_ping_recent_heartbeat(self) -> None:
        bridge = _make_bridge()
        bridge._last_heartbeat_at = time.monotonic()  # just now
        assert await bridge.ping() is True

    async def test_ping_stale_heartbeat(self) -> None:
        bridge = _make_bridge()
        bridge._last_heartbeat_at = time.monotonic() - 10.0  # 10 seconds ago
        assert await bridge.ping() is False


# ---------------------------------------------------------------------------
# TestMT5BridgePlaceOrder
# ---------------------------------------------------------------------------


class TestMT5BridgePlaceOrder:
    def _fill_response(self) -> dict:
        return {
            "status": "ok",
            "order_id": "987654321",
            "fill_price": 42050.0,
            "fill_quantity": 0.01,
        }

    async def test_place_order_success(self) -> None:
        bridge = _make_bridge()
        assessment = _make_assessment()
        with patch.object(bridge, "_enqueue_and_wait", return_value=self._fill_response()):
            result = await bridge.place_order(assessment)
        assert result.status == OrderStatus.FILLED

    async def test_place_order_is_paper_false(self) -> None:
        bridge = _make_bridge()
        assessment = _make_assessment()
        with patch.object(bridge, "_enqueue_and_wait", return_value=self._fill_response()):
            result = await bridge.place_order(assessment)
        assert result.is_paper is False

    async def test_place_order_links_ids(self) -> None:
        bridge = _make_bridge()
        assessment = _make_assessment()
        with patch.object(bridge, "_enqueue_and_wait", return_value=self._fill_response()):
            result = await bridge.place_order(assessment)
        assert result.proposal_id == assessment.proposal_id
        assert result.assessment_id == assessment.assessment_id

    async def test_place_order_rejected(self) -> None:
        bridge = _make_bridge()
        assessment = _make_assessment()
        error_resp = {
            "status": "error",
            "error_code": 10006,
            "error_message": "No money",
        }
        with patch.object(bridge, "_enqueue_and_wait", return_value=error_resp):
            with pytest.raises(OrderRejectedError, match="No money"):
                await bridge.place_order(assessment)

    async def test_place_order_fill_price_stored(self) -> None:
        bridge = _make_bridge()
        assessment = _make_assessment()
        with patch.object(bridge, "_enqueue_and_wait", return_value=self._fill_response()):
            result = await bridge.place_order(assessment)
        assert result.average_fill_price == Decimal("42050.0")

    async def test_place_order_exchange_order_id(self) -> None:
        bridge = _make_bridge()
        assessment = _make_assessment()
        with patch.object(bridge, "_enqueue_and_wait", return_value=self._fill_response()):
            result = await bridge.place_order(assessment)
        assert result.exchange_order_id == "987654321"

    async def test_place_order_computes_volume_from_size_when_qty_none(self) -> None:
        """When approved_quantity is None, volume is computed from effective_size_usd / price."""
        bridge = _make_bridge()
        assessment = _make_assessment(approved_quantity=None)
        with patch.object(
            bridge, "_enqueue_and_wait", return_value=self._fill_response()
        ) as mock_cmd:
            await bridge.place_order(assessment)
        called_command = mock_cmd.call_args[0][0]
        assert called_command["volume"] > 0


# ---------------------------------------------------------------------------
# TestMT5BridgeGetBalance
# ---------------------------------------------------------------------------


class TestMT5BridgeGetBalance:
    async def test_get_balance(self) -> None:
        bridge = _make_bridge()
        bridge._cached_state = {
            "balance": {
                "total_equity": 10000.0,
                "free_margin": 8500.0,
                "used_margin": 1500.0,
                "currency": "USD",
            }
        }
        balance = await bridge.get_balance()
        assert balance.total_equity_usd == Decimal("10000.0")
        assert balance.free_margin_usd == Decimal("8500.0")
        assert balance.used_margin_usd == Decimal("1500.0")
        assert balance.currency == "USD"

    async def test_get_balance_raises_when_no_cache(self) -> None:
        bridge = _make_bridge()
        with pytest.raises(ExchangeConnectionError, match="no heartbeat"):
            await bridge.get_balance()


# ---------------------------------------------------------------------------
# TestMT5BridgeGetPositions
# ---------------------------------------------------------------------------


class TestMT5BridgeGetPositions:
    async def test_get_positions_empty(self) -> None:
        bridge = _make_bridge()
        bridge._cached_state = {"positions": []}
        positions = await bridge.get_positions()
        assert positions == []

    async def test_get_positions_no_cache_returns_empty(self) -> None:
        bridge = _make_bridge()
        positions = await bridge.get_positions()
        assert positions == []

    async def test_get_positions_mapped(self) -> None:
        """MT5 symbol 'BTCUSD' should be mapped back to Python 'BTC/USDT'."""
        bridge = _make_bridge()
        bridge._cached_state = {
            "positions": [
                {
                    "ticket": "111222",
                    "symbol": "BTCUSD",
                    "side": "buy",
                    "quantity": 0.01,
                    "entry_price": 42000.0,
                    "current_price": 42500.0,
                    "unrealised_pnl": 5.0,
                    "stop_loss": 41000.0,
                    "take_profit": 44000.0,
                }
            ]
        }
        positions = await bridge.get_positions()

        assert len(positions) == 1
        pos = positions[0]
        assert pos.symbol == "BTC/USDT"
        assert pos.side == "buy"
        assert pos.quantity == Decimal("0.01")
        assert pos.entry_price == Decimal("42000.0")
        assert pos.stop_loss == Decimal("41000.0")
        assert pos.take_profit == Decimal("44000.0")


# ---------------------------------------------------------------------------
# TestMT5BridgeCancelOrder
# ---------------------------------------------------------------------------


class TestMT5BridgeCancelOrder:
    async def test_cancel_order_success(self) -> None:
        bridge = _make_bridge()
        with patch.object(
            bridge, "_enqueue_and_wait", return_value={"status": "ok", "order_id": "123"}
        ):
            result = await bridge.cancel_order("123", "BTC/USDT")
        assert result is True

    async def test_cancel_order_not_found(self) -> None:
        bridge = _make_bridge()
        with patch.object(
            bridge,
            "_enqueue_and_wait",
            return_value={"status": "not_found", "order_id": "999"},
        ):
            result = await bridge.cancel_order("999", "BTC/USDT")
        assert result is False


# ---------------------------------------------------------------------------
# TestMT5BridgeHTTPHandlers
# ---------------------------------------------------------------------------


class TestMT5BridgeHTTPHandlers:
    async def test_handle_state_updates_cache(self) -> None:
        bridge = _make_bridge()
        body = {
            "balance": {
                "total_equity": 10000.0,
                "free_margin": 8000.0,
                "used_margin": 2000.0,
                "currency": "USD",
            },
            "positions": [],
        }
        request = _make_request(body)
        before = time.monotonic()
        response = await bridge._handle_state(request)

        assert json.loads(response.text)["status"] == "ok"
        assert bridge._cached_state == body
        assert bridge._last_heartbeat_at is not None
        assert bridge._last_heartbeat_at >= before

    async def test_handle_state_invalid_json_returns_400(self) -> None:
        bridge = _make_bridge()
        request = AsyncMock()
        request.json = AsyncMock(side_effect=ValueError("bad json"))
        response = await bridge._handle_state(request)
        assert response.status == 400

    async def test_handle_command_empty_queue(self) -> None:
        bridge = _make_bridge()
        response = await bridge._handle_command(MagicMock())
        data = json.loads(response.text)
        assert data["status"] == "empty"

    async def test_handle_command_returns_queued_command(self) -> None:
        bridge = _make_bridge()
        await bridge._command_queue.put({"action": "PING", "command_id": "abc-123"})
        response = await bridge._handle_command(MagicMock())
        data = json.loads(response.text)
        assert data["status"] == "ok"
        assert data["action"] == "PING"
        assert data["command_id"] == "abc-123"

    async def test_handle_command_dequeues_one(self) -> None:
        bridge = _make_bridge()
        await bridge._command_queue.put({"action": "PING", "command_id": "first"})
        await bridge._command_queue.put({"action": "PING", "command_id": "second"})
        await bridge._handle_command(MagicMock())
        assert bridge._command_queue.qsize() == 1

    async def test_handle_result_resolves_future(self) -> None:
        bridge = _make_bridge()
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        bridge._result_futures["cmd-999"] = future

        body = {"command_id": "cmd-999", "status": "ok", "order_id": "456"}
        request = _make_request(body)
        await bridge._handle_result(request)

        assert future.done()
        assert future.result()["order_id"] == "456"

    async def test_handle_result_unknown_id_does_not_raise(self) -> None:
        bridge = _make_bridge()
        body = {"command_id": "nonexistent-id", "status": "ok"}
        request = _make_request(body)
        # Should not raise
        response = await bridge._handle_result(request)
        assert json.loads(response.text)["status"] == "ok"

    async def test_handle_result_invalid_json_returns_400(self) -> None:
        bridge = _make_bridge()
        request = AsyncMock()
        request.json = AsyncMock(side_effect=ValueError("bad json"))
        response = await bridge._handle_result(request)
        assert response.status == 400


# ---------------------------------------------------------------------------
# TestMT5BridgeEnqueueAndWait
# ---------------------------------------------------------------------------


class TestMT5BridgeEnqueueAndWait:
    async def test_timeout_raises_exchange_connection_error(self) -> None:
        """If the EA never posts a result, ExchangeConnectionError is raised."""
        bridge = _make_bridge()
        bridge._timeout_ms = 100  # 100 ms — fast for tests

        with pytest.raises(ExchangeConnectionError, match="did not respond"):
            await bridge._enqueue_and_wait({"action": "PING"})

    async def test_timeout_cleans_up_future(self) -> None:
        """After timeout, the future is removed from _result_futures."""
        bridge = _make_bridge()
        bridge._timeout_ms = 100

        try:
            await bridge._enqueue_and_wait({"action": "PING"})
        except ExchangeConnectionError:
            pass

        assert len(bridge._result_futures) == 0

    async def test_command_placed_on_queue(self) -> None:
        """The command (with command_id injected) ends up in _command_queue."""
        bridge = _make_bridge()
        bridge._timeout_ms = 100

        try:
            await bridge._enqueue_and_wait({"action": "PING"})
        except ExchangeConnectionError:
            pass

        # Command was put on the queue
        assert not bridge._command_queue.empty()
        cmd = bridge._command_queue.get_nowait()
        assert cmd["action"] == "PING"
        assert "command_id" in cmd

    async def test_success_when_future_resolved(self) -> None:
        """If the EA resolves the future, _enqueue_and_wait returns the result."""
        bridge = _make_bridge()
        bridge._timeout_ms = 2000  # generous timeout

        async def _resolve() -> None:
            # Wait for command to appear on the queue
            cmd = await bridge._command_queue.get()
            cmd_id = cmd["command_id"]
            future = bridge._result_futures.get(cmd_id)
            if future is not None:
                future.set_result({"status": "ok", "command_id": cmd_id})

        task = asyncio.create_task(_resolve())
        result = await bridge._enqueue_and_wait({"action": "PING"})
        await task

        assert result["status"] == "ok"

    async def test_command_id_injected(self) -> None:
        """Each call gets a unique command_id merged into the command dict."""
        bridge = _make_bridge()
        bridge._timeout_ms = 100

        try:
            await bridge._enqueue_and_wait({"action": "PING"})
        except ExchangeConnectionError:
            pass

        cmd = bridge._command_queue.get_nowait()
        assert "command_id" in cmd
        # Verify it looks like a UUID (36 chars with hyphens)
        assert len(cmd["command_id"]) == 36


# ---------------------------------------------------------------------------
# TestMT5BridgeVolumeClamp
# ---------------------------------------------------------------------------


class TestMT5BridgeVolumeClamp:
    """_clamp_volume enforces broker lot-size constraints (step / min / max)."""

    def setup_method(self) -> None:
        self.bridge = _make_bridge()
        # Use Tickmill BTCUSD defaults (already the MT5Settings defaults)
        self.bridge._min_volume = 0.01
        self.bridge._max_volume = 10.0
        self.bridge._volume_step = 0.01

    def test_tiny_volume_rounds_up_to_minimum(self) -> None:
        """0.000142 lots (e.g. $10 / $70,000 BTC) → clamped to 0.01."""
        result = self.bridge._clamp_volume(0.000142)
        assert result == 0.01

    def test_volume_at_minimum_unchanged(self) -> None:
        result = self.bridge._clamp_volume(0.01)
        assert result == 0.01

    def test_volume_above_maximum_clamped(self) -> None:
        result = self.bridge._clamp_volume(15.0)
        assert result == 10.0

    def test_volume_at_maximum_unchanged(self) -> None:
        result = self.bridge._clamp_volume(10.0)
        assert result == 10.0

    def test_volume_rounds_to_nearest_step(self) -> None:
        """0.01499 should round to 0.01 (nearest step down)."""
        result = self.bridge._clamp_volume(0.01499)
        assert result == 0.01

    def test_volume_rounds_up_to_next_step(self) -> None:
        """0.015 should round to 0.02 (nearest step up)."""
        result = self.bridge._clamp_volume(0.015)
        assert result == 0.02

    def test_volume_already_on_step_unchanged(self) -> None:
        result = self.bridge._clamp_volume(0.05)
        assert result == 0.05

    def test_volume_mid_range_rounded_correctly(self) -> None:
        """1.234 lots → rounds to 1.23."""
        result = self.bridge._clamp_volume(1.234)
        assert result == 1.23

    async def test_resolve_volume_clamps_computed_quantity(self) -> None:
        """When approved_quantity is None, computed volume is clamped to minimum."""
        bridge = self.bridge
        # $10 at $70,000/BTC → 0.000142 lots → must clamp to 0.01
        technical = _make_technical(price=70_000.0)
        assessment = _make_assessment(
            approved_usd=10.0,
            approved_quantity=None,
            technical=technical,
        )
        volume = bridge._resolve_volume(assessment)
        assert float(volume) == 0.01

    async def test_resolve_volume_clamps_approved_quantity(self) -> None:
        """Even pre-approved tiny quantities are clamped to the broker minimum."""
        bridge = self.bridge
        assessment = _make_assessment(approved_quantity=Decimal("0.000001"))
        volume = bridge._resolve_volume(assessment)
        assert float(volume) == 0.01

    async def test_place_order_sends_clamped_volume(self) -> None:
        """place_order() sends the clamped volume in the command dict."""
        bridge = self.bridge
        # Use a $10 order at $70,000 BTC — raw=0.000142, clamped=0.01
        technical = _make_technical(price=70_000.0)
        assessment = _make_assessment(
            approved_usd=10.0,
            approved_quantity=None,
            technical=technical,
        )

        fill_response = {
            "status": "ok",
            "order_id": "111",
            "fill_price": 70_000.0,
            "fill_quantity": 0.01,
        }
        with patch.object(bridge, "_enqueue_and_wait", return_value=fill_response) as mock_cmd:
            await bridge.place_order(assessment)

        sent_volume = mock_cmd.call_args[0][0]["volume"]
        assert sent_volume == 0.01
