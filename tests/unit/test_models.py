"""
tests/unit/test_models.py
--------------------------
Unit tests for all core Pydantic models.

These tests have zero I/O — no Redis, no DB, no exchange.
They verify that the contracts between agents are correctly defined
and that validation logic catches bad data at the boundary.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

import pytest

from core.models.market import OHLCVCandle, OrderBook, OrderBookLevel
from core.models.signals import (
    IndicatorName,
    IndicatorReading,
    SignalDirection,
    TechnicalSignal,
)
from core.models.system import AgentHeartbeat, AgentStatus

# ---------------------------------------------------------------------------
# OHLCVCandle
# ---------------------------------------------------------------------------


class TestOHLCVCandle:
    def _make_candle(self, **overrides) -> OHLCVCandle:
        defaults = dict(
            symbol="BTC/USDT",
            timeframe="1m",
            timestamp=datetime(2025, 1, 1, 12, 0, tzinfo=UTC),
            open=Decimal("42000"),
            high=Decimal("42500"),
            low=Decimal("41800"),
            close=Decimal("42200"),
            volume=Decimal("100"),
        )
        defaults.update(overrides)
        return OHLCVCandle(**defaults)

    def test_valid_candle_constructs(self):
        candle = self._make_candle()
        assert candle.symbol == "BTC/USDT"
        assert candle.close == Decimal("42200")

    def test_channel_key_replaces_slash(self):
        candle = self._make_candle()
        assert candle.channel_key == "market.ohlcv.BTC-USDT.1m"

    def test_bullish_candle(self):
        candle = self._make_candle(open=Decimal("42000"), close=Decimal("42500"))
        assert candle.is_bullish is True

    def test_bearish_candle(self):
        candle = self._make_candle(open=Decimal("42500"), close=Decimal("42000"))
        assert candle.is_bullish is False

    def test_invalid_high_less_than_low(self):
        with pytest.raises(ValueError, match="high.*low"):
            self._make_candle(high=Decimal("41000"), low=Decimal("42000"))

    def test_invalid_high_less_than_close(self):
        with pytest.raises(ValueError):
            self._make_candle(
                open=Decimal("42000"),
                high=Decimal("42100"),
                low=Decimal("41800"),
                close=Decimal("42500"),  # close > high
            )

    def test_negative_volume_rejected(self):
        with pytest.raises(ValueError, match="volume"):
            self._make_candle(volume=Decimal("-1"))

    def test_from_ccxt_millisecond_timestamp(self):
        raw = [1_700_000_000_000, 42000.0, 42500.0, 41800.0, 42200.0, 100.0]
        candle = OHLCVCandle.from_ccxt(raw, "BTC/USDT", "1m")
        assert candle.symbol == "BTC/USDT"
        assert candle.open == Decimal("42000")
        assert candle.timestamp.tzinfo is not None  # always UTC

    def test_immutability(self):
        candle = self._make_candle()
        with pytest.raises(Exception):
            candle.close = Decimal("99999")  # frozen model

    def test_json_roundtrip(self):
        candle = self._make_candle()
        restored = OHLCVCandle.from_json(candle.to_json())
        assert restored.close == candle.close
        assert restored.symbol == candle.symbol
        assert restored.timestamp == candle.timestamp


# ---------------------------------------------------------------------------
# OrderBook
# ---------------------------------------------------------------------------


class TestOrderBook:
    def _make_book(self) -> OrderBook:
        return OrderBook(
            symbol="BTC/USDT",
            timestamp=datetime(2025, 1, 1, 12, 0, tzinfo=UTC),
            bids=[
                OrderBookLevel(price=Decimal("42000"), quantity=Decimal("1.5")),
                OrderBookLevel(price=Decimal("41990"), quantity=Decimal("2.0")),
            ],
            asks=[
                OrderBookLevel(price=Decimal("42010"), quantity=Decimal("1.0")),
                OrderBookLevel(price=Decimal("42020"), quantity=Decimal("0.5")),
            ],
        )

    def test_best_bid_ask(self):
        book = self._make_book()
        assert book.best_bid == Decimal("42000")
        assert book.best_ask == Decimal("42010")

    def test_mid_price(self):
        book = self._make_book()
        assert book.mid_price == Decimal("42005")

    def test_spread(self):
        book = self._make_book()
        assert book.spread == Decimal("10")

    def test_empty_book_returns_none(self):
        book = OrderBook(
            symbol="BTC/USDT",
            timestamp=datetime.now(UTC),
            bids=[],
            asks=[],
        )
        assert book.best_bid is None
        assert book.mid_price is None
        assert book.spread is None

    def test_channel_key(self):
        book = self._make_book()
        assert book.channel_key == "market.orderbook.BTC-USDT"


# ---------------------------------------------------------------------------
# TechnicalSignal
# ---------------------------------------------------------------------------


class TestTechnicalSignal:
    def _make_signal(self, direction=SignalDirection.BUY, confidence=0.75) -> TechnicalSignal:
        return TechnicalSignal(
            symbol="BTC/USDT",
            timeframe="1m",
            expires_at=datetime(2099, 1, 1, tzinfo=UTC),
            direction=direction,
            confidence=confidence,
            price=42000.0,
            indicators=[
                IndicatorReading(
                    name=IndicatorName.RSI,
                    value=28.5,
                    signal=SignalDirection.BUY,
                    weight=0.4,
                )
            ],
        )

    def test_signal_constructs(self):
        signal = self._make_signal()
        assert signal.direction == SignalDirection.BUY
        assert isinstance(signal.signal_id, UUID)

    def test_confidence_range_enforced(self):
        with pytest.raises(ValueError):
            self._make_signal(confidence=1.5)
        with pytest.raises(ValueError):
            self._make_signal(confidence=-0.1)

    def test_is_actionable_non_neutral(self):
        signal = self._make_signal(direction=SignalDirection.BUY)
        assert signal.is_actionable is True

    def test_neutral_not_actionable(self):
        signal = self._make_signal(direction=SignalDirection.NEUTRAL)
        assert signal.is_actionable is False

    def test_expired_signal_not_actionable(self):
        signal = TechnicalSignal(
            symbol="BTC/USDT",
            timeframe="1m",
            expires_at=datetime(2000, 1, 1, tzinfo=UTC),  # past
            direction=SignalDirection.BUY,
            confidence=0.8,
            price=42000.0,
        )
        assert signal.is_expired is True
        assert signal.is_actionable is False

    def test_scalar_direction_mapping(self):
        assert self._make_signal(SignalDirection.STRONG_BUY).scalar_direction() == 1.0
        assert self._make_signal(SignalDirection.BUY).scalar_direction() == 0.5
        assert self._make_signal(SignalDirection.NEUTRAL).scalar_direction() == 0.0
        assert self._make_signal(SignalDirection.SELL).scalar_direction() == -0.5
        assert self._make_signal(SignalDirection.STRONG_SELL).scalar_direction() == -1.0

    def test_channel_key(self):
        signal = self._make_signal()
        assert signal.channel_key == "signal.technical.BTC-USDT"

    def test_json_roundtrip(self):
        signal = self._make_signal()
        restored = TechnicalSignal.model_validate_json(signal.to_json())
        assert restored.signal_id == signal.signal_id
        assert restored.direction == signal.direction


# ---------------------------------------------------------------------------
# AgentHeartbeat
# ---------------------------------------------------------------------------


class TestAgentHeartbeat:
    def test_heartbeat_constructs(self):
        hb = AgentHeartbeat(
            agent_name="market_data_agent",
            status=AgentStatus.RUNNING,
            messages_processed=100,
            uptime_seconds=300.0,
        )
        assert hb.agent_name == "market_data_agent"
        assert hb.channel_key == "system.heartbeat"

    def test_json_roundtrip(self):
        hb = AgentHeartbeat(
            agent_name="test_agent",
            status=AgentStatus.RUNNING,
        )
        restored = AgentHeartbeat.model_validate_json(hb.to_json())
        assert restored.agent_name == hb.agent_name
        assert restored.status == hb.status
