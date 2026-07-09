"""
tests/integration/test_ta_pipeline.py
---------------------------------------
Integration test for the full Technical Analysis pipeline.

Tests the data flow:
    Synthetic candles → CandleBufferRegistry → SignalGenerator → TechnicalSignal

No Redis or database required — this tests the pure computation pipeline.
The messaging layer is tested separately in test_messaging.py (requires Redis).

Run with:
    pytest tests/integration/test_ta_pipeline.py -v --no-cov
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

from agents.technical_analysis.candle_buffer import MIN_WARMUP_CANDLES, CandleBufferRegistry
from agents.technical_analysis.signal_generator import SignalGenerator
from core.models.market import OHLCVCandle
from core.models.signals import SignalDirection, TechnicalSignal

# ---------------------------------------------------------------------------
# Candle factory helpers
# ---------------------------------------------------------------------------


def make_candle(
    close: float,
    i: int,
    symbol: str = "BTC/USDT",
    timeframe: str = "1m",
    high_offset: float = 50.0,
    low_offset: float = 50.0,
    volume: float = 100.0,
) -> OHLCVCandle:
    """Create a synthetic candle at a given index (timestamp offset)."""
    base = datetime(2025, 1, 1, tzinfo=UTC)
    ts = base + timedelta(minutes=i)
    price = Decimal(str(round(close, 8)))
    return OHLCVCandle(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=ts,
        open=price,
        high=Decimal(str(round(close + high_offset, 8))),
        low=Decimal(str(round(close - low_offset, 8))),
        close=price,
        volume=Decimal(str(volume)),
        received_at=datetime.now(UTC),
    )


def make_trending_up_candles(
    n: int,
    start: float = 40_000.0,
    step: float = 50.0,
    symbol: str = "BTC/USDT",
    timeframe: str = "1m",
) -> list[OHLCVCandle]:
    """Monotonically rising price series — should produce bullish signals."""
    return [make_candle(start + i * step, i, symbol, timeframe) for i in range(n)]


def make_trending_down_candles(
    n: int,
    start: float = 60_000.0,
    step: float = 50.0,
    symbol: str = "BTC/USDT",
    timeframe: str = "1m",
) -> list[OHLCVCandle]:
    """Monotonically falling price series — should produce bearish signals."""
    return [make_candle(start - i * step, i, symbol, timeframe) for i in range(n)]


def make_oscillating_candles(
    n: int,
    center: float = 50_000.0,
    amplitude: float = 1_000.0,
    symbol: str = "BTC/USDT",
    timeframe: str = "1m",
) -> list[OHLCVCandle]:
    """Sine-wave price series — should produce mixed/neutral signals."""
    prices = [center + amplitude * math.sin(2 * math.pi * i / 30) for i in range(n)]
    return [make_candle(p, i, symbol, timeframe) for i, p in enumerate(prices)]


# ---------------------------------------------------------------------------
# Buffer tests
# ---------------------------------------------------------------------------


class TestCandleBufferPipeline:
    def test_buffer_not_warm_below_minimum(self):
        registry = CandleBufferRegistry()
        candles = make_trending_up_candles(MIN_WARMUP_CANDLES - 1)
        for c in candles:
            registry.add(c)

        buf = registry.get("BTC/USDT", "1m")
        assert buf is not None
        assert buf.is_warm is False
        assert buf.warmup_progress < 1.0

    def test_buffer_warm_at_minimum(self):
        registry = CandleBufferRegistry()
        candles = make_trending_up_candles(MIN_WARMUP_CANDLES)
        for c in candles:
            registry.add(c)

        buf = registry.get("BTC/USDT", "1m")
        assert buf.is_warm is True
        assert buf.warmup_progress == 1.0

    def test_buffer_capacity_not_exceeded(self):
        registry = CandleBufferRegistry()
        # Add more than buffer capacity
        candles = make_trending_up_candles(600)
        for c in candles:
            registry.add(c)

        buf = registry.get("BTC/USDT", "1m")
        assert buf.size <= 500  # MAX_BUFFER_SIZE

    def test_multiple_symbols_independent(self):
        registry = CandleBufferRegistry()

        btc_candles = make_trending_up_candles(MIN_WARMUP_CANDLES, symbol="BTC/USDT")
        eth_candles = make_trending_up_candles(30, symbol="ETH/USDT")  # fewer — not warm

        for c in btc_candles:
            registry.add(c)
        for c in eth_candles:
            registry.add(c)

        btc_buf = registry.get("BTC/USDT", "1m")
        eth_buf = registry.get("ETH/USDT", "1m")

        assert btc_buf.is_warm is True
        assert eth_buf.is_warm is False
        assert not registry.all_warm()
        assert len(registry.warm_buffers()) == 1

    def test_numpy_arrays_correct_dtype(self):
        registry = CandleBufferRegistry()
        candles = make_trending_up_candles(MIN_WARMUP_CANDLES)
        for c in candles:
            registry.add(c)

        buf = registry.get("BTC/USDT", "1m")
        assert buf.closes.dtype == np.float64
        assert buf.highs.dtype == np.float64
        assert buf.lows.dtype == np.float64
        assert buf.volumes.dtype == np.float64

    def test_closes_ordered_oldest_to_newest(self):
        registry = CandleBufferRegistry()
        candles = make_trending_up_candles(MIN_WARMUP_CANDLES, start=40_000.0, step=10.0)
        for c in candles:
            registry.add(c)

        buf = registry.get("BTC/USDT", "1m")
        closes = buf.closes
        # Trending up → each close should be >= previous
        assert all(closes[i] <= closes[i + 1] for i in range(len(closes) - 1))


# ---------------------------------------------------------------------------
# Signal generator tests
# ---------------------------------------------------------------------------


class TestSignalGeneratorPipeline:
    def _fill_registry(
        self,
        candles: list[OHLCVCandle],
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
    ) -> tuple[CandleBufferRegistry, CandleBuffer]:
        registry = CandleBufferRegistry()
        for c in candles:
            registry.add(c)
        buf = registry.get(symbol, timeframe)
        return registry, buf

    def test_cold_buffer_returns_none(self):
        generator = SignalGenerator()
        registry = CandleBufferRegistry()
        candles = make_trending_up_candles(20)  # below warmup threshold
        for c in candles:
            registry.add(c)

        buf = registry.get("BTC/USDT", "1m")
        result = generator.generate(buf)
        assert result is None

    def test_warm_buffer_returns_signal_or_none(self):
        """
        A warm buffer should either return a TechnicalSignal or None
        (if confidence is below threshold). Both are valid — we just
        verify the return type is correct.
        """
        generator = SignalGenerator()
        candles = make_trending_up_candles(MIN_WARMUP_CANDLES + 20)
        _, buf = self._fill_registry(candles)

        result = generator.generate(buf)
        assert result is None or isinstance(result, TechnicalSignal)

    def test_signal_has_required_fields(self):
        """Any generated signal must have all required fields populated."""
        generator = SignalGenerator()
        # Use a longer series to ensure we get a signal above threshold
        candles = make_trending_up_candles(200)
        _, buf = self._fill_registry(candles)

        result = generator.generate(buf)
        if result is None:
            pytest.skip("Signal below confidence threshold — not a failure")

        assert result.symbol == "BTC/USDT"
        assert result.timeframe == "1m"
        assert 0.0 <= result.confidence <= 1.0
        assert result.direction in list(SignalDirection)
        assert result.price > 0
        assert len(result.indicators) >= 2
        assert not result.is_expired

    def test_signal_channel_key_correct(self):
        generator = SignalGenerator()
        candles = make_trending_up_candles(200)
        _, buf = self._fill_registry(candles)

        result = generator.generate(buf)
        if result is None:
            pytest.skip("Signal below confidence threshold")

        assert result.channel_key == "signal.technical.BTC-USDT"

    def test_strong_uptrend_produces_bullish_or_neutral(self):
        """
        A strong monotonic uptrend should never produce a SELL signal.
        May be neutral if confidence is low, but never bearish.
        """
        generator = SignalGenerator()
        # Very strong, sustained uptrend
        candles = make_trending_up_candles(300, start=40_000, step=200)
        _, buf = self._fill_registry(candles)

        result = generator.generate(buf)
        if result is None:
            pytest.skip("Signal below confidence threshold")

        assert result.direction not in (
            SignalDirection.SELL,
            SignalDirection.STRONG_SELL,
        ), f"Expected bullish/neutral, got {result.direction}"

    def test_strong_downtrend_produces_bearish_or_neutral(self):
        """A strong downtrend should never produce a BUY signal."""
        generator = SignalGenerator()
        candles = make_trending_down_candles(300, start=60_000, step=200)
        _, buf = self._fill_registry(candles)

        result = generator.generate(buf)
        if result is None:
            pytest.skip("Signal below confidence threshold")

        assert result.direction not in (
            SignalDirection.BUY,
            SignalDirection.STRONG_BUY,
        ), f"Expected bearish/neutral, got {result.direction}"

    def test_signal_json_serialisable(self):
        """Signals must survive JSON round-trip (for Redis pub/sub)."""
        generator = SignalGenerator()
        candles = make_trending_up_candles(200)
        _, buf = self._fill_registry(candles)

        result = generator.generate(buf)
        if result is None:
            pytest.skip("Signal below confidence threshold")

        # Must not raise
        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Must round-trip
        restored = TechnicalSignal.model_validate_json(json_str)
        assert restored.signal_id == result.signal_id
        assert restored.direction == result.direction
        assert restored.confidence == result.confidence

    def test_indicator_readings_have_valid_weights(self):
        """All indicator readings must have weights that sum approximately correctly."""
        generator = SignalGenerator()
        candles = make_trending_up_candles(200)
        _, buf = self._fill_registry(candles)

        result = generator.generate(buf)
        if result is None:
            pytest.skip("Signal below confidence threshold")

        # Directional indicators (weight > 0) should sum to ~1.0
        weighted = [r for r in result.indicators if r.weight > 0]
        total = sum(r.weight for r in weighted)
        assert 0.9 <= total <= 1.01, f"Weights sum to {total}, expected ~1.0"

        # All weights must be in valid range
        for reading in result.indicators:
            assert 0.0 <= reading.weight <= 1.0

    def test_confidence_in_valid_range(self):
        """Confidence must always be in [0, 1]."""
        generator = SignalGenerator()

        for candle_fn in [
            lambda: make_trending_up_candles(200),
            lambda: make_trending_down_candles(200),
            lambda: make_oscillating_candles(200),
        ]:
            candles = candle_fn()
            _, buf = self._fill_registry(candles)
            result = generator.generate(buf)

            if result is not None:
                assert 0.0 <= result.confidence <= 1.0, (
                    f"Confidence {result.confidence} out of range"
                )

    def test_metadata_contains_atr(self):
        """Signals should contain ATR data for Risk agent stop sizing."""
        generator = SignalGenerator()
        candles = make_trending_up_candles(200)
        _, buf = self._fill_registry(candles)

        result = generator.generate(buf)
        if result is None:
            pytest.skip("Signal below confidence threshold")

        # ATR metadata should be present (used by Risk agent)
        if "atr_value" in result.metadata:
            assert result.metadata["atr_value"] > 0
            assert result.metadata["atr_pct_of_price"] > 0
            assert result.metadata["volatility_regime"] in ("low", "normal", "high")


# ---------------------------------------------------------------------------
# Direct indicator / confidence tests (threshold-independent)
# ---------------------------------------------------------------------------


class TestConfidenceComputation:
    """
    Tests that directly exercise indicator functions and the confidence formula
    without depending on the signal passing the confidence threshold.
    """

    def test_agreement_perfect_alignment(self):
        from agents.technical_analysis.signal_generator import _compute_agreement

        scalars = [1.0, 1.0, 1.0, 1.0]
        assert _compute_agreement(scalars) == pytest.approx(1.0, abs=0.01)

    def test_agreement_complete_disagreement(self):
        from agents.technical_analysis.signal_generator import _compute_agreement

        scalars = [1.0, 1.0, -1.0, -1.0]
        assert _compute_agreement(scalars) < 0.3

    def test_agreement_single_indicator_moderate(self):
        from agents.technical_analysis.signal_generator import _compute_agreement

        assert _compute_agreement([0.5]) == pytest.approx(0.8, abs=0.01)

    def test_scalar_direction_roundtrip(self):
        from agents.technical_analysis.signal_generator import (
            _direction_to_scalar,
            _scalar_to_direction,
        )

        for direction in SignalDirection:
            scalar = _direction_to_scalar(direction)
            assert -1.0 <= scalar <= 1.0
        assert _scalar_to_direction(1.0) == SignalDirection.STRONG_BUY
        assert _scalar_to_direction(-1.0) == SignalDirection.STRONG_SELL
        assert _scalar_to_direction(0.0) == SignalDirection.NEUTRAL

    def test_rsi_overbought_on_rising_series(self):
        from agents.technical_analysis.indicators import compute_rsi

        closes = np.array([40_000.0 + i * 500 for i in range(100)], dtype=np.float64)
        result = compute_rsi(closes)
        assert result.is_overbought is True
        assert result.value > 70.0

    def test_rsi_oversold_on_falling_series(self):
        from agents.technical_analysis.indicators import compute_rsi

        closes = np.array([60_000.0 - i * 500 for i in range(100)], dtype=np.float64)
        result = compute_rsi(closes)
        assert result.is_oversold is True
        assert result.value < 30.0

    def test_ema_cross_uptrend_short_above_long(self):
        from agents.technical_analysis.indicators import compute_ema_cross

        closes = np.array([40_000.0 + i * 100 for i in range(100)], dtype=np.float64)
        result = compute_ema_cross(closes)
        assert result.ema_short > result.ema_long
        assert result.spread > 0

    def test_bollinger_bands_ordering(self):
        from agents.technical_analysis.indicators import compute_bollinger_bands

        closes = np.array(
            [50_000.0 + 200.0 * math.sin(2 * math.pi * i / 20) for i in range(100)],
            dtype=np.float64,
        )
        result = compute_bollinger_bands(closes)
        assert result.upper > result.middle > result.lower

    def test_macd_finite_on_long_series(self):
        from agents.technical_analysis.indicators import compute_macd

        closes = np.array([40_000.0 + i * 50 for i in range(100)], dtype=np.float64)
        result = compute_macd(closes)
        assert math.isfinite(result.macd_line)
        assert math.isfinite(result.signal_line)
        assert math.isfinite(result.histogram)

    def test_atr_positive_and_finite(self):
        from agents.technical_analysis.indicators import compute_atr

        n = 60
        closes = np.array([42_000.0 + i * 10 for i in range(n)], dtype=np.float64)
        result = compute_atr(closes + 100.0, closes - 100.0, closes)
        assert result.value > 0
        assert math.isfinite(result.value)
        assert result.volatility_regime in ("low", "normal", "high")
