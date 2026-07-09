"""
agents/technical_analysis/signal_generator.py
-----------------------------------------------
Signal generator — converts indicator readings into a single weighted,
confidence-scored TechnicalSignal.

This is the core reasoning layer of the TA agent. It combines multiple
independent indicator signals using a configurable weight scheme and
produces a final direction + confidence score that the Decision agent
uses as one input to its trade proposal.

Architecture of signal generation:
    1. Compute all indicators from the candle buffer.
    2. Each indicator contributes a scalar direction [-1, +1] × its weight.
    3. The weighted average gives a composite score in [-1, +1].
    4. Confidence is derived from the agreement between indicators
       (high agreement = high confidence) and the score magnitude.
    5. The composite score is mapped to a SignalDirection enum.

Design decisions:
- Indicator weights are defined here, not in config. They represent
  the relative importance of each indicator in our signal model.
  Changing weights is a strategy decision, not an ops decision.
  TODO: expose weights as config or ML-tunable parameters in future.
- Confidence ≠ strength. A strong BUY signal can have low confidence
  if only one indicator fired. Confidence measures agreement.
- Volume as a multiplier: volume doesn't contribute directional score,
  but it scales confidence up (confirming volume) or down (low volume).
- InsufficientDataError is caught here — if an indicator can't compute
  yet (buffer still warming up), it's excluded from the composite.
  The signal is only emitted once the primary indicators (RSI, MACD)
  are all available.
- ATR is computed here for the signal metadata but NOT for direction —
  it's passed to the Risk agent for stop-loss sizing.

Usage:
    generator = SignalGenerator()
    buffer = CandleBuffer("BTC/USDT", "1m")
    # ... fill buffer with candles ...
    signal = generator.generate(buffer)
    if signal:
        await bus.publish(signal)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from agents.technical_analysis.candle_buffer import CandleBuffer
from agents.technical_analysis.indicators import (
    ATRResult,
    BollingerResult,
    EMACrossResult,
    MACDResult,
    RSIResult,
    VolumeResult,
    compute_atr,
    compute_bollinger_bands,
    compute_ema_cross,
    compute_macd,
    compute_rsi,
    compute_volume_signal,
)
from core.config import get_settings
from core.exceptions import InsufficientDataError
from core.logging import get_logger
from core.models.signals import (
    IndicatorName,
    IndicatorReading,
    SignalDirection,
    SignalSource,
    TechnicalSignal,
)

logger = get_logger("signal_generator")


# ---------------------------------------------------------------------------
# Indicator weight configuration
# ---------------------------------------------------------------------------
# Weights must sum to 1.0 across primary indicators.
# Volume is a confidence multiplier, not a directional weight.
#
# Rationale for defaults:
# - MACD (0.30): best momentum indicator, captures trend + momentum together
# - RSI (0.25):  momentum oscillator, strong on extremes
# - EMA cross (0.25): trend direction confirmation
# - Bollinger (0.20): mean-reversion signal, good for ranging markets
#
# TODO: Make these tunable via config or ML optimization (Phase 5+)

INDICATOR_WEIGHTS: dict[IndicatorName, float] = {
    IndicatorName.MACD: 0.30,
    IndicatorName.RSI: 0.25,
    IndicatorName.EMA_CROSS: 0.25,
    IndicatorName.BOLLINGER_BANDS: 0.20,
}

# Volume ratio thresholds for confidence adjustment
VOLUME_HIGH_THRESHOLD = 1.5  # volume_ratio > this → boost confidence
VOLUME_LOW_THRESHOLD = 0.6  # volume_ratio < this → reduce confidence
VOLUME_BOOST = 0.10  # confidence ± this amount
VOLUME_PENALTY = 0.10


def _direction_to_scalar(direction: SignalDirection) -> float:
    """Map SignalDirection to a scalar in [-1, +1]."""
    return {
        SignalDirection.STRONG_BUY: 1.0,
        SignalDirection.BUY: 0.5,
        SignalDirection.NEUTRAL: 0.0,
        SignalDirection.SELL: -0.5,
        SignalDirection.STRONG_SELL: -1.0,
    }[direction]


def _scalar_to_direction(score: float) -> SignalDirection:
    """
    Map a composite score to a SignalDirection.
    Thresholds are intentionally conservative — many neutral readings
    are better than a false directional signal.
    """
    if score >= 0.65:
        return SignalDirection.STRONG_BUY
    elif score >= 0.25:
        return SignalDirection.BUY
    elif score <= -0.65:
        return SignalDirection.STRONG_SELL
    elif score <= -0.25:
        return SignalDirection.SELL
    else:
        return SignalDirection.NEUTRAL


def _compute_agreement(scalars: list[float]) -> float:
    """
    Compute inter-indicator agreement as a value in [0, 1].

    Agreement = 1.0 when all indicators point the same direction.
    Agreement = 0.0 when indicators are evenly split bull/bear.

    Method: standard deviation of the scalar values.
    Low std dev → high agreement → high confidence.
    Max possible std dev for values in [-1, +1] is ~1.0 (opposite extremes).
    """
    if not scalars:
        return 0.0
    if len(scalars) == 1:
        return 0.8  # single indicator — moderate confidence

    mean = sum(scalars) / len(scalars)
    variance = sum((s - mean) ** 2 for s in scalars) / len(scalars)
    std_dev = math.sqrt(variance)

    # Normalise: std_dev=0 → agreement=1.0, std_dev=1.0 → agreement=0.0
    agreement = max(0.0, 1.0 - std_dev)
    return agreement


class SignalGenerator:
    """
    Combines indicator readings into a typed TechnicalSignal.

    Stateless — can be called on any CandleBuffer at any time.
    Creates a new TechnicalSignal on each call.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._ta_cfg = self._settings.technical_analysis

    def generate(self, buffer: CandleBuffer) -> TechnicalSignal | None:
        """
        Generate a TechnicalSignal from the current state of a CandleBuffer.

        Returns None if:
        - Buffer is not warm (insufficient history)
        - Fewer than 2 primary indicators could compute
        - The resulting confidence is below the configured threshold

        Args:
            buffer: A warm CandleBuffer for the target symbol/timeframe.

        Returns:
            TechnicalSignal or None.
        """
        if not buffer.is_warm:
            logger.debug(
                "buffer_not_warm",
                symbol=buffer.symbol,
                timeframe=buffer.timeframe,
                progress_pct=round(buffer.warmup_progress * 100, 1),
            )
            return None

        closes = buffer.closes
        highs = buffer.highs
        lows = buffer.lows
        volumes = buffer.volumes
        current_price = float(closes[-1])

        # ------------------------------------------------------------------
        # Step 1: Compute all indicators, catching data errors individually
        # ------------------------------------------------------------------
        rsi_result: RSIResult | None = self._try_compute("RSI", compute_rsi, closes)
        macd_result: MACDResult | None = self._try_compute("MACD", compute_macd, closes)
        bb_result: BollingerResult | None = self._try_compute("BB", compute_bollinger_bands, closes)
        ema_result: EMACrossResult | None = self._try_compute("EMA", compute_ema_cross, closes)
        volume_result: VolumeResult | None = self._try_compute(
            "VOL", compute_volume_signal, volumes
        )
        atr_result: ATRResult | None = self._try_compute("ATR", compute_atr, highs, lows, closes)

        # ------------------------------------------------------------------
        # Step 2: Build indicator readings list and compute weighted score
        # ------------------------------------------------------------------
        readings: list[IndicatorReading] = []
        weighted_scores: list[float] = []
        total_weight = 0.0

        indicator_map: list[tuple[IndicatorName, object]] = [
            (IndicatorName.RSI, rsi_result),
            (IndicatorName.MACD, macd_result),
            (IndicatorName.BOLLINGER_BANDS, bb_result),
            (IndicatorName.EMA_CROSS, ema_result),
        ]

        for name, result in indicator_map:
            if result is None:
                continue

            weight = INDICATOR_WEIGHTS[name]
            scalar = _direction_to_scalar(result.signal)

            readings.append(
                IndicatorReading(
                    name=name,
                    value=self._extract_primary_value(name, result),
                    signal=result.signal,
                    weight=weight,
                    metadata=result.metadata,
                )
            )
            weighted_scores.append(scalar * weight)
            total_weight += weight

        if len(readings) < 2:
            logger.debug(
                "insufficient_indicators",
                symbol=buffer.symbol,
                timeframe=buffer.timeframe,
                computed=len(readings),
            )
            return None

        # Normalise by actual total weight (some indicators may have failed)
        composite_score = sum(weighted_scores) / total_weight if total_weight > 0 else 0.0

        # ------------------------------------------------------------------
        # Step 3: Confidence calculation
        # ------------------------------------------------------------------
        # Base confidence from inter-indicator agreement
        raw_scalars = [_direction_to_scalar(r.signal) for r in readings]
        agreement = _compute_agreement(raw_scalars)

        # Scale by composite score magnitude (stronger signal = higher confidence)
        score_magnitude = abs(composite_score)
        base_confidence = (agreement * 0.6) + (score_magnitude * 0.4)

        # Volume modifier
        confidence = base_confidence
        if volume_result is not None:
            if volume_result.volume_ratio >= VOLUME_HIGH_THRESHOLD:
                confidence = min(1.0, confidence + VOLUME_BOOST)
            elif volume_result.volume_ratio <= VOLUME_LOW_THRESHOLD:
                confidence = max(0.0, confidence - VOLUME_PENALTY)

        # Add volume as a reading (directional but low weight — informational)
        if volume_result is not None:
            readings.append(
                IndicatorReading(
                    name=IndicatorName.VOLUME,
                    value=volume_result.volume_ratio,
                    signal=volume_result.signal,
                    weight=0.0,  # 0 weight = informational only, doesn't affect composite
                    metadata=volume_result.metadata,
                )
            )

        # ------------------------------------------------------------------
        # Step 4: Filter below confidence threshold
        # ------------------------------------------------------------------
        min_confidence = self._ta_cfg.min_signal_confidence
        if confidence < min_confidence:
            logger.debug(
                "signal_below_confidence_threshold",
                symbol=buffer.symbol,
                timeframe=buffer.timeframe,
                confidence=round(confidence, 3),
                threshold=min_confidence,
                direction=_scalar_to_direction(composite_score).value,
            )
            return None

        # ------------------------------------------------------------------
        # Step 5: Build the signal
        # ------------------------------------------------------------------
        direction = _scalar_to_direction(composite_score)
        ttl = self._ta_cfg.signal_ttl_seconds
        expires_at = datetime.now(UTC) + timedelta(seconds=ttl)

        # Collect ATR metadata for Risk agent stop-loss sizing
        atr_metadata: dict = {}
        if atr_result is not None:
            atr_metadata = {
                "atr_value": atr_result.value,
                "atr_pct_of_price": atr_result.as_pct_of_price,
                "volatility_regime": atr_result.volatility_regime,
            }

        signal = TechnicalSignal(
            source=SignalSource.TECHNICAL,
            symbol=buffer.symbol,
            timeframe=buffer.timeframe,
            expires_at=expires_at,
            direction=direction,
            confidence=round(confidence, 4),
            indicators=readings,
            price=current_price,
            metadata={
                "composite_score": round(composite_score, 4),
                "indicator_count": len([r for r in readings if r.weight > 0]),
                "total_weight": round(total_weight, 3),
                "agreement": round(agreement, 3),
                **atr_metadata,
            },
        )

        logger.info(
            "signal_generated",
            symbol=buffer.symbol,
            timeframe=buffer.timeframe,
            direction=direction.value,
            confidence=round(confidence, 3),
            composite_score=round(composite_score, 4),
            indicators=[r.name for r in readings if r.weight > 0],
            price=current_price,
        )

        return signal

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_compute(self, name: str, fn: Callable[..., object], *args: object) -> object | None:
        """
        Call an indicator function, catching InsufficientDataError silently.
        Any other exception is logged as a warning (unexpected computation error).
        """
        try:
            return fn(*args)
        except InsufficientDataError as e:
            logger.debug("indicator_insufficient_data", indicator=name, reason=str(e))
            return None
        except Exception as e:
            logger.warning("indicator_compute_error", indicator=name, error=str(e))
            return None

    def _extract_primary_value(self, name: IndicatorName, result: object) -> float:
        """Extract the headline numeric value for display in the signal."""
        match name:
            case IndicatorName.RSI:
                return result.value
            case IndicatorName.MACD:
                return result.histogram
            case IndicatorName.BOLLINGER_BANDS:
                return result.percent_b
            case IndicatorName.EMA_CROSS:
                return result.spread_pct
            case _:
                return 0.0
