"""
agents/technical_analysis/indicators.py
-----------------------------------------
TA-Lib indicator wrappers with typed outputs and pre-computation validation.

Every indicator in this module follows the same contract:
    Input:  np.ndarray of close/high/low/volume prices (oldest → newest)
    Output: A typed dataclass with the current value(s) + interpretation

Design decisions:
- Typed dataclasses, not raw floats. Downstream code reads
  `rsi.value` and `rsi.signal`, never `result[0]` or a magic index.
- Pre-flight validation: check array length before calling TA-Lib.
  TA-Lib silently returns NaN arrays for insufficient data; we raise
  InsufficientDataError immediately so the caller knows why.
- NaN guard: TA-Lib returns NaN for the warmup period. We check the
  last computed value before returning to catch partial-data conditions.
- No side effects: all functions are pure (same input → same output).
  No state, no caches. The signal_generator owns state.
- MT5 note: these indicators run entirely in Python. When MT5 integration
  arrives, MT5's built-in indicators can be used as an alternative source,
  but this module remains the canonical signal source for Python agents.

Each function returns:
    - value(s): the raw numeric output
    - signal: a SignalDirection interpretation of the current value
    - metadata: additional context (e.g. histogram for MACD)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

from core.config import get_settings
from core.exceptions import InsufficientDataError
from core.models.signals import SignalDirection

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RSIResult:
    value: float  # Current RSI value [0, 100]
    signal: SignalDirection
    period: int
    is_overbought: bool
    is_oversold: bool
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MACDResult:
    macd_line: float  # MACD line (fast EMA - slow EMA)
    signal_line: float  # Signal line (EMA of MACD)
    histogram: float  # MACD line - signal line
    signal: SignalDirection
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BollingerResult:
    upper: float
    middle: float  # SMA
    lower: float
    bandwidth: float  # (upper - lower) / middle  [volatility proxy]
    percent_b: float  # Where price sits within the bands [0–1, outside=<0 or >1]
    signal: SignalDirection
    current_price: float
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EMACrossResult:
    ema_short: float
    ema_long: float
    spread: float  # ema_short - ema_long
    spread_pct: float  # spread / ema_long * 100
    signal: SignalDirection
    crossover_detected: bool  # True if cross happened on the last bar
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class VolumeResult:
    current_volume: float
    avg_volume: float  # Rolling average
    volume_ratio: float  # current / avg  (>1.5 = high volume)
    signal: SignalDirection  # HIGH_VOLUME confirms moves; LOW_VOLUME warns
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ATRResult:
    """Average True Range — volatility measure used for stop sizing."""

    value: float
    as_pct_of_price: float  # ATR / current_price * 100
    period: int
    volatility_regime: str  # "low" | "normal" | "high"
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _validate_length(arr: np.ndarray, required: int, indicator: str) -> None:
    """Raise InsufficientDataError if array is too short for the indicator."""
    if len(arr) < required:
        raise InsufficientDataError(
            required=required,
            available=len(arr),
            indicator=indicator,
        )


def _check_nan(value: float, indicator: str) -> None:
    """Raise InsufficientDataError if TA-Lib returned NaN (warmup not complete)."""
    if math.isnan(value):
        raise InsufficientDataError(
            required=-1,  # unknown exact requirement
            available=0,
            indicator=f"{indicator} (NaN — warmup period incomplete)",
        )


def _to_float64(arr: np.ndarray) -> np.ndarray:
    """Ensure array is float64 — TA-Lib requires this dtype."""
    return arr.astype(np.float64)


# ---------------------------------------------------------------------------
# Indicator implementations
# ---------------------------------------------------------------------------


def compute_rsi(
    closes: np.ndarray,
    period: int | None = None,
) -> RSIResult:
    """
    Compute Relative Strength Index.

    RSI measures momentum. Traditionally:
      > 70 → overbought (potential sell signal)
      < 30 → oversold  (potential buy signal)

    We use the configured thresholds from settings rather than hardcoded
    70/30 so the operator can tune sensitivity without code changes.

    Args:
        closes: Close price array, oldest first, minimum length: period + 1
        period: RSI period (defaults to TA_RSI_PERIOD from config)

    Returns:
        RSIResult with value, signal, and overbought/oversold flags.
    """
    cfg = get_settings().technical_analysis
    period = period or cfg.rsi_period
    _validate_length(closes, period + 1, f"RSI({period})")

    closes_f64 = _to_float64(closes)

    if TALIB_AVAILABLE:
        rsi_arr = talib.RSI(closes_f64, timeperiod=period)
    else:
        rsi_arr = _rsi_numpy(closes_f64, period)

    current = float(rsi_arr[-1])
    _check_nan(current, f"RSI({period})")

    is_overbought = current >= cfg.rsi_overbought
    is_oversold = current <= cfg.rsi_oversold
    midpoint = (cfg.rsi_overbought + cfg.rsi_oversold) / 2  # typically 50

    # RSI direction (momentum of RSI itself) — more reliable than absolute level in trends
    prev_rsi = (
        float(rsi_arr[-2]) if len(rsi_arr) >= 2 and not math.isnan(float(rsi_arr[-2])) else current
    )
    rsi_rising = current > prev_rsi

    if is_oversold:
        # Deeply oversold: strong buy. Rising RSI from oversold confirms reversal.
        signal = SignalDirection.STRONG_BUY if current <= 20 else SignalDirection.BUY
    elif is_overbought:
        # Trend-aware overbought interpretation.
        # Rising RSI above 70 in a strong trend = continuation, not reversal.
        # Falling RSI from overbought = exhaustion signal → SELL.
        # This avoids RSI=99 being read as SELL in a powerful bull market.
        if rsi_rising:
            signal = SignalDirection.BUY if current < 90 else SignalDirection.NEUTRAL
        else:
            signal = SignalDirection.STRONG_SELL if current >= 80 else SignalDirection.SELL
    else:
        # Neutral zone: use RSI direction for bias
        if current > midpoint + 5 and rsi_rising:
            signal = SignalDirection.NEUTRAL  # bullish lean, no strong conviction
        elif current < midpoint - 5 and not rsi_rising:
            signal = SignalDirection.NEUTRAL  # bearish lean, no strong conviction
        else:
            signal = SignalDirection.NEUTRAL

    return RSIResult(
        value=current,
        signal=signal,
        period=period,
        is_overbought=is_overbought,
        is_oversold=is_oversold,
        metadata={
            "overbought_threshold": cfg.rsi_overbought,
            "oversold_threshold": cfg.rsi_oversold,
            "prev_value": prev_rsi,
            "rsi_rising": rsi_rising,
        },
    )


def compute_macd(
    closes: np.ndarray,
    fast: int | None = None,
    slow: int | None = None,
    signal_period: int | None = None,
) -> MACDResult:
    """
    Compute MACD (Moving Average Convergence Divergence).

    Signal logic:
    - Histogram > 0 and rising  → bullish momentum building
    - Histogram < 0 and falling → bearish momentum building
    - MACD line crosses above signal line → BUY
    - MACD line crosses below signal line → SELL
    - Histogram direction change → early warning of momentum shift

    Args:
        closes: Close price array, minimum length: slow + signal_period + 1
        fast, slow, signal_period: MACD parameters (default from config)
    """
    cfg = get_settings().technical_analysis
    fast = fast or cfg.macd_fast
    slow = slow or cfg.macd_slow
    signal_period = signal_period or cfg.macd_signal
    min_required = slow + signal_period + 1
    _validate_length(closes, min_required, f"MACD({fast},{slow},{signal_period})")

    closes_f64 = _to_float64(closes)

    if TALIB_AVAILABLE:
        macd_arr, signal_arr, hist_arr = talib.MACD(
            closes_f64,
            fastperiod=fast,
            slowperiod=slow,
            signalperiod=signal_period,
        )
    else:
        macd_arr, signal_arr, hist_arr = _macd_numpy(closes_f64, fast, slow, signal_period)

    current_macd = float(macd_arr[-1])
    current_signal = float(signal_arr[-1])
    current_hist = float(hist_arr[-1])
    prev_hist = float(hist_arr[-2]) if len(hist_arr) >= 2 else 0.0

    _check_nan(current_macd, f"MACD({fast},{slow},{signal_period})")
    _check_nan(current_signal, "MACD signal line")
    _check_nan(current_hist, "MACD histogram")

    # Determine signal from histogram direction and crossover.
    # Use a zero-tolerance band to avoid false SELL on perfectly flat histograms
    # (e.g. synthetic linear price series with identical-step candles).
    # In real markets, histogram is never exactly 0.0 for multiple bars.
    HIST_ZERO_BAND = 1e-8  # treat |histogram| < this as effectively zero
    hist_near_zero = abs(current_hist) < HIST_ZERO_BAND

    if hist_near_zero:
        # Fall back to MACD line vs signal line position for direction
        signal = (
            SignalDirection.BUY
            if current_macd > current_signal
            else SignalDirection.SELL
            if current_macd < current_signal
            else SignalDirection.NEUTRAL
        )
    else:
        hist_rising = current_hist > prev_hist
        hist_positive = current_hist > 0

        if hist_positive and hist_rising:
            signal = (
                SignalDirection.STRONG_BUY
                if current_hist > abs(prev_hist) * 1.5
                else SignalDirection.BUY
            )
        elif hist_positive and not hist_rising:
            signal = SignalDirection.NEUTRAL  # bullish but losing steam
        elif not hist_positive and not hist_rising:
            signal = (
                SignalDirection.STRONG_SELL
                if abs(current_hist) > abs(prev_hist) * 1.5
                else SignalDirection.SELL
            )
        else:
            signal = SignalDirection.NEUTRAL  # bearish but recovering

    return MACDResult(
        macd_line=current_macd,
        signal_line=current_signal,
        histogram=current_hist,
        signal=signal,
        metadata={
            "fast": fast,
            "slow": slow,
            "signal_period": signal_period,
            "prev_histogram": prev_hist,
            "histogram_near_zero": hist_near_zero,
        },
    )


def compute_bollinger_bands(
    closes: np.ndarray,
    period: int | None = None,
    std_dev: float | None = None,
) -> BollingerResult:
    """
    Compute Bollinger Bands.

    Signal logic:
    - Price near lower band (percent_b < 0.05) → oversold → BUY
    - Price near upper band (percent_b > 0.95) → overbought → SELL
    - Price outside bands → strong signal (breakout or mean-revert depending on context)
    - Bandwidth squeeze (low volatility) → context for upcoming breakout

    %B = (Price - Lower Band) / (Upper Band - Lower Band)
    %B < 0  → price is below lower band
    %B > 1  → price is above upper band

    Args:
        closes: Close price array, minimum length: period
        period, std_dev: BB parameters (default from config)
    """
    cfg = get_settings().technical_analysis
    period = period or cfg.bb_period
    std_dev = std_dev or cfg.bb_std_dev
    _validate_length(closes, period, f"BB({period},{std_dev})")

    closes_f64 = _to_float64(closes)
    current_price = float(closes_f64[-1])

    if TALIB_AVAILABLE:
        upper_arr, middle_arr, lower_arr = talib.BBANDS(
            closes_f64,
            timeperiod=period,
            nbdevup=std_dev,
            nbdevdn=std_dev,
            matype=0,  # SMA
        )
    else:
        upper_arr, middle_arr, lower_arr = _bbands_numpy(closes_f64, period, std_dev)

    upper = float(upper_arr[-1])
    middle = float(middle_arr[-1])
    lower = float(lower_arr[-1])

    _check_nan(upper, "BB upper")

    band_range = upper - lower
    percent_b = (current_price - lower) / band_range if band_range > 0 else 0.5
    bandwidth = band_range / middle if middle > 0 else 0.0

    # Signal interpretation
    if percent_b <= 0.0:
        signal = SignalDirection.STRONG_BUY  # below lower band
    elif percent_b <= 0.1:
        signal = SignalDirection.BUY  # near lower band
    elif percent_b >= 1.0:
        signal = SignalDirection.STRONG_SELL  # above upper band
    elif percent_b >= 0.9:
        signal = SignalDirection.SELL  # near upper band
    else:
        signal = SignalDirection.NEUTRAL

    return BollingerResult(
        upper=upper,
        middle=middle,
        lower=lower,
        bandwidth=bandwidth,
        percent_b=percent_b,
        signal=signal,
        current_price=current_price,
        metadata={
            "period": period,
            "std_dev": std_dev,
            "band_range": band_range,
        },
    )


def compute_ema_cross(
    closes: np.ndarray,
    short_period: int | None = None,
    long_period: int | None = None,
) -> EMACrossResult:
    """
    Compute EMA crossover signal.

    A "golden cross" (short EMA crosses above long EMA) is a classic trend
    confirmation signal. A "death cross" is the reverse.

    Signal logic:
    - Short EMA > long EMA → uptrend → BUY (strength depends on spread)
    - Short EMA < long EMA → downtrend → SELL
    - Cross detected on last bar → STRONG signal

    Args:
        closes: Close price array, minimum length: long_period + 1
        short_period, long_period: EMA periods (default from config)
    """
    cfg = get_settings().technical_analysis
    short_period = short_period or cfg.ema_short
    long_period = long_period or cfg.ema_long
    _validate_length(closes, long_period + 1, f"EMA({short_period},{long_period})")

    closes_f64 = _to_float64(closes)

    if TALIB_AVAILABLE:
        ema_short_arr = talib.EMA(closes_f64, timeperiod=short_period)
        ema_long_arr = talib.EMA(closes_f64, timeperiod=long_period)
    else:
        ema_short_arr = _ema_numpy(closes_f64, short_period)
        ema_long_arr = _ema_numpy(closes_f64, long_period)

    current_short = float(ema_short_arr[-1])
    current_long = float(ema_long_arr[-1])
    prev_short = float(ema_short_arr[-2])
    prev_long = float(ema_long_arr[-2])

    _check_nan(current_short, f"EMA({short_period})")
    _check_nan(current_long, f"EMA({long_period})")

    spread = current_short - current_long
    spread_pct = (spread / current_long * 100) if current_long > 0 else 0.0

    # Detect crossover on the most recent bar
    was_above = prev_short > prev_long
    is_above = current_short > current_long
    crossover_detected = was_above != is_above

    if crossover_detected:
        signal = SignalDirection.STRONG_BUY if is_above else SignalDirection.STRONG_SELL
    elif is_above:
        # Uptrend — signal strength from spread magnitude
        signal = SignalDirection.BUY if spread_pct > 0.3 else SignalDirection.NEUTRAL
    else:
        # Downtrend
        signal = SignalDirection.SELL if abs(spread_pct) > 0.3 else SignalDirection.NEUTRAL

    return EMACrossResult(
        ema_short=current_short,
        ema_long=current_long,
        spread=spread,
        spread_pct=spread_pct,
        signal=signal,
        crossover_detected=crossover_detected,
        metadata={
            "short_period": short_period,
            "long_period": long_period,
            "prev_spread": prev_short - prev_long,
        },
    )


def compute_volume_signal(
    volumes: np.ndarray,
    lookback: int = 20,
) -> VolumeResult:
    """
    Assess volume relative to recent average.

    Volume is not a directional signal by itself, but it confirms or
    weakens other signals:
    - High volume on a BUY signal → confirms it
    - Low volume on a BUY signal → weakens confidence
    - Volume surge → potential breakout or reversal

    Args:
        volumes: Volume array, minimum length: lookback
        lookback: Bars to average for comparison
    """
    _validate_length(volumes, lookback, f"Volume({lookback})")

    current = float(volumes[-1])
    avg = float(np.mean(volumes[-lookback:-1]))  # exclude current bar

    ratio = current / avg if avg > 0 else 1.0

    if ratio >= 2.0:
        signal = SignalDirection.STRONG_BUY  # Volume surge — confirms directional move
    elif ratio >= 1.5:
        signal = SignalDirection.BUY  # Above average — mild confirmation
    elif ratio <= 0.5:
        signal = SignalDirection.SELL  # Very low volume — weak/suspect move
    else:
        signal = SignalDirection.NEUTRAL

    return VolumeResult(
        current_volume=current,
        avg_volume=avg,
        volume_ratio=ratio,
        signal=signal,
        metadata={"lookback": lookback},
    )


def compute_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> ATRResult:
    """
    Compute Average True Range for volatility measurement and stop sizing.

    ATR is not directional — it measures how much the market moves on average.
    Used by the Risk agent for dynamic stop-loss sizing:
        stop_loss = entry_price - (atr_multiplier * atr_value)

    Volatility regimes (approximate, crypto):
    - Low:    ATR < 0.5% of price
    - Normal: 0.5% – 2.0%
    - High:   > 2.0%

    Args:
        highs, lows, closes: OHLCV component arrays
        period: ATR period
    """
    _validate_length(closes, period + 1, f"ATR({period})")

    h = _to_float64(highs)
    l = _to_float64(lows)
    c = _to_float64(closes)

    if TALIB_AVAILABLE:
        atr_arr = talib.ATR(h, l, c, timeperiod=period)
    else:
        atr_arr = _atr_numpy(h, l, c, period)

    current_atr = float(atr_arr[-1])
    _check_nan(current_atr, f"ATR({period})")

    current_price = float(closes[-1])
    as_pct = (current_atr / current_price * 100) if current_price > 0 else 0.0

    if as_pct < 0.5:
        regime = "low"
    elif as_pct > 2.0:
        regime = "high"
    else:
        regime = "normal"

    return ATRResult(
        value=current_atr,
        as_pct_of_price=as_pct,
        period=period,
        volatility_regime=regime,
        metadata={"current_price": current_price},
    )


# ---------------------------------------------------------------------------
# Pure NumPy fallbacks (used when TA-Lib is not installed — e.g. CI/CD)
# These are NOT for production use — TA-Lib is significantly faster and
# more numerically robust. Install TA-Lib for all production deployments.
# ---------------------------------------------------------------------------


def _ema_numpy(data: np.ndarray, period: int) -> np.ndarray:
    """EMA via exponential smoothing. Slower than TA-Lib but dependency-free.

    Correctly handles NaN-prefixed input (e.g. MACD signal line computation
    where the input MACD line has NaN values during its own warmup period).
    Finds the first non-NaN index, seeds EMA there, then propagates forward.
    """
    alpha = 2.0 / (period + 1)
    result = np.full_like(data, np.nan)

    # Find first index where we have `period` consecutive non-NaN values
    valid_mask = ~np.isnan(data)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < period:
        return result  # Not enough valid data — return all NaN

    # Seed: mean of first `period` valid values
    start = valid_indices[period - 1]  # index of the period-th valid value
    first_valid = valid_indices[0]
    seed_values = data[first_valid : start + 1]
    result[start] = np.nanmean(seed_values[-period:])

    for i in range(start + 1, len(data)):
        if not np.isnan(data[i]):
            result[i] = data[i] * alpha + result[i - 1] * (1 - alpha)
        # If data[i] is NaN, leave result[i] as NaN

    return result


def _rsi_numpy(closes: np.ndarray, period: int) -> np.ndarray:
    """RSI via Wilder's smoothing method."""
    deltas = np.diff(closes)
    result = np.full(len(closes), np.nan)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    if len(gains) < period:
        return result

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss > 0 else 100.0
        result[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return result


def _macd_numpy(
    closes: np.ndarray,
    fast: int,
    slow: int,
    signal: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD via EMA differences."""
    ema_fast = _ema_numpy(closes, fast)
    ema_slow = _ema_numpy(closes, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema_numpy(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bbands_numpy(
    closes: np.ndarray,
    period: int,
    std_dev: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands via rolling SMA and std dev."""
    result_upper = np.full_like(closes, np.nan)
    result_mid = np.full_like(closes, np.nan)
    result_lower = np.full_like(closes, np.nan)
    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1 : i + 1]
        mid = np.mean(window)
        std = np.std(window, ddof=0)
        result_mid[i] = mid
        result_upper[i] = mid + std_dev * std
        result_lower[i] = mid - std_dev * std
    return result_upper, result_mid, result_lower


def _atr_numpy(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int,
) -> np.ndarray:
    """ATR via true range smoothing."""
    result = np.full(len(closes), np.nan)
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])),
    )
    if len(tr) < period:
        return result
    result[period] = np.mean(tr[:period])
    for i in range(period + 1, len(closes)):
        result[i] = (result[i - 1] * (period - 1) + tr[i - 1]) / period
    return result
