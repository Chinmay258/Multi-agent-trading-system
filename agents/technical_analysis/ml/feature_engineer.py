"""
agents/technical_analysis/ml/feature_engineer.py
--------------------------------------------------
Convert raw OHLCV history into numeric feature vectors suitable for ML.

Two entry points:

1. ``build_features(buffer)`` — produces ONE feature vector for the latest
   bar in a CandleBuffer. Called at inference time by MLSignalGenerator.

2. ``build_feature_matrix(closes, highs, lows, volumes, timestamps)`` —
   produces an [N, F] matrix of feature rows for the training set.
   Called offline by ModelTrainer.

Design decisions:
- No TA-Lib dependency. Indicators are recomputed in pure NumPy here
  so the feature engineer can run in environments where TA-Lib is absent
  (CI, lightweight inference workers, training boxes).
- All features are scale-invariant: returns, ratios, normalised distances.
  Raw price levels are never used directly — a model trained on BTC at
  $30k would otherwise fail when BTC trades at $80k.
- Features are deterministic, stateless functions of the input arrays.
  Same input → identical output, every time. No side effects.
- ``MIN_BARS_REQUIRED`` covers the longest lookback (EMA50 + warmup),
  matching the warmup the rule-based generator already enforces.
- Each feature is finite (no NaN/Inf) by construction. Safe guards
  in helpers convert degenerate cases to 0.0 rather than letting NaN
  poison the training set.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from agents.technical_analysis.candle_buffer import CandleBuffer

# Longest lookback used by any feature: EMA(50) + a small warmup buffer.
# Must be at least 60 to satisfy the spec; we mirror the indicator warmup
# requirements of the rule-based generator.
MIN_BARS_REQUIRED: int = 60


# ---------------------------------------------------------------------------
# Pure-NumPy indicator primitives (no TA-Lib)
# ---------------------------------------------------------------------------


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average. NaN-prefixed result during warmup."""
    out = np.full_like(values, np.nan, dtype=np.float64)
    if len(values) < period:
        return out
    alpha = 2.0 / (period + 1)
    # Seed with the simple mean of the first ``period`` values.
    out[period - 1] = float(np.mean(values[:period]))
    for i in range(period, len(values)):
        out[i] = values[i] * alpha + out[i - 1] * (1 - alpha)
    return out


def _rsi(closes: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RSI. Returns array same length as ``closes``."""
    out = np.full(len(closes), np.nan, dtype=np.float64)
    if len(closes) < period + 1:
        return out
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    rs = avg_gain / avg_loss if avg_loss > 0 else (math.inf if avg_gain > 0 else 0.0)
    out[period] = 100.0 - 100.0 / (1.0 + rs) if math.isfinite(rs) else 100.0

    for i in range(period + 1, len(closes)):
        gain = gains[i - 1]
        loss = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - 100.0 / (1.0 + rs)
        else:
            out[i] = 100.0 if avg_gain > 0 else 50.0
    return out


def _macd(
    closes: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (macd_line, signal_line, histogram) arrays."""
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_line = ema_fast - ema_slow
    # EMA of the macd_line must handle NaN warmup safely.
    out_signal = np.full_like(closes, np.nan, dtype=np.float64)
    valid = np.where(~np.isnan(macd_line))[0]
    if len(valid) >= signal_period:
        start = valid[signal_period - 1]
        seed = float(np.mean(macd_line[valid[:signal_period]]))
        out_signal[start] = seed
        alpha = 2.0 / (signal_period + 1)
        for i in range(start + 1, len(closes)):
            if math.isnan(macd_line[i]):
                continue
            out_signal[i] = macd_line[i] * alpha + out_signal[i - 1] * (1 - alpha)
    histogram = macd_line - out_signal
    return macd_line, out_signal, histogram


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
    """Average True Range, Wilder smoothing."""
    out = np.full(len(closes), np.nan, dtype=np.float64)
    if len(closes) < period + 1:
        return out
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ),
    )
    out[period] = float(np.mean(tr[:period]))
    for i in range(period + 1, len(closes)):
        out[i] = (out[i - 1] * (period - 1) + tr[i - 1]) / period
    return out


def _bollinger(
    closes: np.ndarray,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (upper, middle, lower) Bollinger Band arrays."""
    upper = np.full_like(closes, np.nan, dtype=np.float64)
    middle = np.full_like(closes, np.nan, dtype=np.float64)
    lower = np.full_like(closes, np.nan, dtype=np.float64)
    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1 : i + 1]
        m = float(np.mean(window))
        s = float(np.std(window, ddof=0))
        middle[i] = m
        upper[i] = m + std_dev * s
        lower[i] = m - std_dev * s
    return upper, middle, lower


def _safe_div(num: float, denom: float, default: float = 0.0) -> float:
    """Division that returns ``default`` instead of NaN/Inf for zero or NaN denom."""
    if denom is None or not math.isfinite(denom) or denom == 0.0:
        return default
    val = num / denom
    return val if math.isfinite(val) else default


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_FEATURE_NAMES: list[str] = [
    # Price features
    "returns_1",
    "returns_3",
    "returns_5",
    "returns_10",
    "returns_20",
    "log_return_1",
    "high_low_ratio",
    "close_position",
    # Volume features
    "volume_ratio_5",
    "volume_ratio_20",
    "volume_trend",
    # Momentum
    "rsi_14",
    "rsi_7",
    "macd_hist",
    "macd_signal_cross",
    # Volatility
    "atr_14_pct",
    "bb_width",
    "bb_pct_b",
    # Trend
    "ema_9_dist",
    "ema_21_dist",
    "ema_50_dist",
    "ema_cross_9_21",
    "trend_strength",
    # Microstructure
    "candle_body",
    "upper_wick",
    "lower_wick",
    # Time
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]


class FeatureEngineer:
    """
    Stateless feature builder.

    Inference path:
        engineer = FeatureEngineer()
        vec = engineer.build_features(buffer)   # shape: (n_features,)

    Training path:
        X, names = engineer.build_feature_matrix(closes, highs, lows, volumes, timestamps)
        # X shape: (n_samples, n_features) — first MIN_BARS_REQUIRED-1 bars are skipped.
    """

    MIN_BARS_REQUIRED: int = MIN_BARS_REQUIRED

    @staticmethod
    def feature_names() -> list[str]:
        """Ordered list of feature column names. Matches ``build_features`` output."""
        return list(_FEATURE_NAMES)

    # ------------------------------------------------------------------
    # Single-row inference
    # ------------------------------------------------------------------

    def build_features(self, buffer: CandleBuffer) -> np.ndarray | None:
        """
        Build the feature vector for the most recent bar in the buffer.

        Returns:
            np.ndarray of shape (n_features,) — float64.
            None if the buffer has fewer than MIN_BARS_REQUIRED candles.
        """
        if buffer.size < self.MIN_BARS_REQUIRED:
            return None

        closes = buffer.closes
        opens = buffer.opens
        highs = buffer.highs
        lows = buffer.lows
        volumes = buffer.volumes
        ts = buffer.latest_timestamp  # one datetime

        vec = self._features_for_index(
            idx=len(closes) - 1,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            volumes=volumes,
            timestamp=ts,
        )
        if vec is None:
            return None
        return vec

    # ------------------------------------------------------------------
    # Bulk matrix construction for training
    # ------------------------------------------------------------------

    def build_feature_matrix(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        timestamps: list[datetime],
        opens: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Build feature matrix from full OHLCV history.

        Each row uses bars [0 .. i] as the lookback for the prediction made
        at bar ``i``. Rows where any feature would be undefined (during the
        warmup window) are excluded.

        Args:
            closes:     close prices, oldest → newest.
            highs:      highs.
            lows:       lows.
            volumes:    volumes.
            timestamps: list of UTC datetimes, same length as closes.
            opens:      opens, oldest → newest. If None, defaults to ``closes``
                        (acceptable for tests / synthetic data that lacks O).

        Returns:
            (X, feature_names)
                X            — shape (n_samples, n_features)
                feature_names — list of column names matching X's columns
        """
        n = len(closes)
        if n < self.MIN_BARS_REQUIRED:
            return np.zeros((0, len(_FEATURE_NAMES)), dtype=np.float64), list(_FEATURE_NAMES)

        if opens is None:
            opens = closes.copy()

        # Pre-compute indicator series once for the entire history — much
        # faster than recomputing per-bar inside the loop.
        rsi_14 = _rsi(closes, 14)
        rsi_7 = _rsi(closes, 7)
        _, _, macd_hist = _macd(closes, 12, 26, 9)
        atr_14 = _atr(highs, lows, closes, 14)
        bb_up, bb_mid, bb_low = _bollinger(closes, 20, 2.0)
        ema_9 = _ema(closes, 9)
        ema_21 = _ema(closes, 21)
        ema_50 = _ema(closes, 50)

        rows: list[np.ndarray] = []
        start = self.MIN_BARS_REQUIRED - 1
        for i in range(start, n):
            row = self._row_from_precomputed(
                idx=i,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                volumes=volumes,
                timestamp=timestamps[i],
                rsi_14=rsi_14,
                rsi_7=rsi_7,
                macd_hist=macd_hist,
                atr_14=atr_14,
                bb_up=bb_up,
                bb_mid=bb_mid,
                bb_low=bb_low,
                ema_9=ema_9,
                ema_21=ema_21,
                ema_50=ema_50,
            )
            if row is not None:
                rows.append(row)

        if not rows:
            return np.zeros((0, len(_FEATURE_NAMES)), dtype=np.float64), list(_FEATURE_NAMES)

        X = np.vstack(rows).astype(np.float64)
        # Final hygiene pass: replace any sneaky NaN/Inf with 0.0.
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, list(_FEATURE_NAMES)

    # ------------------------------------------------------------------
    # Internal: single-row helpers
    # ------------------------------------------------------------------

    def _features_for_index(
        self,
        idx: int,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        timestamp: datetime | None,
    ) -> np.ndarray | None:
        """One-shot inference helper that recomputes only the indicators it needs."""
        rsi_14 = _rsi(closes, 14)
        rsi_7 = _rsi(closes, 7)
        _, _, macd_hist = _macd(closes, 12, 26, 9)
        atr_14 = _atr(highs, lows, closes, 14)
        bb_up, bb_mid, bb_low = _bollinger(closes, 20, 2.0)
        ema_9 = _ema(closes, 9)
        ema_21 = _ema(closes, 21)
        ema_50 = _ema(closes, 50)

        return self._row_from_precomputed(
            idx=idx,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            volumes=volumes,
            timestamp=timestamp,
            rsi_14=rsi_14,
            rsi_7=rsi_7,
            macd_hist=macd_hist,
            atr_14=atr_14,
            bb_up=bb_up,
            bb_mid=bb_mid,
            bb_low=bb_low,
            ema_9=ema_9,
            ema_21=ema_21,
            ema_50=ema_50,
        )

    def _row_from_precomputed(
        self,
        idx: int,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        timestamp: datetime | None,
        rsi_14: np.ndarray,
        rsi_7: np.ndarray,
        macd_hist: np.ndarray,
        atr_14: np.ndarray,
        bb_up: np.ndarray,
        bb_mid: np.ndarray,
        bb_low: np.ndarray,
        ema_9: np.ndarray,
        ema_21: np.ndarray,
        ema_50: np.ndarray,
    ) -> np.ndarray | None:
        """Assemble the feature vector for bar ``idx`` from pre-computed series."""
        if idx < self.MIN_BARS_REQUIRED - 1:
            return None

        close = float(closes[idx])
        open_ = float(opens[idx])
        high = float(highs[idx])
        low = float(lows[idx])
        vol = float(volumes[idx])

        # --- Price features ---
        def ret(lag: int) -> float:
            if idx - lag < 0:
                return 0.0
            past = float(closes[idx - lag])
            return _safe_div(close - past, past)

        returns_1 = ret(1)
        returns_3 = ret(3)
        returns_5 = ret(5)
        returns_10 = ret(10)
        returns_20 = ret(20)

        prev_close = float(closes[idx - 1])
        log_return_1 = math.log(close / prev_close) if (close > 0 and prev_close > 0) else 0.0

        high_low_ratio = _safe_div(high - low, close)
        hl_range = high - low
        close_position = _safe_div(close - low, hl_range, default=0.5)

        # --- Volume features ---
        vol5 = float(np.mean(volumes[idx - 5 : idx])) if idx >= 5 else float(np.mean(volumes[:idx]))
        vol20 = (
            float(np.mean(volumes[idx - 20 : idx])) if idx >= 20 else float(np.mean(volumes[:idx]))
        )
        volume_ratio_5 = _safe_div(vol, vol5, default=1.0)
        volume_ratio_20 = _safe_div(vol, vol20, default=1.0)
        vol_lag5 = float(volumes[idx - 5]) if idx >= 5 else float(volumes[0])
        volume_trend = _safe_div(vol - vol_lag5, vol_lag5)

        # --- Momentum ---
        r14 = float(rsi_14[idx]) if not math.isnan(rsi_14[idx]) else 50.0
        r7 = float(rsi_7[idx]) if not math.isnan(rsi_7[idx]) else 50.0
        # Normalise RSI to roughly [-1, +1] around the 50 midline so it's
        # on a comparable scale to the other features.
        rsi_14_norm = (r14 - 50.0) / 50.0
        rsi_7_norm = (r7 - 50.0) / 50.0

        hist_now = float(macd_hist[idx]) if not math.isnan(macd_hist[idx]) else 0.0
        # Scale histogram by price so it's comparable across symbols/regimes.
        macd_hist_norm = _safe_div(hist_now, close)

        macd_signal_cross = 0.0
        for lag in (1, 2, 3):
            if idx - lag < 0:
                break
            prev_h = macd_hist[idx - lag]
            curr_h = macd_hist[idx - lag + 1]
            if math.isnan(prev_h) or math.isnan(curr_h):
                continue
            if (prev_h <= 0 < curr_h) or (prev_h >= 0 > curr_h):
                macd_signal_cross = 1.0
                break

        # --- Volatility ---
        atr_now = float(atr_14[idx]) if not math.isnan(atr_14[idx]) else 0.0
        atr_14_pct = _safe_div(atr_now, close)

        u = float(bb_up[idx]) if not math.isnan(bb_up[idx]) else close
        m = float(bb_mid[idx]) if not math.isnan(bb_mid[idx]) else close
        lo_ = float(bb_low[idx]) if not math.isnan(bb_low[idx]) else close
        bb_width = _safe_div(u - lo_, m)
        bb_pct_b = _safe_div(close - lo_, u - lo_, default=0.5)

        # --- Trend ---
        e9 = float(ema_9[idx]) if not math.isnan(ema_9[idx]) else close
        e21 = float(ema_21[idx]) if not math.isnan(ema_21[idx]) else close
        e50 = float(ema_50[idx]) if not math.isnan(ema_50[idx]) else close
        ema_9_dist = _safe_div(close - e9, close)
        ema_21_dist = _safe_div(close - e21, close)
        ema_50_dist = _safe_div(close - e50, close)
        ema_cross_9_21 = 1.0 if e9 > e21 else -1.0
        trend_strength = abs(ema_9_dist)

        # --- Microstructure ---
        body = abs(close - open_)
        candle_body = _safe_div(body, hl_range, default=0.0)
        upper_wick = _safe_div(high - max(open_, close), hl_range, default=0.0)
        lower_wick = _safe_div(min(open_, close) - low, hl_range, default=0.0)

        # --- Time (cyclical encoding) ---
        if timestamp is None:
            hour_sin = hour_cos = day_sin = day_cos = 0.0
        else:
            hour = timestamp.hour
            dow = timestamp.weekday()
            hour_sin = math.sin(2 * math.pi * hour / 24.0)
            hour_cos = math.cos(2 * math.pi * hour / 24.0)
            day_sin = math.sin(2 * math.pi * dow / 7.0)
            day_cos = math.cos(2 * math.pi * dow / 7.0)

        vec = np.array(
            [
                returns_1,
                returns_3,
                returns_5,
                returns_10,
                returns_20,
                log_return_1,
                high_low_ratio,
                close_position,
                volume_ratio_5,
                volume_ratio_20,
                volume_trend,
                rsi_14_norm,
                rsi_7_norm,
                macd_hist_norm,
                macd_signal_cross,
                atr_14_pct,
                bb_width,
                bb_pct_b,
                ema_9_dist,
                ema_21_dist,
                ema_50_dist,
                ema_cross_9_21,
                trend_strength,
                candle_body,
                upper_wick,
                lower_wick,
                hour_sin,
                hour_cos,
                day_sin,
                day_cos,
            ],
            dtype=np.float64,
        )
        # Hygiene: convert any NaN/Inf the math may have produced.
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return vec
