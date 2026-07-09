"""
agents/technical_analysis/candle_buffer.py
-------------------------------------------
Rolling in-memory OHLCV buffer per (symbol, timeframe) pair.

The TA agent needs the last N candles to compute indicators — RSI needs
14+1, MACD needs 26+9+1, etc. This buffer maintains a fixed-size deque
per symbol/timeframe and exposes numpy arrays for indicator computation.

Design decisions:
- One CandleBuffer per (symbol, timeframe). The TAAgent creates one
  registry (CandleBufferRegistry) that creates buffers on demand.
- Fixed capacity: we keep MAX_BUFFER_SIZE candles, which is enough for
  all indicators with room to spare. No unbounded memory growth.
- numpy extraction is O(n) but called only when computing indicators —
  not on every candle receipt. Acceptable for our frequencies.
- Warmup tracking: buffer reports is_warm when it has enough candles
  for the most demanding indicator. Before warmup, no signals are emitted.
- Thread safety: designed for single-threaded asyncio. No locks needed.

Usage:
    registry = CandleBufferRegistry()
    registry.add("BTC/USDT", "1m", candle)

    buf = registry.get("BTC/USDT", "1m")
    if buf.is_warm:
        closes = buf.closes
        rsi = compute_rsi(closes)
"""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime

import numpy as np

from core.logging import get_logger
from core.models.market import OHLCVCandle

logger = get_logger("candle_buffer")

# Maximum candles kept per symbol/timeframe
# 500 bars covers: MACD(12,26,9)=35, RSI(14)=15, BB(20)=20, EMA(21)=21
# 500 gives ample room for longer lookbacks in future indicators
MAX_BUFFER_SIZE = 500

# Minimum candles required before any indicators are computed.
# MACD(12,26,9) needs slow+signal+1 = 36 bars.
# We use 50 to be safe and allow EMA warmup.
MIN_WARMUP_CANDLES = 50


class CandleBuffer:
    """
    Fixed-size rolling buffer of OHLCV candles for one (symbol, timeframe).

    Maintains the last MAX_BUFFER_SIZE candles in chronological order
    (oldest at index 0, newest at index -1) and extracts numpy arrays
    for indicator computation on demand.
    """

    def __init__(self, symbol: str, timeframe: str, capacity: int = MAX_BUFFER_SIZE) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.capacity = capacity
        self._candles: deque[OHLCVCandle] = deque(maxlen=capacity)

    def add(self, candle: OHLCVCandle) -> None:
        """
        Append a candle to the buffer.

        If the buffer is at capacity, the oldest candle is automatically
        dropped (deque maxlen behaviour). Duplicate timestamps are
        silently rejected — the normaliser handles dedup upstream.
        """
        if self._candles and candle.timestamp <= self._candles[-1].timestamp:
            # Reject out-of-order or duplicate candles
            logger.debug(
                "candle_buffer_rejected_old",
                symbol=self.symbol,
                timeframe=self.timeframe,
                candle_ts=str(candle.timestamp),
                last_ts=str(self._candles[-1].timestamp),
            )
            return
        self._candles.append(candle)

    def add_many(self, candles: list[OHLCVCandle]) -> None:
        """Add a list of candles (e.g. historical warmup batch)."""
        for candle in sorted(candles, key=lambda c: c.timestamp):
            self.add(candle)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._candles)

    @property
    def is_warm(self) -> bool:
        """True when enough candles exist to compute all indicators."""
        return self.size >= MIN_WARMUP_CANDLES

    @property
    def warmup_progress(self) -> float:
        """Fraction of warmup complete [0.0, 1.0]."""
        return min(self.size / MIN_WARMUP_CANDLES, 1.0)

    @property
    def latest_candle(self) -> OHLCVCandle | None:
        return self._candles[-1] if self._candles else None

    @property
    def oldest_candle(self) -> OHLCVCandle | None:
        return self._candles[0] if self._candles else None

    @property
    def latest_close(self) -> float | None:
        c = self.latest_candle
        return float(c.close) if c else None

    @property
    def latest_timestamp(self) -> datetime | None:
        c = self.latest_candle
        return c.timestamp if c else None

    def data_age_seconds(self) -> float | None:
        """Seconds since the most recent candle was received."""
        c = self.latest_candle
        if c is None:
            return None
        return (datetime.now(UTC) - c.received_at).total_seconds()

    # ------------------------------------------------------------------
    # Numpy array extraction (for indicator computation)
    # ------------------------------------------------------------------

    @property
    def closes(self) -> np.ndarray:
        """Close prices as float64 numpy array, oldest → newest."""
        return np.array([float(c.close) for c in self._candles], dtype=np.float64)

    @property
    def opens(self) -> np.ndarray:
        return np.array([float(c.open) for c in self._candles], dtype=np.float64)

    @property
    def highs(self) -> np.ndarray:
        return np.array([float(c.high) for c in self._candles], dtype=np.float64)

    @property
    def lows(self) -> np.ndarray:
        return np.array([float(c.low) for c in self._candles], dtype=np.float64)

    @property
    def volumes(self) -> np.ndarray:
        return np.array([float(c.volume) for c in self._candles], dtype=np.float64)

    def last_n_closes(self, n: int) -> np.ndarray:
        """Return only the last N close prices. Useful for short lookbacks."""
        candles = list(self._candles)[-n:]
        return np.array([float(c.close) for c in candles], dtype=np.float64)

    # ------------------------------------------------------------------
    # Debug / health
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "size": self.size,
            "capacity": self.capacity,
            "is_warm": self.is_warm,
            "warmup_progress_pct": round(self.warmup_progress * 100, 1),
            "latest_timestamp": str(self.latest_timestamp) if self.latest_timestamp else None,
            "latest_close": self.latest_close,
            "data_age_seconds": self.data_age_seconds(),
        }

    def __repr__(self) -> str:
        return (
            f"CandleBuffer({self.symbol}/{self.timeframe}, "
            f"size={self.size}/{self.capacity}, warm={self.is_warm})"
        )


# ---------------------------------------------------------------------------
# Registry — manages all buffers, one per (symbol, timeframe) pair
# ---------------------------------------------------------------------------


class CandleBufferRegistry:
    """
    Creates and manages CandleBuffer instances for all (symbol, timeframe) pairs.

    The TechnicalAnalysisAgent holds one registry and routes incoming
    candles to the correct buffer via add().
    """

    def __init__(self) -> None:
        self._buffers: dict[tuple[str, str], CandleBuffer] = {}

    def add(self, candle: OHLCVCandle) -> CandleBuffer:
        """
        Route a candle to its buffer, creating the buffer if needed.
        Returns the buffer the candle was added to.
        """
        key = (candle.symbol, candle.timeframe)
        if key not in self._buffers:
            self._buffers[key] = CandleBuffer(candle.symbol, candle.timeframe)
            logger.info(
                "candle_buffer_created",
                symbol=candle.symbol,
                timeframe=candle.timeframe,
            )
        self._buffers[key].add(candle)
        return self._buffers[key]

    def get(self, symbol: str, timeframe: str) -> CandleBuffer | None:
        """Return the buffer for a (symbol, timeframe) pair, or None."""
        return self._buffers.get((symbol, timeframe))

    def get_or_create(self, symbol: str, timeframe: str) -> CandleBuffer:
        """Return the buffer, creating it if it doesn't exist yet."""
        key = (symbol, timeframe)
        if key not in self._buffers:
            self._buffers[key] = CandleBuffer(symbol, timeframe)
        return self._buffers[key]

    def all_warm(self) -> bool:
        """True if ALL registered buffers have completed warmup."""
        return bool(self._buffers) and all(b.is_warm for b in self._buffers.values())

    def warm_buffers(self) -> list[CandleBuffer]:
        return [b for b in self._buffers.values() if b.is_warm]

    def symbols(self) -> list[str]:
        return list({symbol for symbol, _ in self._buffers.keys()})

    def to_dict(self) -> dict:
        return {f"{s}/{tf}": buf.to_dict() for (s, tf), buf in self._buffers.items()}

    def __len__(self) -> int:
        return len(self._buffers)

    def __repr__(self) -> str:
        warm = sum(1 for b in self._buffers.values() if b.is_warm)
        return f"CandleBufferRegistry(pairs={len(self._buffers)}, warm={warm})"
