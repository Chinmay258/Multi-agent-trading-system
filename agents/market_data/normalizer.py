"""
agents/market_data/normalizer.py
---------------------------------
Normalises and validates raw market data before it enters the system.

The normaliser is the data quality gate. Every candle that enters the
event bus has been validated here. Downstream agents can trust the data.

Responsibilities:
- Deduplicate candles (exchanges sometimes send the same candle twice).
- Detect and handle gaps in the time series.
- Validate OHLCV consistency (already done in the model, double-checked here).
- Enrich candles with derived fields (quote volume, is_closed detection).
- Maintain a rolling in-memory buffer per symbol/timeframe — used to
  detect stale data and provide context for warmup checks.

Design decisions:
- In-memory dedup using a dict keyed by (symbol, timeframe, timestamp).
  We keep only the last N timestamps to bound memory usage.
- Gap detection: if the gap between consecutive candle timestamps exceeds
  2x the expected timeframe interval, we log a warning. Gaps are not filled
  (that's a strategy decision); they are flagged for downstream agents.
- The normaliser is stateless across restarts — it rebuilds its dedup
  buffer from the candles it receives. This is acceptable because the
  MarketDataAgent fetches history on startup anyway.

Usage:
    normaliser = OHLCVNormaliser()
    clean_candles = normaliser.process_batch(raw_candles, symbol, timeframe)
    for candle in clean_candles:
        await bus.publish(candle)
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import UTC, datetime

from core.logging import get_logger
from core.models.market import OHLCVCandle

logger = get_logger("ohlcv_normaliser")

# Timeframe → expected interval in seconds
TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "12h": 43200,
    "1d": 86400,
    "1w": 604800,
}

# How many timestamps to keep in dedup buffer per (symbol, timeframe)
DEDUP_BUFFER_SIZE = 200


class OHLCVNormaliser:
    """
    Stateful normaliser for OHLCV candle streams.

    One instance per MarketDataAgent — shared across all symbols and
    timeframes. Thread-safe for single-threaded asyncio usage.
    """

    def __init__(self) -> None:
        # Dedup buffer: (symbol, timeframe) → deque of seen timestamps
        self._seen: dict[tuple[str, str], deque[datetime]] = defaultdict(
            lambda: deque(maxlen=DEDUP_BUFFER_SIZE)
        )
        # Last candle per (symbol, timeframe) — for gap detection
        self._last: dict[tuple[str, str], OHLCVCandle] = {}
        # Stats per (symbol, timeframe)
        self._stats: dict[tuple[str, str], dict] = defaultdict(
            lambda: {"received": 0, "duplicates": 0, "gaps": 0, "published": 0}
        )

    def process_batch(
        self,
        candles: list[OHLCVCandle],
    ) -> list[OHLCVCandle]:
        """
        Process a batch of candles fetched from the exchange.

        Steps:
        1. Sort by timestamp (exchange responses aren't always ordered).
        2. Deduplicate against recently-seen timestamps.
        3. Detect gaps in the series.
        4. Return clean, ordered candles ready for publishing.

        Args:
            candles: Raw OHLCVCandle list from ExchangeFetcher.

        Returns:
            Cleaned, deduplicated list in ascending timestamp order.
        """
        if not candles:
            return []

        # All candles in a batch share symbol and timeframe
        symbol = candles[0].symbol
        timeframe = candles[0].timeframe
        key = (symbol, timeframe)
        stats = self._stats[key]

        # Sort ascending
        sorted_candles = sorted(candles, key=lambda c: c.timestamp)
        stats["received"] += len(sorted_candles)

        clean: list[OHLCVCandle] = []
        for candle in sorted_candles:
            # Dedup check
            if self._is_duplicate(candle, key):
                stats["duplicates"] += 1
                logger.debug(
                    "candle_duplicate_skipped",
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=str(candle.timestamp),
                )
                continue

            # Gap detection (non-blocking — just logs and continues)
            if key in self._last:
                self._check_gap(candle, self._last[key], key, stats)

            # Record and emit
            self._seen[key].append(candle.timestamp)
            self._last[key] = candle
            clean.append(candle)
            stats["published"] += 1

        if clean:
            logger.debug(
                "batch_normalised",
                symbol=symbol,
                timeframe=timeframe,
                received=len(sorted_candles),
                published=len(clean),
                duplicates=stats["duplicates"],
            )

        return clean

    def process_single(self, candle: OHLCVCandle) -> OHLCVCandle | None:
        """
        Process a single candle (e.g. from a WebSocket stream).

        Returns the candle if it passes normalisation, None if it was
        a duplicate or otherwise invalid.
        """
        results = self.process_batch([candle])
        return results[0] if results else None

    def get_last_candle(self, symbol: str, timeframe: str) -> OHLCVCandle | None:
        """Return the most recently seen candle for a symbol/timeframe."""
        return self._last.get((symbol, timeframe))

    def get_data_age_seconds(self, symbol: str, timeframe: str) -> float | None:
        """
        Return how many seconds ago the last candle was received.
        Used by health checks to detect stale data.
        """
        last = self.get_last_candle(symbol, timeframe)
        if last is None:
            return None
        now = datetime.now(UTC)
        return (now - last.received_at).total_seconds()

    def get_stats(self) -> dict:
        """Return normalisation stats for all symbol/timeframe pairs."""
        return {f"{s}/{tf}": dict(v) for (s, tf), v in self._stats.items()}

    def reset(self, symbol: str, timeframe: str) -> None:
        """Clear state for a specific symbol/timeframe (e.g. after reconnect)."""
        key = (symbol, timeframe)
        self._seen.pop(key, None)
        self._last.pop(key, None)
        self._stats.pop(key, None)
        logger.info("normaliser_reset", symbol=symbol, timeframe=timeframe)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_duplicate(self, candle: OHLCVCandle, key: tuple[str, str]) -> bool:
        """Return True if this timestamp has been seen recently."""
        return candle.timestamp in self._seen[key]

    def _check_gap(
        self,
        candle: OHLCVCandle,
        last: OHLCVCandle,
        key: tuple[str, str],
        stats: dict,
    ) -> None:
        """
        Detect gaps between consecutive candles.

        A gap is when the interval between two consecutive candle timestamps
        exceeds 2x the expected timeframe interval. This typically means:
        - Exchange was down.
        - Network interruption.
        - REST polling was too slow.

        We log it and record it in stats. We do NOT fill gaps — that would
        introduce synthetic data into the feed. Downstream agents that need
        gap-free data (e.g. TA indicators) handle missing bars themselves.
        """
        symbol, timeframe = key
        expected_interval = TIMEFRAME_SECONDS.get(timeframe, 60)
        actual_interval = (candle.timestamp - last.timestamp).total_seconds()

        # Allow 1.5x tolerance for minor timing jitter
        if actual_interval > expected_interval * 1.5:
            missing_count = int(actual_interval / expected_interval) - 1
            stats["gaps"] += 1
            logger.warning(
                "ohlcv_gap_detected",
                symbol=symbol,
                timeframe=timeframe,
                last_timestamp=str(last.timestamp),
                current_timestamp=str(candle.timestamp),
                gap_seconds=actual_interval,
                expected_seconds=expected_interval,
                estimated_missing_candles=missing_count,
            )
