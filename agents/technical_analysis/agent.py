"""
agents/technical_analysis/agent.py
------------------------------------
TechnicalAnalysisAgent — consumes OHLCV candles, computes indicators,
and publishes typed TechnicalSignal events to the decision pipeline.

Data flow:
    Redis (market.ohlcv.{symbol}.{timeframe})
        → CandleBufferRegistry (per-symbol rolling history)
        → SignalGenerator (indicator computation + weighting)
        → Redis (signal.technical.{symbol})
        → [optionally] Redis cache (for Decision agent fast access)

Key design decisions:
- The agent subscribes to ALL configured symbol/timeframe channels
  simultaneously using subscribe_many(). One subscription loop handles
  all candles — avoids N separate async tasks for N pairs.
- Signal generation is triggered on the PRIMARY timeframe only.
  Lower timeframes (e.g. "1m") feed the buffer but don't independently
  trigger signals. Only the primary timeframe's candle close triggers
  a new signal. This avoids signal spam on every 1m candle.
  The primary timeframe is the first in MARKET_DATA_OHLCV_TIMEFRAMES.
  TODO: Multi-timeframe confluence signals in Phase 5.
- Signal deduplication: we track the last signal time per symbol and
  suppress duplicate signals within signal_ttl_seconds / 2. This
  prevents the Decision agent from being flooded with identical signals
  when candles arrive quickly.
- Warmup handling: the agent logs warmup progress and silently skips
  signal generation while buffers are filling. The MonitoringAgent
  can see warmup_progress in heartbeats.
- MT5 note: this agent is MT5-agnostic. The signals it produces are
  consumed by the Decision agent, which feeds the ExecutionBroker
  interface. MT5Bridge will receive the same signals through the same
  pipeline without any changes here.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agents.base import BaseAgent, run_agent
from agents.technical_analysis.candle_buffer import CandleBufferRegistry
from agents.technical_analysis.ml.ml_signal_generator import MLSignalGenerator
from agents.technical_analysis.ml.model_registry import ModelRegistry
from agents.technical_analysis.signal_generator import SignalGenerator
from core.db.connection import get_session
from core.db.repositories.candle_repo import CandleRepository
from core.logging import get_logger
from core.messaging import Channels
from core.metrics import (
    MESSAGES_PUBLISHED,
    SIGNAL_CONFIDENCE,
    SIGNAL_GENERATION_SECONDS,
    time_histogram,
)
from core.models.market import OHLCVCandle
from core.models.signals import TechnicalSignal

logger = get_logger("technical_analysis_agent")

# Seconds per timeframe — used to compute the DB warmup look-back window
_TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


class TechnicalAnalysisAgent(BaseAgent):
    """
    Technical analysis agent.

    Subscribes to OHLCV candle events, maintains rolling price history
    per symbol/timeframe, and publishes confidence-scored signals.
    """

    name = "technical_analysis_agent"

    def __init__(self) -> None:
        super().__init__()
        self._registry = CandleBufferRegistry()
        self._rule_generator = SignalGenerator()
        # Keep ``_generator`` for any legacy code that still references it.
        self._generator = self._rule_generator
        # ML generator — loaded in setup() when use_ml_signals is True and
        # a trained model exists. None falls through to the rule-based path.
        self._ml_generator: MLSignalGenerator | None = None
        # Last signal emission time per symbol — for dedup
        self._last_signal_at: dict[str, datetime] = {}
        # Primary timeframe (signals only emitted on this timeframe's candles)
        self._primary_timeframe: str = ""
        # Channels to subscribe to
        self._ohlcv_channels: list[str] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        cfg = self.settings.market_data
        self._primary_timeframe = cfg.ohlcv_timeframes[0] if cfg.ohlcv_timeframes else "1m"

        # Build the list of channels to subscribe to
        self._ohlcv_channels = [
            Channels.ohlcv(symbol, timeframe)
            for symbol in cfg.symbols
            for timeframe in cfg.ohlcv_timeframes
        ]

        self.log.info(
            "ta_agent_setup",
            symbols=cfg.symbols,
            timeframes=cfg.ohlcv_timeframes,
            primary_timeframe=self._primary_timeframe,
            channels=self._ohlcv_channels,
        )

        self._maybe_load_ml_model()

        await self._warmup_from_db()

    def _maybe_load_ml_model(self) -> None:
        """
        Attempt to load an ML model for the first configured symbol/primary
        timeframe. Falls back silently to rules when disabled or no model exists.

        TODO: Per-symbol model selection. Today we load a single model for the
        first symbol and use it for all symbols. Per-symbol models are tracked
        in [Phase 9 — symbol-specific ML routing].
        """
        ta_cfg = self.settings.technical_analysis
        if not getattr(ta_cfg, "use_ml_signals", False):
            self.log.info("ml_signals_disabled_by_config")
            return

        cfg = self.settings.market_data
        if not cfg.symbols:
            return

        # Explicit override wins; otherwise consult the registry.
        override = getattr(ta_cfg, "ml_model_path", None)
        if override:
            model_path = Path(override)
        else:
            registry = ModelRegistry()
            latest = registry.get_latest_path(cfg.symbols[0], self._primary_timeframe)
            if latest is None:
                self.log.info(
                    "ml_model_not_found_using_rules",
                    symbol=cfg.symbols[0],
                    timeframe=self._primary_timeframe,
                )
                return
            model_path = latest

        generator = MLSignalGenerator(str(model_path))
        if generator.is_loaded():
            self._ml_generator = generator
            self.log.info("ml_model_loaded", path=str(model_path))
        else:
            self.log.warning("ml_model_load_failed_falling_back_to_rules", path=str(model_path))

    async def _warmup_from_db(self) -> None:
        """
        Pre-fill CandleBuffer instances from TimescaleDB historical data.

        Loads the most recent 200 candles per symbol/timeframe on startup so
        buffers reach is_warm immediately rather than waiting ~50 live candles.
        Falls back gracefully if the DB is empty or unreachable.
        """
        cfg = self.settings.market_data
        warmup_limit = 200

        for symbol in cfg.symbols:
            for timeframe in cfg.ohlcv_timeframes:
                try:
                    tf_seconds = _TIMEFRAME_SECONDS.get(timeframe, 60)
                    since = datetime.now(UTC) - timedelta(seconds=tf_seconds * warmup_limit)

                    async with get_session() as session:
                        repo = CandleRepository(session)
                        candles = await repo.get_candles(
                            symbol, timeframe, since=since, limit=warmup_limit
                        )

                    for candle in candles:
                        self._registry.add(candle)

                    buf = self._registry.get(symbol, timeframe)
                    is_warm = buf.is_warm if buf else False

                    self.log.info(
                        "buffer_warmed_from_db",
                        symbol=symbol,
                        timeframe=timeframe,
                        candles_loaded=len(candles),
                        buffer_warm=is_warm,
                    )
                except Exception as e:
                    self.log.warning(
                        "buffer_db_warmup_failed",
                        symbol=symbol,
                        timeframe=timeframe,
                        error=str(e),
                    )

    async def run_loop(self) -> None:
        """
        Subscribe to all OHLCV channels and process candles as they arrive.
        Uses subscribe_many() for efficient single-loop multi-channel consumption.
        """
        self.log.info(
            "ta_agent_listening",
            channel_count=len(self._ohlcv_channels),
            primary_timeframe=self._primary_timeframe,
        )

        async for channel, candle in self.bus.subscribe_many(
            self._ohlcv_channels,
            OHLCVCandle,
        ):
            if not self._should_continue():
                break

            try:
                await self._process_candle(channel, candle)
                self._record_success()
            except Exception as e:
                self._handle_error(e, context=f"process_candle[{channel}]")

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    async def _process_candle(self, channel: str, candle: OHLCVCandle) -> None:
        """
        Process a single incoming candle:
        1. Add to the appropriate buffer.
        2. If it's a primary-timeframe candle, attempt signal generation.
        3. Publish the signal if generated.
        """
        buffer = self._registry.add(candle)

        self.log.debug(
            "candle_received",
            symbol=candle.symbol,
            timeframe=candle.timeframe,
            close=float(candle.close),
            buffer_size=buffer.size,
            buffer_warm=buffer.is_warm,
        )

        # Only generate signals on primary timeframe candle closes
        if candle.timeframe != self._primary_timeframe:
            return

        # Skip if buffer isn't warm yet
        if not buffer.is_warm:
            self.log.debug(
                "buffer_warming_up",
                symbol=candle.symbol,
                progress_pct=round(buffer.warmup_progress * 100, 1),
            )
            return

        # Signal dedup: don't re-emit within half the TTL window
        if self._is_signal_suppressed(candle.symbol):
            return

        # Generate the signal (timed so we can alert on slow indicator runs).
        # Prefer the ML generator when loaded; fall back to rules otherwise.
        with time_histogram(SIGNAL_GENERATION_SECONDS, symbol=candle.symbol):
            if self._ml_generator is not None and self._ml_generator.is_loaded():
                signal = self._ml_generator.generate(buffer)
                source = "ml"
            else:
                signal = self._rule_generator.generate(buffer)
                source = "rules"

        if signal is not None:
            self.log.info(
                "signal_generated",
                symbol=candle.symbol,
                timeframe=candle.timeframe,
                source=source,
                direction=signal.direction,
                confidence=signal.confidence,
            )
            await self._publish_signal(signal)
            self._last_signal_at[candle.symbol] = datetime.now(UTC)

    async def _publish_signal(self, signal: TechnicalSignal) -> None:
        """Publish a TechnicalSignal to the event bus and cache it."""
        try:
            await self.bus.publish(signal)
            SIGNAL_CONFIDENCE.labels(symbol=signal.symbol).set(float(signal.confidence))
            MESSAGES_PUBLISHED.labels(agent=self.name, channel=signal.channel_key).inc()

            # Also cache in Redis for Decision agent fast-access
            # Key: signal_cache.technical.{symbol}
            cache_key = f"signal_cache.technical.{signal.symbol.replace('/', '-')}"
            await self.bus.cache_set(
                cache_key,
                signal,
                ttl_seconds=self.settings.technical_analysis.signal_ttl_seconds,
            )

            self.log.info(
                "signal_published",
                symbol=signal.symbol,
                direction=signal.direction,
                confidence=signal.confidence,
                composite_score=signal.metadata.get("composite_score"),
                indicators=signal.metadata.get("indicator_count"),
            )
        except Exception as e:
            self.log.error(
                "signal_publish_failed",
                symbol=signal.symbol,
                error=str(e),
            )

    # ------------------------------------------------------------------
    # Signal deduplication
    # ------------------------------------------------------------------

    def _is_signal_suppressed(self, symbol: str) -> bool:
        """
        Return True if we emitted a signal for this symbol too recently.

        Suppression window = signal_ttl_seconds / 3. This means:
        - A 5-minute TTL → suppress duplicate signals within ~100 seconds.
        - Prevents the Decision agent being flooded on fast candle arrival.
        """
        last = self._last_signal_at.get(symbol)
        if last is None:
            return False

        ttl = self.settings.technical_analysis.signal_ttl_seconds
        suppress_window = timedelta(seconds=ttl / 3)
        elapsed = datetime.now(UTC) - last

        if elapsed < suppress_window:
            self.log.debug(
                "signal_suppressed_duplicate",
                symbol=symbol,
                elapsed_seconds=elapsed.total_seconds(),
                suppress_window_seconds=suppress_window.total_seconds(),
            )
            return True

        return False

    # ------------------------------------------------------------------
    # Health reporting
    # ------------------------------------------------------------------

    def health_extra(self) -> dict:
        """Report buffer states and recent signal activity."""
        cfg = self.settings.market_data
        buffer_health = {}

        for symbol in cfg.symbols:
            buf = self._registry.get(symbol, self._primary_timeframe)
            if buf:
                last_signal = self._last_signal_at.get(symbol)
                buffer_health[symbol] = {
                    **buf.to_dict(),
                    "last_signal_seconds_ago": (
                        round((datetime.now(UTC) - last_signal).total_seconds(), 1)
                        if last_signal
                        else None
                    ),
                }

        return {
            "primary_timeframe": self._primary_timeframe,
            "buffer_count": len(self._registry),
            "warm_buffers": len(self._registry.warm_buffers()),
            "all_warm": self._registry.all_warm(),
            "buffers": buffer_health,
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    from core.logging import configure_logging

    configure_logging()
    asyncio.run(run_agent(TechnicalAnalysisAgent()))
