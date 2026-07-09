"""
agents/market_data/websocket_handler.py
----------------------------------------
WebSocket feed handler for live OHLCV and order book streaming.

Upgrades from REST polling (Phase 1) to persistent WebSocket streams.
REST polling gives us a candle every N seconds on a timer.
WebSocket gives us candles as they close — lower latency, fewer API calls.

Design decisions:
- WebSocket is optional and configured per-exchange. Not all CCXT exchanges
  support async WebSocket streaming (ccxt.pro is required for that). We
  detect capability at runtime and fall back to REST if unavailable.
- The handler runs as a background task inside MarketDataAgent. It pushes
  candles to a shared asyncio.Queue, which the agent reads and publishes.
  This decouples network I/O from event publishing.
- Reconnect logic: exponential backoff up to max_attempts. After exhausting
  retries, the handler signals the agent to fall back to REST polling.
- Per-stream health tracking: each (symbol, timeframe) stream has its own
  last_message_at timestamp. The monitoring agent checks these.

TODO: Implement ccxt.pro WebSocket streaming when ready.
      For now this module provides the interface and REST fallback mechanism.
      Phase 2 will complete the WS implementation.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from core.config import get_settings
from core.logging import get_logger

logger = get_logger("websocket_handler")


class StreamHealth:
    """Tracks health of a single WebSocket stream."""

    def __init__(self, symbol: str, timeframe: str) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.connected = False
        self.last_message_at: datetime | None = None
        self.message_count: int = 0
        self.error_count: int = 0
        self.reconnect_count: int = 0

    @property
    def age_seconds(self) -> float | None:
        if self.last_message_at is None:
            return None
        return (datetime.now(UTC) - self.last_message_at).total_seconds()

    def record_message(self) -> None:
        self.message_count += 1
        self.last_message_at = datetime.now(UTC)

    def record_error(self) -> None:
        self.error_count += 1

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "connected": self.connected,
            "last_message_age_seconds": self.age_seconds,
            "message_count": self.message_count,
            "error_count": self.error_count,
            "reconnect_count": self.reconnect_count,
        }


class WebSocketHandler:
    """
    Manages WebSocket streams for multiple symbol/timeframe pairs.

    The handler maintains one stream per (symbol, timeframe) pair and
    pushes received candles to a shared output queue consumed by
    MarketDataAgent.

    Current status: REST fallback mode.
    TODO Phase 2: Integrate ccxt.pro for true WebSocket streaming.
                  Interface is designed; swap _stream_single() implementation.
    """

    def __init__(self, output_queue: asyncio.Queue) -> None:
        self._settings = get_settings()
        self._output_queue: asyncio.Queue = output_queue
        self._streams: dict[tuple[str, str], StreamHealth] = {}
        self._tasks: dict[tuple[str, str], asyncio.Task] = {}
        self._running = False

    async def start_stream(self, symbol: str, timeframe: str) -> None:
        """
        Start streaming candles for a symbol/timeframe pair.
        Launches a background task that pushes candles to output_queue.
        """
        key = (symbol, timeframe)
        if key in self._tasks and not self._tasks[key].done():
            logger.warning(
                "stream_already_running",
                symbol=symbol,
                timeframe=timeframe,
            )
            return

        health = StreamHealth(symbol, timeframe)
        self._streams[key] = health
        self._running = True

        task = asyncio.create_task(
            self._stream_with_reconnect(symbol, timeframe, health),
            name=f"ws_stream_{symbol.replace('/', '-')}_{timeframe}",
        )
        self._tasks[key] = task
        logger.info("stream_started", symbol=symbol, timeframe=timeframe)

    async def stop_stream(self, symbol: str, timeframe: str) -> None:
        """Stop a specific stream."""
        key = (symbol, timeframe)
        task = self._tasks.get(key)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.pop(key, None)
        logger.info("stream_stopped", symbol=symbol, timeframe=timeframe)

    async def stop_all(self) -> None:
        """Stop all active streams."""
        self._running = False
        for key in list(self._tasks.keys()):
            symbol, timeframe = key
            await self.stop_stream(symbol, timeframe)

    def get_health(self) -> dict:
        """Return health status for all streams."""
        return {f"{s}/{tf}": health.to_dict() for (s, tf), health in self._streams.items()}

    # ------------------------------------------------------------------
    # Internal stream management
    # ------------------------------------------------------------------

    async def _stream_with_reconnect(
        self,
        symbol: str,
        timeframe: str,
        health: StreamHealth,
    ) -> None:
        """
        Wrapper that handles reconnection with exponential backoff.
        Retries up to ws_reconnect_attempts times before giving up.
        """
        cfg = self._settings.exchange
        attempt = 0

        while self._running and attempt <= cfg.ws_reconnect_attempts:
            try:
                health.connected = True
                await self._stream_single(symbol, timeframe, health)
            except asyncio.CancelledError:
                break
            except Exception as e:
                health.connected = False
                health.record_error()
                attempt += 1
                health.reconnect_count += 1

                if attempt > cfg.ws_reconnect_attempts:
                    logger.error(
                        "stream_max_reconnects_exceeded",
                        symbol=symbol,
                        timeframe=timeframe,
                        attempts=attempt,
                    )
                    break

                wait = min(cfg.ws_reconnect_delay * (2 ** (attempt - 1)), 60.0)
                logger.warning(
                    "stream_reconnecting",
                    symbol=symbol,
                    timeframe=timeframe,
                    attempt=attempt,
                    wait_seconds=wait,
                    error=str(e),
                )
                await asyncio.sleep(wait)
            else:
                # Clean exit from _stream_single — don't reconnect
                break

        health.connected = False
        logger.info("stream_exited", symbol=symbol, timeframe=timeframe)

    async def _stream_single(
        self,
        symbol: str,
        timeframe: str,
        health: StreamHealth,
    ) -> None:
        """
        The actual stream implementation.

        TODO Phase 2: Replace this with ccxt.pro WebSocket streaming:

            import ccxt.pro as ccxtpro
            exchange = ccxtpro.binance(config)
            while True:
                candles = await exchange.watch_ohlcv(symbol, timeframe)
                for raw in candles:
                    candle = OHLCVCandle.from_ccxt(raw, symbol, timeframe)
                    health.record_message()
                    await self._output_queue.put(candle)

        Current implementation signals that WS is not available so
        MarketDataAgent falls back to REST polling.
        """
        # Signal REST fallback — raise to trigger reconnect loop exit
        logger.info(
            "websocket_not_implemented_using_rest_fallback",
            symbol=symbol,
            timeframe=timeframe,
        )
        # Return cleanly — agent will use REST polling
        return
