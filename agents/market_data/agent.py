"""
agents/market_data/agent.py
----------------------------
MarketDataAgent — the system's data ingestion entry point.

Responsibilities:
- Fetch OHLCV candles, order books, and tickers from the exchange.
- Normalise raw data into typed models via OHLCVNormaliser.
- Publish clean data to Redis pub/sub channels for downstream agents.
- Persist OHLCV candles to TimescaleDB for historical access.
- Maintain data freshness and detect stale feeds.
- Report health via heartbeats.

Data flow:
    Exchange API (CCXT)
        → ExchangeFetcher (raw OHLCV)
        → OHLCVNormaliser (validated, deduplicated)
        → Redis pub/sub (real-time feed to TA agent)
        → TimescaleDB (persistent historical store)

Polling strategy (Phase 1 — REST):
    Each (symbol, timeframe) pair has its own polling loop running
    concurrently via asyncio tasks. Loops are offset by a stagger delay
    to avoid hitting the exchange rate limit with simultaneous requests.

Phase 2 upgrade path:
    Replace polling loops with WebSocketHandler streams.
    The normaliser, publisher, and DB writer are unchanged.
    Only the data source changes.

Design decisions:
- One asyncio task per (symbol, timeframe) poll loop. This is clean and
  easy to monitor individually. Tasks restart on error via the circuit
  breaker in BaseAgent.
- Candles are published AND persisted. Publishing is fire-and-forget
  (Redis). Persistence is async but awaited — we don't want to lose data
  silently. If DB write fails, we log and continue (market data is more
  valuable as a live feed than a historical record).
- The primary timeframe (first in MARKET_DATA_OHLCV_TIMEFRAMES) is used
  for data freshness checks. This is typically "1m".
"""

from __future__ import annotations

import asyncio

from sqlalchemy import text

from agents.base import BaseAgent, run_agent
from agents.market_data.normalizer import TIMEFRAME_SECONDS, OHLCVNormaliser
from core.db.connection import get_session
from core.exceptions import ExchangeConnectionError
from core.logging import get_logger
from core.metrics import DATA_AGE_SECONDS, MESSAGES_PUBLISHED
from core.models.market import OHLCVCandle
from data_sources import DataSource, get_data_source

logger = get_logger("market_data_agent")

# Stagger delay between poll loop startups to spread exchange requests
_STAGGER_SECONDS = 0.5


class MarketDataAgent(BaseAgent):
    """
    Market data ingestion agent.

    Runs N concurrent polling loops (one per symbol/timeframe pair),
    normalises the data, and publishes to the event bus.
    """

    name = "market_data_agent"

    def __init__(self) -> None:
        super().__init__()
        # Market data flows through the pluggable DataSource layer (keyless public
        # exchange by default; optional local MT5). The agent never talks to CCXT
        # or MetaTrader directly.
        self._source: DataSource = get_data_source()
        self._normaliser = OHLCVNormaliser()
        self._poll_tasks: list[asyncio.Task] = []
        self._ticker_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Connect to the exchange and fetch historical warmup data."""
        await self._source.connect()
        self.log.info(
            "data_source_connected",
            source=self._source.source_name,
            sandbox=self._source.is_sandbox,
        )
        await self._warmup_history()

    async def teardown(self) -> None:
        """Cancel all polling tasks and disconnect from exchange."""
        for task in self._poll_tasks:
            if not task.done():
                task.cancel()
        if self._ticker_task and not self._ticker_task.done():
            self._ticker_task.cancel()

        # Wait for tasks to finish cancellation
        all_tasks = self._poll_tasks + ([self._ticker_task] if self._ticker_task else [])
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)

        await self._source.disconnect()
        self.log.info("market_data_agent_teardown_complete")

    async def run_loop(self) -> None:
        """
        Launch concurrent polling tasks for all configured symbol/timeframe pairs.
        Waits until stop is signalled, then exits.
        """
        cfg = self.settings.market_data
        stagger = 0.0

        for symbol in cfg.symbols:
            for timeframe in cfg.ohlcv_timeframes:
                await asyncio.sleep(stagger)
                task = asyncio.create_task(
                    self._poll_ohlcv(symbol, timeframe),
                    name=f"poll_{symbol.replace('/', '-')}_{timeframe}",
                )
                self._poll_tasks.append(task)
                stagger += _STAGGER_SECONDS

        # Ticker polling (less frequent — once per primary timeframe interval)
        self._ticker_task = asyncio.create_task(
            self._poll_tickers(),
            name="poll_tickers",
        )

        # Block until stop is requested
        await self._stop_event.wait()

        self.log.info("run_loop_exiting", active_tasks=len(self._poll_tasks))

    # ------------------------------------------------------------------
    # Polling loops
    # ------------------------------------------------------------------

    async def _poll_ohlcv(self, symbol: str, timeframe: str) -> None:
        """
        Continuous polling loop for one (symbol, timeframe) pair.

        Fetches the latest candles, normalises them, publishes to Redis,
        and persists to TimescaleDB. Sleeps for poll_interval_seconds
        between requests.
        """
        interval = self.settings.market_data.poll_interval_seconds
        expected_tf_seconds = TIMEFRAME_SECONDS.get(timeframe, 60)

        # Use a sensible poll interval: min of config interval and timeframe
        effective_interval = min(interval, expected_tf_seconds)

        self.log.info(
            "poll_loop_started",
            symbol=symbol,
            timeframe=timeframe,
            interval_seconds=effective_interval,
        )

        while self._should_continue():
            try:
                # Fetch the last 2 candles — enough to detect the latest close
                candles = await self._source.fetch_ohlcv(symbol, timeframe, limit=2)
                clean_candles = self._normaliser.process_batch(candles)

                for candle in clean_candles:
                    await self._publish_candle(candle)
                    await self._persist_candle(candle)

                self._record_success()

            except ExchangeConnectionError as e:
                self._handle_error(e, context=f"poll_ohlcv[{symbol}/{timeframe}]")
                await asyncio.sleep(effective_interval * 2)  # longer backoff
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._handle_error(e, context=f"poll_ohlcv[{symbol}/{timeframe}]")

            await asyncio.sleep(effective_interval)

        self.log.info("poll_loop_stopped", symbol=symbol, timeframe=timeframe)

    async def _poll_tickers(self) -> None:
        """
        Poll 24h ticker for all configured symbols.
        Runs every 30 seconds — ticker data doesn't need sub-minute freshness.
        """
        ticker_interval = 30.0

        while self._should_continue():
            for symbol in self.settings.market_data.symbols:
                try:
                    ticker = await self._source.fetch_ticker(symbol)
                    await self.bus.publish(ticker)
                    self.log.debug(
                        "ticker_published",
                        symbol=symbol,
                        last=float(ticker.last),
                    )
                except Exception as e:
                    self.log.warning("ticker_fetch_failed", symbol=symbol, error=str(e))

                if not self._should_continue():
                    break

            await asyncio.sleep(ticker_interval)

    # ------------------------------------------------------------------
    # Publish helpers
    # ------------------------------------------------------------------

    async def _publish_candle(self, candle: OHLCVCandle) -> None:
        """Publish a normalised candle to the Redis event bus."""
        try:
            await self.bus.publish(candle)
            # A candle has just been received and forwarded; its age relative
            # to "now" is effectively zero. Scrapes between candles will see
            # this gauge stay flat, which Prometheus rate panels handle fine.
            DATA_AGE_SECONDS.labels(symbol=candle.symbol).set(0.0)
            MESSAGES_PUBLISHED.labels(agent=self.name, channel=candle.channel_key).inc()
            self.log.debug(
                "candle_published",
                symbol=candle.symbol,
                timeframe=candle.timeframe,
                timestamp=str(candle.timestamp),
                close=float(candle.close),
                volume=float(candle.volume),
            )
        except Exception as e:
            self.log.error(
                "candle_publish_failed",
                symbol=candle.symbol,
                timeframe=candle.timeframe,
                error=str(e),
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _persist_candle(self, candle: OHLCVCandle) -> None:
        """
        Upsert a candle into TimescaleDB.

        Uses INSERT ... ON CONFLICT DO UPDATE to handle duplicate timestamps
        gracefully (e.g. if the same candle is received twice due to reconnect).
        Failure here is logged but does NOT stop the data feed.
        """
        try:
            async with get_session() as session:
                await session.execute(
                    text("""
                        INSERT INTO ohlcv_candles
                            (symbol, timeframe, timestamp, open, high, low, close,
                             volume, quote_volume, received_at)
                        VALUES
                            (:symbol, :timeframe, :timestamp, :open, :high, :low, :close,
                             :volume, :quote_volume, :received_at)
                        ON CONFLICT (symbol, timeframe, timestamp)
                        DO UPDATE SET
                            open         = EXCLUDED.open,
                            high         = EXCLUDED.high,
                            low          = EXCLUDED.low,
                            close        = EXCLUDED.close,
                            volume       = EXCLUDED.volume,
                            quote_volume = EXCLUDED.quote_volume,
                            received_at  = EXCLUDED.received_at
                    """),
                    {
                        "symbol": candle.symbol,
                        "timeframe": candle.timeframe,
                        "timestamp": candle.timestamp,
                        "open": float(candle.open),
                        "high": float(candle.high),
                        "low": float(candle.low),
                        "close": float(candle.close),
                        "volume": float(candle.volume),
                        "quote_volume": float(candle.quote_volume) if candle.quote_volume else None,
                        "received_at": candle.received_at,
                    },
                )
                await session.commit()
        except Exception as e:
            # Non-fatal — live feed continues even if DB write fails
            self.log.error(
                "candle_persist_failed",
                symbol=candle.symbol,
                timeframe=candle.timeframe,
                timestamp=str(candle.timestamp),
                error=str(e),
            )

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    async def _warmup_history(self) -> None:
        """
        Fetch historical candles on startup to warm up the normaliser
        and pre-populate TimescaleDB.

        This ensures the TA agent has enough data to compute indicators
        immediately (RSI needs 14 bars, MACD needs 26 bars, etc.)
        without waiting for live data to accumulate.
        """
        cfg = self.settings.market_data
        limit = cfg.ohlcv_history_limit

        self.log.info(
            "history_warmup_starting",
            symbols=cfg.symbols,
            timeframes=cfg.ohlcv_timeframes,
            candles_per_pair=limit,
        )

        for symbol in cfg.symbols:
            for timeframe in cfg.ohlcv_timeframes:
                try:
                    candles = await self._source.fetch_ohlcv(symbol, timeframe, limit=limit)
                    clean = self._normaliser.process_batch(candles)

                    # Persist and publish all historical candles so the TA
                    # agent buffer warms up via the live feed as a fallback
                    for candle in clean:
                        await self._persist_candle(candle)
                        await self._publish_candle(candle)

                    self.log.info(
                        "history_warmup_complete",
                        symbol=symbol,
                        timeframe=timeframe,
                        candles_fetched=len(candles),
                        candles_stored=len(clean),
                    )

                except Exception as e:
                    # Warmup failure is non-fatal — agent continues with
                    # whatever data it has. TA agent will get signals once
                    # enough live candles have been received.
                    self.log.warning(
                        "history_warmup_failed",
                        symbol=symbol,
                        timeframe=timeframe,
                        error=str(e),
                    )

    # ------------------------------------------------------------------
    # Health reporting
    # ------------------------------------------------------------------

    def health_extra(self) -> dict:
        """Include data freshness info in heartbeat payloads."""
        cfg = self.settings.market_data
        primary_tf = cfg.ohlcv_timeframes[0] if cfg.ohlcv_timeframes else "1m"

        freshness = {}
        for symbol in cfg.symbols:
            age = self._normaliser.get_data_age_seconds(symbol, primary_tf)
            freshness[symbol] = {
                "age_seconds": round(age, 1) if age is not None else None,
                "stale": age is not None and age > self.settings.risk.max_data_staleness_seconds,
            }

        return {
            "symbols": cfg.symbols,
            "timeframes": cfg.ohlcv_timeframes,
            "data_freshness": freshness,
            "normaliser_stats": self._normaliser.get_stats(),
            "active_poll_tasks": sum(1 for t in self._poll_tasks if not t.done()),
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    from core.logging import configure_logging

    configure_logging()
    asyncio.run(run_agent(MarketDataAgent()))
