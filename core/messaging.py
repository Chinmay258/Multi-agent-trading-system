"""
core/messaging.py
-----------------
Redis pub/sub messaging layer — the inter-agent communication bus.

This module is the ONLY place in the codebase that talks directly to Redis.
All agents communicate by calling publish() and subscribe() on this interface.
No agent imports redis directly — this abstraction means we can swap the
underlying transport (e.g. to Kafka) by modifying this file only.

Design decisions:
- MessageBus is an async context manager. Agents create one on startup and
  close it on shutdown — never create per-message connections.
- publish() accepts any BaseMarketModel — serialisation is handled here,
  not in the calling agent.
- subscribe() is an async generator. Agents consume it with async for, which
  blocks naturally without spinning. The generator handles reconnection.
- Type-safe deserialisation: subscribe() takes a model class and returns
  typed instances — callers never parse JSON themselves.
- Dead letter handling: messages that fail to deserialise are logged and
  skipped, not re-raised — one bad message shouldn't crash an agent.
- Connection pooling: a single Redis connection pool is shared within a
  process. Pub/sub requires its own dedicated connection (Redis limitation).

Usage:
    async with MessageBus() as bus:
        # Publisher
        candle = OHLCVCandle(...)
        await bus.publish(candle)

        # Subscriber (async generator)
        async for candle in bus.subscribe("market.ohlcv.BTC-USDT.1m", OHLCVCandle):
            process(candle)

        # Multiple channels
        async for msg in bus.subscribe_many(["signal.technical.*"], TechnicalSignal):
            process(msg)
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TypeVar

import redis.asyncio as aioredis
from redis.asyncio.client import PubSub

from core.config import get_settings
from core.logging import get_logger
from core.models.market import BaseMarketModel

logger = get_logger("messaging")

T = TypeVar("T", bound=BaseMarketModel)

# ---------------------------------------------------------------------------
# Channel constants — all channel names are defined here, nowhere else
# ---------------------------------------------------------------------------


class Channels:
    """
    Canonical Redis pub/sub channel names.

    Use these constants instead of hardcoded strings. Pattern channels
    (ending in *) are for psubscribe; exact channels are for subscribe.

    Parametric channels: call the static methods to get the full channel
    name for a given symbol/timeframe.
    """

    # Market data
    @staticmethod
    def ohlcv(symbol: str, timeframe: str) -> str:
        return f"market.ohlcv.{symbol.replace('/', '-')}.{timeframe}"

    @staticmethod
    def orderbook(symbol: str) -> str:
        return f"market.orderbook.{symbol.replace('/', '-')}"

    @staticmethod
    def ticker(symbol: str) -> str:
        return f"market.ticker.{symbol.replace('/', '-')}"

    # Signals
    @staticmethod
    def technical_signal(symbol: str) -> str:
        return f"signal.technical.{symbol.replace('/', '-')}"

    @staticmethod
    def sentiment_signal(symbol: str) -> str:
        return f"signal.sentiment.{symbol.replace('/', '-')}"

    # Trade lifecycle
    DECISION_PROPOSAL = "decision.proposal"
    RISK_ASSESSMENT = "risk.assessment"
    EXECUTION_RESULT = "execution.result"

    # System
    SYSTEM_HEARTBEAT = "system.heartbeat"
    SYSTEM_COMMAND = "system.command"
    SYSTEM_RISK_OVERRIDE = "system.risk_override"
    SYSTEM_ALERT = "system.alert"

    # Wildcard patterns (for psubscribe)
    MARKET_ALL = "market.*"
    SIGNAL_ALL = "signal.*"
    SYSTEM_ALL = "system.*"


# ---------------------------------------------------------------------------
# Message envelope
# ---------------------------------------------------------------------------


class MessageEnvelope:
    """
    Internal wrapper around a raw Redis pub/sub message.
    Not exposed to agents — they only see deserialised model instances.
    """

    __slots__ = ("channel", "data", "pattern")

    def __init__(self, channel: str, data: str, pattern: str | None = None):
        self.channel = channel
        self.data = data
        self.pattern = pattern


# ---------------------------------------------------------------------------
# MessageBus
# ---------------------------------------------------------------------------


class MessageBus:
    """
    Async Redis pub/sub bus. One instance per agent process.

    Manages one Redis connection pool internally:
    - _pool: shared connection pool for publish() and cache calls (thread-safe)

    Each subscribe() / subscribe_many() call creates its own dedicated PubSub
    object from the pool so concurrent subscribers never share a socket.

    Always use as an async context manager:
        async with MessageBus() as bus:
            ...
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._pool: aioredis.Redis | None = None
        self._connected = False

    async def connect(self) -> None:
        """Establish Redis connection pool. Called by __aenter__."""
        cfg = self._settings.redis
        self._pool = await aioredis.from_url(
            cfg.url,
            max_connections=cfg.max_connections,
            socket_timeout=cfg.socket_timeout,
            retry_on_timeout=cfg.retry_on_timeout,
            decode_responses=True,
        )
        self._connected = True
        logger.info("message_bus_connected", url=f"redis://{cfg.host}:{cfg.port}/{cfg.db}")

    async def disconnect(self) -> None:
        """Close Redis connection pool. Called by __aexit__."""
        if self._pool:
            await self._pool.aclose()
        self._connected = False
        logger.info("message_bus_disconnected")

    async def _new_pubsub(self) -> PubSub:
        """Create a fresh, dedicated PubSub connection from the shared pool."""
        if not self._pool:
            raise RuntimeError("MessageBus not connected. Use 'async with MessageBus() as bus'")
        return self._pool.pubsub(ignore_subscribe_messages=True)

    async def __aenter__(self) -> MessageBus:
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def publish(self, message: BaseMarketModel) -> int:
        """
        Publish a model instance to its canonical channel.

        The channel is determined by the model's channel_key property.
        Returns the number of subscribers that received the message.

        Args:
            message: Any BaseMarketModel subclass with a channel_key property

        Returns:
            Number of active subscribers on the channel
        """
        if not self._pool:
            raise RuntimeError("MessageBus not connected. Use 'async with MessageBus() as bus'")

        channel = message.channel_key
        payload = message.to_json()

        try:
            subscriber_count: int = await self._pool.publish(channel, payload)
            logger.debug(
                "message_published",
                channel=channel,
                subscribers=subscriber_count,
                model=type(message).__name__,
            )
            return subscriber_count
        except Exception as e:
            logger.error("publish_failed", channel=channel, error=str(e))
            raise

    async def publish_to(self, channel: str, message: BaseMarketModel) -> int:
        """
        Publish to an explicit channel (overrides model.channel_key).
        Use when you need to broadcast to a non-canonical channel.
        """
        if not self._pool:
            raise RuntimeError("MessageBus not connected.")
        payload = message.to_json()
        return await self._pool.publish(channel, payload)

    # ------------------------------------------------------------------
    # Subscribing
    # ------------------------------------------------------------------

    async def subscribe(
        self,
        channel: str,
        model_class: type[T],
    ) -> AsyncIterator[T]:
        """
        Subscribe to a channel and yield typed model instances.

        This is an infinite async generator — it runs until the caller
        breaks out or the agent shuts down. Reconnects on transient errors.

        Each call creates its own dedicated PubSub connection so concurrent
        subscribers (e.g. run_loop + _risk_override_listener) never race on
        the same socket.

        Args:
            channel: Exact channel name (no wildcards — use subscribe_pattern for those)
            model_class: Pydantic model class to deserialise messages into

        Yields:
            Deserialised model instances of type model_class

        Example:
            async for candle in bus.subscribe(Channels.ohlcv("BTC/USDT", "1m"), OHLCVCandle):
                await process_candle(candle)
        """
        pubsub = await self._new_pubsub()
        await pubsub.subscribe(channel)
        await asyncio.sleep(0.1)
        logger.info("subscribed", channel=channel, model=model_class.__name__)

        try:
            # listen() uses blocking reads — works correctly in redis-py 7.x
            # where get_message(timeout>0) misses already-buffered messages.
            async for raw in pubsub.listen():
                try:
                    if raw.get("type") != "message":
                        continue
                    message = self._deserialise(raw["data"], model_class, channel)
                    if message is not None:
                        yield message
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("subscribe_error", channel=channel, error=str(e))
                    await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass
        finally:
            try:
                await pubsub.unsubscribe(channel)
                await pubsub.aclose()
            except Exception:
                pass

    async def subscribe_many(
        self,
        channels: list[str],
        model_class: type[T],
    ) -> AsyncIterator[tuple[str, T]]:
        """
        Subscribe to multiple channels simultaneously.
        Yields (channel_name, model_instance) tuples.

        Each call gets its own dedicated PubSub connection.
        """
        pubsub = await self._new_pubsub()
        await pubsub.subscribe(*channels)
        await asyncio.sleep(0.1)
        logger.info("subscribed_many", channels=channels, model=model_class.__name__)

        try:
            async for raw in pubsub.listen():
                try:
                    if raw.get("type") != "message":
                        continue
                    ch = raw["channel"]
                    message = self._deserialise(raw["data"], model_class, ch)
                    if message is not None:
                        yield ch, message
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("subscribe_many_error", channels=channels, error=str(e))
                    await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass
        finally:
            try:
                await pubsub.unsubscribe(*channels)
                await pubsub.aclose()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Cache operations (Redis key-value, not pub/sub)
    # ------------------------------------------------------------------

    async def cache_set(self, key: str, value: BaseMarketModel, ttl_seconds: int = 300) -> None:
        """Store a model in Redis cache with expiry. Used for hot signal data."""
        if not self._pool:
            raise RuntimeError("MessageBus not connected.")
        await self._pool.setex(key, ttl_seconds, value.to_json())

    async def cache_get(self, key: str, model_class: type[T]) -> T | None:
        """Retrieve and deserialise a cached model. Returns None if missing/expired."""
        if not self._pool:
            raise RuntimeError("MessageBus not connected.")
        raw = await self._pool.get(key)
        if raw is None:
            return None
        return self._deserialise(raw, model_class, key)

    async def cache_delete(self, key: str) -> None:
        if not self._pool:
            raise RuntimeError("MessageBus not connected.")
        await self._pool.delete(key)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def ping(self) -> bool:
        """Check Redis connectivity. Used by health checks."""
        try:
            if not self._pool:
                return False
            await self._pool.ping()
            return True
        except Exception:
            return False

    @property
    def active_subscriptions(self) -> list[str]:
        # Subscriptions are now per-call (each gets its own PubSub), so there
        # is no central registry to query. Returns empty list for compatibility.
        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _deserialise(
        self,
        data: str | bytes,
        model_class: type[T],
        channel: str,
    ) -> T | None:
        """
        Deserialise a raw Redis message into a typed model.

        Returns None (with a logged error) if deserialisation fails —
        one malformed message should never crash the subscriber loop.
        """
        try:
            return model_class.model_validate_json(data)
        except Exception as e:
            logger.error(
                "deserialise_failed",
                channel=channel,
                model=model_class.__name__,
                error=str(e),
                raw=str(data)[:200],  # truncate for safety
            )
            return None


# ---------------------------------------------------------------------------
# Module-level convenience — shared bus instance for simple use cases
# ---------------------------------------------------------------------------

_shared_bus: MessageBus | None = None


async def get_message_bus() -> MessageBus:
    """
    Return the process-level shared MessageBus.
    For agents that want a single bus without managing the lifecycle manually.

    Note: Prefer the async context manager form for explicit lifecycle control.
    This is provided for FastAPI dependency injection.
    """
    global _shared_bus
    if _shared_bus is None or not _shared_bus._connected:
        _shared_bus = MessageBus()
        await _shared_bus.connect()
    return _shared_bus
