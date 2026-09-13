"""
core/models/market.py
---------------------
Canonical market data models — the typed contracts published by MarketDataAgent
and consumed by every downstream agent.

These models are immutable (frozen=True) data transfer objects. They are the
single source of truth for what "a candle" or "an orderbook" looks like in
this system. Any agent receiving a market event can rely on these types.

Design decisions:
- frozen=True: models are immutable after creation. This prevents bugs where
  a downstream agent accidentally mutates shared state.
- All timestamps are UTC datetime objects, never raw integers. Timezone
  ambiguity is a common source of subtle trading bugs.
- Decimal is used for prices/volumes where precision matters. float is
  acceptable for derived quantities (indicators, scores) where tiny rounding
  errors are irrelevant.
- Each model has a channel_key() method returning the Redis pub/sub channel
  name for that specific message — keeps channel naming logic co-located with
  the data model, not scattered across agents.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Timeframe(str, Enum):
    """Supported OHLCV timeframes. Extend as needed."""

    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"


class MarketEventType(str, Enum):
    """Discriminator for market data events on the bus."""

    OHLCV = "ohlcv"
    ORDERBOOK = "orderbook"
    TICKER = "ticker"
    TRADE = "trade"


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class BaseMarketModel(BaseModel):
    """
    Shared base for all market data models.
    Provides JSON serialisation config and common utility methods.
    """

    model_config = {
        "frozen": True,  # immutable after construction
        "use_enum_values": True,  # store enum .value, not enum object
        "populate_by_name": True,
    }

    def to_json(self) -> str:
        """Serialise to JSON string for Redis pub/sub."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str | bytes) -> BaseMarketModel:
        """Deserialise from Redis pub/sub message."""
        return cls.model_validate_json(data)


# ---------------------------------------------------------------------------
# OHLCV Candle
# ---------------------------------------------------------------------------


class OHLCVCandle(BaseMarketModel):
    """
    A single OHLCV (Open/High/Low/Close/Volume) candle.

    Published on: market.ohlcv.{symbol}.{timeframe}
    Example channel: market.ohlcv.BTC-USDT.1m

    Note: symbol uses '-' not '/' in channel names to avoid Redis key issues.
    The exchange-facing symbol (BTC/USDT) is stored as-is in the model.
    """

    event_type: MarketEventType = MarketEventType.OHLCV
    symbol: str = Field(description="Trading pair, e.g. 'BTC/USDT'")
    timeframe: str = Field(description="Candle timeframe, e.g. '1m'")
    timestamp: datetime = Field(description="Candle open time (UTC)")
    open: Decimal = Field(description="Open price")
    high: Decimal = Field(description="High price")
    low: Decimal = Field(description="Low price")
    close: Decimal = Field(description="Close price")
    volume: Decimal = Field(description="Base asset volume")
    quote_volume: Decimal | None = Field(default=None, description="Quote asset volume")
    is_closed: bool = Field(default=True, description="False if candle is still forming")
    received_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Wall-clock time this candle was received (for staleness checks)",
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_utc(cls, v: datetime | int | float) -> datetime:
        """Accept CCXT's millisecond timestamps or datetime objects."""
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(v / 1000, tz=UTC)
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v

    @model_validator(mode="after")
    def validate_ohlcv_consistency(self) -> OHLCVCandle:
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) < low ({self.low}) — invalid candle")
        if self.high < self.open or self.high < self.close:
            raise ValueError("high must be >= open and close")
        if self.low > self.open or self.low > self.close:
            raise ValueError("low must be <= open and close")
        if self.volume < 0:
            raise ValueError("volume cannot be negative")
        return self

    @property
    def channel_key(self) -> str:
        safe_symbol = self.symbol.replace("/", "-")
        return f"market.ohlcv.{safe_symbol}.{self.timeframe}"

    @property
    def price_change_pct(self) -> float:
        """Percentage price change over the candle."""
        if self.open == 0:
            return 0.0
        return float((self.close - self.open) / self.open * 100)

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open

    @classmethod
    def from_ccxt(cls, raw: list, symbol: str, timeframe: str) -> OHLCVCandle:
        """
        Construct from CCXT's raw OHLCV list format:
        [timestamp_ms, open, high, low, close, volume]
        """
        return cls(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=raw[0],
            open=Decimal(str(raw[1])),
            high=Decimal(str(raw[2])),
            low=Decimal(str(raw[3])),
            close=Decimal(str(raw[4])),
            volume=Decimal(str(raw[5])),
        )


# ---------------------------------------------------------------------------
# Order Book
# ---------------------------------------------------------------------------


class OrderBookLevel(BaseMarketModel):
    """A single price level in an order book (price, quantity)."""

    price: Decimal
    quantity: Decimal


class OrderBook(BaseMarketModel):
    """
    L2 order book snapshot.
    Published on: market.orderbook.{symbol}
    """

    event_type: MarketEventType = MarketEventType.ORDERBOOK
    symbol: str
    timestamp: datetime
    bids: list[OrderBookLevel] = Field(description="Sorted best bid first (descending price)")
    asks: list[OrderBookLevel] = Field(description="Sorted best ask first (ascending price)")
    received_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_utc(cls, v: datetime | int | float) -> datetime:
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(v / 1000, tz=UTC)
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v

    @property
    def channel_key(self) -> str:
        return f"market.orderbook.{self.symbol.replace('/', '-')}"

    @property
    def best_bid(self) -> Decimal | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Decimal | None:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Decimal | None:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_pct(self) -> float | None:
        if self.mid_price and self.spread:
            return float(self.spread / self.mid_price * 100)
        return None

    @classmethod
    def from_ccxt(cls, raw: dict, symbol: str) -> OrderBook:
        """Construct from CCXT's fetch_order_book() response."""
        return cls(
            symbol=symbol,
            timestamp=raw.get("timestamp") or datetime.now(UTC),
            bids=[
                OrderBookLevel(price=Decimal(str(b[0])), quantity=Decimal(str(b[1])))
                for b in (raw.get("bids") or [])
            ],
            asks=[
                OrderBookLevel(price=Decimal(str(a[0])), quantity=Decimal(str(a[1])))
                for a in (raw.get("asks") or [])
            ],
        )


# ---------------------------------------------------------------------------
# Ticker
# ---------------------------------------------------------------------------


class Ticker(BaseMarketModel):
    """
    24-hour rolling ticker snapshot.
    Published on: market.ticker.{symbol}
    """

    event_type: MarketEventType = MarketEventType.TICKER
    symbol: str
    timestamp: datetime
    last: Decimal = Field(description="Last traded price")
    bid: Decimal | None = None
    ask: Decimal | None = None
    high_24h: Decimal | None = None
    low_24h: Decimal | None = None
    volume_24h: Decimal | None = None
    change_24h_pct: float | None = None
    received_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_utc(cls, v: datetime | int | float) -> datetime:
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(v / 1000, tz=UTC)
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v

    @property
    def channel_key(self) -> str:
        return f"market.ticker.{self.symbol.replace('/', '-')}"

    @classmethod
    def from_ccxt(cls, raw: dict) -> Ticker:
        return cls(
            symbol=raw["symbol"],
            timestamp=raw.get("timestamp") or datetime.now(UTC),
            last=Decimal(str(raw["last"])),
            bid=Decimal(str(raw["bid"])) if raw.get("bid") else None,
            ask=Decimal(str(raw["ask"])) if raw.get("ask") else None,
            high_24h=Decimal(str(raw["high"])) if raw.get("high") else None,
            low_24h=Decimal(str(raw["low"])) if raw.get("low") else None,
            volume_24h=Decimal(str(raw["baseVolume"])) if raw.get("baseVolume") else None,
            change_24h_pct=raw.get("percentage"),
        )
