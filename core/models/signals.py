"""
core/models/signals.py
----------------------
Signal models — the typed contracts published by the Technical Analysis
and Sentiment agents and consumed by the Decision agent.

A signal represents an agent's directional opinion on a symbol at a point
in time, along with a confidence score. The Decision agent combines signals
from multiple sources using a configurable weighting scheme.

Design decisions:
- SignalDirection is a clean enum, not a float: downstream agents shouldn't
  have to guess what 0.7 means. The confidence score is separate.
- Confidence is [0.0, 1.0]. The Decision agent ignores signals below the
  configured threshold (default: 0.6).
- expires_at is computed on construction from signal_ttl_seconds config.
  A stale signal is worse than no signal — the Decision agent must check.
- Signals carry their source agent name for audit logging and debugging.
- The AggregatedSignal model captures the Decision agent's fusion output,
  which is what gets stored in the DB for analysis.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import Enum
from uuid import UUID, uuid4

from pydantic import Field, field_validator

from core.models.market import BaseMarketModel

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SignalDirection(str, Enum):
    """Directional opinion from an analysis agent."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


def _to_direction(v: str | SignalDirection) -> SignalDirection:
    """Normalise a direction value to a SignalDirection enum member.

    After JSON roundtrip with use_enum_values=True, direction fields come back
    as plain strings. This helper accepts both forms so callers never need to
    guard against AttributeError when calling .value or dict-lookup mismatches.
    """
    if isinstance(v, SignalDirection):
        return v
    return SignalDirection(v)


class SignalSource(str, Enum):
    """Which agent produced this signal."""

    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    ML = "ml"  # Future: ML/RL agent
    MANUAL = "manual"  # Future: human override


class IndicatorName(str, Enum):
    """Named indicators contributing to a technical signal."""

    RSI = "rsi"
    MACD = "macd"
    BOLLINGER_BANDS = "bollinger_bands"
    EMA_CROSS = "ema_cross"
    VOLUME = "volume"


# ---------------------------------------------------------------------------
# Indicator reading (sub-component of technical signals)
# ---------------------------------------------------------------------------


class IndicatorReading(BaseMarketModel):
    """
    A single indicator's current reading and its contribution to the signal.
    Included in TechnicalSignal for observability — lets you see exactly
    which indicators fired and why.
    """

    name: IndicatorName
    value: float = Field(description="Raw indicator value (e.g. RSI=72.3)")
    signal: SignalDirection = Field(description="This indicator's directional view")
    weight: float = Field(
        ge=0.0, le=1.0, description="Weight this indicator contributes to the composite signal"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Extra indicator-specific data (e.g. MACD histogram value)",
    )


# ---------------------------------------------------------------------------
# Technical signal
# ---------------------------------------------------------------------------


class TechnicalSignal(BaseMarketModel):
    """
    Composite signal produced by the Technical Analysis agent.
    Published on: signal.technical.{symbol}

    Contains the final direction + confidence, plus the breakdown of which
    indicators fired — essential for debugging bad trades after the fact.
    """

    signal_id: UUID = Field(default_factory=uuid4)
    source: SignalSource = SignalSource.TECHNICAL
    symbol: str
    timeframe: str = Field(description="Primary timeframe this signal was computed on")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime = Field(description="Signal is stale after this time")

    direction: SignalDirection
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score [0,1]. Below threshold → Decision agent discards.",
    )

    # Component breakdown
    indicators: list[IndicatorReading] = Field(
        default_factory=list, description="Individual indicator readings that formed this signal"
    )

    # Current price context (snapshot at signal generation time)
    price: float = Field(description="Current market price at signal time")
    price_change_1h_pct: float | None = None

    metadata: dict = Field(default_factory=dict)

    @field_validator("expires_at", mode="before")
    @classmethod
    def default_expiry(cls, v: datetime | None) -> datetime:
        # If not provided, default to 5 minutes from now.
        # Callers should use the config value: signal_ttl_seconds.
        if v is None:
            return datetime.now(UTC) + timedelta(seconds=300)
        return v

    @property
    def channel_key(self) -> str:
        return f"signal.technical.{self.symbol.replace('/', '-')}"

    @property
    def is_expired(self) -> bool:
        return datetime.now(UTC) > self.expires_at

    @property
    def is_actionable(self) -> bool:
        """True if the signal is fresh and directional (not neutral)."""
        return not self.is_expired and _to_direction(self.direction) != SignalDirection.NEUTRAL

    @property
    def is_bullish(self) -> bool:
        return _to_direction(self.direction) in (SignalDirection.BUY, SignalDirection.STRONG_BUY)

    @property
    def is_bearish(self) -> bool:
        return _to_direction(self.direction) in (SignalDirection.SELL, SignalDirection.STRONG_SELL)

    def scalar_direction(self) -> float:
        """
        Convert direction to a scalar for weighted averaging.
        STRONG_BUY=1.0, BUY=0.5, NEUTRAL=0.0, SELL=-0.5, STRONG_SELL=-1.0
        """
        mapping = {
            SignalDirection.STRONG_BUY: 1.0,
            SignalDirection.BUY: 0.5,
            SignalDirection.NEUTRAL: 0.0,
            SignalDirection.SELL: -0.5,
            SignalDirection.STRONG_SELL: -1.0,
        }
        return mapping[_to_direction(self.direction)]


# ---------------------------------------------------------------------------
# Sentiment signal
# ---------------------------------------------------------------------------


class SentimentSignal(BaseMarketModel):
    """
    Signal produced by the Sentiment/News agent.
    Published on: signal.sentiment.{symbol}
    """

    signal_id: UUID = Field(default_factory=uuid4)
    source: SignalSource = SignalSource.SENTIMENT
    symbol: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime = Field(description="Sentiment signals age faster than TA signals")

    direction: SignalDirection
    confidence: float = Field(ge=0.0, le=1.0)

    # Sentiment-specific fields
    sentiment_score: float = Field(
        ge=-1.0, le=1.0, description="Raw sentiment: -1 (very negative) to +1 (very positive)"
    )
    article_count: int = Field(default=0, description="Number of articles analysed")
    dominant_topics: list[str] = Field(
        default_factory=list,
        description="Key topics detected (e.g. ['regulation', 'etf approval'])",
    )
    sources: list[str] = Field(default_factory=list, description="News sources sampled")

    @property
    def channel_key(self) -> str:
        return f"signal.sentiment.{self.symbol.replace('/', '-')}"

    @property
    def is_expired(self) -> bool:
        return datetime.now(UTC) > self.expires_at

    def scalar_direction(self) -> float:
        mapping = {
            SignalDirection.STRONG_BUY: 1.0,
            SignalDirection.BUY: 0.5,
            SignalDirection.NEUTRAL: 0.0,
            SignalDirection.SELL: -0.5,
            SignalDirection.STRONG_SELL: -1.0,
        }
        return mapping[_to_direction(self.direction)]


# ---------------------------------------------------------------------------
# Aggregated signal (Decision agent output, before Risk review)
# ---------------------------------------------------------------------------


class AggregatedSignal(BaseMarketModel):
    """
    The Decision agent's fusion of all incoming signals.
    This is what the Decision agent publishes to the Risk agent.

    Stored in the DB for post-trade analysis — lets you correlate the
    composite signal score with eventual P&L.
    """

    signal_id: UUID = Field(default_factory=uuid4)
    symbol: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Composite output
    direction: SignalDirection
    confidence: float = Field(ge=0.0, le=1.0)
    composite_score: float = Field(
        ge=-1.0,
        le=1.0,
        description="Weighted average of scalar signal directions. Negative=bearish, positive=bullish.",
    )

    # Component breakdown for audit trail
    technical_signal: TechnicalSignal | None = None
    sentiment_signal: SentimentSignal | None = None
    technical_weight: float = Field(default=0.7, description="Weight given to technical signal")
    sentiment_weight: float = Field(default=0.3, description="Weight given to sentiment signal")

    # Signal counts
    total_signals: int = Field(default=0)
    signals_discarded: int = Field(
        default=0, description="Signals dropped (expired, low confidence)"
    )

    @property
    def channel_key(self) -> str:
        return f"decision.aggregated.{self.symbol.replace('/', '-')}"
