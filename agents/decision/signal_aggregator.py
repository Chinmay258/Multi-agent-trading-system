"""
agents/decision/signal_aggregator.py
-------------------------------------
Signal fusion for the Decision agent.

Combines TechnicalSignal and SentimentSignal into a single AggregatedSignal
using a configurable weighted scoring scheme. Expired and low-confidence
signals are discarded before aggregation.

Design decisions:
- Expired signals are worse than no signal — an old BUY in a SELL market
  is dangerous. Always check is_expired before fusing.
- Confidence is a weighted average of input confidences, not the score
  magnitude alone. Two weak agreeing signals produce moderate confidence.
- Neutral aggregated signals are not suppressed here — the ProposalBuilder
  decides whether to act on them. This keeps the aggregator stateless.
"""

from __future__ import annotations

from core.models.signals import (
    AggregatedSignal,
    SentimentSignal,
    SignalDirection,
    TechnicalSignal,
)


def _scalar_to_direction(score: float) -> SignalDirection:
    """
    Map a composite score [-1, +1] to a SignalDirection.
    Thresholds match the TA agent's signal_generator for consistency.
    """
    if score >= 0.65:
        return SignalDirection.STRONG_BUY
    elif score >= 0.25:
        return SignalDirection.BUY
    elif score <= -0.65:
        return SignalDirection.STRONG_SELL
    elif score <= -0.25:
        return SignalDirection.SELL
    else:
        return SignalDirection.NEUTRAL


class SignalAggregator:
    """
    Fuses TechnicalSignal and SentimentSignal into an AggregatedSignal.

    Weighting (configurable, defaults match spec):
        technical_weight: 0.7
        sentiment_weight: 0.3

    Filtering before fusion:
        - Expired signals (is_expired == True) are dropped.
        - Signals with confidence below min_confidence are dropped.

    Returns None when no signals survive filtering — the caller should
    not propose a trade with no valid signal basis.
    """

    def __init__(
        self,
        technical_weight: float = 0.7,
        sentiment_weight: float = 0.3,
        min_confidence: float = 0.6,
    ) -> None:
        self.technical_weight = technical_weight
        self.sentiment_weight = sentiment_weight
        self.min_confidence = min_confidence

    def aggregate(
        self,
        symbol: str,
        technical: TechnicalSignal | None = None,
        sentiment: SentimentSignal | None = None,
    ) -> AggregatedSignal | None:
        """
        Aggregate available signals into a composite AggregatedSignal.

        Args:
            symbol: Trading pair (e.g. "BTC/USDT")
            technical: Latest TechnicalSignal, or None if unavailable
            sentiment: Latest SentimentSignal, or None if unavailable

        Returns:
            AggregatedSignal, or None if no signals survive filtering.
        """
        total_inputs = 0
        discarded = 0
        valid_technical: TechnicalSignal | None = None
        valid_sentiment: SentimentSignal | None = None

        if technical is not None:
            total_inputs += 1
            if not technical.is_expired and technical.confidence >= self.min_confidence:
                valid_technical = technical
            else:
                discarded += 1

        if sentiment is not None:
            total_inputs += 1
            if not sentiment.is_expired and sentiment.confidence >= self.min_confidence:
                valid_sentiment = sentiment
            else:
                discarded += 1

        if valid_technical is None and valid_sentiment is None:
            return None

        # Weighted composite score
        score_sum = 0.0
        weight_sum = 0.0
        conf_sum = 0.0

        if valid_technical is not None:
            score_sum += valid_technical.scalar_direction() * self.technical_weight
            weight_sum += self.technical_weight
            conf_sum += valid_technical.confidence * self.technical_weight

        if valid_sentiment is not None:
            score_sum += valid_sentiment.scalar_direction() * self.sentiment_weight
            weight_sum += self.sentiment_weight
            conf_sum += valid_sentiment.confidence * self.sentiment_weight

        composite_score = score_sum / weight_sum if weight_sum > 0 else 0.0
        confidence = conf_sum / weight_sum if weight_sum > 0 else 0.0

        # Clamp composite_score to model constraint [-1.0, 1.0]
        composite_score = max(-1.0, min(1.0, composite_score))
        confidence = max(0.0, min(1.0, confidence))

        direction = _scalar_to_direction(composite_score)

        return AggregatedSignal(
            symbol=symbol,
            direction=direction,
            confidence=round(confidence, 4),
            composite_score=round(composite_score, 4),
            technical_signal=valid_technical,
            sentiment_signal=valid_sentiment,
            technical_weight=self.technical_weight,
            sentiment_weight=self.sentiment_weight,
            total_signals=total_inputs,
            signals_discarded=discarded,
        )
