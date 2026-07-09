"""
tests/unit/test_decision.py
-----------------------------
Unit tests for Decision agent components.

Tests the SignalAggregator and ProposalBuilder in isolation — no Redis,
no network, no agent lifecycle. All test scenarios build Pydantic models
directly and invoke the pure-logic methods.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

from agents.decision.proposal_builder import ProposalBuilder
from agents.decision.signal_aggregator import SignalAggregator, _scalar_to_direction
from core.models.signals import (
    AggregatedSignal,
    SentimentSignal,
    SignalDirection,
    TechnicalSignal,
)
from core.models.trade import OrderSide, TradeProposal

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_FUTURE = datetime(2099, 1, 1, tzinfo=UTC)
_PAST = datetime(2000, 1, 1, tzinfo=UTC)


def _make_technical(
    symbol: str = "BTC/USDT",
    direction: SignalDirection = SignalDirection.BUY,
    confidence: float = 0.75,
    expired: bool = False,
    atr_pct: float | None = None,
) -> TechnicalSignal:
    meta: dict = {}
    if atr_pct is not None:
        meta["atr_pct_of_price"] = atr_pct
    return TechnicalSignal(
        symbol=symbol,
        timeframe="1m",
        expires_at=_PAST if expired else _FUTURE,
        direction=direction,
        confidence=confidence,
        price=42000.0,
        metadata=meta,
    )


def _make_sentiment(
    symbol: str = "BTC/USDT",
    direction: SignalDirection = SignalDirection.BUY,
    confidence: float = 0.70,
    expired: bool = False,
) -> SentimentSignal:
    return SentimentSignal(
        symbol=symbol,
        expires_at=_PAST if expired else _FUTURE,
        direction=direction,
        confidence=confidence,
        sentiment_score=0.5,
    )


def _make_aggregated(
    symbol: str = "BTC/USDT",
    direction: SignalDirection = SignalDirection.BUY,
    confidence: float = 0.75,
    composite_score: float = 0.5,
    technical: TechnicalSignal | None = None,
) -> AggregatedSignal:
    return AggregatedSignal(
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        composite_score=composite_score,
        technical_signal=technical,
        total_signals=1,
    )


# ---------------------------------------------------------------------------
# SignalAggregator tests
# ---------------------------------------------------------------------------


class TestSignalAggregatorFiltering:
    def setup_method(self):
        self.agg = SignalAggregator(
            technical_weight=0.7,
            sentiment_weight=0.3,
            min_confidence=0.6,
        )

    def test_both_valid_returns_aggregated_signal(self):
        tech = _make_technical(direction=SignalDirection.BUY, confidence=0.8)
        sent = _make_sentiment(direction=SignalDirection.BUY, confidence=0.7)
        result = self.agg.aggregate("BTC/USDT", technical=tech, sentiment=sent)
        assert result is not None
        assert isinstance(result, AggregatedSignal)

    def test_only_technical_returns_aggregated_signal(self):
        tech = _make_technical(confidence=0.8)
        result = self.agg.aggregate("BTC/USDT", technical=tech, sentiment=None)
        assert result is not None
        assert result.technical_signal is not None
        assert result.sentiment_signal is None

    def test_only_sentiment_returns_aggregated_signal(self):
        sent = _make_sentiment(confidence=0.8)
        result = self.agg.aggregate("BTC/USDT", technical=None, sentiment=sent)
        assert result is not None
        assert result.technical_signal is None
        assert result.sentiment_signal is not None

    def test_both_none_returns_none(self):
        result = self.agg.aggregate("BTC/USDT", technical=None, sentiment=None)
        assert result is None

    def test_expired_technical_discarded(self):
        tech = _make_technical(expired=True, confidence=0.9)
        result = self.agg.aggregate("BTC/USDT", technical=tech, sentiment=None)
        assert result is None

    def test_expired_sentiment_discarded(self):
        sent = _make_sentiment(expired=True, confidence=0.9)
        result = self.agg.aggregate("BTC/USDT", technical=None, sentiment=sent)
        assert result is None

    def test_low_confidence_technical_discarded(self):
        tech = _make_technical(confidence=0.4)  # below 0.6 threshold
        result = self.agg.aggregate("BTC/USDT", technical=tech, sentiment=None)
        assert result is None

    def test_low_confidence_sentiment_discarded(self):
        sent = _make_sentiment(confidence=0.3)  # below threshold
        result = self.agg.aggregate("BTC/USDT", technical=None, sentiment=sent)
        assert result is None

    def test_expired_technical_valid_sentiment_uses_only_sentiment(self):
        tech = _make_technical(expired=True, confidence=0.9, direction=SignalDirection.SELL)
        sent = _make_sentiment(confidence=0.8, direction=SignalDirection.BUY)
        result = self.agg.aggregate("BTC/USDT", technical=tech, sentiment=sent)
        assert result is not None
        # Only sentiment survives; direction should be BUY
        assert result.technical_signal is None
        assert result.sentiment_signal is not None
        assert result.direction in (SignalDirection.BUY.value, "buy", SignalDirection.BUY)

    def test_discarded_counts_tracked(self):
        tech = _make_technical(expired=True, confidence=0.9)
        sent = _make_sentiment(expired=True, confidence=0.9)
        result = self.agg.aggregate("BTC/USDT", technical=tech, sentiment=sent)
        assert result is None  # both discarded → None

    def test_discard_count_on_partial(self):
        tech = _make_technical(expired=True, confidence=0.9)
        sent = _make_sentiment(confidence=0.8)
        result = self.agg.aggregate("BTC/USDT", technical=tech, sentiment=sent)
        assert result is not None
        assert result.total_signals == 2
        assert result.signals_discarded == 1


class TestSignalAggregatorScoring:
    def setup_method(self):
        self.agg = SignalAggregator(
            technical_weight=0.7,
            sentiment_weight=0.3,
            min_confidence=0.0,  # disable confidence filter to test scoring
        )

    def test_both_strong_buy_gives_strong_buy(self):
        tech = _make_technical(direction=SignalDirection.STRONG_BUY, confidence=0.9)
        sent = _make_sentiment(direction=SignalDirection.STRONG_BUY, confidence=0.9)
        result = self.agg.aggregate("BTC/USDT", technical=tech, sentiment=sent)
        assert result is not None
        assert result.composite_score > 0.65
        assert result.direction in (
            SignalDirection.STRONG_BUY.value,
            "strong_buy",
            SignalDirection.STRONG_BUY,
        )

    def test_both_strong_sell_gives_strong_sell(self):
        tech = _make_technical(direction=SignalDirection.STRONG_SELL, confidence=0.9)
        sent = _make_sentiment(direction=SignalDirection.STRONG_SELL, confidence=0.9)
        result = self.agg.aggregate("BTC/USDT", technical=tech, sentiment=sent)
        assert result is not None
        assert result.composite_score < -0.65
        assert result.direction in (
            SignalDirection.STRONG_SELL.value,
            "strong_sell",
            SignalDirection.STRONG_SELL,
        )

    def test_opposing_signals_tend_toward_neutral(self):
        tech = _make_technical(direction=SignalDirection.BUY, confidence=0.8)
        sent = _make_sentiment(direction=SignalDirection.SELL, confidence=0.8)
        result = self.agg.aggregate("BTC/USDT", technical=tech, sentiment=sent)
        assert result is not None
        # Weighted: 0.5*0.7 + (-0.5)*0.3 = 0.35 - 0.15 = 0.2 → below BUY threshold of 0.25 → NEUTRAL
        assert -0.30 < result.composite_score < 0.30

    def test_weights_applied_correctly(self):
        # technical=STRONG_BUY (1.0), sentiment=STRONG_SELL (-1.0)
        # expected score = 1.0*0.7 + (-1.0)*0.3 = 0.4 → BUY direction
        tech = _make_technical(direction=SignalDirection.STRONG_BUY, confidence=0.9)
        sent = _make_sentiment(direction=SignalDirection.STRONG_SELL, confidence=0.9)
        result = self.agg.aggregate("BTC/USDT", technical=tech, sentiment=sent)
        assert result is not None
        expected_score = 1.0 * 0.7 + (-1.0) * 0.3
        assert abs(result.composite_score - expected_score) < 0.0001
        # 0.4 is ≥ 0.25, so direction should be BUY
        assert result.direction in (SignalDirection.BUY.value, "buy", SignalDirection.BUY)

    def test_only_technical_uses_full_technical_weight(self):
        # With only technical, weight_sum = 0.7, score = scalar * 0.7 / 0.7 = scalar
        tech = _make_technical(direction=SignalDirection.BUY, confidence=0.9)  # scalar=0.5
        result = self.agg.aggregate("BTC/USDT", technical=tech, sentiment=None)
        assert result is not None
        assert abs(result.composite_score - 0.5) < 0.0001

    def test_composite_score_clamped_to_unit_interval(self):
        tech = _make_technical(direction=SignalDirection.STRONG_BUY, confidence=0.9)
        result = self.agg.aggregate("BTC/USDT", technical=tech, sentiment=None)
        assert result is not None
        assert -1.0 <= result.composite_score <= 1.0

    def test_confidence_weighted_average(self):
        # tech confidence=0.8 weight=0.7, sent confidence=0.6 weight=0.3
        # expected = (0.8*0.7 + 0.6*0.3) / (0.7+0.3) = (0.56 + 0.18) / 1.0 = 0.74
        tech = _make_technical(direction=SignalDirection.BUY, confidence=0.8)
        sent = _make_sentiment(direction=SignalDirection.BUY, confidence=0.6)
        result = self.agg.aggregate("BTC/USDT", technical=tech, sentiment=sent)
        assert result is not None
        expected_conf = (0.8 * 0.7 + 0.6 * 0.3) / 1.0
        assert abs(result.confidence - expected_conf) < 0.001

    def test_symbol_preserved_in_output(self):
        tech = _make_technical(symbol="ETH/USDT", confidence=0.9)
        result = self.agg.aggregate("ETH/USDT", technical=tech)
        assert result is not None
        assert result.symbol == "ETH/USDT"


class TestScalarToDirection:
    def test_strong_buy_threshold(self):
        assert _scalar_to_direction(0.65) == SignalDirection.STRONG_BUY
        assert _scalar_to_direction(1.0) == SignalDirection.STRONG_BUY

    def test_buy_threshold(self):
        assert _scalar_to_direction(0.25) == SignalDirection.BUY
        assert _scalar_to_direction(0.64) == SignalDirection.BUY

    def test_neutral_zone(self):
        assert _scalar_to_direction(0.0) == SignalDirection.NEUTRAL
        assert _scalar_to_direction(0.24) == SignalDirection.NEUTRAL
        assert _scalar_to_direction(-0.24) == SignalDirection.NEUTRAL

    def test_sell_threshold(self):
        assert _scalar_to_direction(-0.25) == SignalDirection.SELL
        assert _scalar_to_direction(-0.64) == SignalDirection.SELL

    def test_strong_sell_threshold(self):
        assert _scalar_to_direction(-0.65) == SignalDirection.STRONG_SELL
        assert _scalar_to_direction(-1.0) == SignalDirection.STRONG_SELL


# ---------------------------------------------------------------------------
# ProposalBuilder tests
# ---------------------------------------------------------------------------


class TestProposalBuilder:
    def setup_method(self):
        self.builder = ProposalBuilder(
            max_position_pct=0.02,
            default_stop_loss_pct=0.02,
            default_take_profit_pct=0.04,
            max_order_size_usd=1000.0,
        )
        self.portfolio = Decimal("10000.00")

    def _aggregated(
        self,
        direction: SignalDirection = SignalDirection.BUY,
        confidence: float = 0.75,
        technical: TechnicalSignal | None = None,
    ) -> AggregatedSignal:
        return _make_aggregated(
            direction=direction,
            confidence=confidence,
            composite_score=0.5,
            technical=technical,
        )

    def test_buy_direction_gives_buy_order(self):
        agg = self._aggregated(SignalDirection.BUY)
        proposal = self.builder.build(agg, self.portfolio)
        assert proposal is not None
        assert proposal.side in (OrderSide.BUY.value, "buy", OrderSide.BUY)

    def test_strong_buy_gives_buy_order(self):
        agg = self._aggregated(SignalDirection.STRONG_BUY)
        proposal = self.builder.build(agg, self.portfolio)
        assert proposal is not None
        assert proposal.side in (OrderSide.BUY.value, "buy", OrderSide.BUY)

    def test_sell_direction_gives_sell_order(self):
        agg = self._aggregated(SignalDirection.SELL)
        proposal = self.builder.build(agg, self.portfolio)
        assert proposal is not None
        assert proposal.side in (OrderSide.SELL.value, "sell", OrderSide.SELL)

    def test_strong_sell_gives_sell_order(self):
        agg = self._aggregated(SignalDirection.STRONG_SELL)
        proposal = self.builder.build(agg, self.portfolio)
        assert proposal is not None
        assert proposal.side in (OrderSide.SELL.value, "sell", OrderSide.SELL)

    def test_neutral_returns_none(self):
        agg = self._aggregated(SignalDirection.NEUTRAL)
        proposal = self.builder.build(agg, self.portfolio)
        assert proposal is None

    def test_size_is_max_position_pct_of_portfolio(self):
        agg = self._aggregated()
        proposal = self.builder.build(agg, self.portfolio)
        assert proposal is not None
        expected = Decimal("0.02") * self.portfolio  # 2% of 10,000 = 200
        assert proposal.requested_size_usd == expected

    def test_size_capped_at_max_order_size(self):
        large_portfolio = Decimal("1_000_000")  # 2% = 20,000 > 1,000 cap
        agg = self._aggregated()
        proposal = self.builder.build(agg, large_portfolio)
        assert proposal is not None
        assert proposal.requested_size_usd == Decimal("1000.0")

    def test_stop_loss_pct_set_from_builder_default(self):
        agg = self._aggregated()
        proposal = self.builder.build(agg, self.portfolio)
        assert proposal is not None
        assert proposal.suggested_stop_loss_pct == 0.02

    def test_take_profit_pct_set_from_builder_default(self):
        agg = self._aggregated()
        proposal = self.builder.build(agg, self.portfolio)
        assert proposal is not None
        assert proposal.suggested_take_profit_pct == 0.04

    def test_signal_embedded_in_proposal(self):
        agg = self._aggregated()
        proposal = self.builder.build(agg, self.portfolio)
        assert proposal is not None
        assert proposal.signal.symbol == "BTC/USDT"

    def test_proposal_id_is_uuid(self):
        agg = self._aggregated()
        proposal = self.builder.build(agg, self.portfolio)
        assert proposal is not None
        assert isinstance(proposal.proposal_id, UUID)

    def test_reasoning_contains_symbol(self):
        agg = self._aggregated()
        proposal = self.builder.build(agg, self.portfolio)
        assert proposal is not None
        assert "BTC/USDT" in proposal.reasoning

    def test_reasoning_contains_direction(self):
        agg = self._aggregated(SignalDirection.STRONG_BUY)
        proposal = self.builder.build(agg, self.portfolio)
        assert proposal is not None
        assert "strong_buy" in proposal.reasoning

    def test_proposal_is_pydantic_model(self):
        agg = self._aggregated()
        proposal = self.builder.build(agg, self.portfolio)
        assert proposal is not None
        assert isinstance(proposal, TradeProposal)

    def test_channel_key_is_decision_proposal(self):
        agg = self._aggregated()
        proposal = self.builder.build(agg, self.portfolio)
        assert proposal is not None
        assert proposal.channel_key == "decision.proposal"

    def test_json_roundtrip(self):
        agg = self._aggregated()
        proposal = self.builder.build(agg, self.portfolio)
        assert proposal is not None
        restored = TradeProposal.model_validate_json(proposal.to_json())
        assert restored.proposal_id == proposal.proposal_id
        assert restored.symbol == proposal.symbol
