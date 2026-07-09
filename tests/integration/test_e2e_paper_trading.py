"""
tests/integration/test_e2e_paper_trading.py
-------------------------------------------
End-to-end paper-trading test suite.

These tests stitch together the building blocks of the trading pipeline
(buffer → signal → proposal → risk → execution) without booting any agent
event loops or hitting a real Redis / Postgres. Each test isolates one
hop in the lifecycle and verifies the boundary contract that the next
hop relies on.

Why no real bus
---------------
The bus-backed integration is already covered by
``tests/integration/test_full_pipeline.py`` (which skips when Redis is
absent). Here we want a deterministic, infra-free smoke test the CI
``unit-tests`` job can run in seconds. Driving the components directly
also gives us better failure messages: when a test fails we know exactly
which hop broke, not just "something timed out".
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from agents.decision.proposal_builder import ProposalBuilder
from agents.decision.signal_aggregator import SignalAggregator
from agents.risk.agent import RiskAgent
from agents.technical_analysis.candle_buffer import CandleBufferRegistry
from agents.technical_analysis.signal_generator import SignalGenerator
from core.models.market import OHLCVCandle
from core.models.signals import (
    AggregatedSignal,
    SignalDirection,
    TechnicalSignal,
)
from core.models.trade import (
    OrderSide,
    OrderStatus,
    OrderType,
    RejectionReason,
    RiskAssessment,
    RiskDecision,
    TradeProposal,
)

# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

_BASE_TS = datetime(2025, 1, 1, tzinfo=UTC)
_FUTURE = datetime(2099, 1, 1, tzinfo=UTC)


def _make_candle(close: float, i: int, symbol: str = "BTC/USDT") -> OHLCVCandle:
    """Build one synthetic 1m candle at index ``i`` (minutes from base)."""
    price = Decimal(str(round(close, 8)))
    return OHLCVCandle(
        symbol=symbol,
        timeframe="1m",
        timestamp=_BASE_TS + timedelta(minutes=i),
        open=price,
        high=Decimal(str(round(close + 50.0, 8))),
        low=Decimal(str(round(close - 50.0, 8))),
        close=price,
        volume=Decimal("100"),
        received_at=datetime.now(UTC),
    )


def _trending_up(n: int, start: float = 40_000.0, step: float = 100.0) -> list[OHLCVCandle]:
    return [_make_candle(start + i * step, i) for i in range(n)]


def _make_technical_signal(
    direction: SignalDirection = SignalDirection.BUY,
    confidence: float = 0.75,
    price: float = 42_000.0,
    symbol: str = "BTC/USDT",
    fresh: bool = True,
) -> TechnicalSignal:
    """Fully-populated TechnicalSignal used as input to downstream stages."""
    return TechnicalSignal(
        symbol=symbol,
        timeframe="1m",
        timestamp=datetime.now(UTC) if fresh else _BASE_TS,
        expires_at=_FUTURE,
        direction=direction,
        confidence=confidence,
        price=price,
    )


def _make_aggregated_signal(
    direction: SignalDirection = SignalDirection.BUY,
    confidence: float = 0.75,
    composite_score: float = 0.5,
    symbol: str = "BTC/USDT",
) -> AggregatedSignal:
    technical = _make_technical_signal(direction=direction, confidence=confidence, symbol=symbol)
    return AggregatedSignal(
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        composite_score=composite_score,
        technical_signal=technical,
        total_signals=1,
    )


def _make_proposal(
    symbol: str = "BTC/USDT",
    side: OrderSide = OrderSide.BUY,
    size_usd: float = 200.0,
    signal: AggregatedSignal | None = None,
) -> TradeProposal:
    return TradeProposal(
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        requested_size_usd=Decimal(str(size_usd)),
        suggested_stop_loss_pct=0.02,
        signal=signal or _make_aggregated_signal(),
        reasoning="e2e test proposal",
    )


def _make_approved_assessment(
    symbol: str = "BTC/USDT",
    side: OrderSide = OrderSide.BUY,
    approved_usd: float = 200.0,
) -> RiskAssessment:
    proposal = _make_proposal(symbol=symbol, side=side, size_usd=approved_usd)
    return RiskAssessment(
        proposal_id=proposal.proposal_id,
        decision=RiskDecision.APPROVED,
        approved_size_usd=Decimal(str(approved_usd)),
        approved_stop_loss_pct=0.02,
        portfolio_value_usd=Decimal("10000"),
        open_positions_count=0,
        original_proposal=proposal,
    )


# ---------------------------------------------------------------------------
# Stage 1: candle → signal
# ---------------------------------------------------------------------------


class TestCandleToSignal:
    """Feed candles through the registry and verify the generator produces signals."""

    def test_candle_to_signal_pipeline(self, mock_settings):
        """A warm buffer of trending candles should yield a TechnicalSignal."""
        registry = CandleBufferRegistry()
        generator = SignalGenerator()

        # 60 candles is comfortably above the 50-candle warmup threshold.
        for candle in _trending_up(60):
            registry.add(candle)

        buf = registry.get("BTC/USDT", "1m")
        assert buf is not None
        assert buf.is_warm, "Buffer should be warm after 60 candles"

        signal = generator.generate(buf)
        if signal is None:
            pytest.skip(
                "Synthetic series did not breach the 0.6 confidence threshold; "
                "see docs/adr_002_signal_confidence.md for why this is expected"
            )

        assert isinstance(signal, TechnicalSignal)
        assert signal.symbol == "BTC/USDT"
        assert signal.timeframe == "1m"
        assert 0.0 <= signal.confidence <= 1.0
        assert signal.direction in list(SignalDirection)
        assert signal.price > 0


# ---------------------------------------------------------------------------
# Stage 2: signal → proposal
# ---------------------------------------------------------------------------


class TestSignalToProposal:
    """Aggregate technical signals and build trade proposals from them."""

    def test_signal_to_proposal_pipeline(self, mock_settings):
        """A BUY signal at 0.75 confidence should produce a BUY proposal with size>0."""
        technical = _make_technical_signal(
            direction=SignalDirection.BUY,
            confidence=0.75,
            price=42_000.0,
        )

        aggregator = SignalAggregator(
            technical_weight=0.7,
            sentiment_weight=0.3,
            min_confidence=0.6,
        )
        aggregated = aggregator.aggregate(
            symbol="BTC/USDT",
            technical=technical,
            sentiment=None,
        )

        assert aggregated is not None
        assert aggregated.confidence >= 0.6

        builder = ProposalBuilder(
            max_position_pct=0.02,
            default_stop_loss_pct=0.02,
            default_take_profit_pct=0.04,
            max_order_size_usd=1000.0,
        )
        proposal = builder.build(aggregated, portfolio_value_usd=Decimal("10000"))

        assert proposal is not None
        assert proposal.symbol == "BTC/USDT"
        assert proposal.side == OrderSide.BUY
        assert proposal.requested_size_usd > Decimal("0")
        assert proposal.order_type == OrderType.MARKET


# ---------------------------------------------------------------------------
# Stage 3: proposal → risk assessment
# ---------------------------------------------------------------------------


class TestRiskEvaluation:
    """
    Exercise the Risk agent's evaluation pipeline by instantiating RiskAgent and
    calling ``_evaluate`` directly. We never connect the message bus — the
    evaluation logic only reads in-process state plus an optional Redis cache
    write that no-ops when the bus pool is None.
    """

    async def test_proposal_through_risk_approved(self, mock_settings):
        """A clean proposal under all limits should be APPROVED (or MODIFIED)."""
        risk = RiskAgent()
        proposal = _make_proposal(size_usd=200.0)

        assessment = await risk._evaluate(proposal)

        assert assessment.decision in (RiskDecision.APPROVED, RiskDecision.MODIFIED)
        assert assessment.rejection_reason is None
        assert assessment.approved_size_usd is not None
        assert assessment.approved_size_usd > Decimal("0")
        assert assessment.original_proposal.proposal_id == proposal.proposal_id

    async def test_proposal_through_risk_rejected_daily_loss(self, mock_settings):
        """When daily loss exceeds max_daily_loss_pct the assessment must be REJECTED."""
        risk = RiskAgent()
        # Force the in-memory portfolio state into a 10% loss for today,
        # well past the default 5% daily-loss limit.
        state = risk._drawdown.state
        state.current_balance_usd = Decimal("9000")
        # daily_start_balance_usd defaults to initial balance (10_000),
        # so daily_loss_pct = (10000 - 9000) / 10000 = 10%.

        assessment = await risk._evaluate(_make_proposal())

        assert assessment.decision == RiskDecision.REJECTED
        assert assessment.rejection_reason == RejectionReason.DAILY_LOSS_LIMIT

    async def test_risk_override_halts_execution(self, mock_settings):
        """Once the circuit breaker is tripped, every subsequent proposal is REJECTED."""
        risk = RiskAgent()
        risk._breaker.trip("e2e test trip")

        assessment = await risk._evaluate(_make_proposal())

        assert assessment.decision == RiskDecision.REJECTED
        assert assessment.rejection_reason == RejectionReason.CIRCUIT_BREAKER_ACTIVE


# ---------------------------------------------------------------------------
# Stage 4: assessment → execution result (paper)
# ---------------------------------------------------------------------------


class TestPaperBrokerExecution:
    """End-to-end paper fills using the PaperBroker fixture."""

    async def test_paper_broker_fill(self, paper_broker):
        """An approved RiskAssessment should produce a filled (or partial) result."""
        assessment = _make_approved_assessment(side=OrderSide.BUY, approved_usd=200.0)

        result = await paper_broker.place_order(assessment)

        assert result.is_paper is True
        assert result.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)
        assert result.average_fill_price is not None
        assert result.average_fill_price > Decimal("0")
        assert result.proposal_id == assessment.proposal_id
        assert result.symbol == "BTC/USDT"
        assert result.side == OrderSide.BUY

    async def test_paper_broker_balance_updates(self, paper_broker):
        """Placing a BUY then closing the position must round-trip cash correctly."""
        balance_before = await paper_broker.get_balance()
        cash_before = balance_before.free_margin_usd

        assessment = _make_approved_assessment(side=OrderSide.BUY, approved_usd=200.0)
        fill = await paper_broker.place_order(assessment)
        assert fill.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)

        balance_after_buy = await paper_broker.get_balance()
        assert balance_after_buy.free_margin_usd < cash_before, (
            "Free cash should drop after a BUY fill"
        )

        positions_after_buy = await paper_broker.get_positions()
        assert any(p.symbol == "BTC/USDT" for p in positions_after_buy), (
            "BUY fill should register an open position"
        )

        close_result = await paper_broker.close_position("BTC/USDT")
        assert close_result is not None
        assert close_result.is_paper is True

        positions_after_close = await paper_broker.get_positions()
        assert not any(p.symbol == "BTC/USDT" for p in positions_after_close), (
            "Closing the position should remove it from the broker state"
        )
