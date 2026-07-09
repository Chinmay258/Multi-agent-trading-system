"""
tests/integration/test_e2e_pipeline.py
--------------------------------------
The Phase-2 end-to-end pipeline test.

Unlike ``test_e2e_paper_trading.py`` (which exercises each hop in isolation and may
*skip* the candle→signal stage when synthetic data fails to breach the confidence
threshold), this test drives the **entire chain in one shot** and must not skip:

    bundled candles
        → indicators (real TA-Lib / NumPy-fallback computations)
        → TechnicalSignal (real SignalGenerator)
        → AggregatedSignal + TradeProposal (the DecisionAgent's own components)
        → RiskAssessment (real RiskAgent evaluation)
        → paper fill (real PaperBroker)
        → recorded trade (broker position state)

It is fully **keyless and infra-free**: candles come from a bundled CSV
(``tests/data/sample_candles_btc_1m.csv``), execution is simulated, and no Redis,
Postgres, exchange, or MT5 is touched. The CSV is engineered so the real
mean-reversion rule generator produces a deterministic BUY (≈0.59 confidence), so the
test is stable in CI without asserting anything about strategy quality.
"""

from __future__ import annotations

import csv
import os
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

from core.config import Settings, TradingMode, get_settings
from core.models.market import OHLCVCandle
from core.models.signals import SignalDirection, _to_direction
from core.models.trade import OrderSide, OrderStatus, RiskDecision

_SAMPLE_CSV = Path(__file__).parent.parent / "data" / "sample_candles_btc_1m.csv"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline_settings() -> Settings:
    """
    Paper-mode settings with a confidence threshold low enough that the bundled
    sample's ~0.59-confidence signal survives (the production default is 0.6).
    """
    env = {
        "TRADING_MODE": "paper",
        "DATA_SOURCE": "public",
        "MARKET_DATA_SYMBOLS": '["BTC/USDT"]',
        "MARKET_DATA_OHLCV_TIMEFRAMES": '["1m"]',
        "PAPER_INITIAL_BALANCE_USD": "10000",
        "TA_MIN_SIGNAL_CONFIDENCE": "0.5",
    }
    with patch.dict(os.environ, env, clear=False):
        get_settings.cache_clear()
        try:
            yield get_settings()
        finally:
            get_settings.cache_clear()


def _load_sample_candles() -> list[OHLCVCandle]:
    """Load the bundled, version-controlled sample candles (zero external calls)."""
    assert _SAMPLE_CSV.exists(), f"bundled sample data missing: {_SAMPLE_CSV}"
    candles: list[OHLCVCandle] = []
    with _SAMPLE_CSV.open(newline="") as fh:
        for row in csv.DictReader(fh):
            candles.append(
                OHLCVCandle(
                    symbol="BTC/USDT",
                    timeframe="1m",
                    timestamp=row["timestamp"],
                    open=Decimal(row["open"]),
                    high=Decimal(row["high"]),
                    low=Decimal(row["low"]),
                    close=Decimal(row["close"]),
                    volume=Decimal(row["volume"]),
                )
            )
    return candles


# ---------------------------------------------------------------------------
# The end-to-end test
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    """One continuous run from a bundled candle through to a recorded paper trade."""

    async def test_candle_to_recorded_trade(self, pipeline_settings: Settings) -> None:
        from agents.decision.proposal_builder import ProposalBuilder
        from agents.decision.signal_aggregator import SignalAggregator
        from agents.execution.paper_broker import PaperBroker
        from agents.risk.agent import RiskAgent
        from agents.technical_analysis.candle_buffer import CandleBufferRegistry
        from agents.technical_analysis.signal_generator import SignalGenerator

        assert pipeline_settings.trading_mode == TradingMode.PAPER

        # --- Stage 1: candles → warm buffer ---------------------------------
        registry = CandleBufferRegistry()
        for candle in _load_sample_candles():
            registry.add(candle)
        buffer = registry.get("BTC/USDT", "1m")
        assert buffer is not None and buffer.is_warm, "buffer must be warm after 60 candles"

        # --- Stage 2: indicators → TechnicalSignal --------------------------
        signal = SignalGenerator().generate(buffer)
        assert signal is not None, "bundled sample must produce a (non-None) signal"
        assert _to_direction(signal.direction) == SignalDirection.BUY, (
            "engineered sample is designed to yield a BUY via mean-reversion rules"
        )
        assert signal.confidence >= 0.5
        assert signal.price > 0

        # --- Stage 3: decision (aggregate + build proposal) -----------------
        # These are exactly the components DecisionAgent composes internally.
        aggregator = SignalAggregator(
            technical_weight=0.7,
            sentiment_weight=0.3,
            min_confidence=0.5,
        )
        aggregated = aggregator.aggregate(symbol="BTC/USDT", technical=signal, sentiment=None)
        assert aggregated is not None, "aggregator should pass the BUY signal through"

        builder = ProposalBuilder(
            max_position_pct=0.02,
            default_stop_loss_pct=0.02,
            default_take_profit_pct=0.04,
            max_order_size_usd=1000.0,
        )
        proposal = builder.build(aggregated, portfolio_value_usd=Decimal("10000"))
        assert proposal is not None
        assert proposal.side == OrderSide.BUY
        assert proposal.requested_size_usd > Decimal("0")

        # --- Stage 4: risk evaluation ---------------------------------------
        risk = RiskAgent()
        assessment = await risk._evaluate(proposal)
        assert assessment.decision in (RiskDecision.APPROVED, RiskDecision.MODIFIED)
        assert assessment.rejection_reason is None
        assert assessment.approved_size_usd is not None and assessment.approved_size_usd > 0

        # --- Stage 5: paper execution → recorded trade ----------------------
        broker = PaperBroker(pipeline_settings)
        await broker.connect()
        try:
            result = await broker.place_order(assessment)
            assert result.is_paper is True
            assert result.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)
            assert result.side == OrderSide.BUY
            assert result.average_fill_price is not None and result.average_fill_price > 0
            assert result.proposal_id == assessment.proposal_id

            # "recorded trade": the fill is reflected in the broker's position state
            positions = await broker.get_positions()
            assert any(p.symbol == "BTC/USDT" for p in positions), (
                "the executed BUY must be recorded as an open position"
            )

            # ...and cash decreased by roughly the filled notional.
            balance = await broker.get_balance()
            assert balance.free_margin_usd < Decimal("10000")
        finally:
            await broker.disconnect()
