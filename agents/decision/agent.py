"""
agents/decision/agent.py
-------------------------
Decision Agent — fuses signals from analysis agents and publishes trade proposals.

Subscribes to:
    signal.technical.{symbol}   — from TechnicalAnalysisAgent
    signal.sentiment.{symbol}   — from SentimentAgent (when sentiment.enabled=True)

Publishes:
    decision.proposal           — TradeProposal → consumed by RiskAgent

Per-symbol signal cache (in-memory):
    Each arriving signal updates an in-memory dict keyed by symbol.
    Aggregation runs on every new signal arrival using whatever other
    signals are currently cached for that symbol.

Staleness: signals with is_expired=True are discarded in SignalAggregator
before aggregation, so no explicit TTL management is needed here.

Architecture rules:
    - Imports only from core/ and own package (agents.decision.*)
    - Channel names from Channels class only
    - Config via get_settings() only (accessed through self.settings)
    - No direct Redis access — all Redis ops via self.bus
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

from agents.base import BaseAgent
from agents.decision.proposal_builder import ProposalBuilder
from agents.decision.signal_aggregator import SignalAggregator
from core.messaging import Channels
from core.models.signals import SentimentSignal, SignalDirection, TechnicalSignal, _to_direction


class DecisionAgent(BaseAgent):
    """
    Aggregates incoming signals and proposes trades to the Risk agent.

    Signal caching: the latest TechnicalSignal and SentimentSignal per symbol
    are stored in memory. On each new signal arrival the aggregator runs with
    whatever peer signals are currently cached — this means a new technical
    signal triggers re-aggregation even if no new sentiment signal arrived.
    """

    name = "decision_agent"

    def __init__(self) -> None:
        super().__init__()
        ta_cfg = self.settings.technical_analysis
        risk_cfg = self.settings.risk

        self._aggregator = SignalAggregator(
            min_confidence=ta_cfg.min_signal_confidence,
        )
        self._builder = ProposalBuilder(
            max_position_pct=risk_cfg.max_position_pct,
            max_order_size_usd=risk_cfg.max_order_size_usd,
        )
        self._symbols: list[str] = self.settings.market_data.symbols

        # Per-symbol latest signal caches (in-memory, updated on each arrival)
        self._latest_technical: dict[str, TechnicalSignal] = {}
        self._latest_sentiment: dict[str, SentimentSignal] = {}

    async def setup(self) -> None:
        self.log.info(
            "decision_agent_setup",
            symbols=self._symbols,
            min_confidence=self.settings.technical_analysis.min_signal_confidence,
            technical_weight=self._aggregator.technical_weight,
            sentiment_enabled=self.settings.sentiment.enabled,
        )

    async def run_loop(self) -> None:
        """
        Subscribe to signal channels for all configured symbols and process
        each arriving signal through aggregation → proposal pipeline.

        Sentiment subscription runs as a background task (separate coroutine)
        only when sentiment.enabled=True. Both subscriptions share the same
        MessageBus pubsub connection; in practice message cross-contamination
        is rare because technical and sentiment signals arrive at very different
        rates and have different schemas (deserialization failures are logged
        and skipped, never crash the agent).
        """
        technical_channels = [Channels.technical_signal(sym) for sym in self._symbols]
        self.log.info("subscribing_to_channels", channels=technical_channels)

        tasks: list[asyncio.Task] = []

        tech_task = asyncio.create_task(
            self._consume_technical(technical_channels),
            name=f"{self.name}_technical_consumer",
        )
        tasks.append(tech_task)

        if self.settings.sentiment.enabled:
            sentiment_channels = [Channels.sentiment_signal(sym) for sym in self._symbols]
            sent_task = asyncio.create_task(
                self._consume_sentiment(sentiment_channels),
                name=f"{self.name}_sentiment_consumer",
            )
            tasks.append(sent_task)

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            raise

    # ------------------------------------------------------------------
    # Channel consumers
    # ------------------------------------------------------------------

    async def _consume_technical(self, channels: list[str]) -> None:
        """Process TechnicalSignal messages from all symbol channels."""
        self.log.debug("consume_technical_started", channels=channels)
        first = True
        async for _channel, signal in self.bus.subscribe_many(channels, TechnicalSignal):
            if first:
                self.log.info(
                    "first_technical_signal_received",
                    symbol=signal.symbol,
                    channel=_channel,
                    direction=signal.direction,
                    confidence=round(signal.confidence, 3),
                )
                first = False
            if not self._should_continue():
                break
            try:
                await self._on_technical_signal(signal)
                self._record_success()
            except Exception as exc:
                self._handle_error(exc, context=f"technical_signal:{signal.symbol}")

    async def _consume_sentiment(self, channels: list[str]) -> None:
        """Process SentimentSignal messages from all symbol channels."""
        async for _channel, signal in self.bus.subscribe_many(channels, SentimentSignal):
            if not self._should_continue():
                break
            try:
                await self._on_sentiment_signal(signal)
                self._record_success()
            except Exception as exc:
                self._handle_error(exc, context=f"sentiment_signal:{signal.symbol}")

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    async def _on_technical_signal(self, signal: TechnicalSignal) -> None:
        """Update cache, cache to Redis for observability, then re-aggregate."""
        self._latest_technical[signal.symbol] = signal

        # Cache for Risk agent staleness check and monitoring observability
        await self.bus.cache_set(
            f"signal:technical:{signal.symbol.replace('/', '-')}:latest",
            signal,
            ttl_seconds=self.settings.technical_analysis.signal_ttl_seconds,
        )

        await self._try_aggregate_and_propose(signal.symbol)

    async def _on_sentiment_signal(self, signal: SentimentSignal) -> None:
        """Update cache, cache to Redis for observability, then re-aggregate."""
        self._latest_sentiment[signal.symbol] = signal

        await self.bus.cache_set(
            f"signal:sentiment:{signal.symbol.replace('/', '-')}:latest",
            signal,
            ttl_seconds=self.settings.sentiment.sentiment_ttl_seconds,
        )

        await self._try_aggregate_and_propose(signal.symbol)

    # ------------------------------------------------------------------
    # Aggregation and proposal pipeline
    # ------------------------------------------------------------------

    async def _try_aggregate_and_propose(self, symbol: str) -> None:
        """Run aggregation for symbol and publish a proposal if actionable."""
        if self._trading_halted:
            self.log.debug("trading_halted_skipping_proposal", symbol=symbol)
            return

        technical = self._latest_technical.get(symbol)
        sentiment = self._latest_sentiment.get(symbol)

        self.log.debug(
            "aggregating_signals",
            symbol=symbol,
            has_technical=technical is not None,
            has_sentiment=sentiment is not None,
        )

        aggregated = self._aggregator.aggregate(
            symbol=symbol,
            technical=technical,
            sentiment=sentiment,
        )

        self.log.debug(
            "aggregation_result",
            symbol=symbol,
            aggregated=aggregated is not None,
            direction=_to_direction(aggregated.direction).value if aggregated is not None else None,
            confidence=round(aggregated.confidence, 4) if aggregated is not None else None,
        )

        if aggregated is None:
            self.log.debug(
                "no_valid_signals_after_filter",
                symbol=symbol,
                has_technical=technical is not None,
                has_sentiment=sentiment is not None,
            )
            return

        # Trend filter: reject longs below EMA50 and shorts above EMA50.
        if technical is not None:
            ema_50_dist = technical.metadata.get("ema_50_dist")
            if ema_50_dist is not None:
                dist = float(ema_50_dist)
                _agg_dir = _to_direction(aggregated.direction)
                _is_long = _agg_dir in (SignalDirection.BUY, SignalDirection.STRONG_BUY)
                _is_short = _agg_dir in (SignalDirection.SELL, SignalDirection.STRONG_SELL)
                if _is_long and dist < 0:
                    self.log.debug(
                        "trend_filter_result",
                        symbol=symbol,
                        passed=False,
                        reason="price_below_ema50_no_longs",
                        ema_50_dist=round(dist, 6),
                    )
                    self.log.info(
                        "trend_filter_rejected",
                        symbol=symbol,
                        reason="price_below_ema50_no_longs",
                        ema_50_dist=round(dist, 6),
                    )
                    return
                if _is_short and dist > 0:
                    self.log.debug(
                        "trend_filter_result",
                        symbol=symbol,
                        passed=False,
                        reason="price_above_ema50_no_shorts",
                        ema_50_dist=round(dist, 6),
                    )
                    self.log.info(
                        "trend_filter_rejected",
                        symbol=symbol,
                        reason="price_above_ema50_no_shorts",
                        ema_50_dist=round(dist, 6),
                    )
                    return
                self.log.debug(
                    "trend_filter_result",
                    symbol=symbol,
                    passed=True,
                    reason="ema50_aligned",
                    ema_50_dist=round(dist, 6),
                )

        portfolio_value = self._portfolio_value()
        proposal = self._builder.build(aggregated, portfolio_value)

        self.log.debug(
            "proposal_build_result",
            symbol=symbol,
            proposal=proposal is not None,
            direction=_to_direction(aggregated.direction).value,
            confidence=round(aggregated.confidence, 4),
        )

        if proposal is None:
            self.log.debug(
                "neutral_signal_suppressed",
                symbol=symbol,
                direction=aggregated.direction,
            )
            return

        await self.bus.publish(proposal)
        self.log.info(
            "proposal_published",
            symbol=symbol,
            proposal_id=str(proposal.proposal_id),
            side=proposal.side,
            direction=aggregated.direction,
            confidence=aggregated.confidence,
            composite_score=aggregated.composite_score,
            size_usd=float(proposal.requested_size_usd),
        )

    def _portfolio_value(self) -> Decimal:
        """
        Return the portfolio value used for proposal sizing.

        The Decision agent uses the configured initial balance as a stable
        sizing anchor — the Risk agent applies authoritative per-proposal
        sizing based on its live portfolio state.
        """
        return Decimal(str(self.settings.paper_initial_balance_usd))

    # ------------------------------------------------------------------
    # Health reporting
    # ------------------------------------------------------------------

    def health_extra(self) -> dict:
        return {
            "symbols": self._symbols,
            "cached_technical_signals": len(self._latest_technical),
            "cached_sentiment_signals": len(self._latest_sentiment),
            "sentiment_enabled": self.settings.sentiment.enabled,
        }


if __name__ == "__main__":
    import asyncio

    from agents.base import run_agent

    asyncio.run(run_agent(DecisionAgent()))
