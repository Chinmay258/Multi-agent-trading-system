"""
agents/sentiment/agent.py
-------------------------
SentimentAgent — the seventh agent in the system.

Status: DISABLED STUB (MVP).

The sentiment agent is a first-class member of the architecture (it is one of
the seven agents and is referenced by docker-compose and the monitoring layer),
but it ships disabled. With ``SENTIMENT_ENABLED=false`` (the default) the agent
starts, registers a heartbeat so Monitoring counts it as healthy, and then idles
until shutdown. It never publishes anything.

This file exists so that ``python -m agents.sentiment.agent`` resolves and the
``sentiment_agent`` compose service is valid. When sentiment analysis is
implemented, ``run_loop`` will subscribe to news/social feeds, score them, and
publish ``SentimentSignal`` messages on Redis for the DecisionAgent to fuse.

Design decisions:
- No business logic here yet — the heartbeat/lifecycle is inherited from
  BaseAgent, so an idle stub is a few lines.
- Respects the ``sentiment.enabled`` config flag exactly like every other
  agent reads config via ``get_settings()``.
- Heartbeats keep flowing while idle so the system reports 7 healthy agents.
"""

from __future__ import annotations

from agents.base import BaseAgent, run_agent


class SentimentAgent(BaseAgent):
    """Disabled-by-default sentiment agent. Idles until implemented/enabled."""

    name = "sentiment_agent"

    async def run_loop(self) -> None:
        """
        Idle loop.

        When disabled (the MVP default) the agent simply waits for the stop
        event while the base class keeps publishing heartbeats. When enabled,
        sentiment ingestion is not yet implemented, so we log a clear warning
        and idle rather than silently doing nothing.
        """
        if not self.settings.sentiment.enabled:
            self.log.info("sentiment_disabled_idle")
        else:
            # TODO Phase 9: subscribe to news/social feeds, score sentiment,
            # and publish SentimentSignal messages for the DecisionAgent.
            self.log.warning("sentiment_enabled_but_not_implemented_idle")

        await self._stop_event.wait()

    def health_extra(self) -> dict:
        return {"enabled": self.settings.sentiment.enabled, "implemented": False}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    from core.logging import configure_logging

    configure_logging()
    asyncio.run(run_agent(SentimentAgent()))
