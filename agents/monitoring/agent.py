"""
agents/monitoring/agent.py
---------------------------
MonitoringAgent — the system's observability and health layer.

Responsibilities:
- Watch for agent heartbeats and detect stale/crashed agents.
- Expose a health check HTTP endpoint (lightweight aiohttp server).
- Persist heartbeat snapshots to TimescaleDB for dashboards.
- Publish SystemAlert events when thresholds are breached.
- Route alerts to configured channels (Slack, log, etc.).
- Maintain a registry of known agents and their last-seen status.

Design decisions:
- MonitoringAgent is a PASSIVE observer — it subscribes to system.heartbeat
  and system.alert channels but never publishes commands or modifies state.
  The only thing it initiates is alerts.
- It runs its own lightweight HTTP server (aiohttp, not FastAPI) for health
  checks. This is intentionally separate from the main API to ensure health
  checks work even if FastAPI is down.
- Agent registry: populated dynamically from heartbeats. We don't hardcode
  expected agents — the system discovers them. This makes adding new agents
  transparent to Monitoring.
- Alert deduplication: the same alert type for the same agent is suppressed
  for alert_cooldown_seconds to avoid Slack flood during a crash loop.

Health check endpoint:
    GET http://localhost:8081/health
    Returns 200 with JSON if all agents are healthy.
    Returns 503 with JSON if any agent is stale or crashed.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

from aiohttp import web
from sqlalchemy import text

from agents.base import BaseAgent, run_agent
from core.db.connection import get_session
from core.logging import get_logger
from core.messaging import Channels
from core.models.system import (
    AgentHeartbeat,
    AgentStatus,
    AlertSeverity,
    AlertType,
    SystemAlert,
)

logger = get_logger("monitoring_agent")

# How long without a heartbeat before an agent is considered stale
_STALE_THRESHOLD_MULTIPLIER = 3  # missed_heartbeats before alert
# Minimum seconds between repeated alerts for the same issue
_ALERT_COOLDOWN_SECONDS = 300


class AgentRecord:
    """
    In-memory record of a known agent's health state.
    Updated every time a heartbeat is received.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.last_heartbeat: AgentHeartbeat | None = None
        self.last_seen_at: datetime | None = None
        self.status: AgentStatus = AgentStatus.STARTING
        self.consecutive_missed: int = 0
        self.total_heartbeats: int = 0
        self.last_alert_at: datetime | None = None

    def update(self, heartbeat: AgentHeartbeat) -> None:
        self.last_heartbeat = heartbeat
        self.last_seen_at = datetime.now(UTC)
        self.status = AgentStatus(heartbeat.status)
        self.consecutive_missed = 0
        self.total_heartbeats += 1

    @property
    def age_seconds(self) -> float | None:
        if self.last_seen_at is None:
            return None
        return (datetime.now(UTC) - self.last_seen_at).total_seconds()

    def is_stale(self, timeout_seconds: int) -> bool:
        age = self.age_seconds
        return age is not None and age > timeout_seconds

    def is_alert_on_cooldown(self) -> bool:
        if self.last_alert_at is None:
            return False
        age = (datetime.now(UTC) - self.last_alert_at).total_seconds()
        return age < _ALERT_COOLDOWN_SECONDS

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "last_seen_seconds_ago": round(self.age_seconds, 1) if self.age_seconds else None,
            "consecutive_missed_heartbeats": self.consecutive_missed,
            "total_heartbeats": self.total_heartbeats,
            "is_stale": self.is_stale(
                30 * _STALE_THRESHOLD_MULTIPLIER  # approximation without config access
            ),
        }


class MonitoringAgent(BaseAgent):
    """
    System-wide observability agent.

    Runs two concurrent loops:
    1. Heartbeat listener: subscribes to system.heartbeat, updates registry.
    2. Staleness checker: periodically scans registry for stale agents.

    Also runs a lightweight HTTP health check server.
    """

    name = "monitoring_agent"

    def __init__(self) -> None:
        super().__init__()
        self._agents: dict[str, AgentRecord] = {}
        self._http_app: web.Application | None = None
        self._http_runner: web.AppRunner | None = None
        self._http_port = 8081
        self._check_task: asyncio.Task | None = None
        self._alert_listener_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Start the HTTP health check server."""
        await self._start_http_server()

    async def teardown(self) -> None:
        """Stop HTTP server and cancel background tasks."""
        if self._check_task and not self._check_task.done():
            self._check_task.cancel()
        if self._alert_listener_task and not self._alert_listener_task.done():
            self._alert_listener_task.cancel()
        await self._stop_http_server()

    async def run_loop(self) -> None:
        """
        Run the heartbeat listener and staleness checker concurrently.
        """
        self._check_task = asyncio.create_task(
            self._staleness_check_loop(),
            name="staleness_checker",
        )
        self._alert_listener_task = asyncio.create_task(
            self._alert_listener_loop(),
            name="alert_listener",
        )

        # Primary loop: listen for heartbeats
        await self._heartbeat_listener_loop()

    # ------------------------------------------------------------------
    # Heartbeat listener
    # ------------------------------------------------------------------

    async def _heartbeat_listener_loop(self) -> None:
        """Subscribe to system.heartbeat and update agent registry."""
        self.log.info("heartbeat_listener_started")

        async for heartbeat in self.bus.subscribe(
            Channels.SYSTEM_HEARTBEAT,
            AgentHeartbeat,
        ):
            if not self._should_continue():
                break

            try:
                await self._process_heartbeat(heartbeat)
            except Exception as e:
                self.log.error("heartbeat_processing_error", error=str(e))

    async def _process_heartbeat(self, heartbeat: AgentHeartbeat) -> None:
        """Update the agent registry and persist the heartbeat."""
        agent_name = heartbeat.agent_name

        if agent_name not in self._agents:
            self._agents[agent_name] = AgentRecord(agent_name)
            self.log.info("new_agent_discovered", agent=agent_name)

        record = self._agents[agent_name]
        record.update(heartbeat)

        # Log status transitions
        if heartbeat.status == AgentStatus.ERROR.value:
            self.log.warning(
                "agent_in_error_state",
                agent=agent_name,
                errors=heartbeat.errors_since_start,
            )
        elif heartbeat.status == AgentStatus.QUARANTINED.value:
            await self._raise_alert(
                alert_type=AlertType.AGENT_CRASH,
                severity=AlertSeverity.CRITICAL,
                agent_name=agent_name,
                message=f"Agent '{agent_name}' has been quarantined after repeated crashes.",
                record=record,
            )

        # Persist to DB
        await self._persist_heartbeat(heartbeat)

        self.log.debug(
            "heartbeat_received",
            agent=agent_name,
            status=heartbeat.status,
            uptime=round(heartbeat.uptime_seconds, 0),
            messages=heartbeat.messages_processed,
        )

    # ------------------------------------------------------------------
    # Staleness checker
    # ------------------------------------------------------------------

    async def _staleness_check_loop(self) -> None:
        """
        Periodically scan all known agents for stale heartbeats.
        Runs every heartbeat_interval_seconds.
        """
        cfg = self.settings.monitoring
        check_interval = cfg.heartbeat_interval_seconds
        timeout = cfg.heartbeat_timeout_seconds

        while self._should_continue():
            await asyncio.sleep(check_interval)

            for name, record in self._agents.items():
                if name == self.name:
                    continue  # Don't check ourselves

                if record.is_stale(timeout):
                    record.consecutive_missed += 1

                    if not record.is_alert_on_cooldown():
                        age = record.age_seconds
                        await self._raise_alert(
                            alert_type=AlertType.AGENT_STALE,
                            severity=AlertSeverity.CRITICAL,
                            agent_name=name,
                            message=(
                                f"Agent '{name}' has not sent a heartbeat for "
                                f"{age:.0f}s (timeout: {timeout}s). "
                                f"Consecutive missed: {record.consecutive_missed}."
                            ),
                            record=record,
                            extra={"age_seconds": age, "timeout_seconds": timeout},
                        )

    # ------------------------------------------------------------------
    # Alert listener (for alerts raised by other agents)
    # ------------------------------------------------------------------

    async def _alert_listener_loop(self) -> None:
        """
        Listen for SystemAlert events published by other agents.
        Routes them to configured alert channels (Slack, log, etc.).
        """
        async for alert in self.bus.subscribe(Channels.SYSTEM_ALERT, SystemAlert):
            if not self._should_continue():
                break
            await self._route_alert(alert)

    # ------------------------------------------------------------------
    # Alert creation and routing
    # ------------------------------------------------------------------

    async def _raise_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        agent_name: str | None = None,
        record: AgentRecord | None = None,
        extra: dict | None = None,
    ) -> None:
        """Create a SystemAlert, publish it, persist it, and route it."""
        alert = SystemAlert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            agent_name=agent_name,
            extra=extra or {},
        )

        # Update alert cooldown on the record
        if record:
            record.last_alert_at = datetime.now(UTC)

        # Publish to bus (other agents can subscribe to system.alert)
        try:
            await self.bus.publish(alert)
        except Exception as e:
            self.log.error("alert_publish_failed", error=str(e))

        # Route to external channels
        await self._route_alert(alert)

        # Persist
        await self._persist_alert(alert)

    async def _route_alert(self, alert: SystemAlert) -> None:
        """Route an alert to configured notification channels."""
        # Always log
        log_method = {
            AlertSeverity.INFO: self.log.info,
            AlertSeverity.WARNING: self.log.warning,
            AlertSeverity.CRITICAL: self.log.critical,
        }.get(AlertSeverity(alert.severity), self.log.warning)

        log_method(
            "system_alert",
            alert_type=alert.alert_type,
            severity=alert.severity,
            message=alert.message,
            agent=alert.agent_name,
        )

        # Slack (if configured)
        if self.settings.monitoring.slack_webhook_url:
            await self._send_slack_alert(alert)

        # TODO: Add PagerDuty, email, Telegram, etc.

    async def _send_slack_alert(self, alert: SystemAlert) -> None:
        """Send alert to Slack webhook."""
        import aiohttp

        webhook_url = self.settings.monitoring.slack_webhook_url
        if not webhook_url:
            return

        emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(alert.severity, "⚠️")
        payload = {
            "text": f"{emoji} *[{alert.severity.upper()}]* {alert.message}",
            "attachments": [
                {
                    "color": {"info": "#36a64f", "warning": "#ff9f00", "critical": "#cc0000"}.get(
                        alert.severity, "#ff9f00"
                    ),
                    "fields": [
                        {"title": "Type", "value": alert.alert_type, "short": True},
                        {"title": "Agent", "value": alert.agent_name or "system", "short": True},
                        {"title": "Time", "value": str(alert.timestamp), "short": False},
                    ],
                }
            ],
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url.get_secret_value(),
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status != 200:
                        self.log.warning(
                            "slack_alert_failed",
                            status=resp.status,
                            alert_type=alert.alert_type,
                        )
        except Exception as e:
            self.log.error("slack_send_error", error=str(e))

    # ------------------------------------------------------------------
    # HTTP health check server
    # ------------------------------------------------------------------

    async def _start_http_server(self) -> None:
        """Start a lightweight aiohttp health check server on port 8081."""
        self._http_app = web.Application()
        self._http_app.router.add_get("/health", self._handle_health)
        self._http_app.router.add_get("/agents", self._handle_agents)

        self._http_runner = web.AppRunner(self._http_app)
        await self._http_runner.setup()
        site = web.TCPSite(self._http_runner, "0.0.0.0", self._http_port)
        await site.start()

        self.log.info(
            "health_server_started",
            port=self._http_port,
            endpoints=["/health", "/agents"],
        )

    async def _stop_http_server(self) -> None:
        if self._http_runner:
            await self._http_runner.cleanup()

    async def _handle_health(self, request: web.Request) -> web.Response:
        """
        GET /health — returns 200 if healthy, 503 if any agent is stale.
        Used by Docker HEALTHCHECK and load balancers.
        """
        timeout = self.settings.monitoring.heartbeat_timeout_seconds
        stale_agents = [
            name
            for name, record in self._agents.items()
            if record.is_stale(timeout) and name != self.name
        ]

        healthy = len(stale_agents) == 0
        payload = {
            "status": "healthy" if healthy else "degraded",
            "trading_mode": self.settings.trading_mode.value,
            "agent_count": len(self._agents),
            "stale_agents": stale_agents,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        return web.Response(
            status=200 if healthy else 503,
            content_type="application/json",
            text=json.dumps(payload),
        )

    async def _handle_agents(self, request: web.Request) -> web.Response:
        """
        GET /agents — returns full registry of known agents and their status.
        """
        payload = {name: record.to_dict() for name, record in self._agents.items()}
        return web.Response(
            status=200,
            content_type="application/json",
            text=json.dumps(payload),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _persist_heartbeat(self, heartbeat: AgentHeartbeat) -> None:
        """Persist heartbeat snapshot to TimescaleDB."""
        try:
            async with get_session() as session:
                await session.execute(
                    text("""
                        INSERT INTO agent_heartbeats
                            (agent_name, status, messages_processed,
                             errors_since_start, uptime_seconds, recorded_at)
                        VALUES
                            (:agent_name, :status, :messages_processed,
                             :errors_since_start, :uptime_seconds, :recorded_at)
                    """),
                    {
                        "agent_name": heartbeat.agent_name,
                        "status": heartbeat.status,
                        "messages_processed": heartbeat.messages_processed,
                        "errors_since_start": heartbeat.errors_since_start,
                        "uptime_seconds": heartbeat.uptime_seconds,
                        "recorded_at": heartbeat.timestamp,
                    },
                )
                await session.commit()
        except Exception as e:
            # Non-fatal — monitoring continues even if DB write fails
            self.log.error("heartbeat_persist_failed", error=str(e))

    async def _persist_alert(self, alert: SystemAlert) -> None:
        """Persist alert to the database."""
        try:
            async with get_session() as session:
                await session.execute(
                    text("""
                        INSERT INTO system_alerts
                            (alert_id, alert_type, severity, message,
                             agent_name, resolved, created_at)
                        VALUES
                            (:alert_id, :alert_type, :severity, :message,
                             :agent_name, :resolved, :created_at)
                    """),
                    {
                        "alert_id": str(alert.alert_id),
                        "alert_type": alert.alert_type,
                        "severity": alert.severity,
                        "message": alert.message,
                        "agent_name": alert.agent_name,
                        "resolved": alert.resolved,
                        "created_at": alert.timestamp,
                    },
                )
                await session.commit()
        except Exception as e:
            self.log.error("alert_persist_failed", error=str(e))

    # ------------------------------------------------------------------
    # Health reporting
    # ------------------------------------------------------------------

    def health_extra(self) -> dict:
        """Report the number of known agents and stale count."""
        timeout = self.settings.monitoring.heartbeat_timeout_seconds
        stale = [n for n, r in self._agents.items() if r.is_stale(timeout)]
        return {
            "known_agents": len(self._agents),
            "stale_agents": stale,
            "health_endpoint": f"http://localhost:{self._http_port}/health",
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    from core.logging import configure_logging

    configure_logging()
    asyncio.run(run_agent(MonitoringAgent()))
