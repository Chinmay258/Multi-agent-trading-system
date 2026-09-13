"""
core/models/system.py
---------------------
System-level event models: heartbeats, commands, risk overrides, and alerts.

These are the operational messages that keep the system alive and controllable:
- Agents publish heartbeats so Monitoring knows they're alive.
- The control plane publishes commands to start/stop/pause agents.
- The Risk agent publishes overrides to halt all trading immediately.
- Monitoring publishes alerts when thresholds are breached.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import Field

from core.models.market import BaseMarketModel

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AgentStatus(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    QUARANTINED = "quarantined"  # Crashed too many times, awaiting human review


class SystemCommand(str, Enum):
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    RESTART = "restart"
    HALT_TRADING = "halt_trading"  # Stop new trades, keep positions
    EMERGENCY_EXIT = "emergency_exit"  # Close all positions immediately (future)


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    AGENT_CRASH = "agent_crash"
    AGENT_STALE = "agent_stale"  # No heartbeat received
    DAILY_LOSS_THRESHOLD = "daily_loss_threshold"
    DRAWDOWN_THRESHOLD = "drawdown_threshold"
    STALE_MARKET_DATA = "stale_market_data"
    CIRCUIT_BREAKER_TRIPPED = "circuit_breaker_tripped"
    ORDER_REJECTED = "order_rejected"
    EXCHANGE_CONNECTION_LOST = "exchange_connection_lost"


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


class AgentHeartbeat(BaseMarketModel):
    """
    Published by every agent every N seconds on: system.heartbeat

    The Monitoring agent watches for these. If an agent's heartbeat goes
    stale, it triggers an alert and potentially a restart.
    """

    agent_name: str = Field(description="Unique agent identifier")
    status: AgentStatus
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Performance metrics (optional, reported when available)
    messages_processed: int = Field(
        default=0, description="Messages processed since last heartbeat"
    )
    errors_since_start: int = Field(default=0)
    uptime_seconds: float = Field(default=0.0)
    memory_mb: float | None = None
    cpu_pct: float | None = None

    # Agent-specific health data
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent-specific health data (e.g. last_symbol_fetched, signal_count)",
    )

    @property
    def channel_key(self) -> str:
        return "system.heartbeat"


# ---------------------------------------------------------------------------
# System command (control plane → agents)
# ---------------------------------------------------------------------------


class SystemCommandMessage(BaseMarketModel):
    """
    Published by the control plane API to command agents.
    Published on: system.command

    Agents subscribe and act on commands targeting their name or 'all'.
    """

    command_id: UUID = Field(default_factory=uuid4)
    command: SystemCommand
    target_agent: str = Field(description="Agent name to target, or 'all' for broadcast")
    issued_by: str = Field(default="api", description="Who issued the command")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reason: str | None = None

    @property
    def channel_key(self) -> str:
        return "system.command"


# ---------------------------------------------------------------------------
# Risk override (Risk agent → all, emergency halt)
# ---------------------------------------------------------------------------


class RiskOverride(BaseMarketModel):
    """
    Emergency halt signal published by the Risk agent.
    Published on: system.risk_override

    On receipt, the Execution agent immediately stops accepting new proposals.
    The Monitoring agent alerts humans. The system requires manual intervention
    to resume trading after a risk override.
    """

    override_id: UUID = Field(default_factory=uuid4)
    reason: str = Field(description="Why the override was triggered")
    triggered_by: str = Field(description="Agent or human who triggered this")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Snapshot of the condition that triggered the override
    daily_loss_pct: float | None = None
    total_drawdown_pct: float | None = None
    requires_human_reset: bool = Field(
        default=True,
        description="If True, system cannot auto-resume — human must call /api/control/resume",
    )

    @property
    def channel_key(self) -> str:
        return "system.risk_override"


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------


class SystemAlert(BaseMarketModel):
    """
    Alert published by the Monitoring agent when thresholds are breached.
    Published on: system.alert

    Downstream: alerter.py routes to Slack, email, PagerDuty, etc.
    """

    alert_id: UUID = Field(default_factory=uuid4)
    alert_type: AlertType
    severity: AlertSeverity
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    message: str
    agent_name: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    resolved: bool = Field(default=False)
    resolved_at: datetime | None = None

    @property
    def channel_key(self) -> str:
        return "system.alert"
