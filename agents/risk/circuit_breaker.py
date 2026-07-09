"""
agents/risk/circuit_breaker.py
--------------------------------
Manual circuit breaker for the Risk agent.

Trips when daily loss or total drawdown limits are breached. Once tripped,
all trade proposals are rejected until a human operator resets the system.

Design decisions:
- No auto-recovery: a tripped circuit breaker requires explicit human
  intervention via /api/control/resume (Phase 4). Automated reset would
  re-expose capital to the same losing conditions that tripped it.
- Single reason stored: the FIRST trip reason is preserved. Subsequent
  limit breaches don't overwrite it — the original cause is the one that
  matters for incident analysis.
- Thread safety: this is single-threaded asyncio, not needed.
"""

from __future__ import annotations

from datetime import UTC, datetime


class CircuitBreaker:
    """
    Latching circuit breaker — once tripped, stays tripped until reset().

    The Risk agent calls trip() on limit breach and reset() only when
    a human explicitly resumes trading via the control plane API.
    """

    def __init__(self) -> None:
        self._tripped: bool = False
        self._reason: str | None = None
        self._tripped_at: datetime | None = None

    @property
    def is_tripped(self) -> bool:
        """True if the breaker has been tripped and not yet reset."""
        return self._tripped

    @property
    def reason(self) -> str | None:
        """The reason the breaker was tripped, or None if not tripped."""
        return self._reason

    @property
    def tripped_at(self) -> datetime | None:
        """UTC datetime when the breaker tripped, or None if not tripped."""
        return self._tripped_at

    def trip(self, reason: str) -> None:
        """
        Trip the circuit breaker.

        Idempotent: calling trip() on an already-tripped breaker is a no-op
        (the first reason is preserved for diagnostics).

        Args:
            reason: Human-readable explanation for why the breaker tripped.
        """
        if not self._tripped:
            self._tripped = True
            self._reason = reason
            self._tripped_at = datetime.now(UTC)

    def reset(self) -> None:
        """
        Reset the circuit breaker, allowing trading to resume.

        Should only be called after a human operator has reviewed the
        situation and confirmed it is safe to resume trading.
        """
        self._tripped = False
        self._reason = None
        self._tripped_at = None

    def __repr__(self) -> str:
        if self._tripped:
            return f"CircuitBreaker(TRIPPED, reason={self._reason!r})"
        return "CircuitBreaker(CLOSED)"
