"""Intelligent cooldown manager with priority escalation and rate limiting."""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from optivisionhackathon.server.schemas import Priority

logger = logging.getLogger(__name__)

_PRIORITY_RANK = {
    Priority.CRITICAL: 3,
    Priority.WARNING: 2,
    Priority.INFO: 1,
    Priority.NONE: 0,
}


class CooldownManager:
    """Manages speech frequency and prevents repetitive TTS announcements.

    - Prevents repeating the exact same announcement within the cooldown window.
    - Escalations to CRITICAL priority bypass the cooldown to protect user safety.
    - Enforces a minimal pause between consecutive spoken messages.
    """

    def __init__(
        self,
        cooldown_seconds: float | None = None,
        critical_cooldown_seconds: float | None = None,
        min_gap_seconds: float = 0.8,
    ) -> None:
        self.cooldown_seconds = (
            cooldown_seconds
            if cooldown_seconds is not None
            else float(os.getenv("VISIONNAIRE_DECISION_COOLDOWN", "2.5"))
        )
        self.critical_cooldown_seconds = (
            critical_cooldown_seconds
            if critical_cooldown_seconds is not None
            else float(os.getenv("VISIONNAIRE_CRITICAL_COOLDOWN", "1.0"))
        )
        self.min_gap_seconds = min_gap_seconds

        # key -> (timestamp_monotonic, priority)
        self._history: dict[str, tuple[float, Priority]] = {}
        self._last_speech_time: float = 0.0
        self._last_key: Optional[str] = None

    def should_speak(self, key: str, priority: Priority) -> bool:
        """Determine if a message with the given key and priority should be spoken."""
        if not key or priority == Priority.NONE:
            return False

        now = time.monotonic()

        # Enforce minimum spacing between any spoken utterances
        if (now - self._last_speech_time) < self.min_gap_seconds:
            # Only CRITICAL priority can interrupt within the minimal gap
            if priority != Priority.CRITICAL:
                return False

        # If this key has never been spoken, allow it
        if key not in self._history:
            return True

        last_time, last_pri = self._history[key]
        elapsed = now - last_time

        # Escalation: higher priority than last time bypasses cooldown
        if _PRIORITY_RANK[priority] > _PRIORITY_RANK[last_pri]:
            logger.debug("Cooldown bypassed due to priority escalation: %s -> %s", last_pri, priority)
            return True

        # Use shorter cooldown for critical alerts
        allowed_cooldown = (
            self.critical_cooldown_seconds
            if priority == Priority.CRITICAL
            else self.cooldown_seconds
        )

        return elapsed >= allowed_cooldown

    def record_spoken(self, key: str, priority: Priority) -> None:
        """Record that a message was spoken."""
        now = time.monotonic()
        self._history[key] = (now, priority)
        self._last_speech_time = now
        self._last_key = key
        self._prune_history(now)

    def _prune_history(self, now: float) -> None:
        """Prune old entries to prevent memory growth."""
        max_age = max(self.cooldown_seconds, self.critical_cooldown_seconds) * 5.0
        expired = [k for k, (t, _) in self._history.items() if (now - t) > max_age]
        for k in expired:
            del self._history[k]

    def reset(self) -> None:
        """Clear all cooldown history."""
        self._history.clear()
        self._last_speech_time = 0.0
        self._last_key = None
