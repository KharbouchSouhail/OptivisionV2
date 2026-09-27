"""Cooldown and speech-rate management for Visionnaire."""

from __future__ import annotations

import logging
import os
import time

from optivisionhackathon.server.schemas import Priority

logger = logging.getLogger(__name__)


_PRIORITY_RANK = {
    Priority.CRITICAL: 3,
    Priority.WARNING: 2,
    Priority.INFO: 1,
    Priority.NONE: 0,
}


class CooldownManager:
    """Prevent repetitive speech while allowing important alerts through.

    Behaviour:
    - Identical events are not repeated during their cooldown.
    - CRITICAL events can bypass normal cooldowns.
    - A global minimum gap prevents speech flooding.
    - Higher-priority versions of an existing event can interrupt.
    - Old history entries are automatically removed.
    """

    def __init__(
        self,
        cooldown_seconds: float | None = None,
        critical_cooldown_seconds: float | None = None,
        warning_cooldown_seconds: float | None = None,
        min_gap_seconds: float | None = None,
        history_max_age_seconds: float | None = None,
    ) -> None:

        self.cooldown_seconds = (
            cooldown_seconds
            if cooldown_seconds is not None
            else float(
                os.getenv(
                    "VISIONNAIRE_DECISION_COOLDOWN",
                    "4.0",
                )
            )
        )

        self.critical_cooldown_seconds = (
            critical_cooldown_seconds
            if critical_cooldown_seconds is not None
            else float(
                os.getenv(
                    "VISIONNAIRE_CRITICAL_COOLDOWN",
                    "1.0",
                )
            )
        )

        self.warning_cooldown_seconds = (
            warning_cooldown_seconds
            if warning_cooldown_seconds is not None
            else float(
                os.getenv(
                    "VISIONNAIRE_WARNING_COOLDOWN",
                    "3.0",
                )
            )
        )

        self.min_gap_seconds = (
            min_gap_seconds
            if min_gap_seconds is not None
            else float(
                os.getenv(
                    "VISIONNAIRE_SPEECH_MIN_GAP",
                    "2.0",
                )
            )
        )

        self.history_max_age_seconds = (
            history_max_age_seconds
            if history_max_age_seconds is not None
            else float(
                os.getenv(
                    "VISIONNAIRE_COOLDOWN_HISTORY_AGE",
                    "30.0",
                )
            )
        )

        # key -> (last spoken monotonic timestamp, priority)
        self._history: dict[
            str,
            tuple[float, Priority],
        ] = {}

        self._last_speech_time = 0.0
        self._last_key: str | None = None
        self._last_priority = Priority.NONE

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _priority_rank(
        priority: Priority,
    ) -> int:
        return _PRIORITY_RANK.get(
            priority,
            0,
        )

    def _cooldown_for(
        self,
        priority: Priority,
    ) -> float:
        """Return the cooldown appropriate for a priority."""

        if priority == Priority.CRITICAL:
            return self.critical_cooldown_seconds

        if priority == Priority.WARNING:
            return self.warning_cooldown_seconds

        if priority == Priority.INFO:
            return self.cooldown_seconds

        return float("inf")

    # ------------------------------------------------------------------
    # Speech decision
    # ------------------------------------------------------------------

    def should_speak(
        self,
        key: str,
        priority: Priority,
    ) -> bool:
        """Determine whether an event should be spoken."""

        if not key:
            return False

        if priority == Priority.NONE:
            return False

        now = time.monotonic()

        self._prune_history(now)

        # --------------------------------------------------------------
        # CRITICAL alerts
        # --------------------------------------------------------------
        #
        # Critical events can bypass the normal global speech gap.
        # This is important for something suddenly becoming very close.
        #

        if priority == Priority.CRITICAL:
            if (
                key not in self._history
            ):
                logger.debug(
                    "Allowing new CRITICAL event: %s",
                    key,
                )
                return True

            last_time, last_priority = self._history[key]

            elapsed = now - last_time

            # Higher priority escalation.
            if (
                self._priority_rank(priority)
                > self._priority_rank(last_priority)
            ):
                logger.info(
                    "CRITICAL priority escalation: "
                    "%s -> %s",
                    last_priority,
                    priority,
                )
                return True

            if (
                elapsed
                >= self.critical_cooldown_seconds
            ):
                return True

            return False

        # --------------------------------------------------------------
        # Global speech gap
        # --------------------------------------------------------------

        elapsed_since_last_speech = (
            now - self._last_speech_time
        )

        if (
            elapsed_since_last_speech
            < self.min_gap_seconds
        ):
            return False

        # --------------------------------------------------------------
        # First occurrence
        # --------------------------------------------------------------

        if key not in self._history:
            return True

        # --------------------------------------------------------------
        # Existing event
        # --------------------------------------------------------------

        last_time, last_priority = self._history[key]

        elapsed = now - last_time

        # Higher priority can bypass the existing cooldown.
        if (
            self._priority_rank(priority)
            > self._priority_rank(last_priority)
        ):
            logger.info(
                "Priority escalation for %s: %s -> %s",
                key,
                last_priority,
                priority,
            )
            return True

        allowed_cooldown = self._cooldown_for(
            priority
        )

        return elapsed >= allowed_cooldown

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_spoken(
        self,
        key: str,
        priority: Priority,
    ) -> None:
        """Record a successfully scheduled spoken message."""

        if not key:
            return

        if priority == Priority.NONE:
            return

        now = time.monotonic()

        self._history[key] = (
            now,
            priority,
        )

        self._last_speech_time = now
        self._last_key = key
        self._last_priority = priority

        self._prune_history(now)

    # ------------------------------------------------------------------
    # History maintenance
    # ------------------------------------------------------------------

    def _prune_history(
        self,
        now: float,
    ) -> None:
        """Remove old cooldown entries."""

        expired: list[str] = []

        for key, (timestamp, _) in self._history.items():
            if (
                now - timestamp
                > self.history_max_age_seconds
            ):
                expired.append(key)

        for key in expired:
            del self._history[key]

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all cooldown state."""

        self._history.clear()

        self._last_speech_time = 0.0
        self._last_key = None
        self._last_priority = Priority.NONE

        logger.debug(
            "Cooldown history reset"
        )

    @property
    def last_key(self) -> str | None:
        """Return the last spoken event key."""

        return self._last_key

    @property
    def last_priority(self) -> Priority:
        """Return the priority of the last spoken event."""

        return self._last_priority