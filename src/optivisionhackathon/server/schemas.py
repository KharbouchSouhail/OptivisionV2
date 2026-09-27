"""Typed schemas for Visionnaire inference and decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DistanceLevel(str, Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


class Priority(str, Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"
    NONE = "NONE"


@dataclass(frozen=True)
class Detection:
    """A single YOLO detection with coarse distance classification."""

    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    distance_level: DistanceLevel


@dataclass
class InferenceResult:
    """Raw model outputs for a single frame."""

    detections: list[Detection] = field(default_factory=list)


@dataclass
class Decision:
    """Spoken / actionable decision returned to the client."""

    key: str
    message: str
    priority: Priority

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "message": self.message,
            "priority": self.priority.value
            if isinstance(self.priority, Priority)
            else self.priority,
        }

    @classmethod
    def silent(cls) -> Decision:
        return cls(key="silent", message="", priority=Priority.NONE)
