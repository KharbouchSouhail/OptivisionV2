"""Typed schemas for Visionnaire inference and decisions."""

from __future__ import annotations

import time
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "bbox": [float(v) for v in self.bbox],
            "distance_level": self.distance_level.value
            if isinstance(self.distance_level, DistanceLevel)
            else str(self.distance_level),
        }


@dataclass
class InferenceResult:
    """Raw model outputs for a single frame."""

    detections: list[Detection] = field(default_factory=list)


@dataclass
class Decision:
    """Structured result returned to the client over WebSocket JSON."""

    key: str
    message: str
    priority: Priority
    detections: list[Detection] = field(default_factory=list)
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "message": self.message,
            "priority": self.priority.value
            if isinstance(self.priority, Priority)
            else self.priority,
            "detections": [d.to_dict() for d in self.detections],
            "timestamp": float(self.timestamp),
        }

    @classmethod
    def silent(
        cls,
        detections: list[Detection] | None = None,
        key: str = "silent",
    ) -> Decision:
        return cls(
            key=key,
            message="",
            priority=Priority.NONE,
            detections=list(detections or []),
            timestamp=time.time(),
        )
