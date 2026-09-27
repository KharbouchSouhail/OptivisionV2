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
    """A single YOLO detection with distance classification and estimated meters."""

    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    distance_level: DistanceLevel
    distance_meters: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "bbox": [float(v) for v in self.bbox],
            "distance_level": self.distance_level.value
            if isinstance(self.distance_level, DistanceLevel)
            else str(self.distance_level),
        }
        if self.distance_meters > 0.0:
            data["distance_meters"] = round(float(self.distance_meters), 2)
        return data


@dataclass(frozen=True)
class FaceDetection:
    """A detected or recognized face."""

    name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    distance_meters: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [float(v) for v in self.bbox],
        }
        if self.distance_meters > 0.0:
            data["distance_meters"] = round(float(self.distance_meters), 2)
        return data


@dataclass(frozen=True)
class TextDetection:
    """A detected text snippet or signage."""

    text: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(float(self.confidence), 4),
            "bbox": [float(v) for v in self.bbox],
        }


@dataclass
class InferenceResult:
    """Raw model outputs for a single frame across all vision modalities."""

    detections: list[Detection] = field(default_factory=list)
    faces: list[FaceDetection] = field(default_factory=list)
    texts: list[TextDetection] = field(default_factory=list)
    depth_map: Any = None


@dataclass
class Decision:
    """Structured result returned to the client over WebSocket JSON."""

    key: str
    message: str
    priority: Priority
    detections: list[Detection] = field(default_factory=list)
    faces: list[FaceDetection] = field(default_factory=list)
    texts: list[TextDetection] = field(default_factory=list)
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "message": self.message,
            "priority": self.priority.value
            if isinstance(self.priority, Priority)
            else self.priority,
            "detections": [d.to_dict() for d in self.detections],
            "faces": [f.to_dict() for f in self.faces],
            "texts": [t.to_dict() for t in self.texts],
            "timestamp": float(self.timestamp),
        }

    @classmethod
    def silent(
        cls,
        detections: list[Detection] | None = None,
        faces: list[FaceDetection] | None = None,
        texts: list[TextDetection] | None = None,
        key: str = "silent",
        timestamp: float | None = None,
    ) -> Decision:
        return cls(
            key=key,
            message="",
            priority=Priority.NONE,
            detections=list(detections or []),
            faces=list(faces or []),
            texts=list(texts or []),
            timestamp=time.time() if timestamp is None else float(timestamp),
        )
