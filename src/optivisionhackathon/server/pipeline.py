"""Inference pipeline: orchestrates models and produces a Decision.

Current MVP flow:
    frame -> YOLO -> InferenceResult -> basic decision -> Decision

Structured so MiDaS, face, OCR, and fusion can be added later
without changing the WebSocket transport layer.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from optivisionhackathon.server.schemas import (
    Decision,
    Detection,
    DistanceLevel,
    InferenceResult,
    Priority,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from optivisionhackathon.server.models.yolo import YOLODetector

logger = logging.getLogger(__name__)

_DISTANCE_TO_PRIORITY = {
    DistanceLevel.CRITICAL: Priority.CRITICAL,
    DistanceLevel.WARNING: Priority.WARNING,
    DistanceLevel.INFO: Priority.INFO,
}

_PRIORITY_RANK = {
    Priority.CRITICAL: 3,
    Priority.WARNING: 2,
    Priority.INFO: 1,
    Priority.NONE: 0,
}


def build_speech_message(detections: list[Detection]) -> str:
    """Build a short spoken sentence from unique obstacle labels."""
    if not detections:
        return ""

    # Unique labels, ordered by urgency then confidence.
    ranked = sorted(
        detections,
        key=lambda d: (
            _PRIORITY_RANK[_DISTANCE_TO_PRIORITY[d.distance_level]],
            d.confidence,
        ),
        reverse=True,
    )
    labels: list[str] = []
    seen: set[str] = set()
    for det in ranked:
        if det.label in seen:
            continue
        seen.add(det.label)
        labels.append(det.label)

    if len(labels) == 1:
        return f"{labels[0].capitalize()} ahead."
    if len(labels) == 2:
        return f"There is a {labels[0]} and a {labels[1]} ahead."
    leading = ", ".join(labels[:-1])
    return f"There is a {leading}, and a {labels[-1]} ahead."


def decision_key(detections: list[Detection]) -> str:
    """Stable key for speech cooldown (sorted unique labels)."""
    if not detections:
        return "silent"
    labels = sorted({d.label for d in detections})
    top = max(
        detections,
        key=lambda d: (
            _PRIORITY_RANK[_DISTANCE_TO_PRIORITY[d.distance_level]],
            d.confidence,
        ),
    )
    return f"{top.distance_level.value}:{','.join(labels)}"


class Pipeline:
    """Frame processing pipeline (YOLO-only for the current MVP)."""

    def __init__(
        self,
        detector: YOLODetector,
        cooldown_seconds: float = 2.0,
    ) -> None:
        self.detector = detector
        self.cooldown_seconds = cooldown_seconds
        self._last_spoken_key: str | None = None
        self._last_spoken_at: float = 0.0

    def process_frame(self, frame: NDArray[np.uint8]) -> Decision:
        """Run inference on a frame and return a Decision for the client."""
        detections = self.detector.detect(frame)
        inference = InferenceResult(detections=detections)

        # Future extension point:
        # depth = self.depth_estimator.estimate(frame)
        # faces = self.face_recognizer.recognize(frame)
        # text = self.ocr_engine.read(frame)
        # return self.fusion_engine.fuse(inference, depth, faces, text)

        return self._decide(inference)

    def _decide(self, inference: InferenceResult) -> Decision:
        detections = list(inference.detections)
        now_wall = time.time()

        if not detections:
            return Decision.silent(detections=[], key="silent")

        best = max(
            detections,
            key=lambda d: (
                _PRIORITY_RANK[_DISTANCE_TO_PRIORITY[d.distance_level]],
                d.confidence,
            ),
        )
        priority = _DISTANCE_TO_PRIORITY[best.distance_level]
        key = decision_key(detections)
        message = build_speech_message(detections)

        # Cooldown suppresses speech only — detections still returned for overlay.
        now = time.monotonic()
        if (
            key == self._last_spoken_key
            and (now - self._last_spoken_at) < self.cooldown_seconds
        ):
            logger.debug("Suppressing repeated speech: %s", key)
            return Decision(
                key=key,
                message="",
                priority=priority,
                detections=detections,
                timestamp=now_wall,
            )

        self._last_spoken_key = key
        self._last_spoken_at = now
        return Decision(
            key=key,
            message=message,
            priority=priority,
            detections=detections,
            timestamp=now_wall,
        )
