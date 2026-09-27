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
        # --- YOLO ---
        detections = self.detector.detect(frame)
        inference = InferenceResult(detections=detections)

        # Future extension point:
        # depth = self.depth_estimator.estimate(frame)
        # faces = self.face_recognizer.recognize(frame)
        # text = self.ocr_engine.read(frame)
        # return self.fusion_engine.fuse(inference, depth, faces, text)

        return self._decide(inference)

    def _decide(self, inference: InferenceResult) -> Decision:
        if not inference.detections:
            return Decision.silent()

        # Prefer the most urgent (closest / highest priority) obstacle.
        best = max(
            inference.detections,
            key=lambda d: (
                _PRIORITY_RANK[_DISTANCE_TO_PRIORITY[d.distance_level]],
                d.confidence,
            ),
        )

        priority = _DISTANCE_TO_PRIORITY[best.distance_level]
        key = f"{best.distance_level.value}:{best.label}"

        if best.distance_level == DistanceLevel.CRITICAL:
            message = f"Careful! {best.label} very close ahead."
        elif best.distance_level == DistanceLevel.WARNING:
            message = f"Warning: {best.label} ahead."
        else:
            message = f"{best.label} nearby."

        decision = Decision(key=key, message=message, priority=priority)

        # Simple anti-spam: suppress repeating the same alert within cooldown.
        now = time.monotonic()
        if (
            decision.key == self._last_spoken_key
            and (now - self._last_spoken_at) < self.cooldown_seconds
        ):
            logger.debug("Suppressing repeated decision: %s", decision.key)
            return Decision.silent()

        self._last_spoken_key = decision.key
        self._last_spoken_at = now
        return decision
