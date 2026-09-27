"""Inference pipeline: orchestrates models and produces a Decision."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from optivisionhackathon.server.fusion.engine import (
    FusionEngine,
    build_fusion_key,
    build_speech_message,
)
from optivisionhackathon.server.models.depth import DepthEstimator
from optivisionhackathon.server.models.face import FaceRecognizer
from optivisionhackathon.server.models.ocr import OCREngine
from optivisionhackathon.server.schemas import (
    Decision,
    Detection,
    DistanceLevel,
    FaceDetection,
    InferenceResult,
    Priority,
    TextDetection,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from optivisionhackathon.server.models.yolo import YOLODetector

logger = logging.getLogger(__name__)


def decision_key(detections: list[Detection]) -> str:
    """Stable key for speech cooldown (for backwards compatibility)."""
    return build_fusion_key(detections)


class Pipeline:
    """Full Visionnaire inference pipeline orchestrating YOLO, Depth, Face, OCR, and Fusion."""

    def __init__(
        self,
        detector: Any,
        depth_estimator: Optional[DepthEstimator] = None,
        face_recognizer: Optional[FaceRecognizer] = None,
        ocr_engine: Optional[OCREngine] = None,
        fusion_engine: Optional[FusionEngine] = None,
        cooldown_seconds: float = 2.0,
    ) -> None:
        self.detector = detector
        self.depth_estimator = depth_estimator
        self.face_recognizer = face_recognizer
        self.ocr_engine = ocr_engine
        self.cooldown_seconds = cooldown_seconds
        self.fusion_engine = fusion_engine or FusionEngine(cooldown_seconds=cooldown_seconds)

    def process_frame(self, frame: NDArray[np.uint8]) -> Decision:
        """Run all active vision models on the frame and return a fused Decision."""
        if frame is None or frame.size == 0:
            return Decision.silent()

        # 1. Obstacle detection
        detections = self.detector.detect(frame)

        # 2. Depth / distance enrichment
        if self.depth_estimator is not None and detections:
            detections = self.depth_estimator.enrich_detections(detections, frame.shape[:2])

        # 3. Face recognition (using person bounding boxes if available)
        faces: list[FaceDetection] = []
        if self.face_recognizer is not None:
            person_boxes = [d.bbox for d in detections if d.label == "person"]
            faces = self.face_recognizer.recognize(frame, person_boxes=person_boxes)

        # 4. Scene text / OCR
        texts: list[TextDetection] = []
        if self.ocr_engine is not None:
            texts = self.ocr_engine.read(frame)

        # 5. Multimodal fusion & cooldown
        return self.fusion_engine.fuse(
            detections=detections,
            faces=faces,
            texts=texts,
        )
