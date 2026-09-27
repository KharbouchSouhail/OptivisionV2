"""YOLOv8n obstacle detector for Visionnaire."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import torch
from ultralytics import YOLO

from optivisionhackathon.server.schemas import Detection, DistanceLevel

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# COCO obstacle classes relevant to accessibility navigation.
OBSTACLE_CLASSES: frozenset[str] = frozenset(
    {
        "person",
        "car",
        "bicycle",
        "motorcycle",
        "bus",
        "truck",
        "chair",
        "bench",
        "dog",
        "cat",
    }
)

DEFAULT_MODEL_NAME = "yolov8n.pt"


def resolve_device() -> str:
    """Return 'cuda' when a GPU is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def log_device_info() -> str:
    """Log CUDA / GPU info and return the selected inference device."""
    cuda_available = torch.cuda.is_available()
    logger.info("CUDA available: %s", cuda_available)

    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("GPU: %s", gpu_name)
    else:
        logger.info("GPU: none")

    device = resolve_device()
    logger.info("Inference device: %s", device)
    return device


class YOLODetector:
    """YOLOv8n detector loaded once and reused across frames."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        conf_threshold: float | None = None,
        critical_height_ratio: float | None = None,
        warning_height_ratio: float | None = None,
    ) -> None:
        self.model_name = model_name or os.getenv(
            "VISIONNAIRE_YOLO_MODEL", DEFAULT_MODEL_NAME
        )
        self.device = device if device is not None else resolve_device()
        self.conf_threshold = (
            conf_threshold
            if conf_threshold is not None
            else float(os.getenv("VISIONNAIRE_YOLO_CONF", "0.45"))
        )
        # BBox height / frame height thresholds for coarse distance.
        self.critical_height_ratio = (
            critical_height_ratio
            if critical_height_ratio is not None
            else float(os.getenv("VISIONNAIRE_CRITICAL_HEIGHT_RATIO", "0.45"))
        )
        self.warning_height_ratio = (
            warning_height_ratio
            if warning_height_ratio is not None
            else float(os.getenv("VISIONNAIRE_WARNING_HEIGHT_RATIO", "0.25"))
        )

        logger.info("Loading YOLOv8n... (%s)", self.model_name)
        self._model = YOLO(self.model_name)
        # Move / warm the model onto the selected device.
        self._model.to(self.device)
        logger.info("YOLO model ready on %s", self.device)

    def _classify_distance(
        self, bbox_height: float, frame_height: int
    ) -> DistanceLevel:
        if frame_height <= 0:
            return DistanceLevel.INFO
        ratio = bbox_height / float(frame_height)
        if ratio >= self.critical_height_ratio:
            return DistanceLevel.CRITICAL
        if ratio >= self.warning_height_ratio:
            return DistanceLevel.WARNING
        return DistanceLevel.INFO

    def detect(self, frame: NDArray[np.uint8]) -> list[Detection]:
        """Run YOLO on a BGR OpenCV frame and return obstacle detections."""
        if frame is None or frame.size == 0:
            return []

        frame_height = int(frame.shape[0])
        results = self._model.predict(
            source=frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        detections: list[Detection] = []
        if not results:
            return detections

        result = results[0]
        names = result.names or {}
        boxes = result.boxes
        if boxes is None:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0].item())
            label = str(names.get(cls_id, cls_id))
            if label not in OBSTACLE_CLASSES:
                continue

            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            bbox_height = max(0.0, y2 - y1)
            distance_level = self._classify_distance(bbox_height, frame_height)

            detections.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    distance_level=distance_level,
                )
            )

        return detections
