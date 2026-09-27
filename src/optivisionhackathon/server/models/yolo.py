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


# COCO classes that are useful for visual assistance/navigation.
OBSTACLE_CLASSES: frozenset[str] = frozenset(
    {
        # People
        "person",

        # Vehicles
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",

        # Animals
        "dog",
        "cat",

        # Furniture / indoor obstacles
        "chair",
        "bench",
        "couch",
        "bed",
        "dining table",

        # Objects someone may need to avoid
        "backpack",
        "suitcase",
        "skateboard",
        "surfboard",

        # Important environmental objects
        "potted plant",
        "traffic light",
        "stop sign",
        "fire hydrant",
        "parking meter",
    }
)


DEFAULT_MODEL_NAME = "yolov8n.pt"


def resolve_device() -> str:
    """Return CUDA when available, otherwise CPU."""

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def log_device_info() -> str:
    """Log CUDA/GPU information and return the selected device."""

    cuda_available = torch.cuda.is_available()

    logger.info(
        "CUDA available: %s",
        cuda_available,
    )

    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(
                "GPU: %s",
                gpu_name,
            )

            logger.info(
                "CUDA version: %s",
                torch.version.cuda,
            )

        except Exception:
            logger.exception(
                "Failed to read GPU information"
            )
    else:
        logger.info("GPU: none")

    device = resolve_device()

    logger.info(
        "Inference device: %s",
        device,
    )

    return device


class YOLODetector:
    """YOLOv8 detector loaded once and reused across frames."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        conf_threshold: float | None = None,
        critical_height_ratio: float | None = None,
        warning_height_ratio: float | None = None,
    ) -> None:

        self.model_name = model_name or os.getenv(
            "VISIONNAIRE_YOLO_MODEL",
            DEFAULT_MODEL_NAME,
        )

        self.device = (
            device
            if device is not None
            else resolve_device()
        )

        self.conf_threshold = (
            conf_threshold
            if conf_threshold is not None
            else float(
                os.getenv(
                    "VISIONNAIRE_YOLO_CONF",
                    "0.45",
                )
            )
        )

        # Object bounding-box height relative to frame height.
        #
        # Example:
        #
        # 0.10 -> INFO
        # 0.30 -> WARNING
        # 0.50 -> CRITICAL
        #
        self.critical_height_ratio = (
            critical_height_ratio
            if critical_height_ratio is not None
            else float(
                os.getenv(
                    "VISIONNAIRE_CRITICAL_HEIGHT_RATIO",
                    "0.45",
                )
            )
        )

        self.warning_height_ratio = (
            warning_height_ratio
            if warning_height_ratio is not None
            else float(
                os.getenv(
                    "VISIONNAIRE_WARNING_HEIGHT_RATIO",
                    "0.25",
                )
            )
        )

        if not 0.0 < self.conf_threshold <= 1.0:
            raise ValueError(
                "YOLO confidence threshold must be "
                "between 0 and 1."
            )

        if not 0.0 < self.warning_height_ratio < 1.0:
            raise ValueError(
                "Warning height ratio must be "
                "between 0 and 1."
            )

        if not 0.0 < self.critical_height_ratio <= 1.0:
            raise ValueError(
                "Critical height ratio must be "
                "between 0 and 1."
            )

        if (
            self.critical_height_ratio
            <= self.warning_height_ratio
        ):
            raise ValueError(
                "Critical height ratio must be greater "
                "than warning height ratio."
            )

        logger.info(
            "Loading YOLO model: %s",
            self.model_name,
        )

        try:
            self._model = YOLO(
                self.model_name
            )

            # Put the model on the selected device.
            self._model.to(self.device)

        except Exception:
            logger.exception(
                "Failed to load YOLO model"
            )
            raise

        logger.info(
            "YOLO model ready on %s",
            self.device,
        )

        logger.info(
            "YOLO confidence threshold: %.2f",
            self.conf_threshold,
        )

    # ------------------------------------------------------------------
    # Distance classification
    # ------------------------------------------------------------------

    def _classify_distance(
        self,
        bbox_height: float,
        frame_height: int,
    ) -> DistanceLevel:
        """
        Estimate coarse proximity from bounding-box height.

        This is NOT a metric distance measurement.
        It only classifies objects as INFO/WARNING/CRITICAL.
        """

        if frame_height <= 0:
            return DistanceLevel.INFO

        if bbox_height <= 0:
            return DistanceLevel.INFO

        ratio = (
            bbox_height
            / float(frame_height)
        )

        if ratio >= self.critical_height_ratio:
            return DistanceLevel.CRITICAL

        if ratio >= self.warning_height_ratio:
            return DistanceLevel.WARNING

        return DistanceLevel.INFO

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(
        self,
        frame: NDArray[np.uint8],
    ) -> list[Detection]:
        """
        Run YOLO on a BGR OpenCV frame.

        Returns only classes listed in OBSTACLE_CLASSES.
        """

        if frame is None or frame.size == 0:
            return []

        if frame.ndim != 3:
            logger.warning(
                "Invalid frame dimensions: %s",
                frame.shape,
            )
            return []

        frame_height = int(
            frame.shape[0]
        )

        frame_width = int(
            frame.shape[1]
        )

        if frame_height <= 0 or frame_width <= 0:
            return []

        try:
            results = self._model.predict(
                source=frame,
                conf=self.conf_threshold,
                verbose=False,
            )

        except Exception:
            logger.exception(
                "YOLO inference failed"
            )
            return []

        if not results:
            return []

        result = results[0]

        if result is None:
            return []

        boxes = result.boxes

        if boxes is None:
            return []

        names = result.names

        if names is None:
            logger.warning(
                "YOLO result contains no class names"
            )
            return []

        detections: list[Detection] = []

        for box in boxes:

            try:
                cls_id = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                coordinates = (
                    box.xyxy[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )

            except Exception:
                logger.warning(
                    "Failed to parse YOLO detection",
                    exc_info=True,
                )
                continue

            if len(coordinates) != 4:
                continue

            label = str(
                names.get(
                    cls_id,
                    cls_id,
                )
            )

            # Ignore irrelevant COCO classes.
            if label not in OBSTACLE_CLASSES:
                continue

            # Clamp coordinates to frame boundaries.
            x1 = float(
                np.clip(
                    coordinates[0],
                    0,
                    frame_width,
                )
            )

            y1 = float(
                np.clip(
                    coordinates[1],
                    0,
                    frame_height,
                )
            )

            x2 = float(
                np.clip(
                    coordinates[2],
                    0,
                    frame_width,
                )
            )

            y2 = float(
                np.clip(
                    coordinates[3],
                    0,
                    frame_height,
                )
            )

            if x2 <= x1 or y2 <= y1:
                continue

            bbox_height = y2 - y1

            distance_level = (
                self._classify_distance(
                    bbox_height,
                    frame_height,
                )
            )

            detections.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    bbox=(
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                    distance_level=distance_level,
                )
            )

        # Most confident detections first.
        detections.sort(
            key=lambda detection: detection.confidence,
            reverse=True,
        )

        return detections

    # ------------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------------

    def warmup(
        self,
        width: int = 640,
        height: int = 480,
    ) -> None:
        """
        Run one dummy inference to initialize CUDA/model kernels.

        This prevents the first real camera frame from being
        significantly slower.
        """

        if width <= 0 or height <= 0:
            raise ValueError(
                "Warmup dimensions must be positive."
            )

        logger.info(
            "Warming up YOLO on %s...",
            self.device,
        )

        dummy_frame = np.zeros(
            (
                height,
                width,
                3,
            ),
            dtype=np.uint8,
        )

        try:
            self._model.predict(
                source=dummy_frame,
                conf=self.conf_threshold,
                verbose=False,
            )

        except Exception:
            logger.exception(
                "YOLO warmup failed"
            )
            raise

        logger.info(
            "YOLO warmup complete"
        )