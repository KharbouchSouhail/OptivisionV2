"""Monocular depth and obstacle distance estimation."""

from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING

import cv2
import numpy as np

from optivisionhackathon.server.schemas import Detection, DistanceLevel

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Typical real-world heights (in meters) for common obstacle classes
REAL_WORLD_HEIGHTS_METERS: dict[str, float] = {
    "person": 1.70,
    "car": 1.50,
    "bus": 3.20,
    "truck": 3.00,
    "bicycle": 1.05,
    "motorcycle": 1.15,
    "chair": 0.85,
    "bench": 0.80,
    "dog": 0.55,
    "cat": 0.30,
}
DEFAULT_OBJECT_HEIGHT_METERS = 1.00


class DepthEstimator:
    """Estimates obstacle distance in meters and computes relative depth maps."""

    def __init__(
        self,
        fov_degrees: float | None = None,
        critical_distance: float | None = None,
        warning_distance: float | None = None,
    ) -> None:
        self.fov_degrees = (
            fov_degrees
            if fov_degrees is not None
            else float(os.getenv("VISIONNAIRE_CAMERA_FOV", "60.0"))
        )
        self.critical_distance = (
            critical_distance
            if critical_distance is not None
            else float(os.getenv("VISIONNAIRE_CRITICAL_DIST", "1.5"))
        )
        self.warning_distance = (
            warning_distance
            if warning_distance is not None
            else float(os.getenv("VISIONNAIRE_WARNING_DIST", "3.0"))
        )
        logger.info(
            "DepthEstimator initialized (FOV=%.1f°, critical<=%.1fm, warning<=%.1fm)",
            self.fov_degrees,
            self.critical_distance,
            self.warning_distance,
        )

    def _focal_length_pixels(self, frame_height: int) -> float:
        """Pinhole camera focal length from vertical FOV."""
        fov_rad = math.radians(self.fov_degrees)
        return frame_height / (2.0 * math.tan(fov_rad / 2.0))

    def estimate_distance(
        self,
        bbox: tuple[float, float, float, float],
        frame_shape: tuple[int, int],
        label: str,
    ) -> float:
        """Estimate metric distance to an object in meters using pinhole projection.

        Args:
            bbox: (x1, y1, x2, y2) in pixels.
            frame_shape: (height, width) of the frame.
            label: COCO class label.

        Returns:
            Estimated distance in meters (clamped between 0.2m and 50.0m).
        """
        frame_h = frame_shape[0]
        if frame_h <= 0:
            return 2.5

        bbox_h = max(1.0, bbox[3] - bbox[1])
        real_h = REAL_WORLD_HEIGHTS_METERS.get(label.lower(), DEFAULT_OBJECT_HEIGHT_METERS)
        focal_length = self._focal_length_pixels(frame_h)

        dist = (focal_length * real_h) / bbox_h
        return float(np.clip(dist, 0.2, 50.0))

    def classify_distance(self, distance_meters: float) -> DistanceLevel:
        """Classify distance into CRITICAL, WARNING, or INFO."""
        if distance_meters <= self.critical_distance:
            return DistanceLevel.CRITICAL
        if distance_meters <= self.warning_distance:
            return DistanceLevel.WARNING
        return DistanceLevel.INFO

    def estimate_depth_map(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Compute relative depth map [0.0 = closest, 1.0 = farthest].

        Combines ground plane perspective with multi-scale image gradients.
        """
        if frame is None or frame.size == 0:
            return np.zeros((1, 1), dtype=np.float32)

        h, w = frame.shape[:2]
        # Ground plane vertical perspective gradient (bottom is closest, top is farthest)
        y_coords = np.linspace(1.0, 0.1, h, dtype=np.float32).reshape(h, 1)
        perspective = np.tile(y_coords, (1, w))

        # Texture and edge gradient cue (sharper/high-frequency cues generally closer)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(grad_x, grad_y)
        max_grad = np.max(gradient)
        if max_grad > 0:
            gradient /= max_grad

        # Combined depth map: normalized between 0.0 (near) and 1.0 (far)
        depth = 0.75 * perspective + 0.25 * (1.0 - gradient)
        return np.clip(depth, 0.0, 1.0).astype(np.float32)

    def enrich_detections(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int],
    ) -> list[Detection]:
        """Enrich YOLO detections with accurate distance in meters and distance levels."""
        enriched: list[Detection] = []
        for det in detections:
            dist = self.estimate_distance(det.bbox, frame_shape, det.label)
            level = self.classify_distance(dist)
            enriched.append(
                Detection(
                    label=det.label,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    distance_level=level,
                    distance_meters=dist,
                )
            )
        return enriched
