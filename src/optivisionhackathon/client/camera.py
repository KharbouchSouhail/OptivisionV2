"""OpenCV webcam capture and JPEG encoding."""

from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    """Webcam capture helper. No networking or audio logic."""

    def __init__(
        self,
        camera_index: int | None = None,
        jpeg_quality: int | None = None,
    ) -> None:
        self.camera_index = (
            camera_index
            if camera_index is not None
            else int(os.getenv("VISIONNAIRE_CAMERA_INDEX", "0"))
        )
        self.jpeg_quality = (
            jpeg_quality
            if jpeg_quality is not None
            else int(os.getenv("VISIONNAIRE_JPEG_QUALITY", "80"))
        )
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """Open the configured webcam."""
        if self._cap is not None and self._cap.isOpened():
            return

        logger.info("Opening camera index %s", self.camera_index)
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise RuntimeError(
                f"webcam unavailable: failed to open camera index {self.camera_index}"
            )
        logger.info("Camera opened (index=%s)", self.camera_index)

    def read_frame(self) -> np.ndarray:
        """Capture one raw BGR frame."""
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("camera is not open")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError("webcam read failure: could not capture frame")
        return frame

    def encode_jpeg(self, frame: np.ndarray) -> bytes:
        """JPEG-encode an existing BGR frame (no extra capture)."""
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        ok, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            raise RuntimeError("failed to JPEG-encode frame")
        return buffer.tobytes()

    def read_jpeg(self) -> bytes:
        """Capture one frame and return JPEG-encoded bytes."""
        return self.encode_jpeg(self.read_frame())

    def release(self) -> None:
        """Release the webcam cleanly."""
        if self._cap is not None:
            logger.info("Releasing camera")
            self._cap.release()
            self._cap = None

    def __enter__(self) -> Camera:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.release()
