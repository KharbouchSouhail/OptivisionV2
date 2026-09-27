"""Real scene text detection and recognition using PaddleOCR."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import cv2
import numpy as np
from paddleocr import PaddleOCR

from optivisionhackathon.server.schemas import TextDetection

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class OCREngine:
    """Scene text detector and recognizer using PaddleOCR."""

    def __init__(
        self,
        min_confidence: float | None = None,
        max_regions: int = 10,
    ) -> None:
        self.min_confidence = (
            min_confidence
            if min_confidence is not None
            else float(
                os.getenv(
                    "VISIONNAIRE_OCR_CONF",
                    "0.40",
                )
            )
        )

        self.max_regions = max_regions

        logger.info("Initializing PaddleOCR...")

        self._ocr = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )

        logger.info("PaddleOCR ready")

    def read(
        self,
        frame: NDArray[np.uint8],
    ) -> list[TextDetection]:
        """Detect and recognize text in a BGR OpenCV frame."""

        if frame is None or frame.size == 0:
            return []

        try:
            # PaddleOCR expects an image array.
            result = self._ocr.predict(frame)

        except Exception:
            logger.exception("OCR inference failed")
            return []

        detections: list[TextDetection] = []

        for page_result in result:
            try:
                texts = page_result.get("rec_texts", [])
                scores = page_result.get("rec_scores", [])
                boxes = page_result.get("rec_boxes", [])

            except AttributeError:
                logger.warning(
                    "Unexpected PaddleOCR result format"
                )
                continue

            if texts is None:
                continue

            if scores is None:
                scores = []

            if boxes is None:
                boxes = []

            for text, score, box in zip(
                texts,
                scores,
                boxes,
            ):
                text = str(text).strip()

                confidence = float(score)

                if not text:
                    continue

                if confidence < self.min_confidence:
                    continue

                try:
                    coordinates = np.asarray(
                        box,
                        dtype=np.float32,
                    ).reshape(-1, 2)

                    if coordinates.shape[0] < 4:
                        continue

                    x1 = float(
                        np.min(coordinates[:, 0])
                    )
                    y1 = float(
                        np.min(coordinates[:, 1])
                    )
                    x2 = float(
                        np.max(coordinates[:, 0])
                    )
                    y2 = float(
                        np.max(coordinates[:, 1])
                    )

                except Exception:
                    logger.warning(
                        "Could not parse OCR bounding box"
                    )
                    continue

                detections.append(
                    TextDetection(
                        text=text,
                        confidence=confidence,
                        bbox=(
                            x1,
                            y1,
                            x2,
                            y2,
                        ),
                    )
                )

        # Keep the highest-confidence results.
        detections.sort(
            key=lambda detection: detection.confidence,
            reverse=True,
        )

        return detections[: self.max_regions]