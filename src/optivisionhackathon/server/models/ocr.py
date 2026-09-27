"""Aggressive scene-text detection and recognition using EasyOCR.

Designed for Visionnaire:
- Detects both large and small text.
- Uses two scales for better small-text recall.
- GPU accelerated on the NVIDIA T4.
- Deduplicates overlapping detections from multiple passes.
- Sorts text naturally from top-to-bottom, left-to-right.
- Preserves punctuation, numbers, and short text.
- Returns structured TextDetection objects.
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING

import cv2
import easyocr
import numpy as np
import torch

from optivisionhackathon.server.schemas import TextDetection

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class OCREngine:
    def __init__(
        self,
        min_confidence: float | None = None,
        max_regions: int = 50,
    ) -> None:
        self.min_confidence = (
            min_confidence
            if min_confidence is not None
            else float(os.getenv("VISIONNAIRE_OCR_CONF", "0.20"))
        )

        self.max_regions = int(
            os.getenv("VISIONNAIRE_OCR_MAX_REGIONS", str(max_regions))
        )

        self.languages = [
            language.strip()
            for language in os.getenv(
                "VISIONNAIRE_OCR_LANGUAGES",
                "en",
            ).split(",")
            if language.strip()
        ]

        self.gpu_enabled = (
            os.getenv("VISIONNAIRE_OCR_GPU", "true").lower()
            in {"1", "true", "yes", "on"}
            and torch.cuda.is_available()
        )

        # EasyOCR can be expensive. Two passes provide better recall
        # while remaining reasonable on the T4.
        self.scales = (1.0, 1.5)

        # Don't upscale very large frames indefinitely.
        self.max_dimension = int(
            os.getenv("VISIONNAIRE_OCR_MAX_DIMENSION", "1600")
        )

        self._reader: easyocr.Reader | None = None

        self._initialize()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        logger.info(
            "Initializing EasyOCR | languages=%s | GPU=%s",
            self.languages,
            self.gpu_enabled,
        )

        try:
            self._reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu_enabled,
                verbose=True,
            )

            logger.info(
                "EasyOCR ready | GPU=%s | scales=%s",
                self.gpu_enabled,
                self.scales,
            )

        except Exception:
            logger.exception("Failed to initialize EasyOCR")
            self._reader = None

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(
        frame: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """Improve small/low-contrast text without destroying characters."""

        if frame is None or frame.size == 0:
            return frame

        # Mild denoising.
        denoised = cv2.GaussianBlur(frame, (3, 3), 0)

        # Local contrast enhancement.
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        l_channel = clahe.apply(l_channel)

        enhanced = cv2.merge(
            (
                l_channel,
                a_channel,
                b_channel,
            )
        )

        enhanced = cv2.cvtColor(
            enhanced,
            cv2.COLOR_LAB2BGR,
        )

        # Very mild sharpening.
        kernel = np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ],
            dtype=np.float32,
        )

        return cv2.filter2D(enhanced, -1, kernel)

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    def _resize_for_scale(
        self,
        frame: NDArray[np.uint8],
        scale: float,
    ) -> tuple[NDArray[np.uint8], float]:
        """Resize frame while respecting max_dimension.

        Returns:
            resized image,
            actual scale relative to original.
        """

        height, width = frame.shape[:2]

        requested_width = int(width * scale)
        requested_height = int(height * scale)

        actual_scale = scale

        largest_dimension = max(
            requested_width,
            requested_height,
        )

        if largest_dimension > self.max_dimension:
            limiting_scale = self.max_dimension / float(
                max(width, height)
            )

            requested_width = int(width * limiting_scale)
            requested_height = int(height * limiting_scale)

            actual_scale = limiting_scale

        if requested_width == width and requested_height == height:
            return frame, 1.0

        resized = cv2.resize(
            frame,
            (requested_width, requested_height),
            interpolation=cv2.INTER_CUBIC,
        )

        return resized, actual_scale

    # ------------------------------------------------------------------
    # Text normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize whitespace while preserving useful punctuation."""

        text = str(text)

        # Normalize unusual whitespace.
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    @staticmethod
    def _text_key(text: str) -> str:
        """Create a comparison key for duplicate detection."""

        text = text.lower().strip()

        # Remove spaces and punctuation for robust duplicate comparison.
        text = re.sub(r"[^a-z0-9\u00C0-\u024F\u0600-\u06FF]+", "", text)

        return text

    # ------------------------------------------------------------------
    # Bounding boxes
    # ------------------------------------------------------------------

    @staticmethod
    def _box_to_xyxy(
        box: object,
    ) -> tuple[float, float, float, float] | None:
        try:
            points = np.asarray(
                box,
                dtype=np.float32,
            ).reshape(-1, 2)

            if points.shape[0] < 4:
                return None

            x1 = float(np.min(points[:, 0]))
            y1 = float(np.min(points[:, 1]))
            x2 = float(np.max(points[:, 0]))
            y2 = float(np.max(points[:, 1]))

            if x2 <= x1 or y2 <= y1:
                return None

            return x1, y1, x2, y2

        except Exception:
            return None

    @staticmethod
    def _box_area(
        box: tuple[float, float, float, float],
    ) -> float:
        x1, y1, x2, y2 = box

        return max(0.0, x2 - x1) * max(
            0.0,
            y2 - y1,
        )

    @staticmethod
    def _intersection_over_union(
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
    ) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        intersection_width = max(
            0.0,
            ix2 - ix1,
        )

        intersection_height = max(
            0.0,
            iy2 - iy1,
        )

        intersection = (
            intersection_width
            * intersection_height
        )

        if intersection <= 0:
            return 0.0

        union = (
            OCREngine._box_area(a)
            + OCREngine._box_area(b)
            - intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------

    @classmethod
    def _is_duplicate(
        cls,
        candidate: TextDetection,
        existing: TextDetection,
    ) -> bool:
        """Determine whether two OCR results represent the same text."""

        candidate_key = cls._text_key(candidate.text)
        existing_key = cls._text_key(existing.text)

        if not candidate_key or not existing_key:
            return False

        # Exact/near-exact textual match.
        if candidate_key == existing_key:
            return True

        # Bounding-box overlap.
        iou = cls._intersection_over_union(
            candidate.bbox,
            existing.bbox,
        )

        if iou >= 0.50:
            return True

        # Handle OCR differences such as:
        # "EXIT" vs "EXlT"
        # "12.50" vs "12 50"
        shorter = min(
            len(candidate_key),
            len(existing_key),
        )

        longer = max(
            len(candidate_key),
            len(existing_key),
        )

        if shorter >= 4 and longer > 0:
            if shorter / longer >= 0.75:
                if iou >= 0.25:
                    return True

        return False

    @classmethod
    def _deduplicate(
        cls,
        detections: list[TextDetection],
    ) -> list[TextDetection]:
        """Merge duplicate detections from multiple OCR scales."""

        # Highest confidence first.
        detections = sorted(
            detections,
            key=lambda detection: detection.confidence,
            reverse=True,
        )

        unique: list[TextDetection] = []

        for candidate in detections:
            duplicate = False

            for existing in unique:
                if cls._is_duplicate(
                    candidate,
                    existing,
                ):
                    duplicate = True
                    break

            if not duplicate:
                unique.append(candidate)

        return unique

    # ------------------------------------------------------------------
    # Natural reading order
    # ------------------------------------------------------------------

    @staticmethod
    def _sort_reading_order(
        detections: list[TextDetection],
    ) -> list[TextDetection]:
        """Sort text approximately like a human reads it.

        Text is grouped into horizontal lines using bounding-box centers.
        Within each line, text is ordered left-to-right.
        """

        if len(detections) <= 1:
            return detections

        items = []

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox

            height = max(
                1.0,
                y2 - y1,
            )

            center_y = (y1 + y2) / 2.0

            items.append(
                {
                    "detection": detection,
                    "x": x1,
                    "y": y1,
                    "center_y": center_y,
                    "height": height,
                }
            )

        # Largest text first is not what we want here.
        # We need spatial reading order.
        items.sort(
            key=lambda item: item["center_y"]
        )

        lines: list[dict] = []

        for item in items:
            assigned = False

            for line in lines:
                line_center = line["center_y"]
                line_height = line["height"]

                # Dynamic tolerance based on text height.
                tolerance = max(
                    8.0,
                    min(
                        line_height,
                        item["height"],
                    )
                    * 0.65,
                )

                if abs(
                    item["center_y"] - line_center
                ) <= tolerance:
                    line["items"].append(item)

                    # Update line center.
                    line["center_y"] = float(
                        np.mean(
                            [
                                x["center_y"]
                                for x in line["items"]
                            ]
                        )
                    )

                    line["height"] = float(
                        np.mean(
                            [
                                x["height"]
                                for x in line["items"]
                            ]
                        )
                    )

                    assigned = True
                    break

            if not assigned:
                lines.append(
                    {
                        "center_y": item["center_y"],
                        "height": item["height"],
                        "items": [item],
                    }
                )

        # Top-to-bottom.
        lines.sort(
            key=lambda line: line["center_y"]
        )

        ordered: list[TextDetection] = []

        for line in lines:
            line["items"].sort(
                key=lambda item: item["x"]
            )

            ordered.extend(
                item["detection"]
                for item in line["items"]
            )

        return ordered

    # ------------------------------------------------------------------
    # Single OCR pass
    # ------------------------------------------------------------------

    def _read_scale(
        self,
        frame: NDArray[np.uint8],
        scale: float,
    ) -> list[TextDetection]:
        """Run EasyOCR at one image scale."""

        if self._reader is None:
            return []

        processed, actual_scale = self._resize_for_scale(
            frame,
            scale,
        )

        processed = self._preprocess(processed)

        try:
            results = self._reader.readtext(
                processed,

                # Return individual text regions.
                detail=1,

                # Don't combine unrelated text into paragraphs.
                paragraph=False,

                # Aggressive detection settings.
                width_ths=0.5,
                link_threshold=0.20,
                low_text=0.15,
                text_threshold=0.25,

                # Help with small text.
                mag_ratio=1.0,

                # Slightly larger recognition crop.
                add_margin=0.10,

                # Allow rotated text.
                rotation_info=[90, 180, 270],
            )

        except Exception:
            logger.exception(
                "EasyOCR inference failed at scale %.2f",
                actual_scale,
            )
            return []

        detections: list[TextDetection] = []

        for result in results:
            try:
                if not isinstance(
                    result,
                    (list, tuple),
                ):
                    continue

                if len(result) < 3:
                    continue

                raw_box = result[0]
                raw_text = result[1]
                raw_confidence = result[2]

                text = self._normalize_text(
                    str(raw_text)
                )

                confidence = float(
                    raw_confidence
                )

                if not text:
                    continue

                if confidence < self.min_confidence:
                    continue

                coordinates = self._box_to_xyxy(
                    raw_box
                )

                if coordinates is None:
                    continue

                x1, y1, x2, y2 = coordinates

                # Convert coordinates from resized image
                # back to original camera-frame coordinates.
                if actual_scale != 1.0:
                    x1 /= actual_scale
                    y1 /= actual_scale
                    x2 /= actual_scale
                    y2 /= actual_scale

                detections.append(
                    TextDetection(
                        text=text,
                        confidence=confidence,
                        bbox=(
                            float(x1),
                            float(y1),
                            float(x2),
                            float(y2),
                        ),
                    )
                )

            except Exception:
                logger.exception(
                    "Failed to parse EasyOCR result"
                )

        return detections

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(
        self,
        frame: NDArray[np.uint8],
    ) -> list[TextDetection]:
        """Detect and recognize as much visible text as possible."""

        if frame is None or frame.size == 0:
            return []

        if self._reader is None:
            logger.warning(
                "OCR reader is unavailable"
            )
            return []

        all_detections: list[TextDetection] = []

        # Run multiple scales.
        for scale in self.scales:
            detections = self._read_scale(
                frame,
                scale,
            )

            all_detections.extend(
                detections
            )

        if not all_detections:
            return []

        # Remove repeated detections caused by
        # multi-scale processing.
        unique = self._deduplicate(
            all_detections
        )

        # Sort spatially.
        ordered = self._sort_reading_order(
            unique
        )

        # Hard limit to prevent enormous OCR
        # results from overwhelming the fusion/TTS layer.
        return ordered[: self.max_regions]