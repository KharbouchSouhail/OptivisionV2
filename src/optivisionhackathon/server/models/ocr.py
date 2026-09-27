"""Scene text localization and recognition."""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING

import cv2
import numpy as np

from optivisionhackathon.server.schemas import TextDetection

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

COMMON_SIGNS = [
    "EXIT",
    "STOP",
    "ENTRANCE",
    "CAUTION",
    "DANGER",
    "RESTROOM",
    "OFFICE",
    "ROOM",
    "PULL",
    "PUSH",
    "WALK",
    "BUS",
    "OPEN",
    "CLOSED",
    "INFO",
    "HELP",
    "ELEVATOR",
    "STAIRS",
]


class OCREngine:
    """Scene text detector and recognizer using morphological gradients and glyph correlation."""

    def __init__(
        self,
        min_confidence: float | None = None,
        max_regions: int = 5,
    ) -> None:
        self.min_confidence = (
            min_confidence
            if min_confidence is not None
            else float(os.getenv("VISIONNAIRE_OCR_CONF", "0.40"))
        )
        self.max_regions = max_regions
        self._glyph_cache: dict[str, np.ndarray] = {}
        self._init_glyph_cache()

    def _init_glyph_cache(self) -> None:
        """Pre-render canonical glyph templates for A-Z and 0-9."""
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        for ch in chars:
            img = np.zeros((32, 24), dtype=np.uint8)
            cv2.putText(
                img,
                ch,
                (2, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                255,
                2,
                cv2.LINE_AA,
            )
            self._glyph_cache[ch] = img

    def _match_char(self, char_roi: NDArray[np.uint8]) -> tuple[str, float]:
        """Match a single character ROI against the glyph template cache."""
        if char_roi.shape[0] < 8 or char_roi.shape[1] < 4:
            return "", 0.0

        resized = cv2.resize(char_roi, (24, 32))
        best_char = ""
        best_score = -1.0

        for ch, template in self._glyph_cache.items():
            res = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
            score = float(res[0, 0])
            if score > best_score:
                best_score = score
                best_char = ch

        return best_char, max(0.0, best_score)

    def _recognize_word_roi(self, roi_bgr: NDArray[np.uint8]) -> tuple[str, float]:
        """Recognize text within a localized bounding box ROI."""
        if roi_bgr.size == 0 or roi_bgr.shape[0] < 10 or roi_bgr.shape[1] < 16:
            return "", 0.0

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        # Contrast adjustment & binarization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Check inversion: text is usually foreground (smaller area than background)
        if np.mean(thresh) > 127:
            thresh = cv2.bitwise_not(thresh)

        # First check sign matching by rendering known signs
        h, w = roi_bgr.shape[:2]
        best_sign = ""
        best_sign_score = 0.0

        for sign in COMMON_SIGNS:
            canvas = np.zeros((h, w), dtype=np.uint8)
            scale = min(h / 30.0, w / (len(sign) * 16.0))
            if scale <= 0.2:
                continue
            cv2.putText(
                canvas,
                sign,
                (4, int(h * 0.75)),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                255,
                2,
                cv2.LINE_AA,
            )
            res = cv2.matchTemplate(thresh, canvas, cv2.TM_CCOEFF_NORMED)
            score = float(res[0, 0])
            if score > best_sign_score:
                best_sign_score = score
                best_sign = sign

        if best_sign_score >= 0.55:
            return best_sign, best_sign_score

        # Find character contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        char_boxes: list[tuple[int, int, int, int]] = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            # Filter noise contours
            if ch >= int(h * 0.35) and cw >= 3:
                char_boxes.append((x, y, cw, ch))

        if not char_boxes:
            return "", 0.0

        # Sort characters left to right
        char_boxes.sort(key=lambda b: b[0])
        chars: list[str] = []
        scores: list[float] = []

        for x, y, cw, ch in char_boxes:
            char_crop = thresh[y : y + ch, x : x + cw]
            ch_str, score = self._match_char(char_crop)
            if score >= 0.35:
                chars.append(ch_str)
                scores.append(score)

        word = "".join(chars).strip()
        avg_score = float(np.mean(scores)) if scores else 0.0
        return word, avg_score

    def locate_text_regions(
        self,
        frame: NDArray[np.uint8],
    ) -> list[tuple[float, float, float, float]]:
        """Detect rectangular regions with dense horizontal text stroke characteristics."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        # Morphological gradient to highlight text edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

        # Horizontal dilation to connect letters into word strips
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        connected = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rect_kernel)

        _, thresh = cv2.threshold(connected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions: list[tuple[float, float, float, float]] = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / float(bh) if bh > 0 else 0
            area = bw * bh
            # Text lines generally have aspect > 1.2 and occupy reasonable size
            if 1.2 <= aspect <= 10.0 and 400 <= area <= (h * w * 0.3):
                regions.append((float(x), float(y), float(x + bw), float(y + bh)))

        # Limit to the most prominent regions
        regions.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
        return regions[: self.max_regions]

    def read(self, frame: NDArray[np.uint8]) -> list[TextDetection]:
        """Detect and transcribe text in the frame."""
        if frame is None or frame.size == 0:
            return []

        regions = self.locate_text_regions(frame)
        detections: list[TextDetection] = []

        for x1, y1, x2, y2 in regions:
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            roi = frame[iy1:iy2, ix1:ix2]
            text, conf = self._recognize_word_roi(roi)
            if text and conf >= self.min_confidence and len(text) >= 2:
                detections.append(
                    TextDetection(
                        text=text,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                    )
                )

        return detections
