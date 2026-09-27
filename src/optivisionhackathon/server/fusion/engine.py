"""Fusion engine for YOLO, depth, face recognition, and OCR."""

from __future__ import annotations

import logging
import re
import time

from optivisionhackathon.server.fusion.cooldown import CooldownManager
from optivisionhackathon.server.schemas import (
    Decision,
    Detection,
    DistanceLevel,
    FaceDetection,
    Priority,
    TextDetection,
)

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


# Words that should immediately be treated as safety-relevant.
_URGENT_TEXT = frozenset(
    {
        "stop",
        "danger",
        "caution",
        "warning",
        "hazard",
        "emergency",
        "do not enter",
        "no entry",
        "wrong way",
        "keep out",
        "fire",
        "exit",
    }
)


def _normalize_text(text: str) -> str:
    """Normalize OCR text for comparison."""

    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _text_key(text: str) -> str:
    """Create a normalized key for duplicate OCR text."""

    return re.sub(
        r"[^a-z0-9\u00C0-\u024F\u0600-\u06FF]+",
        "",
        text.lower(),
    )


def _is_urgent_text(text: str) -> bool:
    """Check whether OCR text contains a safety-critical phrase."""

    normalized = _normalize_text(text).lower()

    if not normalized:
        return False

    for keyword in _URGENT_TEXT:
        if keyword in normalized:
            return True

    return False


def _deduplicate_texts(
    texts: list[TextDetection],
) -> list[TextDetection]:
    """Remove repeated OCR results while preserving reading order."""

    result: list[TextDetection] = []
    seen: set[str] = set()

    for text_detection in texts:
        text = _normalize_text(text_detection.text)

        if not text:
            continue

        key = _text_key(text)

        if not key:
            continue

        if key in seen:
            continue

        seen.add(key)
        result.append(text_detection)

    return result


def _sort_texts(
    texts: list[TextDetection],
) -> list[TextDetection]:
    """Sort OCR text top-to-bottom and left-to-right."""

    return sorted(
        texts,
        key=lambda text_detection: (
            text_detection.bbox[1],
            text_detection.bbox[0],
        ),
    )


def _format_text_for_speech(text: str) -> str:
    """Clean OCR text before sending it to TTS."""

    text = _normalize_text(text)

    # Avoid producing awkward repeated punctuation.
    text = re.sub(r"[.]{2,}", ".", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    return text


def _build_ocr_sentence(
    texts: list[TextDetection],
) -> str:
    """Build one natural spoken sentence containing detected text."""

    texts = _deduplicate_texts(texts)
    texts = _sort_texts(texts)

    if not texts:
        return ""

    readable: list[str] = []

    for detection in texts:
        text = _format_text_for_speech(
            detection.text
        )

        if not text:
            continue

        readable.append(text)

    if not readable:
        return ""

    if len(readable) == 1:
        return f"Text reads {readable[0]}."

    # Keep all detected text instead of only the first result.
    return "Text reads " + ", ".join(readable) + "."


def build_speech_message(
    detections: list[Detection],
    faces: list[FaceDetection] | None = None,
    texts: list[TextDetection] | None = None,
) -> str:
    """Generate a coherent spoken assistance message."""

    sentences: list[str] = []

    # ---------------------------------------------------------------
    # 1. Known faces
    # ---------------------------------------------------------------

    known_faces = [
        face
        for face in (faces or [])
        if face.name
        and face.name != "Unknown"
    ]

    if known_faces:
        names = sorted(
            {
                face.name
                for face in known_faces
                if face.name
            }
        )

        if len(names) == 1:
            sentences.append(
                f"{names[0]} is ahead."
            )
        elif len(names) == 2:
            sentences.append(
                f"{names[0]} and {names[1]} are ahead."
            )
        else:
            sentences.append(
                f"{', '.join(names[:-1])}, "
                f"and {names[-1]} are ahead."
            )

    # ---------------------------------------------------------------
    # 2. Obstacle detection
    # ---------------------------------------------------------------

    if detections:
        ranked = sorted(
            detections,
            key=lambda detection: (
                _PRIORITY_RANK[
                    _DISTANCE_TO_PRIORITY[
                        detection.distance_level
                    ]
                ],
                detection.confidence,
            ),
            reverse=True,
        )

        top = ranked[0]

        # Critical obstacle.
        if (
            top.distance_level
            == DistanceLevel.CRITICAL
        ):
            label = top.label.capitalize()

            if top.distance_meters > 0.0:
                sentences.append(
                    f"Warning! {label} "
                    f"{top.distance_meters:.1f} "
                    f"meters ahead."
                )
            else:
                sentences.append(
                    f"Warning! {label} very close ahead."
                )

        else:
            labels: list[str] = []
            seen_labels: set[str] = set()

            for detection in ranked:
                label = detection.label

                if label in seen_labels:
                    continue

                seen_labels.add(label)
                labels.append(label)

            if labels:
                if len(labels) == 1:
                    sentences.append(
                        f"{labels[0].capitalize()} ahead."
                    )

                elif len(labels) == 2:
                    sentences.append(
                        f"There is a "
                        f"{labels[0]} and a "
                        f"{labels[1]} ahead."
                    )

                else:
                    leading = ", ".join(
                        labels[:-1]
                    )

                    sentences.append(
                        f"There is a "
                        f"{leading}, and a "
                        f"{labels[-1]} ahead."
                    )

    # ---------------------------------------------------------------
    # 3. OCR
    # ---------------------------------------------------------------

    if texts:
        ordered_texts = _sort_texts(
            _deduplicate_texts(texts)
        )

        if ordered_texts:
            # Put urgent signs first.
            urgent = [
                text
                for text in ordered_texts
                if _is_urgent_text(text.text)
            ]

            normal = [
                text
                for text in ordered_texts
                if not _is_urgent_text(text.text)
            ]

            ordered_texts = urgent + normal

            # Critical text gets its own warning.
            if urgent:
                urgent_text = _format_text_for_speech(
                    urgent[0].text
                )

                sentences.append(
                    f"Warning. Sign reads {urgent_text}."
                )

                # Speak remaining OCR below.
                remaining = ordered_texts[1:]

                if remaining:
                    ocr_sentence = _build_ocr_sentence(
                        remaining
                    )

                    if ocr_sentence:
                        sentences.append(
                            ocr_sentence
                        )

            else:
                ocr_sentence = _build_ocr_sentence(
                    ordered_texts
                )

                if ocr_sentence:
                    sentences.append(
                        ocr_sentence
                    )

    return " ".join(
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ).strip()


def build_fusion_key(
    detections: list[Detection],
    faces: list[FaceDetection] | None = None,
    texts: list[TextDetection] | None = None,
) -> str:
    """Build a stable scene key for cooldown tracking."""

    dets = detections or []

    face_list = [
        face
        for face in (faces or [])
        if face.name
        and face.name != "Unknown"
    ]

    text_list = _deduplicate_texts(
        texts or []
    )

    if (
        not dets
        and not face_list
        and not text_list
    ):
        return "silent"

    parts: list[str] = []

    # ---------------------------------------------------------------
    # Objects
    # ---------------------------------------------------------------

    if dets:
        labels = sorted(
            {
                detection.label
                for detection in dets
            }
        )

        top = max(
            dets,
            key=lambda detection: (
                _PRIORITY_RANK[
                    _DISTANCE_TO_PRIORITY[
                        detection.distance_level
                    ]
                ],
                detection.confidence,
            ),
        )

        parts.append(
            f"{top.distance_level.value}:"
            f"{','.join(labels)}"
        )

    # ---------------------------------------------------------------
    # Faces
    # ---------------------------------------------------------------

    if face_list:
        names = sorted(
            {
                face.name
                for face in face_list
            }
        )

        parts.append(
            f"face:{','.join(names)}"
        )

    # ---------------------------------------------------------------
    # OCR
    # ---------------------------------------------------------------

    if text_list:
        texts_normalized = sorted(
            {
                _text_key(
                    _normalize_text(
                        detection.text
                    )
                )
                for detection in text_list
                if detection.text
            }
        )

        if texts_normalized:
            parts.append(
                f"text:{','.join(texts_normalized)}"
            )

    return "|".join(parts)


class FusionEngine:
    """Fuse visual modalities and manage spoken notifications."""

    def __init__(
        self,
        cooldown_seconds: float = 2.0,
        cooldown_manager: CooldownManager | None = None,
    ) -> None:
        self.cooldown = (
            cooldown_manager
            or CooldownManager(
                cooldown_seconds=cooldown_seconds
            )
        )

    def determine_priority(
        self,
        detections: list[Detection],
        faces: list[FaceDetection] | None = None,
        texts: list[TextDetection] | None = None,
    ) -> Priority:
        """Calculate overall scene priority."""

        dets = detections or []
        face_list = faces or []
        text_list = texts or []

        # Physical obstacles always take precedence.
        if any(
            detection.distance_level
            == DistanceLevel.CRITICAL
            for detection in dets
        ):
            return Priority.CRITICAL

        if any(
            detection.distance_level
            == DistanceLevel.WARNING
            for detection in dets
        ):
            return Priority.WARNING

        # Safety-critical OCR.
        if any(
            _is_urgent_text(text.text)
            for text in text_list
        ):
            return Priority.WARNING

        # Known person.
        if any(
            face.name
            and face.name != "Unknown"
            for face in face_list
        ):
            return Priority.INFO

        if any(
            detection.distance_level
            == DistanceLevel.INFO
            for detection in dets
        ):
            return Priority.INFO

        if dets or text_list or face_list:
            return Priority.INFO

        return Priority.NONE

    def fuse(
        self,
        detections: list[Detection] | None = None,
        faces: list[FaceDetection] | None = None,
        texts: list[TextDetection] | None = None,
    ) -> Decision:
        """Combine all visual detections into a Decision."""

        dets = list(
            detections or []
        )

        face_list = list(
            faces or []
        )

        text_list = list(
            texts or []
        )

        now_wall = time.time()

        if (
            not dets
            and not face_list
            and not text_list
        ):
            return Decision.silent(
                timestamp=now_wall
            )

        priority = self.determine_priority(
            dets,
            face_list,
            text_list,
        )

        key = build_fusion_key(
            dets,
            face_list,
            text_list,
        )

        raw_message = build_speech_message(
            dets,
            face_list,
            text_list,
        )

        spoken_message = ""

        if (
            raw_message
            and self.cooldown.should_speak(
                key,
                priority,
            )
        ):
            spoken_message = raw_message

            self.cooldown.record_spoken(
                key,
                priority,
            )

        return Decision(
            key=key,
            message=spoken_message,
            priority=priority,
            detections=dets,
            faces=face_list,
            texts=text_list,
            timestamp=now_wall,
        )