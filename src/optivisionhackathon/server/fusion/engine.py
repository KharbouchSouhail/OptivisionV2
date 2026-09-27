"""Fusion engine: combines YOLO obstacles, Depth, Faces, and OCR into a unified Decision."""

from __future__ import annotations

import logging
import time
from typing import Optional

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


def build_speech_message(
    detections: list[Detection],
    faces: list[FaceDetection] | None = None,
    texts: list[TextDetection] | None = None,
) -> str:
    """Generate a coherent, natural spoken assistance utterance."""
    sentences: list[str] = []

    # 1. Known faces
    known_faces = [f for f in (faces or []) if f.name and f.name != "Unknown"]
    if known_faces:
        names = sorted({f.name for f in known_faces})
        if len(names) == 1:
            sentences.append(f"{names[0]} is ahead.")
        else:
            sentences.append(f"{' and '.join(names)} are ahead.")

    # 2. Obstacle detection message
    if detections:
        ranked = sorted(
            detections,
            key=lambda d: (
                _PRIORITY_RANK[_DISTANCE_TO_PRIORITY[d.distance_level]],
                d.confidence,
            ),
            reverse=True,
        )
        top = ranked[0]
        if top.distance_level == DistanceLevel.CRITICAL and top.distance_meters > 0.0:
            sentences.append(f"Warning! {top.label.capitalize()} {top.distance_meters:.1f} meters ahead.")
        else:
            labels: list[str] = []
            seen: set[str] = set()
            for det in ranked:
                if det.label in seen:
                    continue
                seen.add(det.label)
                labels.append(det.label)

            if labels:
                if len(labels) == 1:
                    sentences.append(f"{labels[0].capitalize()} ahead.")
                elif len(labels) == 2:
                    sentences.append(f"There is a {labels[0]} and a {labels[1]} ahead.")
                else:
                    leading = ", ".join(labels[:-1])
                    sentences.append(f"There is a {leading}, and a {labels[-1]} ahead.")

    # 3. Signs or text
    if texts:
        for t in texts:
            if t.text:
                sentences.append(f"Sign reads {t.text}.")
                break

    return " ".join(sentences).strip()


def build_fusion_key(
    detections: list[Detection],
    faces: list[FaceDetection] | None = None,
    texts: list[TextDetection] | None = None,
) -> str:
    """Build a stable key reflecting the current scene state for cooldown tracking."""
    dets = detections or []
    face_list = [f for f in (faces or []) if f.name and f.name != "Unknown"]
    text_list = texts or []

    if not dets and not face_list and not text_list:
        return "silent"

    parts: list[str] = []
    if dets:
        labels = sorted({d.label for d in dets})
        top = max(
            dets,
            key=lambda d: (
                _PRIORITY_RANK[_DISTANCE_TO_PRIORITY[d.distance_level]],
                d.confidence,
            ),
        )
        parts.append(f"{top.distance_level.value}:{','.join(labels)}")

    if face_list:
        names = sorted({f.name for f in face_list})
        parts.append(f"face:{','.join(names)}")

    if text_list:
        txts = sorted({t.text for t in text_list})
        parts.append(f"text:{','.join(txts)}")

    return "|".join(parts)


class FusionEngine:
    """Fuses multi-modal visual inputs and manages spoken notifications."""

    def __init__(
        self,
        cooldown_seconds: float = 2.0,
        cooldown_manager: Optional[CooldownManager] = None,
    ) -> None:
        self.cooldown = cooldown_manager or CooldownManager(cooldown_seconds=cooldown_seconds)

    def determine_priority(
        self,
        detections: list[Detection],
        faces: list[FaceDetection] | None = None,
        texts: list[TextDetection] | None = None,
    ) -> Priority:
        """Calculate the overall priority from all present detections."""
        dets = detections or []
        face_list = faces or []
        text_list = texts or []

        if any(d.distance_level == DistanceLevel.CRITICAL for d in dets):
            return Priority.CRITICAL

        if any(d.distance_level == DistanceLevel.WARNING for d in dets):
            return Priority.WARNING

        if any(t.text in ("STOP", "DANGER", "CAUTION") for t in text_list):
            return Priority.WARNING

        if any(f.name != "Unknown" for f in face_list):
            return Priority.INFO

        if any(d.distance_level == DistanceLevel.INFO for d in dets):
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
        """Combine all detections and produce a structured Decision."""
        dets = list(detections or [])
        face_list = list(faces or [])
        text_list = list(texts or [])
        now_wall = time.time()

        if not dets and not face_list and not text_list:
            return Decision.silent(timestamp=now_wall)

        priority = self.determine_priority(dets, face_list, text_list)
        key = build_fusion_key(dets, face_list, text_list)
        raw_message = build_speech_message(dets, face_list, text_list)

        spoken_message = ""
        if raw_message and self.cooldown.should_speak(key, priority):
            spoken_message = raw_message
            self.cooldown.record_spoken(key, priority)

        return Decision(
            key=key,
            message=spoken_message,
            priority=priority,
            detections=dets,
            faces=face_list,
            texts=text_list,
            timestamp=now_wall,
        )
