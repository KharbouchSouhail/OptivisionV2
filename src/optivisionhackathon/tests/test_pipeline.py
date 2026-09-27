"""Basic pipeline tests that do not require a GPU or YOLO weights."""

from __future__ import annotations

import numpy as np

from optivisionhackathon.server.pipeline import Pipeline
from optivisionhackathon.server.schemas import Detection, DistanceLevel, Priority


class _FakeDetector:
    def __init__(self, detections: list[Detection]) -> None:
        self._detections = detections

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return list(self._detections)


def test_pipeline_silent_when_no_detections() -> None:
    pipe = Pipeline(_FakeDetector([]), cooldown_seconds=0.0)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    decision = pipe.process_frame(frame)
    assert decision.message == ""
    assert decision.priority == Priority.NONE


def test_pipeline_critical_person_message() -> None:
    detections = [
        Detection(
            label="person",
            confidence=0.91,
            bbox=(10.0, 10.0, 80.0, 200.0),
            distance_level=DistanceLevel.CRITICAL,
        )
    ]
    pipe = Pipeline(_FakeDetector(detections), cooldown_seconds=0.0)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    decision = pipe.process_frame(frame)
    assert decision.priority == Priority.CRITICAL
    assert "person" in decision.message.lower()
    assert decision.key.startswith("CRITICAL:")
