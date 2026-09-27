"""Tests for decision message building, cooldown, and client parsing."""

from __future__ import annotations

import numpy as np
import pytest

from optivisionhackathon.client.result import parse_decision
from optivisionhackathon.server.pipeline import (
    Pipeline,
    build_speech_message,
    decision_key,
)
from optivisionhackathon.server.schemas import Detection, DistanceLevel, Priority


class _FakeDetector:
    def __init__(self, detections: list[Detection]) -> None:
        self._detections = detections

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return list(self._detections)


def _det(
    label: str,
    conf: float = 0.9,
    level: DistanceLevel = DistanceLevel.WARNING,
) -> Detection:
    return Detection(label, conf, (10.0, 10.0, 80.0, 200.0), level)


def test_pipeline_silent_when_no_detections() -> None:
    pipe = Pipeline(_FakeDetector([]), cooldown_seconds=0.0)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    decision = pipe.process_frame(frame)
    assert decision.message == ""
    assert decision.priority == Priority.NONE
    assert decision.detections == []


def test_pipeline_person_message_and_detections() -> None:
    detections = [_det("person", 0.91, DistanceLevel.CRITICAL)]
    pipe = Pipeline(_FakeDetector(detections), cooldown_seconds=0.0)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    decision = pipe.process_frame(frame)
    assert decision.priority == Priority.CRITICAL
    assert decision.message == "Person ahead."
    assert len(decision.detections) == 1
    assert decision.detections[0].label == "person"
    assert "detections" in decision.to_dict()
    assert "timestamp" in decision.to_dict()


def test_multi_object_speech_message() -> None:
    msg = build_speech_message(
        [_det("person"), _det("chair", 0.8, DistanceLevel.INFO)]
    )
    assert "person" in msg.lower()
    assert "chair" in msg.lower()
    assert "ahead" in msg.lower()


def test_cooldown_suppresses_speech_but_keeps_detections() -> None:
    detections = [_det("person", 0.9, DistanceLevel.WARNING)]
    pipe = Pipeline(_FakeDetector(detections), cooldown_seconds=10.0)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    first = pipe.process_frame(frame)
    assert first.message == "Person ahead."
    second = pipe.process_frame(frame)
    assert second.message == ""
    assert len(second.detections) == 1
    assert second.key == decision_key(detections)


def test_parse_decision_valid() -> None:
    parsed = parse_decision(
        {
            "key": "WARNING:person",
            "message": "Person ahead.",
            "priority": "WARNING",
            "timestamp": 123.4,
            "detections": [
                {
                    "label": "person",
                    "confidence": 0.91,
                    "bbox": [1, 2, 3, 4],
                    "distance_level": "WARNING",
                }
            ],
        }
    )
    assert parsed["message"] == "Person ahead."
    assert parsed["detections"][0]["label"] == "person"
    assert parsed["detections"][0]["bbox"] == [1.0, 2.0, 3.0, 4.0]


def test_parse_decision_malformed_payload() -> None:
    with pytest.raises(ValueError):
        parse_decision(["not", "a", "dict"])
    with pytest.raises(ValueError):
        parse_decision({"detections": "nope"})


def test_parse_decision_skips_bad_detections() -> None:
    parsed = parse_decision(
        {
            "message": "",
            "detections": [
                {"label": "", "confidence": 1, "bbox": [0, 0, 1, 1]},
                {"label": "chair", "confidence": "bad", "bbox": [0, 0, 10, 10]},
                "junk",
            ],
        }
    )
    assert len(parsed["detections"]) == 1
    assert parsed["detections"][0]["label"] == "chair"
    assert parsed["detections"][0]["confidence"] == 0.0
