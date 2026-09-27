"""End-to-end integration tests for Visionnaire full multimodal system."""

from __future__ import annotations

import asyncio
import json
import time
import numpy as np
import pytest

from optivisionhackathon.client.display import annotate_frame
from optivisionhackathon.client.result import parse_decision
from optivisionhackathon.client.speaker import Speaker
from optivisionhackathon.server.fusion.engine import FusionEngine
from optivisionhackathon.server.models.depth import DepthEstimator
from optivisionhackathon.server.models.face import FaceRecognizer
from optivisionhackathon.server.models.ocr import OCREngine
from optivisionhackathon.server.pipeline import Pipeline
from optivisionhackathon.server.schemas import (
    Detection,
    DistanceLevel,
    FaceDetection,
    Priority,
    TextDetection,
)
from optivisionhackathon.shared.protocol import decode_frame, encode_frame


class _MockDetector:
    def __init__(self, detections: list[Detection]) -> None:
        self.detections = detections

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return list(self.detections)


def test_full_pipeline_orchestration(tmp_path: pytest.TempPathFactory) -> None:
    db_file = str(tmp_path / "faces.pkl")
    depth = DepthEstimator(fov_degrees=60.0)
    face = FaceRecognizer(db_path=db_file, device="cpu")
    ocr = OCREngine()
    fusion = FusionEngine(cooldown_seconds=0.0)

    # Register Alice in face DB
    alice_img = np.full((112, 112, 3), 180, dtype=np.uint8)
    face.register_face("Alice", alice_img)

    # Synthetic camera frame
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)

    # Mock person detection
    detector = _MockDetector([
        Detection("person", 0.92, (100.0, 50.0, 300.0, 400.0), DistanceLevel.INFO),
    ])

    pipeline = Pipeline(
        detector=detector,
        depth_estimator=depth,
        face_recognizer=face,
        ocr_engine=ocr,
        fusion_engine=fusion,
        cooldown_seconds=0.0,
    )

    decision = pipeline.process_frame(frame)
    assert decision.priority in (Priority.CRITICAL, Priority.WARNING, Priority.INFO)
    assert len(decision.detections) == 1
    # Check depth enrichment happened
    assert decision.detections[0].distance_meters > 0.0

    # Decision dict serializable to valid JSON
    serialized = json.dumps(decision.to_dict())
    loaded = json.loads(serialized)
    assert "detections" in loaded
    assert "faces" in loaded
    assert "texts" in loaded
    assert "message" in loaded


def test_client_result_parsing_full() -> None:
    payload = {
        "key": "CRITICAL:person|face:Alice|text:EXIT",
        "message": "Alice is ahead. Warning! Person 1.1 meters ahead.",
        "priority": "CRITICAL",
        "timestamp": 1700000000.0,
        "detections": [
            {
                "label": "person",
                "confidence": 0.95,
                "bbox": [10.0, 20.0, 100.0, 200.0],
                "distance_level": "CRITICAL",
                "distance_meters": 1.12,
            }
        ],
        "faces": [
            {
                "name": "Alice",
                "confidence": 0.88,
                "bbox": [20.0, 20.0, 80.0, 80.0],
                "distance_meters": 1.12,
            }
        ],
        "texts": [
            {
                "text": "EXIT",
                "confidence": 0.91,
                "bbox": [300.0, 50.0, 400.0, 90.0],
            }
        ],
    }

    parsed = parse_decision(payload)
    assert parsed["key"] == payload["key"]
    assert parsed["message"] == payload["message"]
    assert parsed["priority"] == "CRITICAL"
    assert len(parsed["detections"]) == 1
    assert parsed["detections"][0]["distance_meters"] == 1.12
    assert len(parsed["faces"]) == 1
    assert parsed["faces"][0]["name"] == "Alice"
    assert len(parsed["texts"]) == 1
    assert parsed["texts"][0]["text"] == "EXIT"


def test_client_display_annotation() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = [
        {
            "label": "chair",
            "confidence": 0.85,
            "bbox": [50, 50, 200, 300],
            "distance_level": "WARNING",
            "distance_meters": 2.1,
        }
    ]
    faces = [
        {
            "name": "Souhail",
            "confidence": 0.92,
            "bbox": [300, 100, 400, 220],
        }
    ]
    texts = [
        {
            "text": "STOP",
            "confidence": 0.99,
            "bbox": [450, 50, 550, 100],
        }
    ]

    annotated = annotate_frame(
        frame,
        detections=detections,
        faces=faces,
        texts=texts,
        fps=12.5,
        connected=True,
        status="ok",
        last_message="Chair ahead",
    )
    assert annotated.shape == (480, 640, 3)
    # Canvas should have non-zero pixels from drawings
    assert np.count_nonzero(annotated) > 0


def test_speaker_non_blocking() -> None:
    speaker = Speaker()
    speaker.start()
    # Speak should return a bool immediately and never block
    res = speaker.speak("Testing non-blocking speech")
    assert isinstance(res, bool)
    speaker.shutdown()


def test_protocol_packet_flow() -> None:
    fake_jpeg = b"\xff\xd8" + b"A" * 500 + b"\xff\xd9"
    packet = encode_frame(fake_jpeg)
    assert len(packet) == len(fake_jpeg) + 8
    recovered = decode_frame(packet)
    assert recovered == fake_jpeg
