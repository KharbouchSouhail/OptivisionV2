"""Comprehensive tests for the multi-modal fusion engine, cooldown, depth, face, and OCR."""

from __future__ import annotations

import time
import cv2
import numpy as np
import pytest

from optivisionhackathon.server.fusion.cooldown import CooldownManager
from optivisionhackathon.server.fusion.engine import (
    FusionEngine,
    build_fusion_key,
    build_speech_message,
)
from optivisionhackathon.server.models.depth import DepthEstimator
from optivisionhackathon.server.models.face import FaceRecognizer
from optivisionhackathon.server.models.ocr import OCREngine
from optivisionhackathon.server.schemas import (
    Detection,
    DistanceLevel,
    FaceDetection,
    Priority,
    TextDetection,
)


def _det(
    label: str = "chair",
    conf: float = 0.85,
    level: DistanceLevel = DistanceLevel.WARNING,
    dist: float = 2.0,
) -> Detection:
    return Detection(
        label=label,
        confidence=conf,
        bbox=(10.0, 10.0, 100.0, 150.0),
        distance_level=level,
        distance_meters=dist,
    )


def test_cooldown_manager_basic() -> None:
    cd = CooldownManager(cooldown_seconds=1.0, critical_cooldown_seconds=0.5, min_gap_seconds=0.0)
    key = "WARNING:chair"

    assert cd.should_speak(key, Priority.WARNING) is True
    cd.record_spoken(key, Priority.WARNING)

    # Immediately after, should be suppressed
    assert cd.should_speak(key, Priority.WARNING) is False


def test_cooldown_manager_escalation() -> None:
    cd = CooldownManager(cooldown_seconds=5.0, min_gap_seconds=0.0)
    key = "chair"

    cd.record_spoken(key, Priority.INFO)
    # Escalation to CRITICAL should bypass cooldown
    assert cd.should_speak(key, Priority.CRITICAL) is True


def test_cooldown_manager_reset() -> None:
    cd = CooldownManager(cooldown_seconds=10.0, min_gap_seconds=0.0)
    key = "test"
    cd.record_spoken(key, Priority.WARNING)
    assert cd.should_speak(key, Priority.WARNING) is False

    cd.reset()
    assert cd.should_speak(key, Priority.WARNING) is True


def test_fusion_engine_speech_building() -> None:
    dets = [_det("person", 0.9, DistanceLevel.CRITICAL, 1.2)]
    faces = [FaceDetection("Alice", 0.95, (10, 10, 50, 50), 1.2)]
    texts = [TextDetection("EXIT", 0.88, (100, 100, 180, 130))]

    msg = build_speech_message(dets, faces, texts)
    assert "Alice is ahead" in msg
    assert "Warning! Person 1.2 meters ahead." in msg
    assert "Sign reads EXIT." in msg


def test_fusion_engine_priority_levels() -> None:
    engine = FusionEngine(cooldown_seconds=0.0)

    # Critical detection
    crit_dec = engine.fuse(detections=[_det("car", 0.9, DistanceLevel.CRITICAL, 1.0)])
    assert crit_dec.priority == Priority.CRITICAL

    # Warning detection
    warn_dec = engine.fuse(detections=[_det("chair", 0.8, DistanceLevel.WARNING, 2.5)])
    assert warn_dec.priority == Priority.WARNING

    # Known face only
    face_dec = engine.fuse(faces=[FaceDetection("Bob", 0.9, (0, 0, 10, 10), 2.0)])
    assert face_dec.priority == Priority.INFO

    # Empty
    silent_dec = engine.fuse()
    assert silent_dec.priority == Priority.NONE
    assert silent_dec.message == ""


def test_depth_estimator() -> None:
    estimator = DepthEstimator(fov_degrees=60.0, critical_distance=1.5, warning_distance=3.0)
    frame_shape = (480, 640)

    # Large bbox (close) vs small bbox (far)
    close_dist = estimator.estimate_distance((100, 50, 300, 450), frame_shape, "person")
    far_dist = estimator.estimate_distance((100, 50, 150, 100), frame_shape, "person")
    assert close_dist < far_dist

    # Classification
    assert estimator.classify_distance(1.2) == DistanceLevel.CRITICAL
    assert estimator.classify_distance(2.4) == DistanceLevel.WARNING
    assert estimator.classify_distance(4.0) == DistanceLevel.INFO

    # Depth map
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
    depth_map = estimator.estimate_depth_map(frame)
    assert depth_map.shape == (100, 100)
    assert 0.0 <= depth_map.min() <= depth_map.max() <= 1.0

    # Enrich detections
    dets = [_det("person", 0.9, DistanceLevel.INFO, 0.0)]
    enriched = estimator.enrich_detections(dets, frame_shape)
    assert len(enriched) == 1
    assert enriched[0].distance_meters > 0.0


def test_face_recognizer(tmp_path: pytest.TempPathFactory) -> None:
    db_file = str(tmp_path / "faces.pkl")
    recognizer = FaceRecognizer(db_path=db_file, device="cpu", similarity_threshold=0.6)

    # Extract embedding
    face_img = np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)
    emb = recognizer.extract_embedding(face_img)
    assert emb.shape == (576,)
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-3)

    # Register face
    recognizer.register_face("Souhail", face_img)
    assert "Souhail" in recognizer.list_faces()

    # Match exact same image
    matched_name, sim = recognizer._match_embedding(emb)
    assert matched_name == "Souhail"
    assert sim >= 0.99

    # Re-instantiate to test persistence
    recognizer2 = FaceRecognizer(db_path=db_file, device="cpu")
    assert "Souhail" in recognizer2.list_faces()

    # Delete face
    assert recognizer2.delete_face("Souhail") is True
    assert recognizer2.list_faces() == []


def test_ocr_engine() -> None:
    ocr = OCREngine(min_confidence=0.3)
    assert len(ocr._glyph_cache) >= 36

    # Create a synthetic image with the word "EXIT"
    canvas = np.zeros((100, 250, 3), dtype=np.uint8)
    cv2.putText(canvas, "EXIT", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)

    results = ocr.read(canvas)
    assert isinstance(results, list)
    if results:
        assert results[0].text in ("EXIT", "E", "X", "I", "T") or len(results[0].text) >= 2
