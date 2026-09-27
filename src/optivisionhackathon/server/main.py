"""Visionnaire inference server entrypoint.

Loads models at startup, builds the multi-modal pipeline (YOLO, Depth, Face, OCR, Fusion),
and starts the WebSocket server. Configuration comes from environment variables.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

from optivisionhackathon.server.fusion.engine import FusionEngine
from optivisionhackathon.server.models.depth import DepthEstimator
from optivisionhackathon.server.models.face import FaceRecognizer
from optivisionhackathon.server.models.ocr import OCREngine
from optivisionhackathon.server.models.yolo import YOLODetector, log_device_info
from optivisionhackathon.server.pipeline import Pipeline
from optivisionhackathon.server.websocket import VisionnaireServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("visionnaire.server")


def _load_env() -> None:
    if load_dotenv():
        return
    here = os.path.abspath(os.path.dirname(__file__))
    for _ in range(5):
        candidate = os.path.join(here, ".env")
        if os.path.isfile(candidate):
            load_dotenv(candidate)
            return
        here = os.path.dirname(here)
    load_dotenv()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


async def _async_main() -> None:
    _load_env()

    host = os.getenv("VISIONNAIRE_HOST", "0.0.0.0")
    port = _env_int("VISIONNAIRE_PORT", 8765)
    cooldown = float(os.getenv("VISIONNAIRE_DECISION_COOLDOWN", "2.5"))

    enable_depth = _env_bool("VISIONNAIRE_ENABLE_DEPTH", True)
    enable_face = _env_bool("VISIONNAIRE_ENABLE_FACE", True)
    enable_ocr = _env_bool("VISIONNAIRE_ENABLE_OCR", True)

    # Detect CUDA and configure compute device
    device = log_device_info()

    logger.info("Initializing YOLO obstacle detector on %s...", device)
    detector = YOLODetector(device=device)

    depth_estimator = DepthEstimator() if enable_depth else None
    if enable_depth:
        logger.info("Depth estimation enabled")

    face_recognizer = FaceRecognizer(device=device) if enable_face else None
    if enable_face:
        logger.info("Face recognition enabled (known faces: %s)", face_recognizer.list_faces() if face_recognizer else [])

    ocr_engine = OCREngine() if enable_ocr else None
    if enable_ocr:
        logger.info("OCR scene text recognition enabled")

    fusion_engine = FusionEngine(cooldown_seconds=cooldown)

    pipeline = Pipeline(
        detector=detector,
        depth_estimator=depth_estimator,
        face_recognizer=face_recognizer,
        ocr_engine=ocr_engine,
        fusion_engine=fusion_engine,
        cooldown_seconds=cooldown,
    )
    server = VisionnaireServer(pipeline=pipeline, host=host, port=port)

    logger.info("Visionnaire server ready with full multi-modal pipeline (YOLO + Depth + Face + OCR + Fusion)")
    await server.run()


def main() -> None:
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        logger.info("Shutting down server (KeyboardInterrupt)")
        sys.exit(0)


if __name__ == "__main__":
    main()
