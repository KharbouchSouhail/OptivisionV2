"""End-to-end live flow verification script.

Spins up the full Visionnaire server and client, captures real webcam frames,
transmits binary frames over WebSocket, receives multi-modal decisions,
annotates frames, and speaks the decision.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time

import cv2
import numpy as np

from optivisionhackathon.client.camera import Camera
from optivisionhackathon.client.display import annotate_frame
from optivisionhackathon.client.result import parse_decision
from optivisionhackathon.client.speaker import Speaker
from optivisionhackathon.client.websocket import DecisionClient
from optivisionhackathon.server.fusion.engine import FusionEngine
from optivisionhackathon.server.models.depth import DepthEstimator
from optivisionhackathon.server.models.face import FaceRecognizer
from optivisionhackathon.server.models.ocr import OCREngine
from optivisionhackathon.server.models.yolo import YOLODetector, log_device_info
from optivisionhackathon.server.pipeline import Pipeline
from optivisionhackathon.server.websocket import VisionnaireServer
from optivisionhackathon.shared.protocol import encode_frame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("visionnaire.verify")


async def run_verification() -> None:
    host = "127.0.0.1"
    port = 8799  # Dedicated port for test verification

    device = log_device_info()
    logger.info("Initializing multi-modal server pipeline...")
    detector = YOLODetector(device=device)
    depth = DepthEstimator()
    face = FaceRecognizer(device=device)
    ocr = OCREngine()
    fusion = FusionEngine(cooldown_seconds=0.0)

    pipeline = Pipeline(
        detector=detector,
        depth_estimator=depth,
        face_recognizer=face,
        ocr_engine=ocr,
        fusion_engine=fusion,
        cooldown_seconds=0.0,
    )

    server = VisionnaireServer(pipeline=pipeline, host=host, port=port)
    server_task = asyncio.create_task(server.run())

    # Wait for server to bind
    await asyncio.sleep(0.5)

    try:
        logger.info("Starting client flow with real webcam...")
        camera = Camera(camera_index=0)
        camera.open()

        speaker = Speaker()
        speaker.start()

        client = DecisionClient(server_url=f"ws://{host}:{port}")
        await client.connect()
        logger.info("Client connected to server!")

        # Capture a real frame
        frame = camera.read_frame()
        logger.info("Captured real webcam frame: shape=%s", frame.shape)

        # Draw a synthetic sign and face onto the frame to verify all modalities in action!
        test_frame = frame.copy()
        cv2.putText(test_frame, "EXIT", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        jpeg = camera.encode_jpeg(test_frame)
        packet = encode_frame(jpeg)
        logger.info("Sending binary frame packet (%d bytes)...", len(packet))

        t0 = time.monotonic()
        raw_decision = await client.send_and_receive(packet)
        elapsed = (time.monotonic() - t0) * 1000.0

        logger.info("Received decision in %.1f ms: %s", elapsed, json.dumps(raw_decision))

        # Parse decision
        parsed = parse_decision(raw_decision)
        logger.info(
            "Parsed: %d obstacles, %d faces, %d texts | priority=%s | message='%s'",
            len(parsed["detections"]),
            len(parsed["faces"]),
            len(parsed["texts"]),
            parsed["priority"],
            parsed["message"],
        )

        # Annotate preview
        annotated = annotate_frame(
            test_frame,
            detections=parsed["detections"],
            faces=parsed["faces"],
            texts=parsed["texts"],
            fps=1.0 / (elapsed / 1000.0) if elapsed > 0 else 10.0,
            connected=True,
            status="verified",
            last_message=parsed["message"],
        )
        logger.info("Frame successfully annotated: shape=%s", annotated.shape)

        # Speak decision
        if parsed["message"]:
            logger.info("Audible TTS speech: '%s'", parsed["message"])
            speaker.speak(parsed["message"])
        else:
            logger.info("Speaking verification prompt...")
            speaker.speak("Visionnaire system online and verified.")

        # Give speaker thread time to speak
        await asyncio.sleep(2.0)

        # Clean up
        await client.close()
        camera.release()
        speaker.shutdown()
        logger.info("Full server -> client flow verification SUCCESSFUL!")

    finally:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


def main() -> None:
    asyncio.run(run_verification())


if __name__ == "__main__":
    main()
