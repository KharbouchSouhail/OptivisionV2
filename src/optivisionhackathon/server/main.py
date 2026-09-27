"""Visionnaire inference server entrypoint.

Loads YOLO once at startup, builds the pipeline, and starts the
WebSocket server. Configuration comes from environment variables.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

from optivisionhackathon.server.models.yolo import YOLODetector, log_device_info
from optivisionhackathon.server.pipeline import Pipeline
from optivisionhackathon.server.websocket import VisionnaireServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("visionnaire.server")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


async def _async_main() -> None:
    load_dotenv()

    host = os.getenv("VISIONNAIRE_HOST", "0.0.0.0")
    port = _env_int("VISIONNAIRE_PORT", 8765)
    cooldown = float(os.getenv("VISIONNAIRE_DECISION_COOLDOWN", "2.0"))

    # Detect CUDA and load the model once for the lifetime of the process.
    device = log_device_info()
    detector = YOLODetector(device=device)
    pipeline = Pipeline(detector=detector, cooldown_seconds=cooldown)
    server = VisionnaireServer(pipeline=pipeline, host=host, port=port)

    logger.info("Visionnaire server ready (YOLO MVP)")
    await server.run()


def main() -> None:
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        logger.info("Shutting down server (KeyboardInterrupt)")
        sys.exit(0)


if __name__ == "__main__":
    main()
