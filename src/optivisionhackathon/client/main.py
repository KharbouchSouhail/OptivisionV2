"""Visionnaire laptop client entrypoint.

Capture loop:
    webcam -> JPEG -> binary WebSocket -> decision JSON -> pyttsx3

Target rate is controlled by VISIONNAIRE_TARGET_FPS (default 5).
Frames are not queued: each iteration waits for the server response
(or drops on failure) before capturing the next frame.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

from dotenv import load_dotenv

from optivisionhackathon.client.camera import Camera
from optivisionhackathon.client.speaker import Speaker
from optivisionhackathon.client.websocket import DecisionClient
from optivisionhackathon.shared.protocol import encode_frame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("visionnaire.client")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


async def run_client() -> None:
    load_dotenv()

    target_fps = _env_float("VISIONNAIRE_TARGET_FPS", 5.0)
    frame_interval = 1.0 / target_fps if target_fps > 0 else 0.2

    camera = Camera()
    speaker = Speaker()
    client = DecisionClient()

    try:
        camera.open()
        speaker.start()
        await client.connect()
    except Exception:
        logger.exception("Failed to start client")
        camera.release()
        speaker.shutdown()
        await client.close()
        raise

    logger.info(
        "Client running at ~%.1f FPS toward %s",
        target_fps,
        client.server_url,
    )

    try:
        while True:
            loop_started = time.monotonic()

            try:
                jpeg = camera.read_jpeg()
            except RuntimeError as exc:
                logger.error("Camera error: %s", exc)
                await asyncio.sleep(0.5)
                continue

            packet = encode_frame(jpeg)

            try:
                decision = await client.send_and_receive(packet)
            except ConnectionError as exc:
                logger.error("WebSocket error: %s — reconnecting", exc)
                try:
                    await client.reconnect()
                except ConnectionError as reconnect_exc:
                    logger.error("Reconnect failed: %s", reconnect_exc)
                # Drop this frame; do not build a backlog.
                continue
            except ValueError as exc:
                logger.error("Bad decision payload: %s", exc)
                continue

            message = str(decision.get("message") or "").strip()
            if message:
                speaker.speak(message)
            else:
                logger.debug("Silent decision: %s", decision)

            elapsed = time.monotonic() - loop_started
            sleep_for = frame_interval - elapsed
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            # If inference took longer than the interval, we naturally
            # drop intermediate webcam frames by reading fresh next loop.

    except asyncio.CancelledError:
        logger.info("Client cancelled")
        raise
    finally:
        logger.info("Cleaning up client resources")
        await client.close()
        camera.release()
        speaker.shutdown()


def main() -> None:
    try:
        asyncio.run(run_client())
    except KeyboardInterrupt:
        logger.info("Shutting down client (KeyboardInterrupt)")
        sys.exit(0)


if __name__ == "__main__":
    main()
