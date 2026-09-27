"""Visionnaire laptop client entrypoint.

Live loop:
    webcam frame -> OpenCV preview (boxes / HUD)
               -> JPEG -> binary WebSocket -> remote YOLO
               -> JSON result -> non-blocking pyttsx3 speech

Target send rate: VISIONNAIRE_TARGET_FPS (default 5).
Only one in-flight inference request at a time (no unbounded queue).
Press `q` in the preview window to quit.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from typing import Any, Optional

from dotenv import load_dotenv

from optivisionhackathon.client.camera import Camera
from optivisionhackathon.client.display import (
    annotate_frame,
    destroy_windows,
    show_frame,
)
from optivisionhackathon.client.result import parse_decision
from optivisionhackathon.client.speaker import Speaker
from optivisionhackathon.client.websocket import DecisionClient
from optivisionhackathon.shared.protocol import encode_frame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("visionnaire.client")


def _load_env() -> None:
    """Load .env from CWD first, then walk up from this file."""
    if load_dotenv():
        return
    here = os.path.abspath(os.path.dirname(__file__))
    for _ in range(5):
        candidate = os.path.join(here, ".env")
        if os.path.isfile(candidate):
            load_dotenv(candidate)
            return
        here = os.path.dirname(here)
    load_dotenv()  # best-effort default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


async def run_client() -> None:
    _load_env()

    target_fps = _env_float("VISIONNAIRE_TARGET_FPS", 5.0)
    frame_interval = 1.0 / target_fps if target_fps > 0 else 0.2

    camera = Camera()
    speaker = Speaker()
    client = DecisionClient()

    detections: list[dict[str, Any]] = []
    faces: list[dict[str, Any]] = []
    texts: list[dict[str, Any]] = []
    last_message = ""
    status = "starting"
    connected = False
    display_fps = 0.0
    pending: Optional[asyncio.Task[dict[str, Any]]] = None
    last_send_at = 0.0
    last_reconnect_at = 0.0
    reconnect_interval = 2.0
    prev_loop_at = time.monotonic()
    running = True

    try:
        camera.open()
    except Exception:
        logger.exception("Failed to open camera")
        raise

    try:
        speaker.start()
    except Exception:
        logger.exception("Speaker start failed — continuing without TTS")

    try:
        await client.connect()
        connected = True
        status = "connected"
    except ConnectionError as exc:
        logger.error("Initial WebSocket connect failed: %s", exc)
        connected = False
        status = "reconnect"
        # Continue with preview; reconnect in-loop.

    logger.info(
        "Client running (~%.1f FPS send) → %s  |  press q to quit",
        target_fps,
        client.server_url,
    )

    try:
        while running:
            loop_started = time.monotonic()
            dt = loop_started - prev_loop_at
            prev_loop_at = loop_started
            if dt > 0:
                instant = 1.0 / dt
                display_fps = (
                    instant if display_fps <= 0 else (0.85 * display_fps + 0.15 * instant)
                )

            # --- capture (single camera pipeline) ---
            try:
                frame = camera.read_frame()
            except RuntimeError as exc:
                logger.error("Camera error: %s", exc)
                status = "camera-error"
                await asyncio.sleep(0.2)
                continue

            # --- collect finished inference ---
            if pending is not None and pending.done():
                try:
                    raw = pending.result()
                    decision = parse_decision(raw)
                    detections = decision["detections"]
                    faces = decision.get("faces", [])
                    texts = decision.get("texts", [])
                    msg = decision["message"]
                    if msg:
                        last_message = msg
                        try:
                            speaker.speak(msg)
                        except Exception:
                            logger.exception("TTS speak failed")
                    connected = True
                    status = "ok"
                except ConnectionError as exc:
                    logger.error("WebSocket error: %s", exc)
                    connected = False
                    status = "disconnected"
                except ValueError as exc:
                    logger.error("Bad server response: %s", exc)
                    status = "bad-response"
                except Exception:
                    logger.exception("Unexpected inference result error")
                    status = "error"
                finally:
                    pending = None

            # --- start new inference if idle and interval elapsed ---
            now = time.monotonic()
            if pending is None and (now - last_send_at) >= frame_interval:
                if not client.connected:
                    # Non-blocking backoff: never sleep the preview loop here.
                    if (now - last_reconnect_at) >= reconnect_interval:
                        last_reconnect_at = now
                        try:
                            await client.connect()
                            connected = True
                            status = "reconnected"
                        except ConnectionError as exc:
                            connected = False
                            status = "reconnect-failed"
                            logger.error("Reconnect failed: %s", exc)
                            last_send_at = now
                if client.connected:
                    try:
                        jpeg = camera.encode_jpeg(frame)
                        packet = encode_frame(jpeg)
                        pending = asyncio.create_task(
                            client.send_and_receive(packet)
                        )
                        last_send_at = now
                        connected = True
                        if status in ("starting", "reconnect", "reconnect-failed"):
                            status = "inferring"
                    except Exception as exc:
                        logger.error("Failed to send frame: %s", exc)
                        connected = False
                        status = "send-failed"
                        pending = None
                        last_send_at = now

            # --- live preview (never blocked by TTS) ---
            annotated = annotate_frame(
                frame,
                detections=detections,
                faces=faces,
                texts=texts,
                fps=display_fps,
                connected=connected and client.connected,
                status=status,
                last_message=last_message,
            )
            key = show_frame(annotated)
            if key == ord("q"):
                logger.info("Quit requested (q)")
                running = False
                break

            # Yield so the pending WebSocket task can progress.
            await asyncio.sleep(0.001)

    except asyncio.CancelledError:
        logger.info("Client cancelled")
        raise
    finally:
        logger.info("Cleaning up client resources")
        if pending is not None and not pending.done():
            pending.cancel()
            try:
                await pending
            except Exception:
                pass
        await client.close()
        camera.release()
        speaker.shutdown()
        destroy_windows()


def main() -> None:
    try:
        asyncio.run(run_client())
    except KeyboardInterrupt:
        logger.info("Shutting down client (KeyboardInterrupt)")
        try:
            destroy_windows()
        except Exception:
            pass
        sys.exit(0)


if __name__ == "__main__":
    main()
