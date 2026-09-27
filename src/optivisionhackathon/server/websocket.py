"""Async WebSocket transport for the Visionnaire inference server.

Handles accept / receive / decode / respond only.
All AI inference is delegated to the Pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import websockets

from optivisionhackathon.shared.protocol import ProtocolError, decode_frame

if TYPE_CHECKING:
    from optivisionhackathon.server.pipeline import Pipeline
    from optivisionhackathon.server.schemas import Decision

logger = logging.getLogger(__name__)


def _jpeg_to_frame(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes into a BGR OpenCV frame."""
    array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("failed to decode JPEG into an image frame")
    return frame


def _error_decision(_reason: str) -> Decision:
    from optivisionhackathon.server.schemas import Decision, Priority

    return Decision(
        key="error",
        message="",
        priority=Priority.NONE,
        detections=[],
        timestamp=time.time(),
    )


class VisionnaireServer:
    """WebSocket server that feeds frames into the inference pipeline."""

    def __init__(
        self,
        pipeline: Pipeline,
        host: str = "0.0.0.0",
        port: int = 8765,
    ) -> None:
        self.pipeline = pipeline
        self.host = host
        self.port = port

    async def _handle_client(self, websocket: Any) -> None:
        remote = getattr(websocket, "remote_address", "unknown")
        logger.info("Client connected: %s", remote)
        try:
            async for message in websocket:
                await self._handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed as exc:
            logger.info("Client disconnected: %s (%s)", remote, exc)
        except Exception:
            logger.exception("Unhandled error in client handler for %s", remote)
        finally:
            logger.info("Connection closed: %s", remote)

    async def _handle_message(
        self,
        websocket: Any,
        message: str | bytes,
    ) -> None:
        if not isinstance(message, (bytes, bytearray)):
            logger.warning("Ignoring non-binary WebSocket message")
            await self._send_decision(
                websocket,
                _error_decision("expected binary frame packet"),
            )
            return

        try:
            jpeg_bytes = decode_frame(bytes(message))
        except ProtocolError as exc:
            logger.warning("Malformed packet: %s", exc)
            await self._send_decision(
                websocket, _error_decision(f"malformed packet: {exc}")
            )
            return

        try:
            frame = _jpeg_to_frame(jpeg_bytes)
        except ValueError as exc:
            logger.warning("Malformed JPEG: %s", exc)
            await self._send_decision(
                websocket, _error_decision(f"malformed jpeg: {exc}")
            )
            return

        try:
            # Run blocking YOLO off the event loop so other clients stay responsive.
            decision = await asyncio.to_thread(self.pipeline.process_frame, frame)
        except Exception as exc:
            logger.exception("Inference error")
            await self._send_decision(
                websocket, _error_decision(f"inference error: {exc}")
            )
            return

        await self._send_decision(websocket, decision)

    async def _send_decision(self, websocket: Any, decision: Decision) -> None:
        payload = json.dumps(decision.to_dict())
        await websocket.send(payload)

    async def run(self) -> None:
        """Bind to host:port and serve until cancelled."""
        logger.info("Starting WebSocket server on %s:%s", self.host, self.port)
        async with websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            max_size=8 * 1024 * 1024,
        ):
            await asyncio.Future()
