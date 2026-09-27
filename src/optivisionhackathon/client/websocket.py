"""Async WebSocket client for Visionnaire."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Optional

import websockets

logger = logging.getLogger(__name__)


class DecisionClient:
    """Connects to the inference server, sends frames, receives decisions."""

    def __init__(
        self,
        server_url: str | None = None,
        reconnect_delay: float = 2.0,
        receive_timeout: float = 10.0,
    ) -> None:
        self.server_url = server_url or os.getenv(
            "VISIONNAIRE_SERVER_URL", "ws://localhost:8765"
        )
        self.reconnect_delay = reconnect_delay
        self.receive_timeout = receive_timeout
        self._ws: Any = None

    @property
    def connected(self) -> bool:
        return self._ws is not None

    async def connect(self) -> None:
        """Connect to the Visionnaire WebSocket server."""
        await self.close()
        logger.info("Connecting to %s", self.server_url)
        try:
            self._ws = await websockets.connect(
                self.server_url, max_size=8 * 1024 * 1024
            )
        except Exception as exc:
            self._ws = None
            raise ConnectionError(
                f"WebSocket connection failure to {self.server_url}: {exc}"
            ) from exc
        logger.info("Connected to %s", self.server_url)

    async def ensure_connected(self) -> None:
        """Connect if needed; reconnect after failures."""
        if self._ws is not None:
            return
        await self.connect()

    async def close(self) -> None:
        """Close the WebSocket connection cleanly."""
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                logger.debug("Error while closing WebSocket", exc_info=True)
            self._ws = None

    async def send_frame(self, packet: bytes) -> None:
        """Send a binary protocol packet."""
        await self.ensure_connected()
        assert self._ws is not None
        try:
            await self._ws.send(packet)
        except Exception as exc:
            await self.close()
            raise ConnectionError(f"WebSocket send failed: {exc}") from exc

    async def receive_decision(self) -> dict[str, Any]:
        """Receive and parse a JSON decision from the server."""
        await self.ensure_connected()
        assert self._ws is not None
        try:
            raw = await asyncio.wait_for(
                self._ws.recv(), timeout=self.receive_timeout
            )
        except asyncio.TimeoutError as exc:
            await self.close()
            raise ConnectionError("WebSocket receive timed out") from exc
        except Exception as exc:
            await self.close()
            raise ConnectionError(f"WebSocket receive failed: {exc}") from exc

        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON decision: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError("decision payload must be a JSON object")
        return data

    async def send_and_receive(self, packet: bytes) -> dict[str, Any]:
        """Send one frame packet and wait for the matching JSON decision."""
        await self.send_frame(packet)
        return await self.receive_decision()

    async def reconnect(self) -> None:
        """Close and reconnect after a short delay."""
        await self.close()
        logger.info("Reconnecting in %.1fs...", self.reconnect_delay)
        await asyncio.sleep(self.reconnect_delay)
        await self.connect()
