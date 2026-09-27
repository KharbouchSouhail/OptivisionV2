"""Text-to-speech output via espeak-ng / espeak or pyttsx3."""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class Speaker:
    """Non-overlapping, non-blocking speech helper.

    TTS runs on a background thread so the camera preview and WebSocket
    loop are never blocked. Failures are logged and never crash the client.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._busy = False
        self._thread: Optional[threading.Thread] = None
        self._available = False
        self._started = False
        self._backend: Optional[str] = None

    def start(self) -> None:
        """Initialize and verify the TTS backend."""
        if self._started:
            return
        self._started = True

        # Check for espeak-ng or espeak CLI binaries
        if shutil.which("espeak-ng"):
            self._backend = "espeak-ng"
            self._available = True
            logger.info("TTS ready (espeak-ng)")
            return
        if shutil.which("espeak"):
            self._backend = "espeak"
            self._available = True
            logger.info("TTS ready (espeak)")
            return

        try:
            import pyttsx3

            engine = pyttsx3.init()
            _ = engine.getProperty("voices")
            engine.stop()
            self._backend = "pyttsx3"
            self._available = True
            logger.info("TTS ready (pyttsx3)")
        except Exception:
            self._available = False
            logger.exception("TTS initialization failed — continuing without speech")

    def speak(self, message: str) -> bool:
        """Speak a message if free. Never blocks the caller.

        Returns True if speech was started, False if skipped / unavailable.
        """
        text = (message or "").strip()
        if not text:
            return False

        if not self._started:
            self.start()

        if not self._available:
            logger.debug("TTS unavailable; skipping: %s", text)
            return False

        with self._lock:
            if self._busy:
                logger.debug("Skipping speech (already speaking): %s", text)
                return False
            self._busy = True

        self._thread = threading.Thread(
            target=self._speak_worker,
            args=(text,),
            daemon=True,
            name="visionnaire-tts",
        )
        self._thread.start()
        return True

    def _speak_worker(self, text: str) -> None:
        """Run speech playback in a worker thread."""
        try:
            logger.info("Speaking: %s", text)
            if self._backend in ("espeak-ng", "espeak"):
                binary = self._backend
                assert binary is not None
                subprocess.run(
                    [binary, "-s", "175", text],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return

            import pyttsx3

            engine = pyttsx3.init()
            try:
                engine.setProperty("rate", 175)
            except Exception:
                pass
            engine.say(text)
            engine.runAndWait()
            try:
                engine.stop()
            except Exception:
                pass
        except Exception:
            logger.exception("TTS error while speaking")
        finally:
            with self._lock:
                self._busy = False

    def shutdown(self) -> None:
        """Stop speech cleanly."""
        with self._lock:
            busy = self._busy
        if busy and self._thread is not None:
            self._thread.join(timeout=2.0)
        self._available = False
        logger.info("Speaker shut down")

    def __enter__(self) -> Speaker:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.shutdown()
