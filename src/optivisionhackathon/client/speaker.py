"""Text-to-speech output via pyttsx3."""

from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class Speaker:
    """Non-overlapping speech helper. Independent from networking."""

    def __init__(self) -> None:
        self._engine = None
        self._lock = threading.Lock()
        self._busy = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Initialize the pyttsx3 engine."""
        if self._engine is not None:
            return
        try:
            import pyttsx3
        except ImportError as exc:
            raise RuntimeError(
                "pyttsx3 is not installed; pip install pyttsx3"
            ) from exc

        logger.info("Initializing pyttsx3 speaker")
        self._engine = pyttsx3.init()

    def speak(self, message: str) -> bool:
        """Speak a message if the speaker is free.

        Returns True if speech was started, False if skipped (busy / empty).
        """
        text = (message or "").strip()
        if not text:
            return False

        if self._engine is None:
            self.start()

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
        try:
            assert self._engine is not None
            logger.info("Speaking: %s", text)
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception:
            logger.exception("TTS error while speaking")
        finally:
            with self._lock:
                self._busy = False

    def shutdown(self) -> None:
        """Stop speech and release the engine."""
        with self._lock:
            busy = self._busy
        if busy and self._thread is not None:
            self._thread.join(timeout=2.0)

        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                logger.debug("Error stopping TTS engine", exc_info=True)
            self._engine = None
        logger.info("Speaker shut down")

    def __enter__(self) -> Speaker:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.shutdown()
