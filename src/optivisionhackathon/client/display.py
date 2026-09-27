"""Live OpenCV preview overlays for the Visionnaire client."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

WINDOW_NAME = "Visionnaire"


def annotate_frame(
    frame: np.ndarray,
    *,
    detections: list[dict[str, Any]],
    fps: float,
    connected: bool,
    status: str,
    last_message: str,
) -> np.ndarray:
    """Draw bounding boxes and HUD text onto a copy of the frame."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    for det in detections:
        bbox = det.get("bbox") or [0, 0, 0, 0]
        x1, y1, x2, y2 = (int(v) for v in bbox[:4])
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        label = str(det.get("label") or "?")
        conf = float(det.get("confidence") or 0.0)
        color = (40, 180, 40)
        level = str(det.get("distance_level") or "").upper()
        if level == "CRITICAL":
            color = (40, 40, 220)
        elif level == "WARNING":
            color = (0, 165, 255)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        caption = f"{label} {conf:.2f}"
        text_y = y1 - 8 if y1 > 20 else y1 + 18
        cv2.putText(
            canvas,
            caption,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    conn_color = (40, 200, 40) if connected else (40, 40, 220)
    hud_lines = [
        f"FPS: {fps:.1f}",
        f"WS: {'CONNECTED' if connected else 'DISCONNECTED'} ({status})",
    ]
    if last_message:
        hud_lines.append(f"Said: {last_message}")

    y = 24
    for line in hud_lines:
        cv2.putText(
            canvas,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            conn_color if line.startswith("WS:") else (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        y += 24

    cv2.putText(
        canvas,
        "Press q to quit",
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return canvas


def show_frame(frame: np.ndarray) -> int:
    """Show a frame; return the OpenCV waitKey code."""
    cv2.imshow(WINDOW_NAME, frame)
    return int(cv2.waitKey(1) & 0xFF)


def destroy_windows() -> None:
    try:
        cv2.destroyWindow(WINDOW_NAME)
    except Exception:
        pass
    cv2.destroyAllWindows()
    # Flush GUI events on some backends.
    try:
        cv2.waitKey(1)
    except Exception:
        pass
