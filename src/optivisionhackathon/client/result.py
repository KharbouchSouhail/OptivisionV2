"""Helpers to parse server JSON decisions on the client."""

from __future__ import annotations

from typing import Any


def parse_decision(payload: Any) -> dict[str, Any]:
    """Normalize a server JSON payload into a client-friendly dict.

    Expected shape (extra keys allowed):
        {
          "detections": [{"label", "confidence", "bbox"}, ...],
          "message": "...",
          "timestamp": ...,
          "key": "...",
          "priority": "..."
        }
    """
    if not isinstance(payload, dict):
        raise ValueError("decision payload must be a JSON object")

    raw_detections = payload.get("detections") or []
    if not isinstance(raw_detections, list):
        raise ValueError("detections must be a list")

    detections: list[dict[str, Any]] = []
    for item in raw_detections:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        bbox_raw = item.get("bbox") or [0, 0, 0, 0]
        if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) < 4:
            continue
        try:
            bbox = [float(bbox_raw[0]), float(bbox_raw[1]), float(bbox_raw[2]), float(bbox_raw[3])]
        except (TypeError, ValueError):
            continue
        detections.append(
            {
                "label": label,
                "confidence": confidence,
                "bbox": bbox,
                "distance_level": str(item.get("distance_level") or ""),
            }
        )

    message = str(payload.get("message") or "").strip()
    try:
        timestamp = float(payload.get("timestamp") or 0.0)
    except (TypeError, ValueError):
        timestamp = 0.0

    return {
        "key": str(payload.get("key") or ""),
        "message": message,
        "priority": str(payload.get("priority") or ""),
        "detections": detections,
        "timestamp": timestamp,
    }
