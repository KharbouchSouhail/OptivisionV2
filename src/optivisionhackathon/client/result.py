"""Helpers to parse server JSON decisions on the client."""

from __future__ import annotations

from typing import Any


def parse_decision(payload: Any) -> dict[str, Any]:
    """Normalize a server JSON payload into a client-friendly dict.

    Expected shape (extra keys allowed):
        {
          "detections": [{"label", "confidence", "bbox", "distance_level", "distance_meters"}, ...],
          "faces": [{"name", "confidence", "bbox", "distance_meters"}, ...],
          "texts": [{"text", "confidence", "bbox"}, ...],
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
        try:
            distance_meters = float(item.get("distance_meters", 0.0))
        except (TypeError, ValueError):
            distance_meters = 0.0

        detections.append(
            {
                "label": label,
                "confidence": confidence,
                "bbox": bbox,
                "distance_level": str(item.get("distance_level") or ""),
                "distance_meters": distance_meters,
            }
        )

    raw_faces = payload.get("faces") or []
    faces: list[dict[str, Any]] = []
    if isinstance(raw_faces, list):
        for item in raw_faces:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
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
            try:
                distance_meters = float(item.get("distance_meters", 0.0))
            except (TypeError, ValueError):
                distance_meters = 0.0
            faces.append(
                {
                    "name": name,
                    "confidence": confidence,
                    "bbox": bbox,
                    "distance_meters": distance_meters,
                }
            )

    raw_texts = payload.get("texts") or []
    texts: list[dict[str, Any]] = []
    if isinstance(raw_texts, list):
        for item in raw_texts:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
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
            texts.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "bbox": bbox,
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
        "faces": faces,
        "texts": texts,
        "timestamp": timestamp,
    }
