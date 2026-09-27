"""CLI tool for registering real ArcFace face embeddings."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import cv2

from optivisionhackathon.server.models.face import (
    DEFAULT_DB_PATH,
    FaceRecognizer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("visionnaire.register_face")


def capture_webcam(
    device: int,
) -> object:
    logger.info(
        "Opening webcam device index %d...",
        device,
    )

    cap = cv2.VideoCapture(device)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera device {device}"
        )

    try:
        # Give the webcam a moment to stabilize.
        for _ in range(10):
            cap.read()
            time.sleep(0.03)

        logger.info(
            "Camera ready. Look directly at the camera."
        )

        ok, frame = cap.read()

        if not ok or frame is None:
            raise RuntimeError(
                "Failed to capture frame from webcam"
            )

        return frame

    finally:
        cap.release()


def load_image(
    path: str,
):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Image file not found: {path}"
        )

    frame = cv2.imread(path)

    if frame is None:
        raise RuntimeError(
            f"Failed to read image: {path}"
        )

    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Register a person using InsightFace/ArcFace."
        )
    )

    parser.add_argument(
        "--name",
        type=str,
        help="Person name, e.g. Souhail",
    )

    parser.add_argument(
        "--image",
        type=str,
        help="Image containing the person's face",
    )

    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Capture the person's face using the webcam",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=int(
            os.getenv(
                "VISIONNAIRE_CAMERA_INDEX",
                "0",
            )
        ),
        help="Webcam device index",
    )

    parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Face database path (default: {DEFAULT_DB_PATH})",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List registered people",
    )

    parser.add_argument(
        "--delete",
        type=str,
        help="Delete a registered person",
    )

    args = parser.parse_args()

    recognizer = FaceRecognizer(
        db_path=args.db,
    )

    # --------------------------------------------------------------
    # List
    # --------------------------------------------------------------

    if args.list:
        faces = recognizer.list_faces()

        if not faces:
            print("No faces currently registered.")
            return

        print(
            f"Registered faces ({len(faces)}):"
        )

        for name in faces:
            count = len(
                recognizer.db.get(
                    name,
                    [],
                )
            )

            print(
                f"  - {name} "
                f"({count} sample"
                f"{'s' if count != 1 else ''})"
            )

        return

    # --------------------------------------------------------------
    # Delete
    # --------------------------------------------------------------

    if args.delete:
        if recognizer.delete_face(
            args.delete
        ):
            print(
                f"Successfully deleted face "
                f"'{args.delete}'."
            )
        else:
            print(
                f"Face '{args.delete}' "
                f"not found."
            )

        return

    # --------------------------------------------------------------
    # Registration validation
    # --------------------------------------------------------------

    if not args.name:
        parser.error(
            "--name is required when registering a face."
        )

    if args.image and args.webcam:
        parser.error(
            "Use either --image or --webcam, not both."
        )

    if not args.image and not args.webcam:
        parser.error(
            "Specify either --image or --webcam."
        )

    # --------------------------------------------------------------
    # Capture image
    # --------------------------------------------------------------

    try:
        if args.image:
            frame = load_image(args.image)
        else:
            frame = capture_webcam(args.device)

    except Exception as exc:
        logger.error(
            "Failed to obtain image: %s",
            exc,
        )
        sys.exit(1)

    # --------------------------------------------------------------
    # Verify that a real face exists
    # --------------------------------------------------------------

    logger.info(
        "Detecting face..."
    )

    faces = recognizer.detect_faces(frame)

    if not faces:
        logger.error(
            "No face detected. "
            "Make sure the face is clearly visible."
        )
        sys.exit(1)

    if len(faces) > 1:
        logger.warning(
            "Detected %d faces. "
            "Using the largest face.",
            len(faces),
        )

    face = max(
        faces,
        key=lambda item: recognizer._bbox_area(
            item.bbox
        ),
    )

    bbox = face.bbox

    logger.info(
        "Face detected at "
        "(%.0f, %.0f) -> (%.0f, %.0f)",
        bbox[0],
        bbox[1],
        bbox[2],
        bbox[3],
    )

    # --------------------------------------------------------------
    # Extract embedding
    # --------------------------------------------------------------

    embedding = getattr(
        face,
        "normed_embedding",
        None,
    )

    if embedding is None:
        embedding = getattr(
            face,
            "embedding",
            None,
        )

    if embedding is None:
        logger.error(
            "InsightFace failed to generate an embedding."
        )
        sys.exit(1)

    embedding = recognizer._normalize(
        embedding
    )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    clean_name = args.name.strip()

    if not clean_name:
        logger.error(
            "Name cannot be empty."
        )
        sys.exit(1)

    if clean_name not in recognizer.db:
        recognizer.db[clean_name] = []

    recognizer.db[clean_name].append(
        embedding
    )

    recognizer.save_db()

    print()
    print(
        f"Successfully registered '{clean_name}'."
    )
    print(
        f"Embedding dimension: {len(embedding)}"
    )
    print(
        f"Database: {recognizer.db_path}"
    )


if __name__ == "__main__":
    main()