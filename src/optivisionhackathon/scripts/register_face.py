"""CLI tool to register known faces for Visionnaire recognition."""

from __future__ import annotations

import argparse
import logging
import os
import sys

import cv2

from optivisionhackathon.server.models.face import DEFAULT_DB_PATH, FaceRecognizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("visionnaire.register_face")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register known faces into the Visionnaire embeddings database."
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Name of the person to register (e.g. 'Souhail')",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to an image file containing the face to register",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Capture a frame from the webcam to register",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=int(os.getenv("VISIONNAIRE_CAMERA_INDEX", "0")),
        help="Camera device index for webcam capture (default: 0)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to face database file (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all currently registered faces in the database",
    )
    parser.add_argument(
        "--delete",
        type=str,
        help="Delete a registered person from the database",
    )

    args = parser.parse_args()

    recognizer = FaceRecognizer(db_path=args.db)

    if args.list:
        faces = recognizer.list_faces()
        if not faces:
            print("No faces currently registered.")
        else:
            print(f"Registered faces ({len(faces)}):")
            for name in faces:
                count = len(recognizer.db.get(name, []))
                print(f"  - {name} ({count} sample{'s' if count != 1 else ''})")
        return

    if args.delete:
        if recognizer.delete_face(args.delete):
            print(f"Successfully deleted face '{args.delete}'.")
        else:
            print(f"Face '{args.delete}' not found in database.")
        return

    if not args.name:
        parser.error("--name is required when registering a face.")

    frame = None

    if args.image:
        if not os.path.isfile(args.image):
            logger.error("Image file not found: %s", args.image)
            sys.exit(1)
        frame = cv2.imread(args.image)
        if frame is None:
            logger.error("Failed to read image from %s", args.image)
            sys.exit(1)
    elif args.webcam:
        logger.info("Opening webcam device index %d...", args.device)
        cap = cv2.VideoCapture(args.device)
        if not cap.isOpened():
            logger.error("Could not open camera device %d", args.device)
            sys.exit(1)
        try:
            # Warm up camera
            for _ in range(5):
                cap.read()
            ok, frame = cap.read()
            if not ok or frame is None:
                logger.error("Failed to capture frame from webcam")
                sys.exit(1)
        finally:
            cap.release()
    else:
        parser.error("Specify either --image <path> or --webcam to register a face.")

    assert frame is not None
    # Locate face or use whole frame if cropped
    regions = recognizer.detect_face_regions(frame)
    if regions:
        x1, y1, x2, y2 = (int(v) for v in regions[0])
        face_crop = frame[y1:y2, x1:x2]
    else:
        logger.warning("No distinct face contour detected; using full image crop.")
        face_crop = frame

    emb = recognizer.register_face(args.name, face_crop)
    print(f"Successfully registered '{args.name}' (embedding dimension: {len(emb)}) to {args.db}")


if __name__ == "__main__":
    main()
