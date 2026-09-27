"""Face detection, ArcFace embedding extraction, and face recognition."""

from __future__ import annotations

import logging
import os
import pickle
from typing import TYPE_CHECKING

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from optivisionhackathon.server.schemas import FaceDetection

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "faces",
    "embeddings.pkl",
)


class FaceRecognizer:
    """Real face detector + ArcFace face recognition."""

    def __init__(
        self,
        db_path: str | None = None,
        device: str | None = None,
        similarity_threshold: float | None = None,
    ) -> None:
        self.db_path = db_path or os.getenv(
            "VISIONNAIRE_FACE_DB",
            DEFAULT_DB_PATH,
        )

        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else float(
                os.getenv(
                    "VISIONNAIRE_FACE_THRESHOLD",
                    "0.45",
                )
            )
        )

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        logger.info("Initializing InsightFace...")

        # CPU is intentional:
        # YOLO already uses the T4 GPU and CPU ArcFace avoids CUDA/ONNX
        # runtime compatibility issues.
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )

        self.app.prepare(
            ctx_id=-1,
            det_size=(640, 640),
        )

        logger.info("InsightFace ready")

        self.db: dict[str, list[np.ndarray]] = {}
        self.load_db()

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    def load_db(self, path: str | None = None) -> None:
        target_path = path or self.db_path

        if not os.path.isfile(target_path):
            logger.info("No face database found at %s", target_path)
            self.db = {}
            return

        if os.path.getsize(target_path) == 0:
            logger.warning("Face database is empty: %s", target_path)
            self.db = {}
            return

        try:
            with open(target_path, "rb") as f:
                loaded = pickle.load(f)

            if not isinstance(loaded, dict):
                raise ValueError("Face database must contain a dictionary")

            self.db = {}

            for name, embeddings in loaded.items():
                if not isinstance(name, str):
                    continue

                if not isinstance(embeddings, list):
                    embeddings = [embeddings]

                valid_embeddings: list[np.ndarray] = []

                for embedding in embeddings:
                    array = np.asarray(
                        embedding,
                        dtype=np.float32,
                    ).reshape(-1)

                    if array.size == 0:
                        continue

                    norm = np.linalg.norm(array)

                    if norm <= 1e-8:
                        continue

                    array = array / norm
                    valid_embeddings.append(array)

                if valid_embeddings:
                    self.db[name] = valid_embeddings

            logger.info(
                "Loaded %d registered faces from %s",
                len(self.db),
                target_path,
            )

        except Exception:
            logger.exception(
                "Failed to load face database from %s",
                target_path,
            )
            self.db = {}

    def save_db(self, path: str | None = None) -> None:
        target_path = path or self.db_path

        directory = os.path.dirname(target_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(target_path, "wb") as f:
            pickle.dump(
                self.db,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        logger.info(
            "Saved %d registered faces to %s",
            len(self.db),
            target_path,
        )

    def list_faces(self) -> list[str]:
        return sorted(self.db.keys())

    def delete_face(self, name: str) -> bool:
        if name not in self.db:
            return False

        del self.db[name]
        self.save_db()

        return True

    # ------------------------------------------------------------------
    # Face detection
    # ------------------------------------------------------------------

    def detect_faces(
        self,
        frame: NDArray[np.uint8],
    ):
        """Return InsightFace detections for a BGR OpenCV frame."""

        if frame is None or frame.size == 0:
            return []

        try:
            return self.app.get(frame)
        except Exception:
            logger.exception("Face detection failed")
            return []

    def detect_face_regions(
        self,
        frame: NDArray[np.uint8],
        person_boxes: list[tuple[float, float, float, float]] | None = None,
    ) -> list[tuple[float, float, float, float]]:
        """Return actual detected face bounding boxes."""

        del person_boxes

        faces = self.detect_faces(frame)

        regions: list[tuple[float, float, float, float]] = []

        for face in faces:
            bbox = getattr(face, "bbox", None)

            if bbox is None or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(float, bbox)

            regions.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                )
            )

        return regions

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(
        embedding: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)

        norm = np.linalg.norm(embedding)

        if norm <= 1e-8:
            return np.zeros_like(embedding)

        return embedding / norm

    def extract_embedding(
        self,
        face_bgr: NDArray[np.uint8],
    ) -> NDArray[np.float32]:
        """
        Extract an ArcFace embedding from a face image.

        This method runs InsightFace's detector and selects the largest
        detected face.
        """

        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("Cannot extract embedding from an empty image")

        faces = self.detect_faces(face_bgr)

        if not faces:
            raise ValueError("No face detected in image")

        face = max(
            faces,
            key=lambda item: self._bbox_area(item.bbox),
        )

        embedding = getattr(face, "normed_embedding", None)

        if embedding is None:
            embedding = getattr(face, "embedding", None)

        if embedding is None:
            raise ValueError("InsightFace did not produce an embedding")

        normalized = self._normalize(
            np.asarray(
                embedding,
                dtype=np.float32,
            )
        )

        if not np.any(normalized):
            raise ValueError("Invalid face embedding")

        return normalized

    @staticmethod
    def _bbox_area(
        bbox,
    ) -> float:
        if bbox is None or len(bbox) != 4:
            return 0.0

        x1, y1, x2, y2 = map(float, bbox)

        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_face(
        self,
        name: str,
        face_bgr: NDArray[np.uint8],
    ) -> NDArray[np.float32]:
        clean_name = name.strip()

        if not clean_name:
            raise ValueError("Face name cannot be empty")

        embedding = self.extract_embedding(face_bgr)

        if clean_name not in self.db:
            self.db[clean_name] = []

        self.db[clean_name].append(embedding)

        self.save_db()

        logger.info(
            "Registered face '%s' with embedding dimension %d",
            clean_name,
            embedding.shape[0],
        )

        return embedding

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _match_embedding(
        self,
        embedding: NDArray[np.float32],
    ) -> tuple[str, float]:
        if not self.db:
            return "Unknown", 0.0

        embedding = self._normalize(embedding)

        if not np.any(embedding):
            return "Unknown", 0.0

        best_name = "Unknown"
        best_similarity = -1.0

        for name, embeddings in self.db.items():
            for known_embedding in embeddings:
                known_embedding = self._normalize(known_embedding)

                if not np.any(known_embedding):
                    continue

                similarity = float(
                    np.dot(
                        embedding,
                        known_embedding,
                    )
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_name = name

        if best_similarity >= self.similarity_threshold:
            return (
                best_name,
                float(
                    np.clip(
                        best_similarity,
                        0.0,
                        1.0,
                    )
                ),
            )

        return (
            "Unknown",
            float(
                np.clip(
                    best_similarity,
                    0.0,
                    1.0,
                )
            ),
        )

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------

    def recognize(
        self,
        frame: NDArray[np.uint8],
        person_boxes: list[tuple[float, float, float, float]] | None = None,
    ) -> list[FaceDetection]:
        del person_boxes

        if frame is None or frame.size == 0:
            return []

        height, width = frame.shape[:2]

        faces = self.detect_faces(frame)

        results: list[FaceDetection] = []

        for face in faces:
            bbox = getattr(face, "bbox", None)

            if bbox is None or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(float, bbox)

            x1 = max(0.0, min(x1, float(width)))
            y1 = max(0.0, min(y1, float(height)))
            x2 = max(0.0, min(x2, float(width)))
            y2 = max(0.0, min(y2, float(height)))

            if x2 <= x1 or y2 <= y1:
                continue

            embedding = getattr(face, "normed_embedding", None)

            if embedding is None:
                embedding = getattr(face, "embedding", None)

            if embedding is None:
                continue

            embedding = self._normalize(
                np.asarray(
                    embedding,
                    dtype=np.float32,
                )
            )

            name, similarity = self._match_embedding(
                embedding
            )

            face_height = max(
                1.0,
                y2 - y1,
            )

            # Approximate distance estimate based on apparent face size.
            # This is only a rough relative estimate.
            focal = height / (
                2.0 * np.tan(np.radians(30.0))
            )

            distance = float(
                np.clip(
                    (focal * 0.22) / face_height,
                    0.3,
                    10.0,
                )
            )

            results.append(
                FaceDetection(
                    name=name,
                    confidence=similarity,
                    bbox=(
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                    distance_meters=distance,
                )
            )

        return results