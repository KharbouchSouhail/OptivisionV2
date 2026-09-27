"""Face detection, embedding extraction, and recognition."""

from __future__ import annotations

import logging
import os
import pickle
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
import torchvision.models as models

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
    """Detects and recognizes faces using deep embeddings and cosine similarity."""

    def __init__(
        self,
        db_path: str | None = None,
        device: str | None = None,
        similarity_threshold: float | None = None,
    ) -> None:
        self.db_path = db_path or os.getenv("VISIONNAIRE_FACE_DB", DEFAULT_DB_PATH)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else float(os.getenv("VISIONNAIRE_FACE_THRESHOLD", "0.65"))
        )

        # Build embedding extractor using MobileNetV3 feature backbone
        self._model = models.mobilenet_v3_small(weights=None)
        self._model.to(self.device)
        self._model.eval()

        self.db: dict[str, list[np.ndarray]] = {}
        self.load_db()

    def load_db(self, path: str | None = None) -> None:
        """Load embeddings database from disk."""
        target_path = path or self.db_path
        if os.path.isfile(target_path) and os.path.getsize(target_path) > 0:
            try:
                with open(target_path, "rb") as f:
                    loaded = pickle.load(f)
                    if isinstance(loaded, dict):
                        # Ensure lists of numpy arrays
                        self.db = {
                            k: [np.asarray(arr, dtype=np.float32) for arr in (v if isinstance(v, list) else [v])]
                            for k, v in loaded.items()
                        }
                        logger.info("Loaded %d registered faces from %s", len(self.db), target_path)
                        return
            except Exception:
                logger.exception("Failed to load face database from %s", target_path)
        self.db = {}

    def save_db(self, path: str | None = None) -> None:
        """Persist embeddings database to disk."""
        target_path = path or self.db_path
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "wb") as f:
            pickle.dump(self.db, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved %d registered faces to %s", len(self.db), target_path)

    def extract_embedding(self, face_bgr: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Extract a 576-D normalized embedding vector from a face crop."""
        if face_bgr is None or face_bgr.size == 0:
            return np.zeros(576, dtype=np.float32)

        # Preprocess crop: resize to 112x112, BGR->RGB, normalized tensor
        resized = cv2.resize(face_bgr, (112, 112))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        # Standard ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.to(self.device)

        with torch.no_grad():
            feat = self._model.features(tensor)
            pool = torch.nn.functional.adaptive_avg_pool2d(feat, 1).flatten(1)
            norm = torch.nn.functional.normalize(pool, p=2, dim=1)
            embedding = norm.squeeze(0).cpu().numpy().astype(np.float32)

        return embedding

    def register_face(self, name: str, face_bgr: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Register a new face embedding under `name` and persist to disk."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Face name cannot be empty")
        embedding = self.extract_embedding(face_bgr)
        if clean_name not in self.db:
            self.db[clean_name] = []
        self.db[clean_name].append(embedding)
        self.save_db()
        return embedding

    def delete_face(self, name: str) -> bool:
        """Delete all embeddings for a person."""
        if name in self.db:
            del self.db[name]
            self.save_db()
            return True
        return False

    def list_faces(self) -> list[str]:
        """List all registered names."""
        return sorted(self.db.keys())

    def _match_embedding(self, embedding: NDArray[np.float32]) -> tuple[str, float]:
        """Find the closest known face in the database using cosine similarity."""
        if not self.db or embedding is None or np.all(embedding == 0):
            return "Unknown", 0.0

        best_name = "Unknown"
        best_sim = -1.0

        for name, emb_list in self.db.items():
            for known_emb in emb_list:
                sim = float(np.dot(embedding, known_emb))
                if sim > best_sim:
                    best_sim = sim
                    best_name = name

        if best_sim >= self.similarity_threshold:
            return best_name, max(0.0, min(1.0, best_sim))
        return "Unknown", max(0.0, min(1.0, best_sim))

    def detect_face_regions(
        self,
        frame: NDArray[np.uint8],
        person_boxes: list[tuple[float, float, float, float]] | None = None,
    ) -> list[tuple[float, float, float, float]]:
        """Detect candidate face bounding boxes.

        If person_boxes are provided from YOLO, the upper 30% region is used.
        Otherwise, face regions are located using multi-scale skin/contrast cues.
        """
        h, w = frame.shape[:2]
        candidates: list[tuple[float, float, float, float]] = []

        if person_boxes:
            for x1, y1, x2, y2 in person_boxes:
                bw = x2 - x1
                bh = y2 - y1
                if bw < 10 or bh < 20:
                    continue
                # Upper 30% of person bounding box represents the head/face region
                head_x1 = max(0.0, x1 + 0.15 * bw)
                head_x2 = min(float(w), x2 - 0.15 * bw)
                head_y1 = max(0.0, y1)
                head_y2 = min(float(h), y1 + 0.32 * bh)
                if (head_x2 - head_x1) >= 8 and (head_y2 - head_y1) >= 8:
                    candidates.append((head_x1, head_y1, head_x2, head_y2))

        if not candidates:
            # Fallback: skin-color thresholding in YCrCb color space
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = (h * w) * 0.01
            for cnt in contours:
                if cv2.contourArea(cnt) > min_area:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    aspect = ch / float(cw) if cw > 0 else 0
                    if 0.8 <= aspect <= 2.2:
                        candidates.append((float(x), float(y), float(x + cw), float(y + ch)))

        return candidates

    def recognize(
        self,
        frame: NDArray[np.uint8],
        person_boxes: list[tuple[float, float, float, float]] | None = None,
    ) -> list[FaceDetection]:
        """Detect and recognize all faces in the frame."""
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        regions = self.detect_face_regions(frame, person_boxes=person_boxes)
        results: list[FaceDetection] = []

        for x1, y1, x2, y2 in regions:
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            face_crop = frame[iy1:iy2, ix1:ix2]
            if face_crop.size == 0:
                continue

            embedding = self.extract_embedding(face_crop)
            name, similarity = self._match_embedding(embedding)

            # Estimate approximate distance from face box height
            focal = (h / (2.0 * np.tan(np.radians(30.0))))
            face_h = max(1.0, float(iy2 - iy1))
            dist = float(np.clip((focal * 0.22) / face_h, 0.3, 10.0))

            results.append(
                FaceDetection(
                    name=name,
                    confidence=similarity,
                    bbox=(x1, y1, x2, y2),
                    distance_meters=dist,
                )
            )

        return results
