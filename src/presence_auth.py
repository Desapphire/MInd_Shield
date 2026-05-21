"""
Local face authentication and presence intelligence.

Runs fully on-device using OpenCV face detection and encrypted local embeddings.
No raw video is stored. Only encrypted embeddings and derived metadata are kept.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - handled by GUI fallback
    cv2 = None

from security import LocalCipher


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


@dataclass
class PresenceResult:
    status: str  # AUTHORIZED | UNKNOWN | ABSENT | NEEDS_ENROLLMENT | LOW_QUALITY
    confidence: float
    security_score: float
    profile_name: str = ""
    message: str = ""
    face_visible: bool = False
    face_count: int = 0
    embedding_similarity: float = 0.0
    anti_spoof_score: float = 0.0
    timestamp: float = 0.0

    @property
    def is_authorized(self) -> bool:
        return self.status == "AUTHORIZED"


class LocalFaceAuthenticator:
    """On-device face authentication with encrypted local profile storage."""

    def __init__(self, profile_name: str = "Primary User", threshold: float = 0.82):
        self.profile_name = profile_name
        self.threshold = threshold
        self.soft_threshold = max(0.6, threshold - 0.08)
        self.min_anti_spoof = 20.0
        self.cipher = LocalCipher()
        self.profile_path = os.path.join(DATA_DIR, "face_profiles.enc")
        self.cascade = self._load_cascade()
        self.profiles = self._load_profiles()
        self.last_result = PresenceResult(
            status="NEEDS_ENROLLMENT" if not self.profiles else "ABSENT",
            confidence=0.0,
            security_score=0.0,
            message="Local face auth ready",
            timestamp=time.time(),
        )

    def _load_cascade(self):
        if cv2 is None:
            return None
        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        return cv2.CascadeClassifier(path)

    def _save_profiles(self) -> None:
        payload = {
            "profiles": [
                {
                    "name": item["name"],
                    "embedding": self.cipher.encrypt_bytes(item["embedding"].astype(np.float32).tobytes()).decode("utf-8"),
                    "shape": list(item["embedding"].shape),
                    "created_at": item.get("created_at", time.time()),
                }
                for item in self.profiles
            ]
        }
        with open(self.profile_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _load_profiles(self) -> List[Dict]:
        if not os.path.exists(self.profile_path):
            return []
        try:
            with open(self.profile_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            profiles = []
            for item in payload.get("profiles", []):
                raw = self.cipher.decrypt_bytes(item["embedding"].encode("utf-8"))
                embedding = np.frombuffer(raw, dtype=np.float32).reshape(item.get("shape", [1, -1]))
                profiles.append(
                    {
                        "name": item.get("name", "Primary User"),
                        "embedding": embedding.flatten(),
                        "created_at": item.get("created_at", time.time()),
                    }
                )
            return profiles
        except Exception:
            return []

    def _largest_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if cv2 is None or self.cascade is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = []
        settings = [
            (1.05, 3, (40, 40)),
            (1.1, 3, (50, 50)),
        ]
        for scale_factor, min_neighbors, min_size in settings:
            detected = self.cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size,
            )
            if len(detected) > 0:
                faces.extend(list(detected))
        if not faces:
            return None
        return max(faces, key=lambda rect: rect[2] * rect[3])

    def _embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        equalized = cv2.equalizeHist(resized)
        vector = equalized.astype(np.float32).flatten()
        vector -= vector.mean()
        norm = np.linalg.norm(vector) or 1.0
        return vector / norm

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a = a.flatten()
        b = b.flatten()
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        return float(np.dot(a, b) / denom)

    def _anti_spoof_score(self, face_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = float(np.mean(gray))
        face_area = face_bgr.shape[0] * face_bgr.shape[1]
        quality = 0.0
        quality += min(blur / 150.0, 1.0) * 40.0
        quality += min(max(brightness, 25) / 200.0, 1.0) * 30.0
        quality += min(face_area / (220 * 220), 1.0) * 30.0
        return max(0.0, min(100.0, quality))

    def enroll(self, name: Optional[str], frame: np.ndarray) -> Tuple[bool, str]:
        if cv2 is None:
            return False, "OpenCV not available"
        rect = self._largest_face(frame)
        if rect is None:
            return False, "No face detected for enrollment"
        x, y, w, h = rect
        face = frame[max(0, y):y + h, max(0, x):x + w]
        if face.size == 0:
            return False, "Face crop failed"
        embedding = self._embedding(face)
        profile_name = name.strip() if name and name.strip() else self.profile_name

        existing = next((p for p in self.profiles if p["name"].lower() == profile_name.lower()), None)
        if existing is not None:
            existing["embedding"] = (existing["embedding"] * 0.7 + embedding * 0.3)
            existing["created_at"] = time.time()
        else:
            self.profiles.append({"name": profile_name, "embedding": embedding, "created_at": time.time()})

        self._save_profiles()
        return True, f"Enrolled locally as {profile_name}"

    def delete_profile(self, name: str) -> bool:
        before = len(self.profiles)
        self.profiles = [p for p in self.profiles if p["name"].lower() != name.lower()]
        if len(self.profiles) != before:
            self._save_profiles()
            return True
        return False

    def delete_all_profiles(self) -> None:
        self.profiles = []
        if os.path.exists(self.profile_path):
            os.remove(self.profile_path)

    @property
    def enrolled_profiles(self) -> List[str]:
        return [p["name"] for p in self.profiles]

    def list_enrolled_profiles(self) -> List[str]:
        """Compatibility alias for older callers."""
        return self.enrolled_profiles

    def verify_frame(self, frame: Optional[np.ndarray]) -> PresenceResult:
        now = time.time()
        if frame is None:
            result = PresenceResult(
                status="ABSENT",
                confidence=0.0,
                security_score=0.0,
                message="No webcam frame",
                timestamp=now,
            )
            self.last_result = result
            return result

        if cv2 is None or self.cascade is None:
            result = PresenceResult(
                status="LOW_QUALITY",
                confidence=0.0,
                security_score=0.0,
                message="Face auth unavailable",
                timestamp=now,
            )
            self.last_result = result
            return result

        rect = self._largest_face(frame)
        if rect is None:
            result = PresenceResult(
                status="ABSENT",
                confidence=0.0,
                security_score=0.0,
                message="Face not visible",
                timestamp=now,
            )
            self.last_result = result
            return result

        x, y, w, h = rect
        face = frame[max(0, y):y + h, max(0, x):x + w]
        if face.size == 0:
            result = PresenceResult(
                status="LOW_QUALITY",
                confidence=0.0,
                security_score=0.0,
                message="Face crop failed",
                face_visible=True,
                face_count=1,
                timestamp=now,
            )
            self.last_result = result
            return result

        current_embedding = self._embedding(face)
        anti_spoof = self._anti_spoof_score(face)

        if not self.profiles:
            result = PresenceResult(
                status="NEEDS_ENROLLMENT",
                confidence=0.0,
                security_score=anti_spoof,
                message="Enroll a local face profile",
                face_visible=True,
                face_count=1,
                anti_spoof_score=anti_spoof,
                timestamp=now,
            )
            self.last_result = result
            return result

        best_name = ""
        best_similarity = -1.0
        for profile in self.profiles:
            similarity = self._cosine_similarity(current_embedding, profile["embedding"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_name = profile["name"]

        confidence = max(0.0, min(100.0, best_similarity * 100.0))
        security_score = max(0.0, min(100.0, confidence * 0.7 + anti_spoof * 0.3))

        is_match = best_similarity >= self.threshold
        is_soft_match = best_similarity >= self.soft_threshold and anti_spoof >= self.min_anti_spoof
        if is_match or is_soft_match:
            result = PresenceResult(
                status="AUTHORIZED",
                confidence=confidence,
                security_score=security_score,
                profile_name=best_name,
                message="Authorized user detected",
                face_visible=True,
                face_count=1,
                embedding_similarity=best_similarity,
                anti_spoof_score=anti_spoof,
                timestamp=now,
            )
        elif anti_spoof < 15.0:
            result = PresenceResult(
                status="LOW_QUALITY",
                confidence=confidence,
                security_score=security_score,
                profile_name=best_name,
                message="Face quality too low",
                face_visible=True,
                face_count=1,
                embedding_similarity=best_similarity,
                anti_spoof_score=anti_spoof,
                timestamp=now,
            )
        else:
            result = PresenceResult(
                status="UNKNOWN",
                confidence=confidence,
                security_score=security_score,
                profile_name=best_name,
                message="Unknown face detected",
                face_visible=True,
                face_count=1,
                embedding_similarity=best_similarity,
                anti_spoof_score=anti_spoof,
                timestamp=now,
            )

        self.last_result = result
        return result
