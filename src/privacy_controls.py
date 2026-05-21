"""
Privacy control helpers for local-only Mind-Shield+.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class PrivacySettings:
    webcam_enabled: bool = True
    emotion_ai_enabled: bool = True
    posture_ai_enabled: bool = True
    face_auth_enabled: bool = True
    paused: bool = False


class PrivacyController:
    """Local privacy toggles and user control state."""

    def __init__(self):
        self.settings = PrivacySettings()

    def toggle_webcam(self, enabled: bool) -> PrivacySettings:
        self.settings.webcam_enabled = enabled
        return self.settings

    def toggle_emotion_ai(self, enabled: bool) -> PrivacySettings:
        self.settings.emotion_ai_enabled = enabled
        return self.settings

    def toggle_posture_ai(self, enabled: bool) -> PrivacySettings:
        self.settings.posture_ai_enabled = enabled
        return self.settings

    def toggle_face_auth(self, enabled: bool) -> PrivacySettings:
        self.settings.face_auth_enabled = enabled
        return self.settings

    def pause(self, paused: bool) -> PrivacySettings:
        self.settings.paused = paused
        return self.settings

    def as_dict(self) -> Dict:
        return self.settings.__dict__.copy()
