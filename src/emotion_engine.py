"""
Offline emotion and stress intelligence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class EmotionState:
    stress: float = 0.0
    frustration: float = 0.0
    boredom: float = 0.0
    emotional_fatigue: float = 0.0
    concentration_quality: float = 100.0
    recovery: float = 100.0
    wellness: float = 100.0


class EmotionEngine:
    """Heuristic offline emotion/stress model for local-only use."""

    def assess(self, metrics: Dict, posture_score: float = 70.0, face_confidence: float = 0.0) -> EmotionState:
        focus = float(metrics.get("focus_score", 0))
        drift = float(metrics.get("behavioral_drift", metrics.get("app_switch_rate", 0) * 10))
        fatigue_raw = float(metrics.get("fatigue_prob", 0))
        fatigue = fatigue_raw * 100.0 if fatigue_raw <= 1.0 else fatigue_raw
        cognitive = float(metrics.get("cognitive_load", 0))
        idle = float(metrics.get("idle_time", 0))
        posture_penalty = max(0.0, 100.0 - posture_score)

        stress = np.clip(0.35 * cognitive + 0.25 * drift + 0.2 * fatigue + 0.2 * posture_penalty, 0, 100)
        frustration = np.clip(0.4 * drift + 0.3 * posture_penalty + 0.3 * (100 - focus), 0, 100)
        boredom = np.clip(0.6 * idle + 0.2 * (100 - focus) + 0.2 * (50 - cognitive), 0, 100)
        emotional_fatigue = np.clip(0.5 * fatigue + 0.25 * stress + 0.25 * posture_penalty, 0, 100)
        concentration_quality = np.clip(100 - (0.45 * stress + 0.25 * frustration + 0.2 * boredom), 0, 100)
        recovery = np.clip(100 - emotional_fatigue + max(0.0, 30 - idle) * 0.5, 0, 100)
        wellness = np.clip(0.45 * concentration_quality + 0.35 * recovery + 0.2 * (100 - stress), 0, 100)

        if face_confidence and face_confidence < 50:
            wellness = max(0.0, wellness - 5.0)

        return EmotionState(
            stress=float(stress),
            frustration=float(frustration),
            boredom=float(boredom),
            emotional_fatigue=float(emotional_fatigue),
            concentration_quality=float(concentration_quality),
            recovery=float(recovery),
            wellness=float(wellness),
        )
