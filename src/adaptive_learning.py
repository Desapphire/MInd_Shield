"""
Personalized adaptive learning for local baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class AdaptiveProfile:
    typing_speed_mean: float = 0.0
    typing_speed_std: float = 1.0
    focus_mean: float = 0.0
    focus_std: float = 1.0
    posture_mean: float = 70.0
    posture_std: float = 10.0
    samples: int = 0
    feedback_score: float = 0.0
    history: List[Dict] = field(default_factory=list)


class AdaptiveLearningEngine:
    """Tracks personal baselines and updates thresholds from feedback."""

    def __init__(self):
        self.profile = AdaptiveProfile()

    def update(self, metrics: Dict, feedback: Dict | None = None) -> AdaptiveProfile:
        focus = float(metrics.get("focus_score", 0))
        typing_speed = float(metrics.get("typing_speed_wpm", 0))
        posture = float(metrics.get("posture_score", 70))

        self.profile.history.append({"focus": focus, "typing_speed": typing_speed, "posture": posture})
        self.profile.samples += 1

        values = self.profile.history[-240:]
        if values:
            self.profile.focus_mean = float(np.mean([v["focus"] for v in values]))
            self.profile.focus_std = float(max(np.std([v["focus"] for v in values]), 1.0))
            self.profile.typing_speed_mean = float(np.mean([v["typing_speed"] for v in values]))
            self.profile.typing_speed_std = float(max(np.std([v["typing_speed"] for v in values]), 1.0))
            self.profile.posture_mean = float(np.mean([v["posture"] for v in values]))
            self.profile.posture_std = float(max(np.std([v["posture"] for v in values]), 1.0))

        if feedback:
            self.profile.feedback_score += float(feedback.get("score", 0.0))

        return self.profile

    def adaptive_thresholds(self) -> Dict[str, float]:
        return {
            "focus_low": max(35.0, self.profile.focus_mean - self.profile.focus_std),
            "fatigue_high": min(75.0, self.profile.focus_mean + self.profile.focus_std),
            "posture_low": max(50.0, self.profile.posture_mean - self.profile.posture_std),
        }
