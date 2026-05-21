"""
Long-term burnout prediction and recovery intelligence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class BurnoutState:
    burnout_risk: float = 0.0
    chronic_stress: float = 0.0
    recovery_quality: float = 100.0
    sustainability_score: float = 100.0
    trend: str = "stable"


class BurnoutEngine:
    def forecast(self, history: List[Dict]) -> BurnoutState:
        if not history:
            return BurnoutState()

        recent = history[-120:]
        risk = np.mean([float(x.get("risk_score", 0)) for x in recent])
        stress = np.mean([float(x.get("stress", x.get("cognitive_load", 0))) for x in recent])
        recovery = np.mean([float(x.get("recovery", 100)) for x in recent])
        focus = np.mean([float(x.get("focus_score", 0)) for x in recent])

        burnout_risk = np.clip(0.35 * risk + 0.35 * stress + 0.2 * (100 - recovery) + 0.1 * (100 - focus), 0, 100)
        chronic_stress = np.clip(stress, 0, 100)
        recovery_quality = np.clip(recovery, 0, 100)
        sustainability_score = np.clip(100 - burnout_risk + recovery_quality * 0.15, 0, 100)

        if len(recent) >= 10:
            first = np.mean([float(x.get("risk_score", 0)) for x in recent[: len(recent)//2]])
            second = np.mean([float(x.get("risk_score", 0)) for x in recent[len(recent)//2:]])
            if second - first > 5:
                trend = "worsening"
            elif second - first < -5:
                trend = "recovering"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return BurnoutState(
            burnout_risk=float(burnout_risk),
            chronic_stress=float(chronic_stress),
            recovery_quality=float(recovery_quality),
            sustainability_score=float(sustainability_score),
            trend=trend,
        )
