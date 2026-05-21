"""
Hybrid decision engine for recommendations and alerts.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def generate_recommendations(analysis: Dict, metrics: Dict) -> Tuple[List[str], List[str]]:
    """Return (recommendations, reasons)."""
    recs: List[str] = []
    reasons: List[str] = []

    fatigue = float(analysis.get("fatigue_prob", 0))
    focus = float(analysis.get("focus_score", 0))
    drift = float(analysis.get("behavioral_drift", 0))
    cognitive = float(analysis.get("cognitive_load", 0))
    risk = float(analysis.get("risk_score", 0))
    trend = analysis.get("trend", "stable")
    idle = float(metrics.get("idle_time", 0))

    # Rule-based triggers
    if fatigue > 40 and focus < 50:
        recs.append("😴 Take a 5-minute reset break")
        reasons.append("Fatigue high and focus low")

    if drift > 55 or trend == "worsening":
        recs.append("🚨 Behavior drift rising: reduce multitasking")
        reasons.append("Behavioral drift increasing")

    if cognitive > 65:
        recs.append("🧠 Break tasks into smaller steps")
        reasons.append("Cognitive load elevated")

    if idle > 120 and risk > 50:
        recs.append("🔄 Reset posture + hydrate")
        reasons.append("Idle time with high risk")

    # ML-weighted suggestion
    if risk >= 75:
        recs.append("🚨 Critical risk: step away for 10 minutes")
        reasons.append("Risk score critical")
    elif risk >= 55:
        recs.append("⚠️ High risk: schedule a short break soon")
        reasons.append("Risk score high")

    if not recs:
        recs.append("✅ Stable state: keep focus for 10 more minutes")
        reasons.append("Signals stable")

    return recs[:3], reasons[:3]
