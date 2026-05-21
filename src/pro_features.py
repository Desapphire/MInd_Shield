"""
Mind-Shield+ Pro Features

Adds production-style capabilities:
- Per-user baseline calibration and personalization
- Prediction confidence estimation
- Smart intervention recommendations
- Local SQLite persistence for analytics history
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class PersonalBaselineManager:
    """Tracks user-specific behavior baseline and adapts scores accordingly."""

    def __init__(self, calibration_samples: int = 120):
        self.calibration_samples = calibration_samples
        self.profile_path = os.path.join(DATA_DIR, "baseline_profile.json")
        self.samples: List[Dict] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.profile_path):
            return
        try:
            with open(self.profile_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.samples = payload.get("samples", [])[-1200:]
        except Exception:
            self.samples = []

    def _save(self) -> None:
        payload = {
            "updated_at": datetime.now().isoformat(),
            "samples": self.samples[-1200:],
        }
        with open(self.profile_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def update(self, metrics: Dict) -> None:
        sample = {
            "timestamp": time.time(),
            "typing_speed_wpm": float(metrics.get("typing_speed_wpm", 0)),
            "typing_speed_variance": float(metrics.get("typing_speed_variance", 0)),
            "error_rate": float(metrics.get("error_rate", 0)),
            "app_switch_rate": float(metrics.get("app_switch_rate", 0)),
            "recent_keys": float(metrics.get("recent_keys", 0)),
        }
        self.samples.append(sample)
        if len(self.samples) > 1200:
            self.samples = self.samples[-1200:]

        # Save periodically to avoid excess IO.
        if len(self.samples) % 30 == 0:
            self._save()

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def is_calibrated(self) -> bool:
        return len(self.samples) >= self.calibration_samples

    def _stats(self) -> Dict[str, Tuple[float, float]]:
        if not self.samples:
            return {}
        keys = [
            "typing_speed_wpm",
            "typing_speed_variance",
            "error_rate",
            "app_switch_rate",
            "recent_keys",
        ]
        stats = {}
        for key in keys:
            values = [float(s.get(key, 0.0)) for s in self.samples]
            mu = float(np.mean(values))
            sigma = float(np.std(values))
            stats[key] = (mu, max(sigma, 1e-6))
        return stats

    def get_baseline_stats(self) -> Dict[str, float]:
        """Return baseline means for explainability features."""
        stats = self._stats()
        if not stats:
            return {}
        return {key: stats[key][0] for key in stats}

    def _z(self, value: float, mu: float, sigma: float) -> float:
        return (value - mu) / max(sigma, 1e-6)

    def adapt_analysis(self, analysis: Dict, metrics: Dict) -> Dict:
        """Inject personalized deviation and confidence, and adjust selected scores."""
        updated = dict(analysis)

        if not self.samples:
            updated["confidence"] = 45.0
            updated["personalized_deviation"] = 0.0
            updated["calibration_progress"] = 0
            return updated

        stats = self._stats()
        z_wpm = self._z(float(metrics.get("typing_speed_wpm", 0)), *stats["typing_speed_wpm"])
        z_var = self._z(float(metrics.get("typing_speed_variance", 0)), *stats["typing_speed_variance"])
        z_err = self._z(float(metrics.get("error_rate", 0)), *stats["error_rate"])
        z_switch = self._z(float(metrics.get("app_switch_rate", 0)), *stats["app_switch_rate"])

        # Deviation uses absolute z-score with robust clipping for stability.
        avg_abs_z = float(np.mean([
            min(abs(z_wpm), 4.0),
            min(abs(z_var), 4.0),
            min(abs(z_err), 4.0),
            min(abs(z_switch), 4.0),
        ]))

        personalized_deviation = _clamp((avg_abs_z / 4.0) * 100.0, 0.0, 100.0)

        data_factor = _clamp(self.sample_count / max(self.calibration_samples, 1), 0.25, 1.0)
        stability_factor = _clamp(1.0 - (avg_abs_z / 5.0), 0.3, 1.0)
        confidence = _clamp(100.0 * data_factor * stability_factor, 25.0, 95.0)

        # Personalization reduces false positives for naturally slower/faster typists.
        if self.is_calibrated:
            speed_penalty = 0.0
            if z_wpm < -1.0:
                speed_penalty = min(abs(z_wpm) * 2.5, 8.0)
            elif z_wpm > 1.5:
                speed_penalty = -min(abs(z_wpm) * 1.5, 5.0)

            updated["cognitive_load"] = round(_clamp(float(updated.get("cognitive_load", 0)) + speed_penalty, 5.0, 95.0), 1)
            updated["risk_score"] = round(_clamp(float(updated.get("risk_score", 0)) + speed_penalty * 0.6, 5.0, 95.0), 1)

        updated["confidence"] = round(confidence, 1)
        updated["personalized_deviation"] = round(personalized_deviation, 1)
        updated["calibration_progress"] = int(_clamp((self.sample_count / max(self.calibration_samples, 1)) * 100, 0, 100))
        return updated


@dataclass
class Intervention:
    icon: str
    message: str


class InterventionEngine:
    """Generates actionable interventions from live state."""

    def recommend(self, analysis: Dict, metrics: Dict, focus_active: bool = False, posture_enabled: bool = False) -> List[Intervention]:
        out: List[Intervention] = []
        risk = float(analysis.get("risk_score", 0))
        fatigue = float(analysis.get("fatigue_prob", 0))
        cognitive = float(analysis.get("cognitive_load", 0))
        focus = float(analysis.get("focus_score", 0))
        idle = float(metrics.get("idle_time", 0))
        switches = float(metrics.get("app_switch_rate", 0))
        hours = float(metrics.get("session_duration", 0)) / 3600.0

        if risk >= 75:
            out.append(Intervention("🚨", "Critical risk: take a 10-minute break away from screen"))
        elif risk >= 55:
            out.append(Intervention("⚠️", "High risk: run a 3-minute reset (water + breathing)"))

        if fatigue >= 70:
            out.append(Intervention("😴", "Eye recovery: follow 20-20-20 for 2 minutes"))

        if cognitive >= 65 and switches >= 2:
            out.append(Intervention("🧠", "Reduce context switching: close non-task tabs for 15 minutes"))

        if focus < 45 and not focus_active:
            out.append(Intervention("🎯", "Start Focus Mode (25 min) to stabilize attention"))

        if hours >= 1.5:
            out.append(Intervention("⏱️", "Long session detected: schedule a micro-walk now"))

        if idle > 30 and risk >= 45:
            out.append(Intervention("🔄", "Low activity + high strain: do a short posture reset"))

        if not posture_enabled:
            out.append(Intervention("🧘", "Enable posture monitoring to improve physical strain detection"))

        if not out:
            out.append(Intervention("✅", "Stable state: continue current task, check again in 10 minutes"))

        return out[:3]


class SessionStore:
    """Persists analytics snapshots and provides lightweight daily stats."""

    def __init__(self):
        self.db_path = os.path.join(DATA_DIR, "mindshield.db")
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    risk_score REAL,
                    fatigue_prob REAL,
                    cognitive_load REAL,
                    focus_score REAL,
                    productivity REAL,
                    confidence REAL,
                    current_app TEXT,
                    typing_speed_wpm REAL,
                    error_rate REAL,
                    app_switch_rate REAL,
                    recent_keys INTEGER,
                    idle_time REAL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def save_analysis(self, analysis: Dict, metrics: Dict) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO analysis_events (
                    ts, risk_score, fatigue_prob, cognitive_load, focus_score, productivity,
                    confidence, current_app, typing_speed_wpm, error_rate, app_switch_rate,
                    recent_keys, idle_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(timespec="seconds"),
                    float(analysis.get("risk_score", 0)),
                    float(analysis.get("fatigue_prob", 0)),
                    float(analysis.get("cognitive_load", 0)),
                    float(analysis.get("focus_score", 0)),
                    float(analysis.get("productivity", 0)),
                    float(analysis.get("confidence", 0)),
                    str(metrics.get("current_app", ""))[:120],
                    float(metrics.get("typing_speed_wpm", 0)),
                    float(metrics.get("error_rate", 0)),
                    float(metrics.get("app_switch_rate", 0)),
                    int(metrics.get("recent_keys", 0)),
                    float(metrics.get("idle_time", 0)),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_today_summary(self) -> Dict:
        today_prefix = datetime.now().strftime("%Y-%m-%d")
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    COUNT(*),
                    AVG(risk_score),
                    MAX(risk_score),
                    AVG(focus_score),
                    AVG(confidence)
                FROM analysis_events
                WHERE ts LIKE ?
                """,
                (f"{today_prefix}%",),
            )
            row = cur.fetchone() or (0, 0, 0, 0, 0)
            return {
                "samples": int(row[0] or 0),
                "avg_risk": float(row[1] or 0.0),
                "peak_risk": float(row[2] or 0.0),
                "avg_focus": float(row[3] or 0.0),
                "avg_confidence": float(row[4] or 0.0),
            }
        finally:
            conn.close()

    def get_best_focus_hour(self) -> str:
        """Return the hour (HH) with the highest average focus score today."""
        today_prefix = datetime.now().strftime("%Y-%m-%d")
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT SUBSTR(ts, 12, 2) AS hour, AVG(focus_score) AS avg_focus
                FROM analysis_events
                WHERE ts LIKE ?
                GROUP BY hour
                ORDER BY avg_focus DESC
                LIMIT 1
                """,
                (f"{today_prefix}%",),
            )
            row = cur.fetchone()
            return row[0] if row else "--"
        finally:
            conn.close()
