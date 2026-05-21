"""
Gamification tracker for streaks and levels.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


@dataclass
class GamificationState:
    focus_streak_seconds: int = 0
    best_streak_seconds: int = 0
    total_focus_seconds: int = 0
    level: str = "Beginner"
    productivity_score: float = 0.0
    last_update_ts: float = 0.0


class GamificationTracker:
    """Tracks focus streaks and progress levels locally."""

    def __init__(self):
        self.path = os.path.join(DATA_DIR, "gamification.json")
        self.state = self._load()

    def _load(self) -> GamificationState:
        if not os.path.exists(self.path):
            return GamificationState()
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Legacy key support
            if "focus_streak" in data:
                data["focus_streak_seconds"] = int(data.get("focus_streak", 0)) * 60
            if "best_streak" in data:
                data["best_streak_seconds"] = int(data.get("best_streak", 0)) * 60
            if "total_focus_minutes" in data:
                data["total_focus_seconds"] = int(data.get("total_focus_minutes", 0)) * 60
            return GamificationState(**data)
        except Exception:
            return GamificationState()

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.state.__dict__, f, indent=2)

    def _level_for_minutes(self, minutes: int) -> str:
        if minutes >= 600:
            return "Elite"
        if minutes >= 300:
            return "Pro"
        if minutes >= 120:
            return "Intermediate"
        return "Beginner"

    def update(self, focus_score: float, risk_score: float) -> GamificationState:
        now = time.time()
        if not self.state.last_update_ts:
            self.state.last_update_ts = now
            return self.state

        elapsed = int(max(1, now - self.state.last_update_ts))
        self.state.last_update_ts = now

        if focus_score >= 70 and risk_score <= 50:
            self.state.focus_streak_seconds += elapsed
            self.state.total_focus_seconds += elapsed
        else:
            self.state.focus_streak_seconds = 0

        self.state.best_streak_seconds = max(self.state.best_streak_seconds, self.state.focus_streak_seconds)
        total_minutes = int(self.state.total_focus_seconds / 60)
        self.state.level = self._level_for_minutes(total_minutes)

        # Simple productivity score based on total focus minutes and streaks
        self.state.productivity_score = min(100.0, 20 + total_minutes * 0.2 + (self.state.best_streak_seconds / 60) * 0.5)

        self._save()
        return self.state

    def as_dict(self) -> Dict:
        data = self.state.__dict__.copy()
        data["focus_streak_minutes"] = round(self.state.focus_streak_seconds / 60, 1)
        data["best_streak_minutes"] = round(self.state.best_streak_seconds / 60, 1)
        data["total_focus_minutes"] = int(self.state.total_focus_seconds / 60)
        return data
