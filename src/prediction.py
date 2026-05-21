"""
Lightweight prediction utilities for short-term focus/fatigue trends.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def _linear_forecast(values: List[float], steps: int) -> float:
    if len(values) < 3:
        return float(values[-1]) if values else 0.0

    y = np.array(values, dtype=float)
    x = np.arange(len(y), dtype=float)
    # Linear regression fit
    slope, intercept = np.polyfit(x, y, 1)
    future_x = len(y) - 1 + steps
    return float(intercept + slope * future_x)


def forecast_metrics(history: List[Dict], keys: List[str], seconds_ahead: int = 300) -> Dict[str, float]:
    """Forecast metrics for the next time window using recent history."""
    if not history:
        return {key: 0.0 for key in keys}

    # Use last 30 points (approx 30s if 1s cadence)
    window = history[-30:]
    result: Dict[str, float] = {}
    for key in keys:
        series = [float(item.get(key, 0.0)) for item in window]
        # Convert seconds_ahead to steps based on 1s cadence
        steps = max(int(seconds_ahead), 1)
        result[key] = max(0.0, min(100.0, _linear_forecast(series, steps)))
    return result
