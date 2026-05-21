"""
Alert explainability helpers.
"""

from __future__ import annotations

from typing import Dict, List


def _fmt_delta(value: float, unit: str) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value):.1f}{unit}"


def build_alert_explanations(baseline: Dict[str, float], metrics: Dict) -> List[str]:
    """Return top 3 drivers based on deviation from baseline."""
    if not baseline:
        return [
            "Baseline building... keep working for a more accurate alert",
            "Typing consistency and error rate are being learned",
            "App switching trends will appear after calibration",
        ]

    drivers = []

    # Error rate: convert to percentage points.
    base_error = float(baseline.get("error_rate", 0))
    cur_error = float(metrics.get("error_rate", 0))
    error_pp = (cur_error - base_error) * 100.0
    drivers.append((abs(error_pp), f"Error rate {_fmt_delta(error_pp, '%')} vs baseline"))

    # Typing speed: WPM delta.
    base_wpm = float(baseline.get("typing_speed_wpm", 0))
    cur_wpm = float(metrics.get("typing_speed_wpm", 0))
    wpm_delta = cur_wpm - base_wpm
    drivers.append((abs(wpm_delta), f"Typing speed {_fmt_delta(wpm_delta, ' WPM')} vs usual"))

    # App switching: switches per minute.
    base_switch = float(baseline.get("app_switch_rate", 0))
    cur_switch = float(metrics.get("app_switch_rate", 0))
    switch_delta = cur_switch - base_switch
    drivers.append((abs(switch_delta), f"App switching {_fmt_delta(switch_delta, '/min')} vs baseline"))

    drivers.sort(key=lambda d: d[0], reverse=True)
    return [d[1] for d in drivers[:3]]
