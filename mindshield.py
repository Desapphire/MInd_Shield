"""
Mind-Shield+ Desktop Monitor v2.2
=================================
With integrated Focus AI Mode - No server required!
"""

import sys
import time
import threading
import json
import os
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Optional
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import math
import winsound  # For alert sounds on Windows

# System monitoring
try:
    from pynput import keyboard, mouse
    from pynput.keyboard import Key
except ImportError:
    print("Please install pynput: pip install pynput")
    sys.exit(1)

try:
    import psutil
except ImportError:
    print("Please install psutil: pip install psutil")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Please install numpy: pip install numpy")
    sys.exit(1)

# Import our ML modules
sys.path.insert(0, 'd:/EDI_Project/src')
try:
    from fatigue_model import FatiguePredictionModel
    from anomaly_detection import BehavioralDriftDetector
    from pro_features import PersonalBaselineManager, InterventionEngine, SessionStore
    from alert_explain import build_alert_explanations
    from decision_engine import generate_recommendations
    from prediction import forecast_metrics
    from gamification import GamificationTracker
    from adaptive_learning import AdaptiveLearningEngine
    from emotion_engine import EmotionEngine
    from burnout_engine import BurnoutEngine
    from local_storage import LocalStorage
    from privacy_controls import PrivacyController
    from event_bus import EventBus
except ImportError as e:
    print(f"Error importing ML modules: {e}")
    sys.exit(1)

# Posture detection (optional - requires mediapipe and opencv)
try:
    from posture_detection import PostureDetector, MEDIAPIPE_AVAILABLE, CV2_AVAILABLE
    import cv2
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
    from presence_auth import LocalFaceAuthenticator
except (ImportError, AttributeError, Exception) as e:
    print(f"Posture detection not available: {e}")
    MEDIAPIPE_AVAILABLE = False
    CV2_AVAILABLE = False
    PIL_AVAILABLE = False
    PostureDetector = None
    LocalFaceAuthenticator = None


# ============================================================================
# FOCUS AI MODE - Distracting Sites
# ============================================================================

DISTRACTING_SITES = [
    'youtube', 'instagram', 'facebook', 'twitter', 'reddit',
    'tiktok', 'netflix', 'twitch', 'discord', 'snapchat',
    'pinterest', 'tumblr', 'whatsapp', 'telegram', 'spotify',
    'steam', 'epic games', 'valorant', 'minecraft', 'fortnite',
    'games', 'gaming', 'manga', 'anime'
]


class FocusSession:
    """Tracks Focus Mode sessions for productivity enhancement."""
    
    def __init__(self):
        self.is_active = False
        self.start_time = None
        self.duration_minutes = 25
        self.end_time = None
        
        # Focus metrics
        self.distraction_count = 0
        self.distraction_log = []
        self.focus_score = 100
        
        # Current distraction tracking
        self.current_distraction = None
        self.distraction_start = None
        self.last_warning_time = None
        
        # History of completed sessions
        self.completed_sessions = []
        self.total_focus_time = 0
        
    def start(self, duration_minutes: int = 25) -> tuple:
        """Start a new focus session."""
        if self.is_active:
            return False, "Focus mode already active"
        
        self.is_active = True
        self.start_time = time.time()
        self.duration_minutes = duration_minutes
        self.end_time = self.start_time + (duration_minutes * 60)
        
        # Reset metrics
        self.distraction_count = 0
        self.distraction_log = []
        self.focus_score = 100
        self.current_distraction = None
        self.distraction_start = None
        
        return True, f"Focus mode started for {duration_minutes} minutes"
    
    def stop(self) -> tuple:
        """Stop the current focus session."""
        if not self.is_active:
            return False, "Focus mode not active"
        
        # Calculate actual focus time
        elapsed = time.time() - self.start_time
        distraction_time = sum(d.get('duration', 0) for d in self.distraction_log)
        actual_focus_time = max(0, elapsed - distraction_time)
        
        # Save session
        self.completed_sessions.append({
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time': datetime.now().isoformat(),
            'planned_duration': self.duration_minutes * 60,
            'actual_focus_time': actual_focus_time,
            'distraction_count': self.distraction_count,
            'focus_score': self.focus_score,
        })
        
        self.total_focus_time += actual_focus_time
        self.is_active = False
        
        return True, f"Session complete! Focus score: {self.focus_score}"
    
    def check_expiry(self) -> bool:
        """Check if session has expired."""
        if self.is_active and self.end_time:
            if time.time() >= self.end_time:
                self.stop()
                return True
        return False
    
    def get_remaining_time(self) -> tuple:
        """Get remaining time as (minutes, seconds)."""
        if not self.is_active:
            return (self.duration_minutes, 0)
        
        remaining = max(0, self.end_time - time.time())
        return (int(remaining // 60), int(remaining % 60))
    
    def get_remaining_formatted(self) -> str:
        """Get remaining time as MM:SS string."""
        m, s = self.get_remaining_time()
        return f"{m:02d}:{s:02d}"
    
    def check_window_distraction(self, window_title: str) -> dict:
        """Check if current window is a distraction."""
        if not self.is_active:
            return {'is_distraction': False}
        
        # Check expiry
        if self.check_expiry():
            return {'is_distraction': False, 'session_ended': True}
        
        window_lower = window_title.lower()
        
        # Check if window contains any distracting site name
        is_distracting = any(site in window_lower for site in DISTRACTING_SITES)
        
        if is_distracting:
            now = time.time()
            
            # New distraction
            if self.current_distraction != window_title:
                # Log previous distraction
                if self.current_distraction and self.distraction_start:
                    duration = now - self.distraction_start
                    self.distraction_log.append({
                        'timestamp': datetime.fromtimestamp(self.distraction_start).strftime('%H:%M:%S'),
                        'site': self.current_distraction[:30],
                        'duration': duration
                    })
                
                # Start tracking new distraction
                self.current_distraction = window_title
                self.distraction_start = now
                self.distraction_count += 1
                
                # Reduce focus score
                self.focus_score = max(0, self.focus_score - min(10, 5 + self.distraction_count))
                
                # Should show warning?
                show_warning = True
                if self.last_warning_time:
                    # Don't spam warnings - wait at least 5 seconds
                    show_warning = (now - self.last_warning_time) > 5
                
                if show_warning:
                    self.last_warning_time = now
                
                return {
                    'is_distraction': True,
                    'site': window_title[:30],
                    'show_warning': show_warning,
                    'distraction_count': self.distraction_count,
                    'focus_score': self.focus_score
                }
            else:
                # Ongoing distraction
                duration = now - self.distraction_start
                return {
                    'is_distraction': True,
                    'ongoing': True,
                    'duration': duration
                }
        else:
            # Not distracting - close out any previous distraction
            if self.current_distraction and self.distraction_start:
                duration = time.time() - self.distraction_start
                self.distraction_log.append({
                    'timestamp': datetime.fromtimestamp(self.distraction_start).strftime('%H:%M:%S'),
                    'site': self.current_distraction[:30],
                    'duration': duration
                })
                self.current_distraction = None
                self.distraction_start = None
            
            return {'is_distraction': False}
    
    def get_status(self) -> dict:
        """Get current focus status."""
        self.check_expiry()
        
        if not self.is_active:
            return {
                'is_active': False,
                'total_focus_time': self.total_focus_time,
                'completed_sessions': len(self.completed_sessions)
            }
        
        elapsed = time.time() - self.start_time
        distraction_time = sum(d.get('duration', 0) for d in self.distraction_log)
        if self.current_distraction and self.distraction_start:
            distraction_time += time.time() - self.distraction_start
        
        productivity = max(0, 100 - (distraction_time / max(elapsed, 1) * 100))
        
        return {
            'is_active': True,
            'remaining': self.get_remaining_formatted(),
            'focus_score': self.focus_score,
            'productivity': round(productivity, 1),
            'distraction_count': self.distraction_count,
            'distractions': self.distraction_log[-5:]
        }


class SystemBehaviorCapture:
    """Captures system-wide keyboard, mouse, and application behavior."""
    
    def __init__(self):
        self.running = False
        
        # Keyboard metrics
        self.keystrokes = deque(maxlen=500)
        self.key_intervals = deque(maxlen=200)
        self.last_key_time = None
        self.error_keys = 0
        self.total_keys = 0
        self.word_count = 0
        self.char_since_space = 0
        
        # Mouse metrics
        self.mouse_clicks = deque(maxlen=200)
        self.mouse_distances = deque(maxlen=100)
        self.last_mouse_pos = None
        self.total_mouse_distance = 0
        
        # Scroll metrics
        self.scroll_events = deque(maxlen=100)
        self.total_scroll = 0
        
        # Application tracking
        self.app_switches = deque(maxlen=50)
        self.last_active_window = None
        self.current_app = "Unknown"
        
        # Timing
        self.start_time = None
        self.last_activity = None
        
        # Listeners
        self.keyboard_listener = None
        self.mouse_listener = None
        
    def on_key_press(self, key):
        """Handle keyboard events."""
        current_time = time.time()
        self.last_activity = current_time
        
        # Track key intervals
        if self.last_key_time:
            interval = current_time - self.last_key_time
            if 0.01 < interval < 3.0:  # Valid typing interval
                self.key_intervals.append(interval)
        
        self.last_key_time = current_time
        self.total_keys += 1
        
        # Track error keys
        if key in [Key.backspace, Key.delete]:
            self.error_keys += 1
            if self.char_since_space > 0:
                self.char_since_space -= 1
        
        # Word counting
        try:
            if hasattr(key, 'char') and key.char:
                self.char_since_space += 1
                if key.char == ' ':
                    if self.char_since_space > 1:
                        self.word_count += 1
                    self.char_since_space = 0
        except:
            pass

        try:
            if key in [Key.space, Key.enter]:
                if self.char_since_space > 1:
                    self.word_count += 1
                self.char_since_space = 0
        except:
            pass
        
        self.keystrokes.append({'time': current_time})
        
    def on_mouse_move(self, x, y):
        """Handle mouse movement."""
        self.last_activity = time.time()
        
        if self.last_mouse_pos:
            dx = x - self.last_mouse_pos[0]
            dy = y - self.last_mouse_pos[1]
            distance = math.sqrt(dx*dx + dy*dy)
            if distance > 3:
                self.mouse_distances.append(distance)
                self.total_mouse_distance += distance
        
        self.last_mouse_pos = (x, y)
        
    def on_mouse_click(self, x, y, button, pressed):
        """Handle mouse clicks."""
        if pressed:
            self.last_activity = time.time()
            self.mouse_clicks.append({'time': time.time()})
            
    def on_scroll(self, x, y, dx, dy):
        """Handle scroll events."""
        self.last_activity = time.time()
        self.total_scroll += abs(dy)
        self.scroll_events.append({'time': time.time(), 'dy': dy})
        
    def check_active_window(self):
        """Check for application switches (Windows only)."""
        try:
            import ctypes
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            length = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            
            window_title = buf.value[:40]  # Truncate
            
            if window_title and window_title != self.last_active_window:
                if self.last_active_window is not None:
                    self.app_switches.append({
                        'time': time.time(),
                        'from': self.last_active_window,
                        'to': window_title
                    })
                self.last_active_window = window_title
                self.current_app = window_title
        except:
            pass
            
    def start(self):
        """Start capturing."""
        self.running = True
        self.start_time = time.time()
        self.last_activity = time.time()
        
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
        self.keyboard_listener.start()
        
        self.mouse_listener = mouse.Listener(
            on_move=self.on_mouse_move,
            on_click=self.on_mouse_click,
            on_scroll=self.on_scroll
        )
        self.mouse_listener.start()
        
        def track_windows():
            while self.running:
                self.check_active_window()
                time.sleep(0.5)
                
        threading.Thread(target=track_windows, daemon=True).start()
        
    def stop(self):
        """Stop capturing."""
        self.running = False
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        if self.mouse_listener:
            self.mouse_listener.stop()
            
    def get_metrics(self) -> Dict:
        """Calculate current behavioral metrics."""
        now = time.time()
        session_duration = now - self.start_time if self.start_time else 0
        session_minutes = max(session_duration / 60, 0.5)
        
        # Typing metrics
        recent_intervals = list(self.key_intervals)[-30:]
        if len(recent_intervals) > 3:
            avg_interval = np.mean(recent_intervals)
            typing_speed_cpm = 60 / max(avg_interval, 0.1)
            speed_variance = np.std(recent_intervals)
        else:
            avg_interval = 0.3
            typing_speed_cpm = 0
            speed_variance = 0
        
        # Activity in last minute
        recent_keys = len([k for k in self.keystrokes if now - k['time'] < 60])
        recent_clicks = len([c for c in self.mouse_clicks if now - c['time'] < 60])

        wpm_from_intervals = typing_speed_cpm / 5 if typing_speed_cpm > 0 else 0
        wpm_from_recent = recent_keys / 5 if recent_keys > 0 else 0
        wpm_from_words = self.word_count / session_minutes
        typing_speed_wpm = max(wpm_from_intervals, wpm_from_recent, wpm_from_words)
        
        # Error rate
        error_rate = self.error_keys / max(self.total_keys, 1)
        
        # App switches per minute (last 5 min)
        recent_switches = [s for s in self.app_switches if now - s['time'] < 300]
        switch_rate = len(recent_switches) / 5
        
        # Idle time
        idle_time = now - self.last_activity if self.last_activity else 0
        
        # System metrics
        try:
            cpu_percent = psutil.cpu_percent(interval=0)
            memory_percent = psutil.virtual_memory().percent
        except:
            cpu_percent = 0
            memory_percent = 0
        
        return {
            'session_duration': session_duration,
            'total_keystrokes': self.total_keys,
            'word_count': self.word_count,
            'typing_speed_wpm': typing_speed_wpm,
            'typing_speed_variance': speed_variance,
            'error_rate': error_rate,
            'error_count': self.error_keys,
            'recent_keys': recent_keys,
            'recent_clicks': recent_clicks,
            'total_clicks': len(self.mouse_clicks),
            'total_scroll': self.total_scroll,
            'app_switch_rate': switch_rate,
            'app_switches_total': len(self.app_switches),
            'recent_switches': recent_switches[-5:],  # Last 5 switches
            'current_app': self.current_app,
            'idle_time': idle_time,
            'cpu_usage': cpu_percent,
            'memory_usage': memory_percent,
        }


class MindShieldEngine:
    """ML engine for fatigue detection - Fixed version."""
    
    def __init__(self):
        self.fatigue_model = FatiguePredictionModel()
        self.anomaly_detector = BehavioralDriftDetector(contamination=0.2)
        self._initialize_models()
        
        # Baseline tracking
        self.baseline_samples = []
        self.warmup_count = 0
        
        # History
        self.history = deque(maxlen=60)
        self.risk_history = deque(maxlen=30)
        
        # Stats
        self.peak_risk = 0
        self.lowest_risk = 100
        
    def _initialize_models(self):
        """Initialize models."""
        np.random.seed(42)
        n = 300
        
        # Normal behavior
        normal = {
            'typing_speed_wpm': np.random.normal(40, 12, n//2),
            'typing_speed_variance': np.abs(np.random.normal(0.1, 0.05, n//2)),
            'error_rate': np.random.beta(2, 25, n//2),
            'app_switch_rate': np.random.exponential(1.5, n//2),
        }
        
        # Fatigued behavior
        fatigued = {
            'typing_speed_wpm': np.random.normal(25, 10, n//2),
            'typing_speed_variance': np.abs(np.random.normal(0.3, 0.1, n//2)),
            'error_rate': np.random.beta(4, 12, n//2),
            'app_switch_rate': np.random.exponential(4, n//2),
        }
        
        import pandas as pd
        normal_df = pd.DataFrame(normal)
        normal_df['label'] = 0
        fatigued_df = pd.DataFrame(fatigued)
        fatigued_df['label'] = 1
        
        train_df = pd.concat([normal_df, fatigued_df], ignore_index=True)
        X = train_df.drop('label', axis=1).values
        y = train_df['label'].values
        
        self.fatigue_model.train(X, y)
        self.anomaly_detector.fit_baseline(normal_df.drop('label', axis=1).values)
        
    def analyze(self, metrics: Dict) -> Dict:
        """Analyze metrics and return assessment."""
        # Build features
        features = np.array([[
            max(metrics.get('typing_speed_wpm', 35), 0),
            max(metrics.get('typing_speed_variance', 0.1), 0),
            min(max(metrics.get('error_rate', 0.05), 0), 0.5),
            max(metrics.get('app_switch_rate', 1), 0),
        ]])
        
        # Fatigue prediction
        try:
            fatigue_prob = float(self.fatigue_model.predict_proba(features)[0])
            fatigue_prob = min(max(fatigue_prob, 0), 1)
        except:
            fatigue_prob = 0.3
            
        # Simple cognitive load - based on YOUR behavior, not system memory!
        cognitive_load = self._calc_cognitive_load(metrics)
        
        # Focus score
        focus_score = self._calc_focus(metrics)
        
        # Productivity
        productivity = self._calc_productivity(metrics)
        
        # Calculate drift
        behavioral_drift = self._calc_drift(metrics)
        
        # Overall risk
        risk_score = self._calc_risk(fatigue_prob, cognitive_load, behavioral_drift, metrics)
        
        # Update stats
        self.peak_risk = max(self.peak_risk, risk_score)
        self.lowest_risk = min(self.lowest_risk, risk_score)
        self.risk_history.append(risk_score)
        
        # Trend
        trend = self._calc_trend()
        
        # Eye strain based on time
        hours = metrics.get('session_duration', 0) / 3600
        eye_strain = min(hours / 2, 1) * 100
        
        # Recommendations
        recommendations = self._get_recommendations(risk_score, fatigue_prob, cognitive_load, hours, focus_score)
        
        result = {
            'risk_score': round(risk_score, 1),
            'risk_level': self._get_level(risk_score),
            'fatigue_prob': round(fatigue_prob * 100, 1),
            'cognitive_load': round(cognitive_load, 1),
            'focus_score': round(focus_score, 1),
            'productivity': round(productivity, 1),
            'behavioral_drift': round(behavioral_drift, 1),
            'eye_strain': round(eye_strain, 1),
            'trend': trend,
            'recommendations': recommendations,
        }
        
        self.history.append(result)
        return result
        
    def _calc_cognitive_load(self, m: Dict) -> float:
        """Calculate cognitive load - FIXED to be realistic."""
        load = 0
        
        # Error rate contribution (0-25%)
        error_rate = m.get('error_rate', 0)
        load += min(error_rate * 100, 25)
        
        # Typing variance (0-20%) - higher variance = struggling
        variance = m.get('typing_speed_variance', 0)
        load += min(variance * 40, 20)
        
        # App switching (0-25%) - more switches = context switching
        switches = m.get('app_switch_rate', 0)
        load += min(switches * 5, 25)
        
        # Typing speed - very slow might indicate thinking (0-20%)
        wpm = m.get('typing_speed_wpm', 40)
        if wpm < 15:
            load += 20
        elif wpm < 25:
            load += 10
        
        # Recent activity - if idle, might be thinking (0-10%)
        idle = m.get('idle_time', 0)
        if idle > 5:
            load += min(idle, 10)
        
        return min(max(load, 5), 95)  # Always between 5-95
        
    def _calc_focus(self, m: Dict) -> float:
        """Calculate focus score."""
        focus = 70  # Start at decent focus
        
        # App switches reduce focus
        switches = m.get('app_switch_rate', 0)
        focus -= min(switches * 8, 40)
        
        # Consistent typing increases focus
        variance = m.get('typing_speed_variance', 0)
        if variance < 0.15:
            focus += 15
            
        # Active typing increases focus
        recent_keys = m.get('recent_keys', 0)
        if recent_keys > 30:
            focus += 10
        elif recent_keys < 5:
            focus -= 10
            
        # Low error rate = focused
        error = m.get('error_rate', 0)
        if error < 0.05:
            focus += 5
            
        return min(max(focus, 10), 95)
        
    def _calc_productivity(self, m: Dict) -> float:
        """Calculate productivity index."""
        prod = 50  # Base
        
        # Typing output
        wpm = m.get('typing_speed_wpm', 0)
        prod += min(wpm, 40)  # Up to +40 for fast typing
        
        # Recent activity
        recent_keys = m.get('recent_keys', 0)
        if recent_keys > 50:
            prod += 10
        elif recent_keys < 10:
            prod -= 20
            
        # Idle penalty
        idle = m.get('idle_time', 0)
        if idle > 10:
            prod -= min(idle * 2, 30)
            
        return min(max(prod, 5), 95)
        
    def _calc_drift(self, m: Dict) -> float:
        """Calculate behavioral drift using ML anomaly detection."""
        current = {
            'wpm': m.get('typing_speed_wpm', 35),
            'error': m.get('error_rate', 0.05),
            'switches': m.get('app_switch_rate', 1),
        }
        
        self.baseline_samples.append(current)
        self.warmup_count += 1
        
        # Warmup period - return low drift
        if self.warmup_count < 8:
            return 5 + self.warmup_count * 2
        
        # Try ML-based anomaly detection first
        ml_drift = None
        if hasattr(self, 'anomaly_detector') and self.anomaly_detector.is_fitted:
            try:
                features = np.array([[current['wpm'], current['error'], current['switches']]])
                labels, scores = self.anomaly_detector.detect_anomaly(features)
                # Convert score to drift percentage (positive = normal, negative = anomaly)
                ml_drift = max(0, 50 - scores[0] * 50)  # Map to 0-100 range
            except Exception as e:
                pass  # Fall back to statistical method
        
        if ml_drift is not None:
            return min(max(ml_drift, 5), 80)
            
        # Fallback: Use first 5 samples as baseline
        baseline = self.baseline_samples[:5]
        baseline_avg = {
            'wpm': np.mean([s['wpm'] for s in baseline]),
            'error': np.mean([s['error'] for s in baseline]),
            'switches': np.mean([s['switches'] for s in baseline]),
        }
        
        # Calculate drift
        drift = 0
        if baseline_avg['wpm'] > 0:
            drift += abs(current['wpm'] - baseline_avg['wpm']) / max(baseline_avg['wpm'], 10) * 30
        drift += abs(current['error'] - baseline_avg['error']) * 200
        drift += abs(current['switches'] - baseline_avg['switches']) * 10
        
        return min(max(drift, 5), 80)
        
    def _calc_risk(self, fatigue: float, cognitive: float, drift: float, m: Dict) -> float:
        """Calculate overall risk - balanced formula."""
        # Fatigue contribution (0-35)
        risk = fatigue * 35
        
        # Cognitive load (0-25)
        risk += (cognitive / 100) * 25
        
        # Drift (0-15)
        risk += (drift / 100) * 15
        
        # Time factor (0-15)
        hours = m.get('session_duration', 0) / 3600
        risk += min(hours / 2, 1) * 15
        
        # Idle bonus - if active, slight reduction
        if m.get('recent_keys', 0) > 20:
            risk -= 5
            
        return min(max(risk, 5), 95)
        
    def _calc_trend(self) -> str:
        """Calculate trend."""
        if len(self.risk_history) < 6:
            return "stable"
        recent = list(self.risk_history)[-12:]
        first = np.mean(recent[:len(recent)//2])
        second = np.mean(recent[len(recent)//2:])
        diff = second - first
        if diff > 5:
            return "worsening"
        elif diff < -5:
            return "improving"
        return "stable"
        
    def _get_level(self, score: float) -> str:
        if score < 25:
            return "LOW"
        elif score < 50:
            return "MODERATE"
        elif score < 75:
            return "HIGH"
        return "CRITICAL"
        
    def _get_recommendations(self, risk, fatigue, cognitive, hours, focus) -> List[str]:
        recs = []
        
        if hours >= 2:
            recs.append("🕐 2+ hours - Take a 15-min break")
        elif hours >= 1:
            recs.append("⏰ Consider a 5-min break soon")
        
        if risk >= 75:
            recs.append("🚨 CRITICAL: Step away now")
        elif risk >= 50:
            recs.append("⚠️ Take a break in 10 mins")
        elif risk >= 25:
            recs.append("📊 Monitor your energy")
        else:
            recs.append("✅ You're doing great!")
            
        if fatigue > 0.6:
            recs.append("😴 Rest eyes (20-20-20 rule)")
        if focus < 40:
            recs.append("🎯 Close extra tabs/apps")
        if cognitive > 60:
            recs.append("🧠 Break tasks into smaller steps")
            
        return recs[:3]


class MindShieldGUI:
    """Compact GUI that fits on screen."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Mind-Shield+ v2.2")
        self.root.geometry("1200x900")
        self.root.minsize(1000, 760)
        self.root.configure(bg='#0b0f1a')
        self.root.resizable(True, True)
        
        # Colors
        self.c = {
            'bg': '#0b0f1a',
            'card': '#121826',
            'border': '#2a3246',
            'green': '#00f5a0',
            'yellow': '#f5c542',
            'orange': '#ff8a3d',
            'red': '#ff4d6d',
            'blue': '#4cc9f0',
            'pink': '#ff6fb5',
            'text': '#eaf2ff',
            'dim': '#9aa4b2',
            'focus_bg': '#1b2a3d',
        }
        
        # State
        self.monitoring = False
        self.capture = None
        self.engine = MindShieldEngine()
        self.baseline_manager = PersonalBaselineManager(calibration_samples=120)
        self.intervention_engine = InterventionEngine()
        self.session_store = SessionStore()
        self.gamification = GamificationTracker()
        self.adaptive_learning = AdaptiveLearningEngine()
        self.emotion_engine = EmotionEngine()
        self.burnout_engine = BurnoutEngine()
        self.local_storage = LocalStorage()
        self.privacy = PrivacyController()
        self.event_bus = EventBus()
        self.last_persist_time = 0
        self.last_summary_refresh = 0
        self.latest_posture_score = None
        self.study_mode_enabled = False
        self._risk_pulse_active = False
        self._risk_pulse_on = False
        self._risk_pulse_job = None
        self._streak_pulse_on = False
        self.latest_emotion_state = None
        self.latest_burnout_state = None
        
        # Focus Mode
        self.focus_session = FocusSession()
        self.focus_warning_window = None
        self.focus_check_running = False
        
        # Posture Detection
        self.posture_detector = None
        self.posture_enabled = False
        self.last_posture_good = None
        self.last_camera_frame = None
        self.face_auth = None
        self.face_auth_enabled = False
        self.face_auth_locked = False
        self.face_auth_state = None
        self.face_auth_name_var = tk.StringVar(value="Primary User")
        self.face_auth_last_auth_ts = 0.0
        self.face_auth_last_seen_ts = 0.0
        self.face_auth_grace_seconds = 2
        self.face_auth_absent_seconds = 12
        self.face_auth_unknown_since = 0.0
        self.face_auth_unknown_seconds = 12
        self.face_auth_enroll_in_progress = False
        self.face_auth_last_auth_ts = 0.0
        self.face_auth_last_seen_ts = 0.0
        self.face_auth_grace_seconds = 10
        self.face_auth_absent_seconds = 10
        self.face_auth_unknown_since = 0.0
        self.face_auth_unknown_seconds = 12
        if PostureDetector:
            try:
                self.posture_detector = PostureDetector()
                if not self.posture_detector.is_available():
                    self.posture_detector = None
            except Exception as e:
                print(f"Posture detection not available: {e}")
                self.posture_detector = None
        if LocalFaceAuthenticator:
            try:
                self.face_auth = LocalFaceAuthenticator(
                    profile_name=self.face_auth_name_var.get(),
                    threshold=0.75,
                )
            except Exception as e:
                print(f"Local face auth not available: {e}")
                self.face_auth = None
        
        # Graph data (last 30 data points)
        self.graph_data = {
            'risk': deque(maxlen=30),
            'fatigue': deque(maxlen=30),
            'cognitive': deque(maxlen=30),
            'focus': deque(maxlen=30),
        }
        
        self._build_ui()
        self._update_loop()
        
    def _build_ui(self):
        """Build compact UI."""
        # Header
        header = tk.Frame(self.root, bg=self.c['card'], height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        title = tk.Frame(header, bg=self.c['card'])
        title.pack(side=tk.LEFT, padx=15, pady=8)

        tk.Label(title, text="🛡️ Mind-Shield+", font=('Bahnschrift', 16, 'bold'),
            bg=self.c['card'], fg=self.c['text']).pack(anchor=tk.W)
        tk.Label(title, text="Focus. Flow. Finish.", font=('Bahnschrift', 9),
            bg=self.c['card'], fg=self.c['dim']).pack(anchor=tk.W)
        
        self.status = tk.Label(header, text="● Stopped", font=('Bahnschrift', 10, 'bold'),
                      bg=self.c['card'], fg=self.c['red'])
        self.status.pack(side=tk.RIGHT, padx=12)

        self.status_badge = tk.Label(header, text="OFFLINE", font=('Bahnschrift', 9, 'bold'),
                         bg=self.c['border'], fg=self.c['dim'], padx=10, pady=4)
        self.status_badge.pack(side=tk.RIGHT, padx=8)

        self.privacy_badge = tk.Label(header, text="LOCAL ONLY", font=('Bahnschrift', 9, 'bold'),
                 bg=self.c['focus_bg'], fg=self.c['blue'], padx=10, pady=4)
        self.privacy_badge.pack(side=tk.RIGHT, padx=(0, 8))
        
        # Main area
        main = tk.Frame(self.root, bg=self.c['bg'])
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel (risk + metrics)
        left = tk.Frame(main, bg=self.c['bg'], width=300)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        left.pack_propagate(False)
        
        # Risk card
        risk_card = tk.Frame(left, bg=self.c['card'], highlightbackground=self.c['border'], highlightthickness=1)
        risk_card.pack(fill=tk.X, pady=(0, 8))
        self.risk_card = risk_card
        self._add_card_accent(risk_card, self.c['pink'], self.c['blue'])
        
        tk.Label(risk_card, text="Risk Level", font=('Bahnschrift', 11),
                bg=self.c['card'], fg=self.c['dim']).pack(pady=(10, 0))
        
        self.risk_val = tk.Label(risk_card, text="--", font=('Bahnschrift', 46, 'bold'),
                                 bg=self.c['card'], fg=self.c['green'])
        self.risk_val.pack()
        
        self.risk_lvl = tk.Label(risk_card, text="READY", font=('Bahnschrift', 12, 'bold'),
                                bg=self.c['card'], fg=self.c['dim'])
        self.risk_lvl.pack()

        # Risk progress bar
        self.risk_bar = tk.Canvas(risk_card, height=8, bg=self.c['border'], highlightthickness=0)
        self.risk_bar.pack(fill=tk.X, padx=12, pady=(6, 6))
        self.risk_bar_fill = self.risk_bar.create_rectangle(0, 0, 0, 8, fill=self.c['green'], width=0)
        
        self.trend_lbl = tk.Label(risk_card, text="", font=('Bahnschrift', 10),
                                 bg=self.c['card'], fg=self.c['dim'])
        self.trend_lbl.pack(pady=(0, 10))
        
        # Metrics
        self.metrics = {}
        metrics_data = [
            ('cognitive', '🧠 Cognitive Load'),
            ('fatigue', '😴 Fatigue'),
            ('state', '🧩 Unified State'),
            ('focus', '🎯 Focus'),
            ('productivity', '📈 Productivity'),
            ('drift', '📊 Drift'),
            ('eye_strain', '👁️ Eye Strain'),
            ('confidence', '🔍 Model Confidence'),
            ('typing', '⌨️ Typing Speed'),
            ('errors', '❌ Errors'),
        ]
        
        for key, label in metrics_data:
            card = tk.Frame(left, bg=self.c['card'], highlightbackground=self.c['border'], highlightthickness=1)
            card.pack(fill=tk.X, pady=2)
            
            inner = tk.Frame(card, bg=self.c['card'])
            inner.pack(fill=tk.X, padx=10, pady=6)
            
            tk.Label(inner, text=label, font=('Bahnschrift', 9),
                    bg=self.c['card'], fg=self.c['dim']).pack(side=tk.LEFT)
            
            val = tk.Label(inner, text="--", font=('Bahnschrift', 12, 'bold'),
                          bg=self.c['card'], fg=self.c['text'])
            val.pack(side=tk.RIGHT)
            self.metrics[key] = val
        
        # Right panel - SCROLLABLE
        right_container = tk.Frame(main, bg=self.c['bg'])
        right_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))
        
        right_canvas = tk.Canvas(right_container, bg=self.c['bg'], highlightthickness=0)
        right_scrollbar = tk.Scrollbar(right_container, orient="vertical", command=right_canvas.yview)
        right = tk.Frame(right_canvas, bg=self.c['bg'])
        self.right_frame = right  # Store reference
        
        right.bind("<Configure>", lambda e: right_canvas.configure(scrollregion=right_canvas.bbox("all")))
        right_canvas.create_window((0, 0), window=right, anchor="nw", width=right_canvas.winfo_width())
        right_canvas.configure(yscrollcommand=right_scrollbar.set)
        
        # Update width on resize
        def _on_right_canvas_configure(event):
            right_canvas.itemconfig(right_canvas.find_all()[0], width=event.width)
        right_canvas.bind("<Configure>", _on_right_canvas_configure)
        
        right_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Enable mousewheel scrolling for right panel
        def _on_mousewheel(event):
            delta = event.delta
            if delta == 0 and hasattr(event, "num"):
                delta = 120 if event.num == 4 else -120
            right_canvas.yview_scroll(int(-1 * (delta / 120)), "units")

        def _on_enter(_event):
            right_canvas.focus_set()

        right_canvas.bind("<Enter>", _on_enter)
        right_canvas.bind("<MouseWheel>", _on_mousewheel)
        right.bind("<MouseWheel>", _on_mousewheel)
        self.root.bind_all("<MouseWheel>", _on_mousewheel)
        self.root.bind_all("<Button-4>", _on_mousewheel)
        self.root.bind_all("<Button-5>", _on_mousewheel)
        self.right_canvas = right_canvas

        self.secure_overlay = tk.Frame(main, bg='#060a12')
        self.secure_overlay_label = tk.Label(
            self.secure_overlay,
            text="",
            font=('Bahnschrift', 20, 'bold'),
            bg='#060a12',
            fg=self.c['text'],
            justify=tk.CENTER,
        )
        self.secure_overlay_label.pack(expand=True)
        self.secure_overlay_desc = tk.Label(
            self.secure_overlay,
            text="",
            font=('Segoe UI', 10),
            bg='#060a12',
            fg=self.c['dim'],
            justify=tk.CENTER,
        )
        self.secure_overlay_desc.pack(pady=(0, 10))
        self.secure_overlay.place_forget()

        # === LOCAL FACE AUTH CARD ===
        auth_card = tk.Frame(right, bg=self.c['card'], highlightbackground=self.c['border'], highlightthickness=1)
        auth_card.pack(fill=tk.X, pady=(0, 8))
        self.auth_card = auth_card
        self._add_card_accent(auth_card, self.c['blue'], self.c['pink'])

        auth_header = tk.Frame(auth_card, bg=self.c['card'])
        auth_header.pack(fill=tk.X, padx=12, pady=(10, 6))

        tk.Label(auth_header, text="🔐 Local Face Auth", font=('Bahnschrift', 12, 'bold'),
            bg=self.c['card'], fg=self.c['text']).pack(side=tk.LEFT)

        self.auth_status_lbl = tk.Label(auth_header, text="OFF", font=('Bahnschrift', 10, 'bold'),
                         bg=self.c['card'], fg=self.c['dim'])
        self.auth_status_lbl.pack(side=tk.RIGHT)

        auth_body = tk.Frame(auth_card, bg=self.c['card'])
        auth_body.pack(fill=tk.X, padx=12, pady=(0, 8))

        tk.Label(auth_body, text="Profile", font=('Bahnschrift', 9), bg=self.c['card'], fg=self.c['dim']).grid(row=0, column=0, sticky='w')
        self.face_name_entry = tk.Entry(auth_body, textvariable=self.face_auth_name_var, font=('Segoe UI', 10),
                                        bg='#182033', fg=self.c['text'], insertbackground=self.c['text'], relief=tk.FLAT)
        self.face_name_entry.grid(row=0, column=1, sticky='ew', padx=(8, 0))

        self.auth_conf_lbl = tk.Label(auth_body, text="Confidence: --", font=('Bahnschrift', 9),
                         bg=self.c['card'], fg=self.c['dim'])
        self.auth_conf_lbl.grid(row=1, column=0, columnspan=2, sticky='w', pady=(6, 0))

        self.auth_security_lbl = tk.Label(auth_body, text="Security: --", font=('Bahnschrift', 9),
                         bg=self.c['card'], fg=self.c['dim'])
        self.auth_security_lbl.grid(row=2, column=0, columnspan=2, sticky='w', pady=(2, 0))

        self.auth_message_lbl = tk.Label(auth_body, text="Use the webcam to enroll the local user.", font=('Bahnschrift', 8),
                         bg=self.c['card'], fg=self.c['dim'])
        self.auth_message_lbl.grid(row=3, column=0, columnspan=2, sticky='w', pady=(2, 0))

        auth_body.columnconfigure(1, weight=1)

        auth_ctrl = tk.Frame(auth_card, bg=self.c['card'])
        auth_ctrl.pack(fill=tk.X, padx=12, pady=(0, 10))

        self.auth_toggle_btn = tk.Button(auth_ctrl, text="▶ ENABLE", font=('Bahnschrift', 9),
                                         bg=self.c['green'], fg='white', relief=tk.FLAT,
                                         command=self.toggle_face_auth, cursor='hand2')
        self.auth_toggle_btn.pack(side=tk.LEFT, ipadx=8, ipady=2)

        self.auth_enroll_btn = tk.Button(auth_ctrl, text="➕ Enroll", font=('Bahnschrift', 9),
                                         bg='#20354f', fg=self.c['text'], relief=tk.FLAT,
                                         command=self.enroll_face_profile, cursor='hand2')
        self.auth_enroll_btn.pack(side=tk.LEFT, padx=(8, 0), ipadx=8, ipady=2)

        self.auth_reset_btn = tk.Button(auth_ctrl, text="🗑 Reset", font=('Bahnschrift', 9),
                                        bg='#20354f', fg=self.c['text'], relief=tk.FLAT,
                                        command=self.reset_face_profiles, cursor='hand2')
        self.auth_reset_btn.pack(side=tk.LEFT, padx=(8, 0), ipadx=8, ipady=2)

        if not self.face_auth:
            self.auth_toggle_btn.config(state=tk.DISABLED, bg=self.c['dim'])
            self.auth_enroll_btn.config(state=tk.DISABLED, bg=self.c['dim'])
            self.auth_reset_btn.config(state=tk.DISABLED, bg=self.c['dim'])
            self.auth_message_lbl.config(text="Install OpenCV to enable local face auth.")
        
        # === FOCUS AI MODE CARD ===
        focus_card = tk.Frame(right, bg=self.c['focus_bg'], highlightbackground=self.c['blue'], highlightthickness=2)
        focus_card.pack(fill=tk.X, pady=(0, 8))
        self.focus_card = focus_card
        self._add_card_accent(focus_card, self.c['blue'], self.c['green'])
        
        # Focus header with toggle
        focus_header = tk.Frame(focus_card, bg=self.c['focus_bg'])
        focus_header.pack(fill=tk.X, padx=12, pady=8)
        
        tk.Label(focus_header, text="🎯 Focus AI Mode", font=('Bahnschrift', 12, 'bold'),
            bg=self.c['focus_bg'], fg=self.c['text']).pack(side=tk.LEFT)
        
        self.focus_toggle_btn = tk.Button(focus_header, text="▶ START", font=('Segoe UI', 9, 'bold'),
                                          bg=self.c['green'], fg='white', relief=tk.FLAT,
                                          command=self.toggle_focus_mode, cursor='hand2')
        self.focus_toggle_btn.pack(side=tk.RIGHT, ipadx=8, ipady=2)
        
        self.focus_status_lbl = tk.Label(focus_header, text="OFF", font=('Bahnschrift', 10, 'bold'),
                         bg=self.c['focus_bg'], fg=self.c['dim'])
        self.focus_status_lbl.pack(side=tk.RIGHT, padx=(0, 10))
        
        # Focus control buttons row
        focus_ctrl_frame = tk.Frame(focus_card, bg=self.c['focus_bg'])
        focus_ctrl_frame.pack(fill=tk.X, padx=12, pady=(0, 5))
        
        self.focus_reset_btn = tk.Button(focus_ctrl_frame, text="🔄 Reset", font=('Bahnschrift', 9),
                          bg='#20354f', fg=self.c['text'], relief=tk.FLAT,
                                          command=self.reset_focus_mode, cursor='hand2')
        self.focus_reset_btn.pack(side=tk.LEFT, ipadx=8, ipady=2)
        
        self.focus_export_btn = tk.Button(focus_ctrl_frame, text="📊 Export", font=('Bahnschrift', 9),
                          bg='#20354f', fg=self.c['text'], relief=tk.FLAT,
                                          command=self.export_focus_data, cursor='hand2')
        self.focus_export_btn.pack(side=tk.LEFT, padx=(8, 0), ipadx=8, ipady=2)
        
        # Focus timer and duration
        focus_timer_frame = tk.Frame(focus_card, bg=self.c['focus_bg'])
        focus_timer_frame.pack(fill=tk.X, padx=12)
        
        self.focus_timer_lbl = tk.Label(focus_timer_frame, text="25:00", 
                        font=('Bahnschrift', 30, 'bold'),
                        bg=self.c['focus_bg'], fg=self.c['text'])
        self.focus_timer_lbl.pack(side=tk.LEFT)
        
        # Duration selector
        dur_frame = tk.Frame(focus_timer_frame, bg=self.c['focus_bg'])
        dur_frame.pack(side=tk.RIGHT)
        
        tk.Label(dur_frame, text="Duration:", font=('Bahnschrift', 9),
            bg=self.c['focus_bg'], fg=self.c['dim']).pack(side=tk.LEFT, padx=(0, 5))
        
        self.focus_duration_var = tk.StringVar(value="25")
        duration_menu = ttk.Combobox(dur_frame, textvariable=self.focus_duration_var,
                                     values=["15", "25", "30", "45", "60"], width=5,
                                     state="readonly")
        duration_menu.pack(side=tk.LEFT)
        
        # Focus metrics row
        focus_metrics = tk.Frame(focus_card, bg=self.c['focus_bg'])
        focus_metrics.pack(fill=tk.X, padx=12, pady=8)
        
        # Focus Score
        fs_frame = tk.Frame(focus_metrics, bg='#264653', highlightbackground='#2a9d8f', highlightthickness=1)
        fs_frame.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        tk.Label(fs_frame, text="🏆", font=('Segoe UI', 12), bg='#264653', fg=self.c['text']).pack(side=tk.LEFT, padx=6)
        fs_inner = tk.Frame(fs_frame, bg='#264653')
        fs_inner.pack(side=tk.LEFT, padx=4, pady=4)
        self.focus_score_lbl = tk.Label(fs_inner, text="100", font=('Segoe UI', 14, 'bold'),
                                        bg='#264653', fg=self.c['green'])
        self.focus_score_lbl.pack()
        tk.Label(fs_inner, text="Focus Score", font=('Segoe UI', 7), bg='#264653', fg=self.c['dim']).pack()
        
        # Productivity
        pr_frame = tk.Frame(focus_metrics, bg='#264653', highlightbackground='#2a9d8f', highlightthickness=1)
        pr_frame.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        tk.Label(pr_frame, text="⚡", font=('Segoe UI', 12), bg='#264653', fg=self.c['text']).pack(side=tk.LEFT, padx=6)
        pr_inner = tk.Frame(pr_frame, bg='#264653')
        pr_inner.pack(side=tk.LEFT, padx=4, pady=4)
        self.productivity_lbl = tk.Label(pr_inner, text="100%", font=('Segoe UI', 14, 'bold'),
                                         bg='#264653', fg=self.c['green'])
        self.productivity_lbl.pack()
        tk.Label(pr_inner, text="Productivity", font=('Segoe UI', 7), bg='#264653', fg=self.c['dim']).pack()
        
        # Distractions
        dc_frame = tk.Frame(focus_metrics, bg='#264653', highlightbackground='#2a9d8f', highlightthickness=1)
        dc_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Label(dc_frame, text="⚠️", font=('Segoe UI', 12), bg='#264653', fg=self.c['text']).pack(side=tk.LEFT, padx=6)
        dc_inner = tk.Frame(dc_frame, bg='#264653')
        dc_inner.pack(side=tk.LEFT, padx=4, pady=4)
        self.distraction_count_lbl = tk.Label(dc_inner, text="0", font=('Segoe UI', 14, 'bold'),
                                              bg='#264653', fg=self.c['green'])
        self.distraction_count_lbl.pack()
        tk.Label(dc_inner, text="Distractions", font=('Segoe UI', 7), bg='#264653', fg=self.c['dim']).pack()
        
        # Distraction log
        distraction_log_frame = tk.Frame(focus_card, bg='#1e3a5f')
        distraction_log_frame.pack(fill=tk.X, padx=12, pady=(0, 8))
        
        tk.Label(distraction_log_frame, text="📋 Distraction Log:", font=('Segoe UI', 9),
                bg='#1e3a5f', fg=self.c['dim']).pack(anchor=tk.W)
        
        self.distraction_log_text = tk.Text(distraction_log_frame, height=2, font=('Consolas', 8),
                                            bg='#0d1117', fg=self.c['orange'], relief=tk.FLAT,
                                            wrap=tk.WORD, state=tk.DISABLED)
        self.distraction_log_text.pack(fill=tk.X, pady=(4, 0))
        
        # Focus Session Stats
        focus_stats_frame = tk.Frame(focus_card, bg=self.c['focus_bg'])
        focus_stats_frame.pack(fill=tk.X, padx=12, pady=(0, 10))
        
        self.focus_sessions_lbl = tk.Label(focus_stats_frame, text="Sessions: 0 | Total Focus: 0m",
                           font=('Bahnschrift', 8), bg=self.c['focus_bg'], fg=self.c['dim'],
                                           cursor='hand2')
        self.focus_sessions_lbl.pack(side=tk.LEFT)
        self.focus_sessions_lbl.bind('<Button-1>', lambda e: self.show_session_history())
        
        tk.Label(focus_stats_frame, text="  (click for details)", font=('Bahnschrift', 7),
            bg=self.c['focus_bg'], fg='#4a6a8f').pack(side=tk.LEFT)
        
        # ===== Posture Detection Card =====
        posture_card = tk.Frame(right, bg=self.c['card'], highlightbackground=self.c['border'], highlightthickness=1)
        posture_card.pack(fill=tk.X, pady=(0, 8))
        self.posture_card = posture_card
        
        posture_header = tk.Frame(posture_card, bg=self.c['card'])
        posture_header.pack(fill=tk.X, padx=12, pady=8)
        
        tk.Label(posture_header, text="🧘 Posture Detection", font=('Segoe UI', 11, 'bold'),
                bg=self.c['card'], fg=self.c['text']).pack(side=tk.LEFT)
        
        self.posture_status_lbl = tk.Label(posture_header, text="OFF", font=('Segoe UI', 10, 'bold'),
                                           bg=self.c['card'], fg=self.c['dim'])
        self.posture_status_lbl.pack(side=tk.RIGHT, padx=(0, 10))
        
        self.posture_toggle_btn = tk.Button(posture_header, text="▶ START", font=('Segoe UI', 9, 'bold'),
                                            bg=self.c['green'], fg='white', relief=tk.FLAT,
                                            command=self.toggle_posture, cursor='hand2')
        self.posture_toggle_btn.pack(side=tk.RIGHT, ipadx=8, ipady=2)
        
        # Posture metrics
        posture_metrics = tk.Frame(posture_card, bg=self.c['card'])
        posture_metrics.pack(fill=tk.X, padx=12, pady=(0, 8))
        
        # Score
        score_frame = tk.Frame(posture_metrics, bg=self.c['card'])
        score_frame.pack(side=tk.LEFT, expand=True)
        self.posture_score_lbl = tk.Label(score_frame, text="--", font=('Segoe UI', 18, 'bold'),
                                          bg=self.c['card'], fg=self.c['green'])
        self.posture_score_lbl.pack()
        tk.Label(score_frame, text="Score", font=('Segoe UI', 8), bg=self.c['card'], fg=self.c['dim']).pack()
        
        # Good Time
        good_frame = tk.Frame(posture_metrics, bg=self.c['card'])
        good_frame.pack(side=tk.LEFT, expand=True)
        self.posture_good_lbl = tk.Label(good_frame, text="0m", font=('Segoe UI', 14, 'bold'),
                                         bg=self.c['card'], fg=self.c['green'])
        self.posture_good_lbl.pack()
        tk.Label(good_frame, text="Good", font=('Segoe UI', 8), bg=self.c['card'], fg=self.c['dim']).pack()
        
        # Bad Time
        bad_frame = tk.Frame(posture_metrics, bg=self.c['card'])
        bad_frame.pack(side=tk.LEFT, expand=True)
        self.posture_bad_lbl = tk.Label(bad_frame, text="0m", font=('Segoe UI', 14, 'bold'),
                                        bg=self.c['card'], fg=self.c['red'])
        self.posture_bad_lbl.pack()
        tk.Label(bad_frame, text="Bad", font=('Segoe UI', 8), bg=self.c['card'], fg=self.c['dim']).pack()
        
        # Posture issues display
        self.posture_issues_lbl = tk.Label(posture_card, text="📷 Camera ready - Click START to monitor",
                                           font=('Segoe UI', 9), bg=self.c['card'], fg=self.c['dim'])
        self.posture_issues_lbl.pack(padx=12, pady=(0, 5))

        # Posture change alert (small, orange)
        self.posture_alert_lbl = tk.Label(posture_card, text="", font=('Segoe UI', 8),
                          bg=self.c['card'], fg=self.c['orange'])
        self.posture_alert_lbl.pack(padx=12, pady=(0, 6), anchor=tk.W)
        
        # Video preview frame
        preview_container = tk.Frame(posture_card, bg='#1a1a2e', relief=tk.SUNKEN, bd=1)
        preview_container.pack(padx=12, pady=(0, 8))
        
        # Video preview label (will show webcam feed with skeleton overlay)
        self.posture_preview_lbl = tk.Label(preview_container, bg='#1a1a2e',
                                            text="📷 Click START to begin\nposture monitoring", 
                                            fg=self.c['dim'], font=('Segoe UI', 9),
                                            width=28, height=10)
        self.posture_preview_lbl.pack(padx=2, pady=2)
        self.posture_preview_image = None  # Hold reference to prevent garbage collection
        
        # Check if posture detection available
        if not self.posture_detector:
            self.posture_toggle_btn.config(state=tk.DISABLED, bg=self.c['dim'])
            self.posture_issues_lbl.config(text="⚠️ Install mediapipe: pip install mediapipe opencv-python")
        
        # Recommendations
        rec_card = tk.Frame(right, bg=self.c['card'], highlightbackground=self.c['border'], highlightthickness=1)
        rec_card.pack(fill=tk.X, pady=(0, 8))
        
        tk.Label(rec_card, text="💡 Recommendations", font=('Bahnschrift', 11, 'bold'),
                bg=self.c['card'], fg=self.c['text']).pack(anchor=tk.W, padx=12, pady=(10, 5))
        
        self.recs = []
        for i in range(3):
            lbl = tk.Label(rec_card, text="", font=('Segoe UI', 10),
                          bg=self.c['card'], fg=self.c['text'], anchor=tk.W)
            lbl.pack(anchor=tk.W, padx=12, pady=1)
            self.recs.append(lbl)
        self.rec_reason_lbl = tk.Label(rec_card, text="", font=('Bahnschrift', 8),
                                       bg=self.c['card'], fg=self.c['dim'], anchor=tk.W)
        self.rec_reason_lbl.pack(anchor=tk.W, padx=12, pady=(4, 4))
        tk.Frame(rec_card, height=8, bg=self.c['card']).pack()

        # Explainable alert drivers
        explain_card = tk.Frame(right, bg=self.c['card'], highlightbackground=self.c['border'], highlightthickness=1)
        explain_card.pack(fill=tk.X, pady=(0, 8))

        tk.Label(explain_card, text="🧭 Why This Alert?", font=('Bahnschrift', 11, 'bold'),
                bg=self.c['card'], fg=self.c['text']).pack(anchor=tk.W, padx=12, pady=(10, 5))

        self.alert_labels = []
        for i in range(3):
            lbl = tk.Label(explain_card, text="No active alerts", font=('Bahnschrift', 9),
                          bg=self.c['card'], fg=self.c['dim'], anchor=tk.W)
            lbl.pack(anchor=tk.W, padx=12, pady=1)
            self.alert_labels.append(lbl)
        tk.Frame(explain_card, height=6, bg=self.c['card']).pack()

        # Daily analytics summary card
        daily_card = tk.Frame(right, bg=self.c['card'], highlightbackground=self.c['border'], highlightthickness=1)
        daily_card.pack(fill=tk.X, pady=(0, 8))

        tk.Label(daily_card, text="📅 Today Summary", font=('Bahnschrift', 12, 'bold'),
            bg=self.c['card'], fg=self.c['text']).pack(anchor=tk.W, padx=12, pady=(10, 5))

        self.today_samples_lbl = tk.Label(daily_card, text="Samples: 0", font=('Bahnschrift', 10),
                          bg=self.c['card'], fg=self.c['dim'])
        self.today_samples_lbl.pack(anchor=tk.W, padx=12, pady=1)

        self.today_risk_lbl = tk.Label(daily_card, text="Avg Risk: -- | Peak Risk: --", font=('Bahnschrift', 10),
                           bg=self.c['card'], fg=self.c['text'])
        self.today_risk_lbl.pack(anchor=tk.W, padx=12, pady=1)

        self.today_focus_lbl = tk.Label(daily_card, text="Avg Focus: -- | Avg Confidence: --", font=('Bahnschrift', 10),
                        bg=self.c['card'], fg=self.c['text'])
        self.today_focus_lbl.pack(anchor=tk.W, padx=12, pady=(1, 8))

        self.today_peak_lbl = tk.Label(daily_card, text="Best Focus Hour: --", font=('Bahnschrift', 10),
                           bg=self.c['card'], fg=self.c['dim'])
        self.today_peak_lbl.pack(anchor=tk.W, padx=12, pady=(0, 8))

        # Privacy controls card
        privacy_card = tk.Frame(right, bg=self.c['card'], highlightbackground=self.c['border'], highlightthickness=1)
        privacy_card.pack(fill=tk.X, pady=(0, 8))
        self._add_card_accent(privacy_card, self.c['pink'], self.c['blue'])

        tk.Label(privacy_card, text="🛡 Privacy Control Room", font=('Bahnschrift', 12, 'bold'),
            bg=self.c['card'], fg=self.c['text']).pack(anchor=tk.W, padx=12, pady=(10, 5))

        self.privacy_status_lbl = tk.Label(privacy_card, text="Local-only active | Webcam on", font=('Bahnschrift', 10),
                         bg=self.c['card'], fg=self.c['green'])
        self.privacy_status_lbl.pack(anchor=tk.W, padx=12, pady=1)

        privacy_btns = tk.Frame(privacy_card, bg=self.c['card'])
        privacy_btns.pack(fill=tk.X, padx=12, pady=(4, 10))

        self.pause_privacy_btn = tk.Button(privacy_btns, text="⏸ Pause AI", font=('Bahnschrift', 9),
                                           bg='#20354f', fg=self.c['text'], relief=tk.FLAT,
                                           command=self.toggle_privacy_pause, cursor='hand2')
        self.pause_privacy_btn.pack(side=tk.LEFT, ipadx=8, ipady=2)

        self.emotion_privacy_btn = tk.Button(privacy_btns, text="🧠 Emotion AI", font=('Bahnschrift', 9),
                                             bg='#20354f', fg=self.c['text'], relief=tk.FLAT,
                                             command=self.toggle_emotion_ai, cursor='hand2')
        self.emotion_privacy_btn.pack(side=tk.LEFT, padx=(8, 0), ipadx=8, ipady=2)

        self.webcam_privacy_btn = tk.Button(privacy_btns, text="📷 Webcam", font=('Bahnschrift', 9),
                                            bg='#20354f', fg=self.c['text'], relief=tk.FLAT,
                                            command=self.toggle_webcam_privacy, cursor='hand2')
        self.webcam_privacy_btn.pack(side=tk.LEFT, padx=(8, 0), ipadx=8, ipady=2)

        self.privacy_hint_lbl = tk.Label(privacy_card, text="No telemetry. All processing stays on-device.", font=('Bahnschrift', 8),
                         bg=self.c['card'], fg=self.c['dim'])
        self.privacy_hint_lbl.pack(anchor=tk.W, padx=12, pady=(0, 8))

        # Emotion intelligence card
        emotion_card = tk.Frame(right, bg=self.c['card'], highlightbackground=self.c['border'], highlightthickness=1)
        emotion_card.pack(fill=tk.X, pady=(0, 8))
        self._add_card_accent(emotion_card, self.c['orange'], self.c['pink'])

        tk.Label(emotion_card, text="🫀 Emotional Signal", font=('Bahnschrift', 12, 'bold'),
            bg=self.c['card'], fg=self.c['text']).pack(anchor=tk.W, padx=12, pady=(10, 5))

        self.emotion_stress_lbl = tk.Label(emotion_card, text="Stress: --", font=('Bahnschrift', 10),
                         bg=self.c['card'], fg=self.c['text'])
        self.emotion_stress_lbl.pack(anchor=tk.W, padx=12, pady=1)

        self.emotion_energy_lbl = tk.Label(emotion_card, text="Recovery: -- | Wellness: --", font=('Bahnschrift', 10),
                         bg=self.c['card'], fg=self.c['text'])
        self.emotion_energy_lbl.pack(anchor=tk.W, padx=12, pady=(1, 8))

        # Burnout intelligence card
        burnout_card = tk.Frame(right, bg=self.c['card'], highlightbackground=self.c['border'], highlightthickness=1)
        burnout_card.pack(fill=tk.X, pady=(0, 8))
        self._add_card_accent(burnout_card, self.c['red'], self.c['orange'])

        tk.Label(burnout_card, text="🔥 Burnout Forecast", font=('Bahnschrift', 12, 'bold'),
            bg=self.c['card'], fg=self.c['text']).pack(anchor=tk.W, padx=12, pady=(10, 5))

        self.burnout_risk_lbl = tk.Label(burnout_card, text="Risk: -- | Trend: --", font=('Bahnschrift', 10),
                         bg=self.c['card'], fg=self.c['text'])
        self.burnout_risk_lbl.pack(anchor=tk.W, padx=12, pady=1)

        self.burnout_recovery_lbl = tk.Label(burnout_card, text="Recovery: -- | Sustainability: --", font=('Bahnschrift', 10),
                         bg=self.c['card'], fg=self.c['text'])
        self.burnout_recovery_lbl.pack(anchor=tk.W, padx=12, pady=(1, 8))

        # Forecast card
        forecast_card = tk.Frame(right, bg=self.c['card'], highlightbackground=self.c['border'], highlightthickness=1)
        forecast_card.pack(fill=tk.X, pady=(0, 8))
        self._add_card_accent(forecast_card, self.c['blue'], self.c['pink'])

        tk.Label(forecast_card, text="🔮 5-10 Min Forecast", font=('Bahnschrift', 12, 'bold'),
            bg=self.c['card'], fg=self.c['text']).pack(anchor=tk.W, padx=12, pady=(10, 5))

        self.forecast_focus_lbl = tk.Label(forecast_card, text="Focus: --%", font=('Bahnschrift', 10),
                           bg=self.c['card'], fg=self.c['text'])
        self.forecast_focus_lbl.pack(anchor=tk.W, padx=12, pady=1)

        self.forecast_fatigue_lbl = tk.Label(forecast_card, text="Fatigue: --%", font=('Bahnschrift', 10),
                             bg=self.c['card'], fg=self.c['text'])
        self.forecast_fatigue_lbl.pack(anchor=tk.W, padx=12, pady=(1, 8))

        # Gamification card
        game_card = tk.Frame(right, bg=self.c['card'], highlightbackground=self.c['border'], highlightthickness=1)
        game_card.pack(fill=tk.X, pady=(0, 8))
        self._add_card_accent(game_card, self.c['green'], self.c['yellow'])

        tk.Label(game_card, text="🏆 Focus Streaks", font=('Bahnschrift', 12, 'bold'),
            bg=self.c['card'], fg=self.c['text']).pack(anchor=tk.W, padx=12, pady=(10, 5))

        self.gamification_lbl = tk.Label(game_card, text="Level: Beginner | Streak: 0m", font=('Bahnschrift', 10),
                         bg=self.c['card'], fg=self.c['text'])
        self.gamification_lbl.pack(anchor=tk.W, padx=12, pady=1)

        self.gamification_score_lbl = tk.Label(game_card, text="Productivity Score: 0", font=('Bahnschrift', 10),
                               bg=self.c['card'], fg=self.c['dim'])
        self.gamification_score_lbl.pack(anchor=tk.W, padx=12, pady=(1, 8))

        self.streak_badge = tk.Label(game_card, text="🔥 Streak", font=('Bahnschrift', 10, 'bold'),
                         bg=self.c['card'], fg=self.c['dim'])
        self.streak_badge.pack(anchor=tk.W, padx=12, pady=(0, 8))

        # AI Study Assistant card
        study_card = tk.Frame(right, bg=self.c['card'], highlightbackground=self.c['border'], highlightthickness=1)
        study_card.pack(fill=tk.X, pady=(0, 8))
        self._add_card_accent(study_card, self.c['blue'], self.c['orange'])

        study_header = tk.Frame(study_card, bg=self.c['card'])
        study_header.pack(fill=tk.X, padx=12, pady=(10, 5))

        tk.Label(study_header, text="📚 AI Study Assistant", font=('Bahnschrift', 12, 'bold'),
            bg=self.c['card'], fg=self.c['text']).pack(side=tk.LEFT)

        self.study_toggle_btn = tk.Button(study_header, text="OFF", font=('Bahnschrift', 9, 'bold'),
                          bg=self.c['dim'], fg='white', relief=tk.FLAT,
                          command=self.toggle_study_mode, cursor='hand2')
        self.study_toggle_btn.pack(side=tk.RIGHT, ipadx=8)

        self.study_status_lbl = tk.Label(study_card, text="Mode: Disabled", font=('Bahnschrift', 10),
                         bg=self.c['card'], fg=self.c['dim'])
        self.study_status_lbl.pack(anchor=tk.W, padx=12, pady=1)

        self.study_cycle_lbl = tk.Label(study_card, text="Suggested cycle: --", font=('Bahnschrift', 10),
                        bg=self.c['card'], fg=self.c['text'])
        self.study_cycle_lbl.pack(anchor=tk.W, padx=12, pady=(1, 8))
        
        # App switches card
        switch_card = tk.Frame(right, bg=self.c['card'], highlightbackground=self.c['border'], highlightthickness=1)
        switch_card.pack(fill=tk.X, pady=(0, 8))
        
        tk.Label(switch_card, text="🔄 Recent App Switches", font=('Bahnschrift', 12, 'bold'),
                bg=self.c['card'], fg=self.c['text']).pack(anchor=tk.W, padx=12, pady=(10, 5))
        
        self.switch_labels = []
        for i in range(3):
            lbl = tk.Label(switch_card, text="", font=('Consolas', 9),
                          bg=self.c['card'], fg=self.c['blue'], anchor=tk.W)
            lbl.pack(anchor=tk.W, padx=12, pady=1)
            self.switch_labels.append(lbl)
        tk.Frame(switch_card, height=6, bg=self.c['card']).pack()
        
        # Real-time Graph
        graph_card = tk.Frame(right, bg=self.c['card'], highlightbackground=self.c['border'], highlightthickness=1)
        graph_card.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        
        tk.Label(graph_card, text="📈 Real-Time Analysis", font=('Bahnschrift', 12, 'bold'),
                bg=self.c['card'], fg=self.c['text']).pack(anchor=tk.W, padx=12, pady=(10, 5))
        
        # Graph legend
        legend_frame = tk.Frame(graph_card, bg=self.c['card'])
        legend_frame.pack(fill=tk.X, padx=12)
        for name, color in [('Risk', self.c['red']), ('Fatigue', self.c['orange']), 
                           ('Cognitive', self.c['yellow']), ('Focus', self.c['green'])]:
            tk.Label(legend_frame, text=f"● {name}", font=('Segoe UI', 8),
                    bg=self.c['card'], fg=color).pack(side=tk.LEFT, padx=(0, 10))
        
        # Canvas for graph - FIXED: proper height and scaling
        graph_frame = tk.Frame(graph_card, bg=self.c['bg'])
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(5, 10))
        
        self.graph_canvas = tk.Canvas(graph_frame, bg=self.c['bg'], height=180, width=400,
                                      highlightthickness=0)
        self.graph_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Draw graph on configure and after initial display
        def _on_graph_configure(event):
            self.root.after(100, self.draw_graph)
        self.graph_canvas.bind('<Configure>', _on_graph_configure)
        
        # Activity log (smaller)
        log_card = tk.Frame(right, bg=self.c['card'], highlightbackground=self.c['border'], highlightthickness=1)
        log_card.pack(fill=tk.X, pady=(0, 8))
        
        tk.Label(log_card, text="📋 Log", font=('Segoe UI', 10, 'bold'),
                bg=self.c['card'], fg=self.c['text']).pack(anchor=tk.W, padx=12, pady=(8, 3))
        
        self.log = tk.Text(log_card, height=4, font=('Consolas', 8),
                          bg=self.c['bg'], fg=self.c['green'], relief=tk.FLAT,
                          wrap=tk.WORD, state=tk.DISABLED)
        self.log.pack(fill=tk.X, padx=12, pady=(0, 8))
        
        # Stats bar
        stats = tk.Frame(right, bg=self.c['card'], highlightbackground=self.c['border'], highlightthickness=1)
        stats.pack(fill=tk.X)
        
        stats_inner = tk.Frame(stats, bg=self.c['card'])
        stats_inner.pack(fill=tk.X, padx=12, pady=8)
        
        self.stat_labels = {}
        for key, label in [('keys', '⌨️'), ('clicks', '🖱️'), ('words', '📝'), 
                           ('switches', '🔄'), ('cpu', '💻'), ('mem', '🧮')]:
            f = tk.Frame(stats_inner, bg=self.c['card'])
            f.pack(side=tk.LEFT, padx=(0, 15))
            tk.Label(f, text=label, font=('Segoe UI', 9), bg=self.c['card'], fg=self.c['dim']).pack(side=tk.LEFT)
            v = tk.Label(f, text="0", font=('Segoe UI', 10, 'bold'), bg=self.c['card'], fg=self.c['text'])
            v.pack(side=tk.LEFT, padx=(3, 0))
            self.stat_labels[key] = v
        
        # Control bar
        ctrl = tk.Frame(self.root, bg=self.c['bg'])
        ctrl.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.start_btn = tk.Button(ctrl, text="▶ Start", font=('Segoe UI', 11, 'bold'),
                                   bg=self.c['green'], fg='white', relief=tk.FLAT,
                                   command=self.toggle, cursor='hand2')
        self.start_btn.pack(side=tk.LEFT, ipadx=15, ipady=4)
        
        tk.Button(ctrl, text="🔄 Reset", font=('Segoe UI', 10),
                 bg=self.c['card'], fg=self.c['text'], relief=tk.FLAT,
                 command=self.reset, cursor='hand2').pack(side=tk.LEFT, padx=(8, 0), ipadx=10, ipady=4)
        
        tk.Button(ctrl, text="📊 Export", font=('Segoe UI', 10),
                 bg=self.c['card'], fg=self.c['text'], relief=tk.FLAT,
                 command=self.export, cursor='hand2').pack(side=tk.LEFT, padx=(8, 0), ipadx=10, ipady=4)
        
        tk.Button(ctrl, text="☕ Break", font=('Segoe UI', 10),
                 bg=self.c['card'], fg=self.c['text'], relief=tk.FLAT,
                 command=self.record_break_taken, cursor='hand2').pack(side=tk.LEFT, padx=(8, 0), ipadx=10, ipady=4)
        
        self.timer = tk.Label(ctrl, text="00:00:00", font=('Segoe UI', 11),
                             bg=self.c['bg'], fg=self.c['dim'])
        self.timer.pack(side=tk.RIGHT)
        
        tk.Label(ctrl, text="Session:", font=('Segoe UI', 10),
                bg=self.c['bg'], fg=self.c['dim']).pack(side=tk.RIGHT, padx=(0, 5))
        
    def toggle(self):
        if self.monitoring:
            self.stop()
        else:
            self.start()
            
    def start(self):
        self.monitoring = True
        self.capture = SystemBehaviorCapture()
        self.capture.start()
        self.engine = MindShieldEngine()
        self._refresh_daily_summary()
        
        self.start_btn.config(text="⏹ Stop", bg=self.c['red'])
        self.status.config(text="● Monitoring", fg=self.c['green'])
        self.status_badge.config(text="LIVE", bg=self.c['blue'], fg='white')
        self.add_log("✅ Started monitoring")
        
    def stop(self):
        self.monitoring = False
        if self.capture:
            self.capture.stop()
        self._set_secure_overlay(False)
        self.start_btn.config(text="▶ Start", bg=self.c['green'])
        self.status.config(text="● Stopped", fg=self.c['red'])
        self.status_badge.config(text="OFFLINE", bg=self.c['border'], fg=self.c['dim'])
        self.add_log("⏹ Stopped")

    def toggle_privacy_pause(self):
        """Pause or resume the local AI loop."""
        self.privacy.pause(not self.privacy.settings.paused)
        if self.privacy.settings.paused:
            self.add_log("⏸ Local AI paused")
            self.privacy_status_lbl.config(text="Local-only active | Paused", fg=self.c['yellow'])
        else:
            self.add_log("▶ Local AI resumed")

    def toggle_emotion_ai(self):
        """Enable or disable emotion intelligence locally."""
        enabled = not self.privacy.settings.emotion_ai_enabled
        self.privacy.toggle_emotion_ai(enabled)
        self.add_log(f"🧠 Emotion AI {'enabled' if enabled else 'disabled'}")

    def toggle_webcam_privacy(self):
        """Enable or disable webcam-based monitoring features."""
        enabled = not self.privacy.settings.webcam_enabled
        self.privacy.toggle_webcam(enabled)
        if not enabled:
            self.add_log("📷 Webcam disabled")
            if self.posture_enabled:
                self.stop_posture()
            self.face_auth_enabled = False
            self.face_auth_locked = False
            self._set_secure_overlay(False)
        else:
            self.add_log("📷 Webcam enabled")

    def toggle_face_auth(self):
        """Enable or disable local face authentication."""
        if not self.face_auth:
            messagebox.showerror("Local Face Auth", "Local face authentication is unavailable.")
            return

        if not self.posture_detector:
            messagebox.showerror("Local Face Auth", "Webcam monitoring is required for local face authentication.")
            return

        if self.face_auth_enabled:
            self.face_auth_enabled = False
            self.face_auth_locked = False
            self._set_secure_overlay(False)
            self.auth_toggle_btn.config(text="▶ ENABLE", bg=self.c['green'])
            self.auth_status_lbl.config(text="OFF", fg=self.c['dim'])
            self.auth_message_lbl.config(text="Local face auth disabled.", fg=self.c['dim'])
            self.status_badge.config(text="LIVE", bg=self.c['blue'], fg='white' if self.monitoring else self.c['dim'])
            self.add_log("🔓 Local face auth disabled")
            return

        if self.posture_detector and not self.posture_enabled:
            self.start_posture()
            if not self.posture_enabled:
                messagebox.showerror("Local Face Auth", "Start webcam/posture monitoring first so the app can see your face.")
                return

        self.face_auth_enabled = True
        self.face_auth_locked = True if not (self.face_auth and self.face_auth.enrolled_profiles) else False
        self.auth_toggle_btn.config(text="⏹ LOCKED", bg=self.c['orange'])
        self.auth_status_lbl.config(text="ON", fg=self.c['green'])
        self.auth_message_lbl.config(text="Show your face to unlock the dashboard.", fg=self.c['dim'])
        self.add_log("🔐 Local face auth enabled")

    def enroll_face_profile(self):
        """Enroll the current webcam face as the local trusted user."""
        if self.face_auth_enroll_in_progress:
            return
        if not self.face_auth:
            messagebox.showerror("Local Face Auth", "Face authentication is unavailable.")
            return

        if self.last_camera_frame is None:
            messagebox.showinfo("Enroll Face", "Start webcam monitoring first so the app can capture a face frame.")
            return

        name = self.face_auth_name_var.get().strip() or "Primary User"
        self.face_auth_enroll_in_progress = True
        self.auth_enroll_btn.config(state=tk.DISABLED, bg=self.c['dim'])
        self.auth_message_lbl.config(text="Enrolling...", fg=self.c['dim'])

        threading.Thread(
            target=self._run_face_enroll,
            args=(name,),
            daemon=True,
        ).start()

    def _run_face_enroll(self, name: str) -> None:
        ok = False
        msg = "No face detected for enrollment"
        for _ in range(12):
            frame = self.last_camera_frame
            if frame is None:
                time.sleep(0.2)
                continue
            ok, msg = self.face_auth.enroll(name, frame)
            if ok:
                break
            if msg not in {"No face detected for enrollment", "Face crop failed"}:
                break
            time.sleep(0.2)
        self.root.after(0, lambda: self._finish_face_enroll(ok, msg, name))

    def _finish_face_enroll(self, ok: bool, msg: str, name: str) -> None:
        self.face_auth_enroll_in_progress = False
        self.auth_enroll_btn.config(state=tk.NORMAL, bg='#20354f')
        if ok:
            self.face_auth_enabled = True
            self.face_auth_locked = False
            self._set_secure_overlay(False)
            self.auth_status_lbl.config(text=f"ON • {name}", fg=self.c['green'])
            self.auth_message_lbl.config(text=msg, fg=self.c['green'])
            self.add_log(f"✅ {msg}")
        else:
            self.auth_message_lbl.config(text=msg, fg=self.c['orange'])
            messagebox.showerror("Enroll Face", msg)

    def reset_face_profiles(self):
        """Remove all stored local face profiles."""
        if not self.face_auth:
            return

        if not messagebox.askyesno("Reset Face Profiles", "Delete all local face profiles from this device?"):
            return

        self.face_auth.delete_all_profiles()
        self.face_auth_enabled = False
        self.face_auth_locked = False
        self._set_secure_overlay(False)
        self.auth_toggle_btn.config(text="▶ ENABLE", bg=self.c['green'])
        self.auth_status_lbl.config(text="OFF", fg=self.c['dim'])
        self.auth_conf_lbl.config(text="Confidence: --", fg=self.c['dim'])
        self.auth_security_lbl.config(text="Security: --", fg=self.c['dim'])
        self.auth_message_lbl.config(text="Local face profiles cleared.", fg=self.c['dim'])
        self.add_log("🗑 Local face profiles cleared")

    def _set_secure_overlay(self, locked: bool, result=None):
        """Show or hide the secure-session overlay."""
        self.face_auth_locked = locked
        if locked:
            status_text = "Secure Session Locked"
            message = "Authorized face required to continue monitoring."
            if result:
                if result.status == "ABSENT":
                    message = "No face detected. Sit in front of the webcam to unlock."
                elif result.status == "NEEDS_ENROLLMENT":
                    message = "Enroll this device user to begin local face authentication."
                elif result.message:
                    message = result.message
            self.secure_overlay_label.config(text=status_text)
            self.secure_overlay_desc.config(text=message)
            self.secure_overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
            self.secure_overlay.lift()
            self.status_badge.config(text="LOCKED", bg=self.c['orange'], fg='white')
        else:
            self.secure_overlay.place_forget()
            if self.monitoring:
                self.status_badge.config(text="LIVE", bg=self.c['blue'], fg='white')

    def _update_face_auth_ui(self, result):
        """Render local face-auth state in the dashboard."""
        self.face_auth_state = result
        now = time.time()
        if result.face_visible:
            self.face_auth_last_seen_ts = now
        self.auth_conf_lbl.config(text=f"Confidence: {result.confidence:.0f}%")
        self.auth_security_lbl.config(text=f"Security: {result.security_score:.0f}%")

        if self.face_auth_locked and result.status != "AUTHORIZED":
            self.auth_status_lbl.config(text="LOCKED", fg=self.c['orange'])
            self.auth_message_lbl.config(text="Show your face to unlock.", fg=self.c['orange'])
            self.auth_toggle_btn.config(text="⏹ LOCKED", bg=self.c['orange'])
            self._set_secure_overlay(True, result)
            return

        if result.status != "AUTHORIZED" and (now - self.face_auth_last_auth_ts) < self.face_auth_grace_seconds:
            self.auth_status_lbl.config(text="ON • Trusted", fg=self.c['green'])
            self.auth_message_lbl.config(text="Rechecking face...", fg=self.c['green'])
            self.auth_toggle_btn.config(text="✅ ACTIVE", bg=self.c['green'])
            self._set_secure_overlay(False, result)
            return

        if result.status == "AUTHORIZED":
            self.face_auth_last_auth_ts = now
            self.face_auth_unknown_since = 0.0
            self.auth_status_lbl.config(text=f"ON • {result.profile_name or 'Trusted'}", fg=self.c['green'])
            self.auth_message_lbl.config(text=result.message or "Authorized", fg=self.c['green'])
            self.auth_toggle_btn.config(text="✅ ACTIVE", bg=self.c['green'])
            self._set_secure_overlay(False, result)
        elif result.status == "NEEDS_ENROLLMENT":
            self.face_auth_unknown_since = 0.0
            self.auth_status_lbl.config(text="ENROLL", fg=self.c['yellow'])
            self.auth_message_lbl.config(text=result.message or "Enroll a face profile.", fg=self.c['yellow'])
            self.auth_toggle_btn.config(text="⏹ LOCKED", bg=self.c['orange'])
            self._set_secure_overlay(True, result)
        elif result.status == "ABSENT":
            self.auth_status_lbl.config(text="ABSENT", fg=self.c['orange'])
            self.auth_message_lbl.config(text=result.message or "Face not visible.", fg=self.c['orange'])
            should_lock = (
                (now - self.face_auth_last_seen_ts) >= self.face_auth_absent_seconds
                and (now - self.face_auth_last_auth_ts) >= self.face_auth_grace_seconds
            )
            self.auth_toggle_btn.config(text="⏹ LOCKED" if should_lock else "✅ ACTIVE", bg=self.c['orange'])
            self._set_secure_overlay(should_lock, result)
        else:
            if self.face_auth_unknown_since == 0.0:
                self.face_auth_unknown_since = now
            self.auth_status_lbl.config(text="CHECK", fg=self.c['orange'])
            self.auth_message_lbl.config(text=result.message or "Unknown face detected.", fg=self.c['orange'])
            should_lock = (
                (now - self.face_auth_unknown_since) >= self.face_auth_unknown_seconds
                and (now - self.face_auth_last_auth_ts) >= self.face_auth_grace_seconds
            )
            self.auth_toggle_btn.config(text="⏹ LOCKED" if should_lock else "✅ ACTIVE", bg=self.c['orange'])
            self._set_secure_overlay(should_lock, result)
    
    # ========== FOCUS MODE METHODS ==========
    
    def toggle_focus_mode(self):
        """Toggle Focus AI Mode on/off."""
        if self.focus_session.is_active:
            self.stop_focus_mode()
        else:
            self.start_focus_mode()
    
    def start_focus_mode(self):
        """Start Focus AI Mode."""
        try:
            duration = int(self.focus_duration_var.get())
        except:
            duration = 25
            
        success, msg = self.focus_session.start(duration)
        
        if success:
            self.focus_toggle_btn.config(text="⏹ STOP", bg=self.c['red'])
            self.focus_status_lbl.config(text="ACTIVE", fg=self.c['green'])
            self.focus_card.config(highlightbackground=self.c['green'])
            self.add_log(f"🎯 Focus Mode started ({duration} min)")
            
            # Auto-start main monitoring if not running
            if not self.monitoring:
                self.start()
                self.add_log("📊 Auto-started monitoring with Focus Mode")
            
            # Start fast focus check loop
            self._start_focus_check_loop()
            
            # Play start sound
            try:
                winsound.Beep(800, 200)
            except:
                pass
    
    def stop_focus_mode(self):
        """Stop Focus AI Mode."""
        # Stop the focus check loop
        self.focus_check_running = False
        
        success, msg = self.focus_session.stop()
        
        if success:
            self.focus_toggle_btn.config(text="▶ START", bg=self.c['green'])
            self.focus_status_lbl.config(text="OFF", fg=self.c['dim'])
            self.focus_card.config(highlightbackground=self.c['blue'])
            self.add_log(f"🏁 {msg}")
            
            # Update session stats
            self._update_focus_stats()
            
            # Reset timer display
            try:
                duration = int(self.focus_duration_var.get())
            except:
                duration = 25
            self.focus_timer_lbl.config(text=f"{duration:02d}:00")
            
            # Show completion message
            messagebox.showinfo("Focus Session Complete", 
                               f"Great job! Your focus score: {self.focus_session.completed_sessions[-1]['focus_score']}\n"
                               f"Distractions: {self.focus_session.completed_sessions[-1]['distraction_count']}")
    
    def reset_focus_mode(self):
        """Reset Focus Mode completely."""
        # Stop if running
        if self.focus_session.is_active:
            self.focus_check_running = False
            self.focus_session.is_active = False
        
        # Create new session
        self.focus_session = FocusSession()
        
        # Reset UI
        self.focus_toggle_btn.config(text="▶ START", bg=self.c['green'])
        self.focus_status_lbl.config(text="OFF", fg=self.c['dim'])
        self.focus_card.config(highlightbackground=self.c['blue'])
        
        try:
            duration = int(self.focus_duration_var.get())
        except:
            duration = 25
        self.focus_timer_lbl.config(text=f"{duration:02d}:00")
        
        self.focus_score_lbl.config(text="100", fg=self.c['green'])
        self.productivity_lbl.config(text="100%", fg=self.c['green'])
        self.distraction_count_lbl.config(text="0", fg=self.c['green'])
        
        self.distraction_log_text.config(state=tk.NORMAL)
        self.distraction_log_text.delete(1.0, tk.END)
        self.distraction_log_text.insert(tk.END, "No distractions - Stay focused! 🎯")
        self.distraction_log_text.config(state=tk.DISABLED)
        
        self.focus_sessions_lbl.config(text="Sessions: 0 | Total Focus: 0m")
        
        self.add_log("🔄 Focus Mode reset")

    # ===== STUDY ASSISTANT =====

    def toggle_study_mode(self):
        self.study_mode_enabled = not self.study_mode_enabled
        if self.study_mode_enabled:
            self.study_toggle_btn.config(text="ON", bg=self.c['blue'])
            self.study_status_lbl.config(text="Mode: Adaptive Pomodoro", fg=self.c['text'])
            self.add_log("📚 Study Assistant enabled")
        else:
            self.study_toggle_btn.config(text="OFF", bg=self.c['dim'])
            self.study_status_lbl.config(text="Mode: Disabled", fg=self.c['dim'])
            self.study_cycle_lbl.config(text="Suggested cycle: --")
            self.add_log("📚 Study Assistant disabled")

    def _update_study_plan(self, analysis: Dict):
        if not self.study_mode_enabled:
            return
        fatigue = float(analysis.get('fatigue_prob', 0))
        focus = float(analysis.get('focus_score', 0))

        if fatigue >= 65:
            cycle = "20 min focus / 5 min break"
        elif focus >= 75:
            cycle = "50 min focus / 10 min break"
        else:
            cycle = "25 min focus / 5 min break"

        self.study_cycle_lbl.config(text=f"Suggested cycle: {cycle}")
    
    def _update_focus_stats(self):
        """Update focus session statistics display."""
        sessions = len(self.focus_session.completed_sessions)
        total_mins = int(self.focus_session.total_focus_time / 60)
        self.focus_sessions_lbl.config(text=f"Sessions: {sessions} | Total Focus: {total_mins}m")
    
    def export_focus_data(self):
        """Export focus session data to a file."""
        from tkinter import filedialog
        import json
        from datetime import datetime
        
        sessions = self.focus_session.completed_sessions
        
        if not sessions:
            messagebox.showinfo("Export Focus Data", "No focus sessions to export yet.\nComplete at least one focus session first.")
            return
        
        # Prepare export data
        export_data = {
            'export_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_sessions': len(sessions),
            'total_focus_time_minutes': int(self.focus_session.total_focus_time / 60),
            'average_focus_score': np.mean([s.get('focus_score', 100) for s in sessions]),
            'total_distractions': sum(s.get('distraction_count', 0) for s in sessions),
            'sessions': []
        }
        
        for i, session in enumerate(sessions, 1):
            session_data = {
                'session_number': i,
                'duration_minutes': int(session.get('duration', 0) / 60),
                'focus_score': session.get('focus_score', 100),
                'distraction_count': session.get('distraction_count', 0),
                'distracting_apps': session.get('distracting_apps', [])
            }
            export_data['sessions'].append(session_data)
        
        # Ask user for save location
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")],
            initialfilename=f"focus_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            title="Export Focus Session Data"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                self.add_log(f"📊 Focus data exported to {filename}")
                messagebox.showinfo("Export Successful", f"Focus data exported to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Export Failed", f"Failed to export data:\n{str(e)}")
    
    # ===== POSTURE DETECTION =====
    
    def toggle_posture(self):
        """Toggle posture detection on/off."""
        if not self.posture_detector:
            messagebox.showerror("Posture Detection", 
                               "Posture detection not available.\n\n"
                               "Install required packages:\n"
                               "pip install mediapipe opencv-python")
            return
        
        if self.posture_enabled:
            self.stop_posture()
        else:
            self.start_posture()
    
    def start_posture(self):
        """Start posture monitoring."""
        if not self.posture_detector:
            return
        
        success, msg = self.posture_detector.start(callback=self._on_posture_update)
        
        if success:
            self.posture_enabled = True
            self.posture_toggle_btn.config(text="⏹ STOP", bg=self.c['red'])
            self.posture_status_lbl.config(text="ACTIVE", fg=self.c['green'])
            self.posture_card.config(highlightbackground=self.c['green'])
            self.posture_issues_lbl.config(text="📷 Monitoring...", fg=self.c['green'])
            self.add_log("🧘 Posture monitoring started")
        else:
            messagebox.showerror("Posture Detection", f"Failed to start:\n{msg}")
    
    def stop_posture(self):
        """Stop posture monitoring."""
        if not self.posture_detector:
            return
        
        success, msg = self.posture_detector.stop()
        
        self.posture_enabled = False
        self.posture_toggle_btn.config(text="▶ START", bg=self.c['green'])
        self.posture_status_lbl.config(text="OFF", fg=self.c['dim'])
        self.posture_card.config(highlightbackground=self.c['border'])
        
        if success:
            stats = self.posture_detector.get_posture_stats()
            self.posture_issues_lbl.config(
                text=f"✓ Session: {stats['good_percentage']:.0f}% good posture",
                fg=self.c['green'] if stats['good_percentage'] > 70 else self.c['yellow']
            )
            self.add_log(f"🧘 Posture stopped - {stats['good_percentage']:.0f}% good")

        if self.face_auth_enabled:
            self._set_secure_overlay(True, self.face_auth_state)
    
    def _on_posture_update(self, status, frame=None, raw_frame=None):
        """Callback for posture updates (from background thread)."""
        # Schedule UI update on main thread
        self.root.after(0, lambda s=status, f=frame, rf=raw_frame: self._update_posture_ui(s, f, rf))
    
    def _update_posture_ui(self, status, frame=None, raw_frame=None):
        """Update posture UI elements."""
        if not self.posture_enabled:
            return

        if raw_frame is not None:
            self.last_camera_frame = raw_frame
        else:
            self.last_camera_frame = frame

        if self.face_auth and self.face_auth_enabled and self.last_camera_frame is not None:
            try:
                presence = self.face_auth.verify_frame(self.last_camera_frame)
                self._update_face_auth_ui(presence)
            except Exception as e:
                self.add_log(f"⚠️ Face auth error: {e}")
        
        # Update score
        score = status.score
        score_color = self.c['green'] if score >= 70 else self.c['yellow'] if score >= 40 else self.c['red']
        self.posture_score_lbl.config(text=f"{score:.0f}", fg=score_color)
        self.latest_posture_score = score
        
        # Update times
        if self.posture_detector:
            stats = self.posture_detector.get_posture_stats()
            good_mins = int(stats['good_time'] / 60)
            bad_mins = int(stats['bad_time'] / 60)
            self.posture_good_lbl.config(text=f"{good_mins}m")
            self.posture_bad_lbl.config(text=f"{bad_mins}m")
        
        # Update video preview
        if frame is not None:
            self._update_posture_preview(frame, status)
        
        # Detect posture change and show small alert
        if status.confidence > 0.5:
            if self.last_posture_good is None:
                self.last_posture_good = status.is_good
            elif status.is_good != self.last_posture_good:
                self.last_posture_good = status.is_good
                self._flash_posture_change_alert("Posture changed")

        # Update issues
        if status.issues and status.issues[0] != "No person detected":
            issues_text = " | ".join(status.issues[:2])
            self.posture_issues_lbl.config(text=f"⚠️ {issues_text}", fg=self.c['orange'])
            
            # Show posture alert if bad for too long
            if self.posture_detector and self.posture_detector.bad_posture_time > 30:
                if time.time() - self.posture_detector.last_alert_time > 60:  # Alert once per minute
                    self.posture_detector.last_alert_time = time.time()
                    self._show_posture_alert(issues_text)
        elif status.confidence > 0.5:
            self.posture_issues_lbl.config(text="✓ Good posture! Keep it up!", fg=self.c['green'])
        else:
            self.posture_issues_lbl.config(text="📷 Looking for you...", fg=self.c['dim'])

    def _flash_posture_change_alert(self, message: str):
        """Show a small orange posture-change alert briefly."""
        self.posture_alert_lbl.config(text=f"⚠️ {message}")
        self.root.after(3000, lambda: self.posture_alert_lbl.config(text=""))
    
    def _show_posture_alert(self, issues: str):
        """Show posture correction alert."""
        self.add_log(f"⚠️ Posture alert: {issues}")
        
        # Play alert sound
        try:
            import winsound
            winsound.PlaySound("SystemHand", winsound.SND_ASYNC)
        except:
            pass
    
    def _update_posture_preview(self, frame, status):
        """Update the video preview with the current webcam frame (skeleton already drawn)."""
        if not PIL_AVAILABLE:
            return
        
        try:
            # Frame already has skeleton drawn by posture_detection module
            # Convert BGR to RGB for display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame to fit in the preview area (larger for better visibility)
            preview_width = 220
            preview_height = 165
            h, w = rgb_frame.shape[:2]
            scale = min(preview_width / w, preview_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            rgb_frame = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Add border based on posture status
            border_color = (0, 255, 0) if status.is_good else (255, 165, 0) if status.score >= 50 else (255, 0, 0)
            cv2.rectangle(rgb_frame, (0, 0), (new_w - 1, new_h - 1), border_color, 3)
            
            # Add score and status text overlay
            score_text = f"Score: {status.score:.0f}"
            status_text = "GOOD" if status.is_good else "FIX"
            cv2.putText(rgb_frame, score_text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(rgb_frame, score_text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, border_color, 1)
            
            # Status indicator in top right
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.putText(rgb_frame, status_text, (new_w - text_size[0] - 8, 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, border_color, 2)
            
            # Convert to PIL Image then to PhotoImage
            pil_image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(image=pil_image)
            
            # Update label
            self.posture_preview_lbl.config(image=photo, text="", width=new_w, height=new_h)
            self.posture_preview_image = photo  # Keep reference to prevent garbage collection
        except Exception as e:
            print(f"Preview update error: {e}")
    
    def show_session_history(self):
        """Show detailed session history popup."""
        popup = tk.Toplevel(self.root)
        popup.title("📊 Focus Session History")
        popup.geometry("400x350")
        popup.configure(bg=self.c['bg'])
        popup.transient(self.root)
        popup.grab_set()
        
        # Header
        tk.Label(popup, text="📊 Focus Session History", font=('Segoe UI', 14, 'bold'),
                bg=self.c['bg'], fg=self.c['text']).pack(pady=(15, 10))
        
        # Summary stats
        sessions = self.focus_session.completed_sessions
        total_sessions = len(sessions)
        total_mins = int(self.focus_session.total_focus_time / 60)
        total_distractions = sum(s.get('distraction_count', 0) for s in sessions)
        avg_score = np.mean([s.get('focus_score', 100) for s in sessions]) if sessions else 0
        
        summary_frame = tk.Frame(popup, bg=self.c['card'], highlightbackground=self.c['border'], highlightthickness=1)
        summary_frame.pack(fill=tk.X, padx=15, pady=5)
        
        tk.Label(summary_frame, text=f"Total Sessions: {total_sessions}", font=('Segoe UI', 10),
                bg=self.c['card'], fg=self.c['text']).pack(anchor='w', padx=10, pady=2)
        tk.Label(summary_frame, text=f"Total Focus Time: {total_mins} minutes", font=('Segoe UI', 10),
                bg=self.c['card'], fg=self.c['text']).pack(anchor='w', padx=10, pady=2)
        tk.Label(summary_frame, text=f"Total Distractions: {total_distractions}", font=('Segoe UI', 10),
                bg=self.c['card'], fg=self.c['text']).pack(anchor='w', padx=10, pady=2)
        tk.Label(summary_frame, text=f"Average Focus Score: {avg_score:.0f}%", font=('Segoe UI', 10),
                bg=self.c['card'], fg=self.c['green'] if avg_score > 70 else self.c['yellow'] if avg_score > 40 else self.c['red']).pack(anchor='w', padx=10, pady=2)
        
        # Session list
        tk.Label(popup, text="Recent Sessions:", font=('Segoe UI', 10, 'bold'),
                bg=self.c['bg'], fg=self.c['text']).pack(anchor='w', padx=15, pady=(10, 5))
        
        list_frame = tk.Frame(popup, bg=self.c['card'], highlightbackground=self.c['border'], highlightthickness=1)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)
        
        if not sessions:
            tk.Label(list_frame, text="No completed sessions yet.\nStart a Focus session to track your productivity!",
                    font=('Segoe UI', 9), bg=self.c['card'], fg=self.c['dim']).pack(pady=20)
        else:
            for i, session in enumerate(reversed(sessions[-10:]), 1):
                duration = int(session.get('duration', 0) / 60)
                score = session.get('focus_score', 0)
                distractions = session.get('distraction_count', 0)
                score_color = self.c['green'] if score > 70 else self.c['yellow'] if score > 40 else self.c['red']
                
                session_text = f"#{len(sessions) - i + 1}: {duration}min | Score: {score}% | Distractions: {distractions}"
                tk.Label(list_frame, text=session_text, font=('Segoe UI', 9),
                        bg=self.c['card'], fg=score_color).pack(anchor='w', padx=10, pady=2)
        
        # Close button
        tk.Button(popup, text="Close", font=('Segoe UI', 10),
                 bg=self.c['blue'], fg='white', relief=tk.FLAT,
                 command=popup.destroy, cursor='hand2').pack(pady=15, ipadx=20, ipady=4)
    
    def _start_focus_check_loop(self):
        """Start the fast focus distraction check loop."""
        self.focus_check_running = True
        self._focus_check_iteration()
    
    def _focus_check_iteration(self):
        """Fast iteration - check for distractions every 500ms."""
        if not self.focus_check_running or not self.focus_session.is_active:
            return
        
        try:
            # Get current active window directly
            window_title = self._get_active_window_title()
            
            if window_title:
                result = self.focus_session.check_window_distraction(window_title)
                
                if result.get('is_distraction'):
                    if not self.focus_warning_window:
                        self.show_distraction_warning(result.get('site', window_title))
                    if result.get('show_warning'):
                        self.add_log(f"⚠️ Distraction: {result.get('site', '')[:25]}")
                else:
                    if self.focus_warning_window:
                        try:
                            self.focus_warning_window.destroy()
                        except:
                            pass
                        self.focus_warning_window = None
                
                # Check if session ended
                if result.get('session_ended'):
                    self.focus_toggle_btn.config(text="▶ START", bg=self.c['green'])
                    self.focus_status_lbl.config(text="COMPLETED", fg=self.c['blue'])
                    self.add_log("🎯 Focus session completed!")
                    self.focus_check_running = False
                    return
        except Exception as e:
            pass
        
        # Schedule next check in 500ms
        if self.focus_check_running:
            self.root.after(500, self._focus_check_iteration)
    
    def _get_active_window_title(self) -> str:
        """Get the title of the currently active window."""
        try:
            import ctypes
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            length = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            return buf.value
        except:
            return ""
    
    def show_distraction_warning(self, site_name):
        """Show an aggressive distraction warning overlay."""
        # Close existing warning
        if self.focus_warning_window:
            try:
                self.focus_warning_window.destroy()
            except:
                pass
        
        # Play warning sound
        def play_alarm():
            try:
                for _ in range(3):
                    winsound.Beep(800, 150)
                    winsound.Beep(600, 150)
            except:
                pass
        
        threading.Thread(target=play_alarm, daemon=True).start()
        
        # Create large warning overlay
        warning = tk.Toplevel(self.root)
        warning.title("⚠️ FOCUS MODE ALERT")
        
        # Make it larger and more visible
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        w, h = 500, 200
        x = (screen_w - w) // 2
        y = 50  # Near top of screen
        
        warning.geometry(f"{w}x{h}+{x}+{y}")
        warning.configure(bg='#8b0000')
        warning.overrideredirect(True)  # No title bar
        warning.attributes('-topmost', True)
        warning.attributes('-alpha', 0.95)
        
        self.focus_warning_window = warning
        
        # Content with large text
        tk.Label(warning, text="🚨 DISTRACTION DETECTED! 🚨", 
                font=('Segoe UI', 18, 'bold'),
                bg='#8b0000', fg='white').pack(pady=(20, 10))
        
        tk.Label(warning, text=f"You opened: {site_name[:40]}", 
                font=('Segoe UI', 12),
                bg='#8b0000', fg='#ffcccc').pack(pady=5)
        
        tk.Label(warning, text="⏰ GET BACK TO WORK! ⏰", 
                font=('Segoe UI', 14, 'bold'),
                bg='#8b0000', fg='#ffff00').pack(pady=10)
        
        btn_frame = tk.Frame(warning, bg='#8b0000')
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="✓ I'm back to work!", 
                 font=('Segoe UI', 11, 'bold'),
                 bg='#28a745', fg='white', relief=tk.FLAT,
                 command=warning.destroy, cursor='hand2').pack(side=tk.LEFT, padx=5, ipadx=15, ipady=5)
        
        # Bring to front
        warning.lift()
        warning.focus_force()
    
    def update_focus_ui(self):
        """Update Focus Mode UI elements."""
        if not self.focus_session.is_active:
            # Session might have just ended
            self.focus_check_running = False
            return
        
        status = self.focus_session.get_status()
        
        # Check if session ended (timer expired)
        if not status.get('is_active', True):
            self.focus_toggle_btn.config(text="▶ START", bg=self.c['green'])
            self.focus_status_lbl.config(text="COMPLETED", fg=self.c['blue'])
            self.focus_check_running = False
            return
        
        # Timer
        self.focus_timer_lbl.config(text=status['remaining'])
        
        # Metrics
        score = status['focus_score']
        self.focus_score_lbl.config(text=str(score), fg=self.color_for(100 - score))
        
        prod = status['productivity']
        self.productivity_lbl.config(text=f"{prod:.0f}%", fg=self.color_for(100 - prod))
        
        dc = status['distraction_count']
        dc_color = self.c['green'] if dc == 0 else (self.c['yellow'] if dc < 3 else self.c['red'])
        self.distraction_count_lbl.config(text=str(dc), fg=dc_color)
        
        # Distraction log
        distractions = status.get('distractions', [])
        if distractions:
            log_text = "\n".join([f"[{d['timestamp']}] {d['site']} ({d['duration']:.0f}s)" 
                                  for d in distractions[-3:]])
        else:
            log_text = "No distractions - Stay focused! 🎯"
        
        self.distraction_log_text.config(state=tk.NORMAL)
        self.distraction_log_text.delete(1.0, tk.END)
        self.distraction_log_text.insert(tk.END, log_text)
        self.distraction_log_text.config(state=tk.DISABLED)
    
    def check_focus_distractions(self, metrics):
        """Check for distractions during Focus Mode."""
        if not self.focus_session.is_active:
            return
        
        current_app = metrics.get('current_app', '')
        if not current_app:
            return
        
        result = self.focus_session.check_window_distraction(current_app)
        
        if result.get('is_distraction') and result.get('show_warning'):
            self.show_distraction_warning(result.get('site', current_app))
            self.add_log(f"⚠️ Distraction: {result.get('site', '')[:25]}")
        
        # Check if session ended
        if result.get('session_ended'):
            self.focus_toggle_btn.config(text="▶ START", bg=self.c['green'])
            self.focus_status_lbl.config(text="COMPLETED", fg=self.c['blue'])
            self.add_log("🎯 Focus session completed!")
        
    # ========================================
        
    def reset(self):
        """Full reset of all monitoring data and metrics."""
        # Ask for confirmation
        if self.monitoring or self.focus_session.is_active:
            confirm = messagebox.askyesno(
                "Confirm Reset",
                "This will reset ALL data including:\n\n"
                "• Risk level and metrics\n"
                "• Focus session history\n"
                "• Analysis history\n"
                "• Graph data\n\n"
                "Are you sure?",
                icon='warning'
            )
            if not confirm:
                return
        
        # Stop monitoring
        if self.capture:
            self.capture.stop()
        self.capture = None
        self.monitoring = False
        
        # Reset engine (fresh ML models, clear history)
        self.engine = MindShieldEngine()
        
        # Reset Focus Mode completely
        self.focus_check_running = False
        if self.focus_session.is_active:
            self.focus_session.is_active = False
        self.focus_session = FocusSession()
        self.focus_toggle_btn.config(text="▶ START", bg=self.c['green'])
        self.focus_status_lbl.config(text="OFF", fg=self.c['dim'])
        self.focus_card.config(highlightbackground=self.c['blue'])
        self.focus_timer_lbl.config(text="25:00")
        self.focus_score_lbl.config(text="100", fg=self.c['green'])
        self.productivity_lbl.config(text="100%", fg=self.c['green'])
        self.distraction_count_lbl.config(text="0", fg=self.c['green'])
        self.distraction_log_text.config(state=tk.NORMAL)
        self.distraction_log_text.delete(1.0, tk.END)
        self.distraction_log_text.insert(tk.END, "No distractions - Stay focused! 🎯")
        self.distraction_log_text.config(state=tk.DISABLED)
        self.focus_sessions_lbl.config(text="Sessions: 0 | Total Focus: 0m")
        
        # Reset main display
        self.risk_val.config(text="--", fg=self.c['green'])
        self.risk_lvl.config(text="READY", fg=self.c['dim'])
        self.trend_lbl.config(text="")
        self.start_btn.config(text="▶ Start", bg=self.c['green'])
        self.status.config(text="● Stopped", fg=self.c['red'])
        self.timer.config(text="00:00:00")
        
        # Reset all metric labels
        for v in self.metrics.values():
            v.config(text="--")
        for v in self.stat_labels.values():
            v.config(text="0")
        for lbl in self.recs:
            lbl.config(text="")
        for lbl in self.switch_labels:
            lbl.config(text="")
        
        # Clear graph data
        for key in self.graph_data:
            self.graph_data[key].clear()
        self.graph_canvas.delete("all")
        
        # Reset break tracking
        if hasattr(self, 'last_break_time'):
            del self.last_break_time
        if hasattr(self, 'break_count'):
            self.break_count = 0
        if hasattr(self, '_break_shown'):
            del self._break_shown
        
        # Reset posture detection
        if self.posture_enabled:
            self.stop_posture()
        self.posture_score_lbl.config(text="--", fg=self.c['green'])
        self.posture_good_lbl.config(text="0m")
        self.posture_bad_lbl.config(text="0m")
        self.posture_issues_lbl.config(text="📷 Camera ready - Click START to monitor", fg=self.c['dim'])
            
        self.add_log("🔄 Full reset complete - All data cleared")
        
    def export(self):
        if not self.engine.history and not self.focus_session.completed_sessions:
            messagebox.showinfo("Export", "No data yet")
            return
        
        today_summary = self.session_store.get_today_summary()
        export_data = {
            'analysis_history': list(self.engine.history),
            'focus_sessions': self.focus_session.completed_sessions,
            'total_focus_time': self.focus_session.total_focus_time,
            'today_summary': today_summary,
            'baseline_profile': {
                'sample_count': self.baseline_manager.sample_count,
                'calibrated': self.baseline_manager.is_calibrated,
                'calibration_target': self.baseline_manager.calibration_samples,
            },
            'exported_at': datetime.now().isoformat()
        }
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile=f"mindshield_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        if filename:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            self.add_log(f"📊 Exported to {os.path.basename(filename)}")
            
    def add_log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.config(state=tk.NORMAL)
        self.log.insert(tk.END, f"[{ts}] {msg}\n")
        self.log.see(tk.END)
        self.log.config(state=tk.DISABLED)

    def _add_card_accent(self, card: tk.Frame, left_color: str, right_color: str):
        """Add a thin accent bar to a card to simulate gradient energy."""
        bar = tk.Frame(card, bg=left_color, height=4)
        bar.pack(fill=tk.X, side=tk.TOP)
        right = tk.Frame(bar, bg=right_color)
        right.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)

    def _start_risk_pulse(self):
        if self._risk_pulse_active:
            return
        self._risk_pulse_active = True
        self._risk_pulse_tick()

    def _risk_pulse_tick(self):
        if not self._risk_pulse_active:
            return
        self._risk_pulse_on = not self._risk_pulse_on
        color = self.c['red'] if self._risk_pulse_on else self.c['pink']
        self.risk_card.config(highlightbackground=color)
        self.risk_val.config(fg=color)
        self._risk_pulse_job = self.root.after(450, self._risk_pulse_tick)

    def _stop_risk_pulse(self, normal_color: str):
        if not self._risk_pulse_active:
            self.risk_card.config(highlightbackground=self.c['border'])
            self.risk_val.config(fg=normal_color)
            return
        self._risk_pulse_active = False
        if self._risk_pulse_job:
            try:
                self.root.after_cancel(self._risk_pulse_job)
            except:
                pass
            self._risk_pulse_job = None
        self.risk_card.config(highlightbackground=self.c['border'])
        self.risk_val.config(fg=normal_color)
        
    def color_for(self, val, invert=False):
        if invert:
            val = 100 - val
        if val < 25:
            return self.c['green']
        elif val < 50:
            return self.c['yellow']
        elif val < 75:
            return self.c['orange']
        return self.c['red']
    
    def draw_graph(self):
        """Draw the real-time graph."""
        canvas = self.graph_canvas
        canvas.delete("all")
        
        # Get canvas dimensions
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        
        # Minimum size check - use actual dimensions
        if w < 50:
            w = 400
        if h < 50:
            h = 150
        
        padding_left = 35
        padding_right = 10
        padding_top = 10
        padding_bottom = 15
        
        graph_w = w - padding_left - padding_right
        graph_h = h - padding_top - padding_bottom
        
        if graph_w < 10 or graph_h < 10:
            return
        
        # Draw background fill
        canvas.create_rectangle(padding_left, padding_top, w - padding_right, h - padding_bottom,
                       fill=self.c['bg'], outline=self.c['border'])
        
        # Draw background grid
        for i in range(5):
            y = padding_top + int(graph_h * i / 4)
            canvas.create_line(padding_left, y, w - padding_right, y, 
                             fill=self.c['border'], dash=(2, 4))
            # Y-axis labels
            val = 100 - (i * 25)
            canvas.create_text(padding_left - 5, y, text=str(val), 
                             font=('Bahnschrift', 8), fill=self.c['dim'], anchor=tk.E)
        
        # Draw vertical axis
        canvas.create_line(padding_left, padding_top, padding_left, h - padding_bottom,
                          fill=self.c['border'])
        
        # Check if we have any data
        has_data = any(len(self.graph_data[k]) >= 2 for k in self.graph_data)
        
        if not has_data:
            # Show message when no data
            canvas.create_text(w // 2, h // 2, text="Start monitoring to see real-time data",
                             font=('Bahnschrift', 10), fill=self.c['dim'], anchor=tk.CENTER)
            return
        
        # Draw data lines
        colors = {
            'risk': self.c['red'],
            'fatigue': self.c['orange'],
            'cognitive': self.c['yellow'],
            'focus': self.c['green'],
        }
        
        for key, color in colors.items():
            data = list(self.graph_data[key])
            if len(data) < 2:
                continue
            
            points = []
            for i, val in enumerate(data):
                # Clamp value to 0-100
                val = max(0, min(100, val))
                x = padding_left + int(graph_w * i / max(len(data) - 1, 1))
                y = padding_top + int(graph_h * (100 - val) / 100)
                points.append((x, y))
            
            # Draw line segments
            for i in range(len(points) - 1):
                canvas.create_line(points[i][0], points[i][1], 
                                  points[i+1][0], points[i+1][1],
                                  fill=color, width=2)
            
            # Draw last point marker
            if points:
                lx, ly = points[-1]
                canvas.create_oval(lx-4, ly-4, lx+4, ly+4, fill=color, outline='white', width=1)
        
    def update_display(self, analysis, metrics):
        # Risk
        risk = analysis['risk_score']
        self.risk_val.config(text=f"{risk:.0f}", fg=self.color_for(risk))
        self.risk_lvl.config(text=analysis['risk_level'], fg=self.color_for(risk))

        # Risk progress bar update
        bar_w = self.risk_bar.winfo_width() or 200
        fill_w = int(bar_w * min(max(risk, 0), 100) / 100)
        self.risk_bar.coords(self.risk_bar_fill, 0, 0, fill_w, 8)
        self.risk_bar.itemconfig(self.risk_bar_fill, fill=self.color_for(risk))

        # Critical pulse
        if risk >= 75:
            self._start_risk_pulse()
        else:
            self._stop_risk_pulse(self.color_for(risk))
        
        # Trend
        trend = analysis.get('trend', 'stable')
        if trend == 'improving':
            self.trend_lbl.config(text="📉 Improving", fg=self.c['green'])
        elif trend == 'worsening':
            self.trend_lbl.config(text="📈 Worsening", fg=self.c['orange'])
        else:
            self.trend_lbl.config(text="➡️ Stable", fg=self.c['dim'])
        
        # Metrics
        self.metrics['cognitive'].config(text=f"{analysis['cognitive_load']:.0f}%",
                                         fg=self.color_for(analysis['cognitive_load']))
        self.metrics['fatigue'].config(text=f"{analysis['fatigue_prob']:.0f}%",
                                       fg=self.color_for(analysis['fatigue_prob']))
        self.metrics['state'].config(text=f"{analysis.get('cognitive_state', 0):.0f}%",
                         fg=self.color_for(analysis.get('cognitive_state', 0)))
        self.metrics['focus'].config(text=f"{analysis['focus_score']:.0f}%",
                                     fg=self.color_for(analysis['focus_score'], True))
        self.metrics['productivity'].config(text=f"{analysis['productivity']:.0f}%",
                                            fg=self.color_for(analysis['productivity'], True))
        self.metrics['drift'].config(text=f"{analysis['behavioral_drift']:.0f}%",
                                     fg=self.color_for(analysis['behavioral_drift']))
        self.metrics['eye_strain'].config(text=f"{analysis['eye_strain']:.0f}%",
                                          fg=self.color_for(analysis['eye_strain']))
        confidence = analysis.get('confidence', 0)
        self.metrics['confidence'].config(text=f"{confidence:.0f}%",
                          fg=self.color_for(100 - confidence))
        self.metrics['typing'].config(text=f"{metrics.get('typing_speed_wpm', 0):.1f} WPM")
        self.metrics['errors'].config(text=f"{metrics.get('error_count', 0)}")

        emotion = analysis.get('emotion_state', {})
        if emotion:
            self.emotion_stress_lbl.config(
                text=f"Stress: {emotion.get('stress', 0):.0f}% | Frustration: {emotion.get('frustration', 0):.0f}%",
                fg=self.color_for(emotion.get('stress', 0)),
            )
            self.emotion_energy_lbl.config(
                text=f"Recovery: {emotion.get('recovery', 0):.0f}% | Wellness: {emotion.get('wellness', 0):.0f}%",
                fg=self.color_for(emotion.get('wellness', 0), True),
            )

        burnout = analysis.get('burnout_state', {})
        if burnout:
            self.burnout_risk_lbl.config(
                text=f"Risk: {burnout.get('burnout_risk', 0):.0f}% | Trend: {burnout.get('trend', 'stable').title()}",
                fg=self.color_for(burnout.get('burnout_risk', 0)),
            )
            self.burnout_recovery_lbl.config(
                text=f"Recovery: {burnout.get('recovery_quality', 0):.0f}% | Sustainability: {burnout.get('sustainability_score', 0):.0f}%",
                fg=self.color_for(burnout.get('sustainability_score', 0), True),
            )

        privacy = self.privacy.as_dict()
        privacy_text = ["Local-only active"]
        privacy_text.append("Webcam on" if privacy.get('webcam_enabled', True) else "Webcam off")
        privacy_text.append("Emotion AI on" if privacy.get('emotion_ai_enabled', False) else "Emotion AI off")
        if privacy.get('paused'):
            privacy_text.append("Paused")
        self.privacy_status_lbl.config(text=" | ".join(privacy_text), fg=self.c['yellow'] if privacy.get('paused') else self.c['green'])
        self.pause_privacy_btn.config(text="▶ Resume AI" if privacy.get('paused') else "⏸ Pause AI")
        self.emotion_privacy_btn.config(text="🧠 Emotion AI ON" if privacy.get('emotion_ai_enabled') else "🧠 Emotion AI OFF")
        self.webcam_privacy_btn.config(text="📷 Webcam ON" if privacy.get('webcam_enabled') else "📷 Webcam OFF")
        
        # Timer
        dur = metrics.get('session_duration', 0)
        h, m, s = int(dur//3600), int((dur%3600)//60), int(dur%60)
        self.timer.config(text=f"{h:02d}:{m:02d}:{s:02d}")
        
        # Stats
        self.stat_labels['keys'].config(text=f"{metrics.get('total_keystrokes', 0)}")
        self.stat_labels['clicks'].config(text=f"{metrics.get('total_clicks', 0)}")
        self.stat_labels['words'].config(text=f"{metrics.get('word_count', 0)}")
        self.stat_labels['switches'].config(text=f"{metrics.get('app_switches_total', 0)}")
        self.stat_labels['cpu'].config(text=f"{metrics.get('cpu_usage', 0):.0f}%")
        self.stat_labels['mem'].config(text=f"{metrics.get('memory_usage', 0):.0f}%")
        
        # Recommendations
        recs = analysis.get('recommendations', [])
        for i, lbl in enumerate(self.recs):
            lbl.config(text=recs[i] if i < len(recs) else "")
        reasons = analysis.get('decision_reasons', [])
        if reasons:
            self.rec_reason_lbl.config(text=f"Why: {', '.join(reasons[:2])}")
        else:
            self.rec_reason_lbl.config(text="")

        # Forecast
        forecast = analysis.get('forecast', {})
        if forecast:
            f5 = forecast.get('focus_5m', None)
            f10 = forecast.get('focus_10m', None)
            fa5 = forecast.get('fatigue_5m', None)
            fa10 = forecast.get('fatigue_10m', None)
            if f5 is None or f10 is None or fa5 is None or fa10 is None:
                self.forecast_focus_lbl.config(text="Focus: --% / --%")
                self.forecast_fatigue_lbl.config(text="Fatigue: --% / --%")
            else:
                self.forecast_focus_lbl.config(text=f"Focus: {f5:.0f}% / {f10:.0f}%")
                self.forecast_fatigue_lbl.config(text=f"Fatigue: {fa5:.0f}% / {fa10:.0f}%")

        # Gamification
        game = analysis.get('gamification', {})
        if game and not isinstance(game, dict):
            if hasattr(game, 'as_dict'):
                game = game.as_dict()
            else:
                try:
                    game = dict(game.__dict__)
                except Exception:
                    game = {}
        if game:
            streak_min = float(game.get('focus_streak_minutes', game.get('focus_streak', 0)))
            self.gamification_lbl.config(text=f"Level: {game.get('level', 'Beginner')} | Streak: {streak_min:.1f}m")
            self.gamification_score_lbl.config(text=f"Productivity Score: {game.get('productivity_score', 0):.0f}")
            streak = float(streak_min)
            if streak >= 3:
                self._streak_pulse_on = not self._streak_pulse_on
                badge_color = self.c['orange'] if self._streak_pulse_on else self.c['yellow']
                self.streak_badge.config(text=f"🔥 Streak {streak:.1f}m", fg=badge_color)
            else:
                self.streak_badge.config(text="🔥 Streak", fg=self.c['dim'])

        # Explainable alert drivers
        alert_trigger = analysis.get('risk_score', 0) >= 50 or analysis.get('risk_level') in ("HIGH", "CRITICAL")
        if alert_trigger:
            baseline = self.baseline_manager.get_baseline_stats()
            drivers = build_alert_explanations(baseline, metrics)
            for i, lbl in enumerate(self.alert_labels):
                text = drivers[i] if i < len(drivers) else ""
                lbl.config(text=text, fg=self.c['orange'])
        else:
            for i, lbl in enumerate(self.alert_labels):
                lbl.config(text="No active alerts" if i == 0 else "", fg=self.c['dim'])
            
        # App switches
        recent = metrics.get('recent_switches', [])
        for i, lbl in enumerate(self.switch_labels):
            if i < len(recent):
                sw = recent[-(i+1)]  # Most recent first
                lbl.config(text=f"→ {sw['to'][:30]}")
            else:
                lbl.config(text="")
        
        # Update graph data
        self.graph_data['risk'].append(analysis['risk_score'])
        self.graph_data['fatigue'].append(analysis['fatigue_prob'])
        self.graph_data['cognitive'].append(analysis['cognitive_load'])
        self.graph_data['focus'].append(analysis['focus_score'])
        
        # Draw the graph
        self.draw_graph()

        # Calibration state hint in trend line for easy onboarding feedback
        progress = analysis.get('calibration_progress', 100)
        if progress < 100:
            self.trend_lbl.config(text=f"🧪 Calibrating profile: {progress}%", fg=self.c['blue'])

    def _refresh_daily_summary(self):
        """Refresh daily summary section from local analytics store."""
        summary = self.session_store.get_today_summary()
        best_hour = self.session_store.get_best_focus_hour()
        self.today_samples_lbl.config(text=f"Samples: {summary['samples']}")
        self.today_risk_lbl.config(
            text=f"Avg Risk: {summary['avg_risk']:.1f} | Peak Risk: {summary['peak_risk']:.1f}",
            fg=self.color_for(summary['avg_risk'])
        )
        self.today_focus_lbl.config(
            text=f"Avg Focus: {summary['avg_focus']:.1f} | Avg Confidence: {summary['avg_confidence']:.1f}",
            fg=self.c['text']
        )
        self.today_peak_lbl.config(text=f"Best Focus Hour: {best_hour}:00")
    
    def _check_break_reminders(self, analysis: Dict, metrics: Dict):
        """Smart break reminders based on fatigue and time."""
        if not hasattr(self, 'last_break_time'):
            self.last_break_time = time.time()
            self.break_count = 0
        
        minutes_since_break = (time.time() - self.last_break_time) / 60
        risk = analysis.get('risk_score', 0)
        fatigue = analysis.get('fatigue_prob', 0)
        
        # Smart break suggestions
        should_break = False
        break_reason = ""
        
        if minutes_since_break >= 45 and risk > 50:
            should_break = True
            break_reason = "High fatigue detected after 45+ minutes"
        elif minutes_since_break >= 60:
            should_break = True
            break_reason = "60 minutes continuous work"
        elif fatigue > 70 and minutes_since_break >= 30:
            should_break = True
            break_reason = "High fatigue level (70%+)"
        
        if should_break and not hasattr(self, '_break_shown'):
            self._show_break_suggestion(break_reason)
            self._break_shown = True
            
    def _show_break_suggestion(self, reason: str):
        """Show smart break suggestion."""
        self.add_log(f"😴 Break suggested: {reason}")
        
        # Show break notification
        response = messagebox.askyesno(
            "🧘 Break Time!",
            f"{reason}\n\n"
            "Taking a 5-minute break can boost productivity by 15%.\n\n"
            "Would you like to take a break now?",
            icon='question'
        )
        
        if response:
            self.last_break_time = time.time()
            self.break_count += 1
            if hasattr(self, '_break_shown'):
                del self._break_shown
            self.add_log("☕ Break started - see you in 5 minutes!")
            
    def record_break_taken(self):
        """User manually records a break."""
        self.last_break_time = time.time()
        if not hasattr(self, 'break_count'):
            self.break_count = 0
        self.break_count += 1
        if hasattr(self, '_break_shown'):
            del self._break_shown
        self.add_log(f"☕ Break #{self.break_count} recorded")
        
    def _update_loop(self):
        if self.monitoring and self.capture:
            try:
                if self.privacy.settings.paused:
                    self.privacy_status_lbl.config(text="Local-only active | Paused", fg=self.c['yellow'])
                    self.root.after(500, self._update_loop)
                    return

                if self.face_auth_enabled and self.face_auth_locked:
                    self._set_secure_overlay(True, self.face_auth_state)
                    self.root.after(1000, self._update_loop)
                    return

                metrics = self.capture.get_metrics()
                if self.latest_posture_score is not None:
                    metrics['posture_score'] = self.latest_posture_score
                analysis = self.engine.analyze(metrics)

                # Learn personal baseline continuously and adapt scoring confidence.
                self.baseline_manager.update(metrics)
                analysis = self.baseline_manager.adapt_analysis(analysis, metrics)

                # Generate intervention recommendations with richer context.
                interventions = self.intervention_engine.recommend(
                    analysis,
                    metrics,
                    focus_active=self.focus_session.is_active,
                    posture_enabled=self.posture_enabled,
                )
                default_recs = [f"{i.icon} {i.message}" for i in interventions]

                # Hybrid decision engine recommendations + explainability reasons.
                try:
                    recs, reasons = generate_recommendations(analysis, metrics)
                    if len(recs) < 3:
                        for r in default_recs:
                            if r not in recs:
                                recs.append(r)
                            if len(recs) >= 3:
                                break
                    analysis['recommendations'] = recs
                    analysis['decision_reasons'] = reasons
                except Exception as e:
                    analysis['recommendations'] = default_recs[:3]
                    analysis['decision_reasons'] = []
                    self.add_log(f"⚠️ Decision engine error: {e}")

                # Unified cognitive state (multi-modal fusion)
                posture_score = float(metrics.get('posture_score', 70))
                posture_strain = max(0.0, 100.0 - posture_score)
                cognitive_state = 0.5 * analysis.get('risk_score', 0) + 0.3 * analysis.get('cognitive_load', 0) + 0.2 * posture_strain
                analysis['cognitive_state'] = max(0.0, min(100.0, cognitive_state))

                # Short-term forecast (5-10 minutes)
                try:
                    forecast_5m = forecast_metrics(list(self.engine.history), ['focus_score', 'fatigue_prob'], seconds_ahead=300)
                    forecast_10m = forecast_metrics(list(self.engine.history), ['focus_score', 'fatigue_prob'], seconds_ahead=600)
                    analysis['forecast'] = {
                        'focus_5m': forecast_5m.get('focus_score', None),
                        'fatigue_5m': forecast_5m.get('fatigue_prob', None),
                        'focus_10m': forecast_10m.get('focus_score', None),
                        'fatigue_10m': forecast_10m.get('fatigue_prob', None),
                    }
                except Exception as e:
                    analysis['forecast'] = {}
                    self.add_log(f"⚠️ Forecast error: {e}")

                
                # If Focus Mode is active, integrate its scores
                if self.focus_session.is_active:
                    focus_status = self.focus_session.get_status()
                    # Use Focus Mode's actual focus score
                    analysis['focus_score'] = focus_status.get('focus_score', analysis['focus_score'])
                    # Use Focus Mode's productivity
                    analysis['productivity'] = focus_status.get('productivity', analysis['productivity'])

                # Adaptive learning and emotional intelligence (local-only)
                self.adaptive_learning.update(metrics)
                thresholds = self.adaptive_learning.adaptive_thresholds()
                emotion_state = self.emotion_engine.assess(
                    analysis,
                    posture_score=float(metrics.get('posture_score', 70)),
                    face_confidence=float(self.face_auth_state.confidence if self.face_auth_state else 0),
                )
                analysis['emotion_state'] = emotion_state.__dict__.copy()

                burnout_state = self.burnout_engine.forecast(list(self.engine.history))
                analysis['burnout_state'] = burnout_state.__dict__.copy()
                analysis['adaptive_thresholds'] = thresholds

                try:
                    self.event_bus.emit('analysis', analysis)
                except Exception:
                    pass

                # Gamification update (after focus override)
                try:
                    game_state = self.gamification.update(analysis.get('focus_score', 0), analysis.get('risk_score', 0))
                    analysis['gamification'] = game_state.as_dict()
                except Exception as e:
                    analysis['gamification'] = {}
                    self.add_log(f"⚠️ Gamification error: {e}")
                
                self.update_display(analysis, metrics)

                try:
                    self.local_storage.save_payload('fatigue_logs', {
                        'ts': datetime.now().isoformat(),
                        'risk_score': analysis.get('risk_score', 0),
                        'emotion_state': analysis.get('emotion_state', {}),
                        'burnout_state': analysis.get('burnout_state', {}),
                        'focus_score': analysis.get('focus_score', 0),
                    })
                except Exception:
                    pass

                # Update study assistant suggestions
                self._update_study_plan(analysis)

                # Persist snapshots every 5 seconds for local analytics.
                now = time.time()
                if now - self.last_persist_time >= 5:
                    self.session_store.save_analysis(analysis, metrics)
                    self.last_persist_time = now

                # Refresh summary less frequently to keep UI responsive.
                if now - self.last_summary_refresh >= 10:
                    self._refresh_daily_summary()
                    self.last_summary_refresh = now
                
                # Smart break reminders
                self._check_break_reminders(analysis, metrics)
                
            except Exception as e:
                self.add_log(f"⚠️ Update error: {e}")
        
        # Update Focus Mode UI (timer, score, etc.)
        if self.focus_session.is_active:
            self.update_focus_ui()
        
        self.root.after(1000, self._update_loop)
        
    def run(self):
        consent = messagebox.askyesno(
            "Mind-Shield+ Consent",
            "Mind-Shield+ v2.2 with Focus AI Mode\n\n"
            "This monitors keyboard, mouse, and app activity.\n\n"
            "✓ All data processed locally\n"
            "✓ No data sent anywhere\n"
            "✓ Focus Mode detects distracting apps\n\n"
            "Allow monitoring?"
        )
        if consent:
            self.add_log("🛡️ Mind-Shield+ v2.2 ready")
            self.root.mainloop()
        else:
            self.root.destroy()


if __name__ == "__main__":
    print("Starting Mind-Shield+ v2.2 with Focus AI Mode...")
    app = MindShieldGUI()
    app.run()
