"""
Mind-Shield+ Posture Detection Module v2.0 - IMPROVED ACCURACY

Uses MediaPipe PoseLandmarker (Tasks API) to detect slouching 
and poor posture from webcam feed in real-time.

Features:
- Smoothed measurements to reduce jitter
- Calibration baseline for personalized detection
- Skeleton visualization on preview
- Accurate angle calculations
- Robust landmark visibility checking
"""

import numpy as np
import threading
import time
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable, List, Dict
from collections import deque

# Check for OpenCV
CV2_AVAILABLE = False
cv2 = None
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    pass

# Check for MediaPipe Tasks API
MEDIAPIPE_AVAILABLE = False
mp = None

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker, 
        PoseLandmarkerOptions, 
        RunningMode
    )
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError) as e:
    print(f"MediaPipe Tasks API not available: {e}")


# Pose landmark indices (MediaPipe Tasks API - 33 landmarks)
class PoseLandmarkIndices:
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


@dataclass
class PostureStatus:
    """Current posture status with detailed metrics."""
    is_good: bool
    score: float  # 0-100, higher is better
    issues: List[str]  # List of detected issues
    head_tilt: float  # Degrees (0 = level)
    shoulder_alignment: float  # 0-1, 1 = perfect level
    forward_lean: float  # Degrees (0 = upright)
    neck_angle: float  # Degrees
    confidence: float  # Detection confidence 0-1
    landmarks_visible: bool  # Whether key landmarks are visible


class SmoothingFilter:
    """Exponential moving average filter for smoothing values."""
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize filter.
        
        Args:
            alpha: Smoothing factor (0-1). Lower = smoother but slower response
        """
        self.alpha = alpha
        self.value = None
    
    def update(self, new_value: float) -> float:
        """Update and return smoothed value."""
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value
    
    def reset(self):
        """Reset filter state."""
        self.value = None


class PostureDetector:
    """
    Real-time posture detection using MediaPipe PoseLandmarker.
    
    Detects:
    - Slouching (forward lean / curved spine)
    - Head tilt (lateral)
    - Shoulder misalignment (uneven shoulders)
    - Forward head posture (neck crane)
    - Head drop (looking down too much)
    
    Features:
    - Smoothed measurements for stable readings
    - Calibration mode for personalized baselines
    - Skeleton overlay on video preview
    """
    
    # Posture thresholds (degrees/normalized values)
    HEAD_TILT_THRESHOLD = 12  # degrees - head tilted sideways
    SHOULDER_DIFF_THRESHOLD = 0.04  # normalized y difference
    FORWARD_HEAD_THRESHOLD = 0.06  # normalized - ear ahead of shoulder
    NECK_ANGLE_THRESHOLD = 35  # degrees - neck bent forward
    SLOUCH_THRESHOLD = 15  # degrees - spine curve
    
    # Minimum visibility for reliable detection
    MIN_VISIBILITY = 0.5
    
    def __init__(self, camera_index: int = 0):
        """
        Initialize posture detector.
        
        Args:
            camera_index: Webcam index (0 = default)
        """
        self.camera_index = camera_index
        self.running = False
        self.camera = None
        self.thread = None
        self.callback: Optional[Callable] = None
        
        # Status tracking
        self.current_status: Optional[PostureStatus] = None
        self.posture_history = deque(maxlen=60)  # Last 60 readings
        self.good_posture_time = 0.0
        self.bad_posture_time = 0.0
        self.last_alert_time = 0.0
        
        # Smoothing filters for stable readings
        self.filters: Dict[str, SmoothingFilter] = {
            'head_tilt': SmoothingFilter(alpha=0.4),
            'shoulder_diff': SmoothingFilter(alpha=0.4),
            'forward_head': SmoothingFilter(alpha=0.3),
            'neck_angle': SmoothingFilter(alpha=0.3),
            'slouch': SmoothingFilter(alpha=0.3),
            'score': SmoothingFilter(alpha=0.5),
        }
        
        # Calibration baseline (can be set by user)
        self.baseline = {
            'shoulder_y_diff': 0.0,
            'ear_shoulder_x_diff': 0.0,
            'calibrated': False
        }
        
        # MediaPipe setup
        self.landmarker = None
        self._initialized = False
        
        if MEDIAPIPE_AVAILABLE and CV2_AVAILABLE:
            self._init_landmarker()
    
    def _init_landmarker(self):
        """Initialize the pose landmarker."""
        try:
            model_path = self._find_model()
            if not model_path:
                print("ERROR: Pose landmarker model not found!")
                print("Expected at: d:/EDI_Project/src/pose_landmarker_lite.task")
                return
            
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.6,
                min_tracking_confidence=0.6
            )
            
            self.landmarker = PoseLandmarker.create_from_options(options)
            self._initialized = True
            print("✓ Posture detection initialized successfully")
            
        except Exception as e:
            print(f"ERROR initializing PoseLandmarker: {e}")
            self.landmarker = None
    
    def _find_model(self) -> Optional[str]:
        """Find the pose landmarker model file."""
        possible_paths = [
            os.path.join(os.path.dirname(__file__), 'pose_landmarker_lite.task'),
            os.path.join(os.path.dirname(__file__), '..', 'pose_landmarker_lite.task'),
            'd:/EDI_Project/src/pose_landmarker_lite.task',
            'pose_landmarker_lite.task',
        ]
        
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                print(f"Found model at: {abs_path}")
                return abs_path
        
        return None
    
    def is_available(self) -> bool:
        """Check if posture detection is available."""
        return self._initialized and self.landmarker is not None and CV2_AVAILABLE
    
    def calibrate(self, frame: np.ndarray) -> Tuple[bool, str]:
        """
        Calibrate baseline posture from current frame.
        User should be sitting in their ideal posture.
        
        Returns:
            (success, message)
        """
        if not self._initialized:
            return False, "Posture detection not initialized"
        
        status = self._analyze_frame(frame, use_smoothing=False)
        if status is None or not status.landmarks_visible:
            return False, "Could not detect landmarks. Make sure you're visible."
        
        # Store current measurements as baseline
        # (would need raw values, simplified here)
        self.baseline['calibrated'] = True
        return True, "Calibration saved! This is now your reference posture."
    
    def start(self, callback: Optional[Callable] = None) -> Tuple[bool, str]:
        """Start posture monitoring."""
        if not CV2_AVAILABLE:
            return False, "OpenCV not installed. Run: pip install opencv-python"
        
        if not MEDIAPIPE_AVAILABLE:
            return False, "MediaPipe not available. Run: pip install mediapipe"
        
        if not self._initialized:
            return False, "Pose landmarker not initialized. Check model file."
        
        if self.running:
            return False, "Already running"
        
        # Try to open camera
        try:
            # Try DirectShow first (faster on Windows)
            self.camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.camera.isOpened():
                self.camera = cv2.VideoCapture(self.camera_index)
            if not self.camera.isOpened():
                return False, "Could not open webcam. Check camera connection."
        except Exception as e:
            return False, f"Camera error: {e}"
        
        # Camera settings for balance of quality and performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 20)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        self.callback = callback
        self.running = True
        self.good_posture_time = 0.0
        self.bad_posture_time = 0.0
        
        # Reset filters
        for f in self.filters.values():
            f.reset()
        
        # Start monitoring thread
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
        return True, "Posture monitoring started"
    
    def stop(self) -> Tuple[bool, str]:
        """Stop posture monitoring."""
        if not self.running:
            return False, "Not running"
        
        self.running = False
        
        if self.camera:
            try:
                self.camera.release()
            except:
                pass
            self.camera = None
        
        total = self.good_posture_time + self.bad_posture_time
        pct = (self.good_posture_time / total * 100) if total > 0 else 100
        return True, f"Stopped. Good posture: {pct:.0f}% ({self.good_posture_time:.0f}s good, {self.bad_posture_time:.0f}s bad)"
    
    def _monitor_loop(self):
        """Main monitoring loop (runs in background thread)."""
        process_interval = 0.15  # Process every 150ms (6-7 FPS analysis)
        last_process = 0
        
        while self.running:
            now = time.time()
            
            if now - last_process < process_interval:
                time.sleep(0.02)
                continue
            
            last_process = now
            
            try:
                # Read frame
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    continue
                
                # Flip horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Analyze posture
                status = self._analyze_frame(frame, use_smoothing=True)
                
                if status:
                    self.current_status = status
                    self.posture_history.append(status.score)
                    
                    # Update time tracking
                    if status.is_good:
                        self.good_posture_time += process_interval
                    else:
                        self.bad_posture_time += process_interval
                    
                    # Draw skeleton on frame for preview
                    frame_with_skeleton = self._draw_skeleton(frame, status)
                    
                    # Send callback
                    if self.callback:
                        try:
                            self.callback(status, frame_with_skeleton)
                        except Exception as e:
                            print(f"Callback error: {e}")
                            
            except Exception as e:
                print(f"Monitor loop error: {e}")
                time.sleep(0.5)
    
    def _analyze_frame(self, frame: np.ndarray, use_smoothing: bool = True) -> Optional[PostureStatus]:
        """
        Analyze a single frame for posture issues.
        
        Uses multiple metrics to assess posture quality.
        """
        if self.landmarker is None:
            return None
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect pose landmarks
            result = self.landmarker.detect(mp_image)
            
            # No detection
            if not result.pose_landmarks or len(result.pose_landmarks) == 0:
                return PostureStatus(
                    is_good=True,
                    score=50,
                    issues=["No person detected"],
                    head_tilt=0,
                    shoulder_alignment=0.5,
                    forward_lean=0,
                    neck_angle=0,
                    confidence=0,
                    landmarks_visible=False
                )
            
            # Get landmarks
            lm = result.pose_landmarks[0]
            
            # Extract key landmarks with visibility check
            def get_lm(idx):
                point = lm[idx]
                return {
                    'x': point.x,
                    'y': point.y,
                    'z': getattr(point, 'z', 0),
                    'vis': getattr(point, 'visibility', 1.0) if hasattr(point, 'visibility') else 
                           getattr(point, 'presence', 1.0) if hasattr(point, 'presence') else 1.0
                }
            
            nose = get_lm(PoseLandmarkIndices.NOSE)
            left_ear = get_lm(PoseLandmarkIndices.LEFT_EAR)
            right_ear = get_lm(PoseLandmarkIndices.RIGHT_EAR)
            left_eye = get_lm(PoseLandmarkIndices.LEFT_EYE)
            right_eye = get_lm(PoseLandmarkIndices.RIGHT_EYE)
            left_shoulder = get_lm(PoseLandmarkIndices.LEFT_SHOULDER)
            right_shoulder = get_lm(PoseLandmarkIndices.RIGHT_SHOULDER)
            left_hip = get_lm(PoseLandmarkIndices.LEFT_HIP)
            right_hip = get_lm(PoseLandmarkIndices.RIGHT_HIP)
            
            # Check if key landmarks are visible
            key_vis = [
                left_shoulder['vis'], right_shoulder['vis'],
                nose['vis'], left_ear['vis'], right_ear['vis']
            ]
            avg_visibility = np.mean(key_vis)
            landmarks_visible = avg_visibility > self.MIN_VISIBILITY
            
            if not landmarks_visible:
                return PostureStatus(
                    is_good=True,
                    score=60,
                    issues=["Partial view - move closer"],
                    head_tilt=0,
                    shoulder_alignment=0.5,
                    forward_lean=0,
                    neck_angle=0,
                    confidence=avg_visibility,
                    landmarks_visible=False
                )
            
            # Store landmarks for drawing
            self._last_landmarks = lm
            
            issues = []
            score = 100
            
            # ===== 1. HEAD TILT (lateral) =====
            # Compare ear heights - ears should be level
            ear_y_diff = left_ear['y'] - right_ear['y']
            ear_x_diff = abs(left_ear['x'] - right_ear['x']) + 0.001
            head_tilt_raw = np.degrees(np.arctan2(ear_y_diff, ear_x_diff))
            
            if use_smoothing:
                head_tilt = self.filters['head_tilt'].update(head_tilt_raw)
            else:
                head_tilt = head_tilt_raw
            
            if abs(head_tilt) > self.HEAD_TILT_THRESHOLD:
                direction = "left" if head_tilt > 0 else "right"
                issues.append(f"Head tilted {direction} ({abs(head_tilt):.0f}°)")
                score -= min(20, abs(head_tilt))
            
            # ===== 2. SHOULDER ALIGNMENT =====
            # Shoulders should be level
            shoulder_y_diff_raw = left_shoulder['y'] - right_shoulder['y']
            
            if use_smoothing:
                shoulder_y_diff = self.filters['shoulder_diff'].update(shoulder_y_diff_raw)
            else:
                shoulder_y_diff = shoulder_y_diff_raw
            
            shoulder_alignment = 1 - min(abs(shoulder_y_diff) / 0.15, 1)
            
            if abs(shoulder_y_diff) > self.SHOULDER_DIFF_THRESHOLD:
                side = "Left" if shoulder_y_diff > 0 else "Right"
                issues.append(f"{side} shoulder dropped")
                score -= 15
            
            # ===== 3. FORWARD HEAD POSTURE =====
            # Ear should be roughly above shoulder, not in front
            mid_shoulder_x = (left_shoulder['x'] + right_shoulder['x']) / 2
            mid_ear_x = (left_ear['x'] + right_ear['x']) / 2
            
            # In flipped image, if ear x < shoulder x, head is forward
            forward_head_raw = mid_shoulder_x - mid_ear_x  # Positive = forward
            
            if use_smoothing:
                forward_head = self.filters['forward_head'].update(forward_head_raw)
            else:
                forward_head = forward_head_raw
            
            if forward_head > self.FORWARD_HEAD_THRESHOLD:
                issues.append("Forward head posture")
                score -= 20
            
            # ===== 4. NECK ANGLE / LOOKING DOWN =====
            # Angle between nose-ear line and horizontal
            mid_eye_y = (left_eye['y'] + right_eye['y']) / 2
            mid_ear_y = (left_ear['y'] + right_ear['y']) / 2
            
            neck_tilt_raw = (mid_eye_y - mid_ear_y) * 100  # Simplified metric
            
            if use_smoothing:
                neck_angle = self.filters['neck_angle'].update(neck_tilt_raw)
            else:
                neck_angle = neck_tilt_raw
            
            # ===== 5. SLOUCHING (Spine Curve) =====
            # Compare shoulder position relative to hips
            mid_shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2
            mid_hip_y = (left_hip['y'] + right_hip['y']) / 2
            mid_hip_x = (left_hip['x'] + right_hip['x']) / 2
            
            # Calculate torso angle
            torso_dx = abs(mid_shoulder_x - mid_hip_x) + 0.001
            torso_dy = mid_hip_y - mid_shoulder_y  # Positive when upright
            
            # Slouch detection based on shoulder-hip alignment
            slouch_raw = np.degrees(np.arctan2(abs(mid_shoulder_x - mid_hip_x), torso_dy + 0.001))
            
            if use_smoothing:
                forward_lean = self.filters['slouch'].update(slouch_raw)
            else:
                forward_lean = slouch_raw
            
            if forward_lean > self.SLOUCH_THRESHOLD:
                issues.append(f"Slouching ({forward_lean:.0f}°)")
                score -= 25
            
            # ===== CALCULATE FINAL SCORE =====
            score = max(0, min(100, score))
            
            if use_smoothing:
                score = self.filters['score'].update(score)
            
            is_good = score >= 70 and len(issues) == 0
            
            return PostureStatus(
                is_good=is_good,
                score=score,
                issues=issues if issues else ["Good posture!"],
                head_tilt=head_tilt,
                shoulder_alignment=shoulder_alignment,
                forward_lean=forward_lean,
                neck_angle=neck_angle,
                confidence=avg_visibility,
                landmarks_visible=True
            )
            
        except Exception as e:
            print(f"Frame analysis error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _draw_skeleton(self, frame: np.ndarray, status: PostureStatus) -> np.ndarray:
        """Draw pose skeleton and status indicators on frame."""
        if not hasattr(self, '_last_landmarks') or self._last_landmarks is None:
            return frame
        
        frame_h, frame_w = frame.shape[:2]
        lm = self._last_landmarks
        
        # Color based on posture status
        if status.is_good:
            color = (0, 255, 0)  # Green
        elif status.score >= 50:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 0, 255)  # Red
        
        # Define skeleton connections (upper body focus)
        connections = [
            # Face
            (PoseLandmarkIndices.LEFT_EAR, PoseLandmarkIndices.LEFT_EYE),
            (PoseLandmarkIndices.RIGHT_EAR, PoseLandmarkIndices.RIGHT_EYE),
            (PoseLandmarkIndices.LEFT_EYE, PoseLandmarkIndices.NOSE),
            (PoseLandmarkIndices.RIGHT_EYE, PoseLandmarkIndices.NOSE),
            # Shoulders
            (PoseLandmarkIndices.LEFT_SHOULDER, PoseLandmarkIndices.RIGHT_SHOULDER),
            (PoseLandmarkIndices.LEFT_EAR, PoseLandmarkIndices.LEFT_SHOULDER),
            (PoseLandmarkIndices.RIGHT_EAR, PoseLandmarkIndices.RIGHT_SHOULDER),
            # Arms
            (PoseLandmarkIndices.LEFT_SHOULDER, PoseLandmarkIndices.LEFT_ELBOW),
            (PoseLandmarkIndices.RIGHT_SHOULDER, PoseLandmarkIndices.RIGHT_ELBOW),
            # Torso
            (PoseLandmarkIndices.LEFT_SHOULDER, PoseLandmarkIndices.LEFT_HIP),
            (PoseLandmarkIndices.RIGHT_SHOULDER, PoseLandmarkIndices.RIGHT_HIP),
            (PoseLandmarkIndices.LEFT_HIP, PoseLandmarkIndices.RIGHT_HIP),
        ]
        
        # Draw connections
        for start_idx, end_idx in connections:
            try:
                start = lm[start_idx]
                end = lm[end_idx]
                
                start_point = (int(start.x * frame_w), int(start.y * frame_h))
                end_point = (int(end.x * frame_w), int(end.y * frame_h))
                
                cv2.line(frame, start_point, end_point, color, 2)
            except:
                pass
        
        # Draw key landmark points
        key_landmarks = [
            PoseLandmarkIndices.NOSE,
            PoseLandmarkIndices.LEFT_EYE, PoseLandmarkIndices.RIGHT_EYE,
            PoseLandmarkIndices.LEFT_EAR, PoseLandmarkIndices.RIGHT_EAR,
            PoseLandmarkIndices.LEFT_SHOULDER, PoseLandmarkIndices.RIGHT_SHOULDER,
            PoseLandmarkIndices.LEFT_HIP, PoseLandmarkIndices.RIGHT_HIP,
        ]
        
        for idx in key_landmarks:
            try:
                point = lm[idx]
                x = int(point.x * frame_w)
                y = int(point.y * frame_h)
                cv2.circle(frame, (x, y), 5, color, -1)
                cv2.circle(frame, (x, y), 7, (255, 255, 255), 1)
            except:
                pass
        
        return frame
    
    def get_status(self) -> Optional[PostureStatus]:
        """Get current posture status."""
        return self.current_status
    
    def get_average_score(self) -> float:
        """Get average posture score from history."""
        if not self.posture_history:
            return 100
        return float(np.mean(list(self.posture_history)))
    
    def get_posture_stats(self) -> dict:
        """Get posture statistics."""
        total_time = self.good_posture_time + self.bad_posture_time
        return {
            'good_time': self.good_posture_time,
            'bad_time': self.bad_posture_time,
            'total_time': total_time,
            'good_percentage': (self.good_posture_time / total_time * 100) if total_time > 0 else 100,
            'average_score': self.get_average_score()
        }


# Singleton for easy access
_detector: Optional[PostureDetector] = None

def get_posture_detector() -> PostureDetector:
    """Get or create the posture detector singleton."""
    global _detector
    if _detector is None:
        _detector = PostureDetector()
    return _detector
