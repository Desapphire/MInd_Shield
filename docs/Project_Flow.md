# Mind-Shield+ Project Flow (End-to-End)

This document explains the complete runtime flow, module by module, in a single place for presentation preparation.

## 1) App Startup

1. User launches the GUI entry point: `mindshield.py`.
2. Core controllers initialize:
   - Behavioral capture (keyboard/mouse/system)
   - Feature engineering and ML models
   - Posture detection (if available)
   - Local face authentication (if enabled)
3. UI renders the dashboard with live cards and controls.

## 2) Live Monitoring Loop

1. SystemBehaviorCapture collects raw signals (keystrokes, mouse, active window, idle time).
2. FeatureEngineer normalizes and transforms raw signals into feature vectors.
3. FatiguePredictionModel outputs fatigue probability.
4. BehavioralDriftDetector flags drift from baseline.
5. CognitiveLoadEstimator outputs the current load score.
6. RiskEvaluationEngine fuses signals into a unified risk score + risk level.
7. Decision engine and intervention engine generate recommendations.
8. UI renders metrics, graphs, and alerts.

## 3) Focus AI Mode Flow

1. User starts a timed focus session.
2. Active window title is checked every 500 ms.
3. Distractions increase the counter and reduce focus score.
4. Session data is stored in memory and can be exported to a file if needed.

## 4) Posture Detection Flow (Optional)

1. Webcam frames are read by the posture detector.
2. MediaPipe PoseLandmarker extracts body landmarks.
3. Posture metrics are computed (head tilt, shoulder alignment, forward head, slouch).
4. A posture score is derived and smoothed.
5. UI shows the score, issues, and skeleton overlay.

## 5) Local Face Authentication Flow (Optional)

1. User enrolls once while sitting normally.
2. A face embedding is created from the current frame.
3. During monitoring, each new frame is compared to the enrolled embedding.
4. If the face is missing for more than the configured threshold, the UI locks.
5. The UI only unlocks when an authorized face is detected.

## 6) Data Handling and Storage

- Live session metrics are kept in memory.
- Optional exports write JSON files when the user requests it.
- Face auth profiles are stored locally in encrypted form.
- No cloud services and no telemetry.

## 7) Shutdown

1. Capture loops stop.
2. Webcam resources are released.
3. The GUI exits safely.

## Reference Map

- Entry point: `mindshield.py`
- ML models: `src/fatigue_model.py`, `src/anomaly_detection.py`, `src/cognitive_load.py`
- Risk fusion: `src/risk_evaluation.py`
- Posture: `src/posture_detection.py`
- Face auth: `src/presence_auth.py`
