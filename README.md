<<<<<<< HEAD
# Mind-Shield+ v2.2

**AI-Powered Cognitive Fatigue Detection & Wellness System**

## Overview

Mind-Shield+ is an intelligent **desktop application** that monitors your behavioral interaction patterns in real-time to estimate cognitive load, predict mental fatigue, detect behavioral drift, and monitor posture. It provides proactive wellness interventions before fatigue impacts your health and productivity.

## Key Features

- **🖥️ Real-Time Monitoring**: Captures keyboard, mouse, and system activity in real-time
- **🧠 Fatigue Prediction**: ML-powered fatigue probability estimation using Random Forest
- **📊 Cognitive Load Scoring**: Weighted multi-factor mental effort calculation
- **🔍 Behavioral Drift Detection**: Isolation Forest algorithm detects deviation from baseline
- **🧘 Posture Detection**: Webcam-based posture monitoring with skeleton overlay (MediaPipe)
- **🎯 Focus AI Mode**: Pomodoro-style deep work sessions with distraction detection
- **📈 Real-Time Dashboard**: Live graphs, metrics, and recommendations
- **💡 Smart Recommendations**: Contextual break and wellness suggestions

## Project Structure

```
EDI_Project/
├── mindshield.py                # Main desktop application (GUI)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── PROJECT_DOCUMENTATION.md     # Detailed technical documentation
├── .venv/                       # Virtual environment
└── src/
    ├── __init__.py              # Package initialization
    ├── data_simulation.py       # Synthetic data for model training
    ├── feature_engineering.py   # Feature extraction and transformation
    ├── cognitive_load.py        # Cognitive load scoring algorithm
    ├── fatigue_model.py         # Random Forest fatigue prediction
    ├── anomaly_detection.py     # Isolation Forest drift detection
    ├── risk_evaluation.py       # Integrated risk assessment engine
    ├── posture_detection.py     # MediaPipe webcam posture analysis
    ├── visualization.py         # Plotting and dashboards
    ├── pose_landmarker_lite.task # MediaPipe pose model
    └── main.py                  # Console demonstration script
```

## Installation

### Prerequisites

- Python 3.10 or newer
- Webcam (optional, for posture detection)

### Setup

1. **Clone or download the project**

2. **Create and activate virtual environment:**
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Activate (Linux/macOS)
source .venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt

# For posture detection (optional but recommended)
pip install mediapipe opencv-python pillow
```

## Usage

### 🖥️ Run the Desktop Application (Recommended)

```bash
# Activate virtual environment first
.\.venv\Scripts\Activate.ps1

# Run the application
python mindshield.py
```

Or directly with venv Python:
```bash
.\.venv\Scripts\python.exe mindshield.py
```

**Features:**
- Real-time behavioral monitoring (typing speed, mouse velocity, errors)
- Live fatigue probability display with risk levels
- Dynamic cognitive load estimation
- Focus AI Mode with distraction detection
- Posture monitoring with webcam preview
- Smart break reminders and recommendations
- Session analytics and graphs

### 📊 Run the Console Demo (Model Training)

```bash
cd src
python main.py
```

This will:
1. Generate synthetic behavioral data (for model training)
2. Extract cognitive features
3. Train fatigue prediction models
4. Run behavioral drift detection
5. Estimate cognitive load
6. Evaluate integrated risk
7. Simulate a 90-minute work session
8. Generate visualization outputs

### Run Individual Modules

```bash
# Test fatigue prediction model
python src/fatigue_model.py

# Test anomaly detection
python src/anomaly_detection.py

# Test cognitive load estimator
python src/cognitive_load.py

# Generate synthetic training data
python src/data_simulation.py
```

## Behavioral Variables Monitored

| Variable | Description | Normal | Fatigued |
|----------|-------------|--------|----------|
| Typing Speed | Keystrokes per minute | ~60 WPM | ~35 WPM |
| Typing Delay | Time between keystrokes (ms) | ~120ms | ~200ms |
| Backspace Rate | Error correction frequency | ~5% | ~15% |
| Tab Switch Rate | Context switches per minute | ~2/min | ~5/min |
| Mouse Velocity | Navigation speed (px/s) | ~500 | ~300 |
| Idle Time | Pause duration (seconds) | ~3s | ~8s |
| Posture Score | Webcam posture analysis | 85-100 | <70 |

## Machine Learning Models

### 1. Fatigue Prediction (Random Forest)
- **Algorithm**: Random Forest Classifier (100 trees)
- **Input**: Typing speed, error rate, mouse velocity, idle time, etc.
- **Output**: Probability of fatigue (0-100%)
- **Performance**: ~92% accuracy, ~97% ROC AUC

### 2. Behavioral Drift Detection (Isolation Forest)
- **Algorithm**: Isolation Forest (unsupervised)
- **Input**: Current behavioral features
- **Output**: Anomaly score indicating deviation from baseline
- **Use**: Detects when behavior significantly changes

### 3. Cognitive Load Estimator (Weighted Scoring)
- **Algorithm**: Multi-factor weighted combination
- **Components**: Typing load (25%), Error load (25%), Attention (20%), Idle (15%), Engagement (15%)
- **Output**: Cognitive load score (0-100)

### 4. Posture Detection (MediaPipe)
- **Algorithm**: MediaPipe PoseLandmarker + geometric analysis
- **Input**: Webcam frames
- **Output**: Posture score, detected issues (head tilt, slouching, etc.)

## Risk Levels

| Level | Score Range | Action |
|-------|-------------|--------|
| 🟢 Low | 0-25 | Continue working normally |
| 🟡 Moderate | 26-50 | Consider a short break |
| 🟠 High | 51-75 | Take a break soon |
| 🔴 Critical | 76-100 | Immediate break recommended |

## Focus AI Mode

Dedicated deep work mode with:
- **Configurable timer** (default: 25 minutes)
- **Distraction detection** (YouTube, Facebook, Twitter, etc.)
- **Focus score** based on concentration quality
- **Session history** and statistics
- **Auto-start monitoring** when focus begins

## Posture Detection

Real-time webcam-based posture monitoring:
- **Head tilt** detection (>12° threshold)
- **Slouching** detection (>15° deviation)
- **Forward head posture** detection
- **Uneven shoulders** detection
- **Skeleton overlay** on video preview
- **Smoothed readings** to reduce jitter

## Dependencies

| Package | Purpose |
|---------|---------|
| pynput | Keyboard and mouse monitoring |
| psutil | System and process information |
| numpy | Numerical computations |
| pandas | Data manipulation |
| scikit-learn | Machine learning models |
| tkinter | Desktop GUI (included with Python) |
| mediapipe | Pose detection (optional) |
| opencv-python | Webcam capture (optional) |
| pillow | Image processing (optional) |

## Technical Documentation

For detailed technical information including:
- Algorithm explanations
- Code architecture
- Module-by-module breakdown
- Future enhancements
- Patent potential

See: **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)**

## License

This project is developed for educational and research purposes.

## Authors

Mind-Shield+ Team
=======
# Mind-Shield+ v2.2

**AI-Powered Cognitive Fatigue Detection & Wellness System**

## Overview

Mind-Shield+ is an intelligent **desktop application** that monitors your behavioral interaction patterns in real-time to estimate cognitive load, predict mental fatigue, detect behavioral drift, and monitor posture. It provides proactive wellness interventions before fatigue impacts your health and productivity.

## Key Features

- **🖥️ Real-Time Monitoring**: Captures keyboard, mouse, and system activity in real-time
- **🧠 Fatigue Prediction**: ML-powered fatigue probability estimation using Random Forest
- **📊 Cognitive Load Scoring**: Weighted multi-factor mental effort calculation
- **🔍 Behavioral Drift Detection**: Isolation Forest algorithm detects deviation from baseline
- **🧘 Posture Detection**: Webcam-based posture monitoring with skeleton overlay (MediaPipe)
- **🎯 Focus AI Mode**: Pomodoro-style deep work sessions with distraction detection
- **📈 Real-Time Dashboard**: Live graphs, metrics, and recommendations
- **💡 Smart Recommendations**: Contextual break and wellness suggestions

## Project Structure

```
EDI_Project/
├── mindshield.py                # Main desktop application (GUI)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── Project_Documentation.md     # Detailed technical documentation
├── .venv/                       # Virtual environment
└── src/
    ├── __init__.py              # Package initialization
    ├── data_simulation.py       # Synthetic data for model training
    ├── feature_engineering.py   # Feature extraction and transformation
    ├── cognitive_load.py        # Cognitive load scoring algorithm
    ├── fatigue_model.py         # Random Forest fatigue prediction
    ├── anomaly_detection.py     # Isolation Forest drift detection
    ├── risk_evaluation.py       # Integrated risk assessment engine
    ├── posture_detection.py     # MediaPipe webcam posture analysis
    ├── presence_auth.py         # Local face auth + presence validation
    ├── visualization.py         # Plotting and dashboards
    ├── pose_landmarker_lite.task # MediaPipe pose model
    └── main.py                  # Console demonstration script
```

## Installation

### Prerequisites

- Python 3.10 or newer
- Webcam (optional, for posture detection)

### Setup

1. **Clone or download the project**

2. **Create and activate virtual environment:**
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Activate (Linux/macOS)
source .venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt

# For posture detection (optional but recommended)
pip install mediapipe opencv-python pillow
```

## Usage

### 🖥️ Run the Desktop Application (Recommended)

```bash
# Activate virtual environment first
.\.venv\Scripts\Activate.ps1

# Run the application
python mindshield.py
```

Or directly with venv Python:
```bash
.\.venv\Scripts\python.exe mindshield.py
```

**Features:**
- Real-time behavioral monitoring (typing speed, mouse velocity, errors)
- Live fatigue probability display with risk levels
- Dynamic cognitive load estimation
- Focus AI Mode with distraction detection
- Posture monitoring with webcam preview
- Smart break reminders and recommendations
- Session analytics and graphs

### 🔐 Local Face Auth (Presence + Seat Validation)

Local Face Auth verifies the enrolled user and also checks that the same face stays in the usual seat position.

**What it enforces:**
- One enrolled face per profile name.
- Single face in view (multiple faces = lock).
- Seat position check (face center + size must match enrollment).
- Local-only encrypted embeddings. No raw video stored.

**Enrollment steps:**
1. Start Mind-Shield+.
2. Click **Posture** to start the webcam (or enable Local Face Auth and it will start for you).
3. Sit in your normal position and make sure only your face is visible.
4. Enter a profile name and click **Enroll** when the preview is steady.
5. Enable **Local Face Auth** to lock/unlock monitoring based on presence.

If no profile exists, enabling face auth will prompt you to enroll first.

**Flowchart:**
See [face_auth_flow.svg](face_auth_flow.svg).

### 📊 Run the Console Demo (Model Training)

```bash
cd src
python main.py
```

This will:
1. Generate synthetic behavioral data (for model training)
2. Extract cognitive features
3. Train fatigue prediction models
4. Run behavioral drift detection
5. Estimate cognitive load
6. Evaluate integrated risk
7. Simulate a 90-minute work session
8. Generate visualization outputs

### Run Individual Modules

```bash
# Test fatigue prediction model
python src/fatigue_model.py

# Test anomaly detection
python src/anomaly_detection.py

# Test cognitive load estimator
python src/cognitive_load.py

# Generate synthetic training data
python src/data_simulation.py
```

### 🌐 Run the Local API Server (Optional)

```bash
python src/api_server.py
```

Health check: `http://127.0.0.1:8000/health`

## Behavioral Variables Monitored

| Variable | Description | Normal | Fatigued |
|----------|-------------|--------|----------|
| Typing Speed | Keystrokes per minute | ~60 WPM | ~35 WPM |
| Typing Delay | Time between keystrokes (ms) | ~120ms | ~200ms |
| Backspace Rate | Error correction frequency | ~5% | ~15% |
| Tab Switch Rate | Context switches per minute | ~2/min | ~5/min |
| Mouse Velocity | Navigation speed (px/s) | ~500 | ~300 |
| Idle Time | Pause duration (seconds) | ~3s | ~8s |
| Posture Score | Webcam posture analysis | 85-100 | <70 |

## Machine Learning Models

### 1. Fatigue Prediction (Random Forest)
- **Algorithm**: Random Forest Classifier (100 trees)
- **Input**: Typing speed, error rate, mouse velocity, idle time, etc.
- **Output**: Probability of fatigue (0-100%)
- **Performance**: ~92% accuracy, ~97% ROC AUC

### 2. Behavioral Drift Detection (Isolation Forest)
- **Algorithm**: Isolation Forest (unsupervised)
- **Input**: Current behavioral features
- **Output**: Anomaly score indicating deviation from baseline
- **Use**: Detects when behavior significantly changes

### 3. Cognitive Load Estimator (Weighted Scoring)
- **Algorithm**: Multi-factor weighted combination
- **Components**: Typing load (25%), Error load (25%), Attention (20%), Idle (15%), Engagement (15%)
- **Output**: Cognitive load score (0-100)

### 4. Posture Detection (MediaPipe)
- **Algorithm**: MediaPipe PoseLandmarker + geometric analysis
- **Input**: Webcam frames
- **Output**: Posture score, detected issues (head tilt, slouching, etc.)

## Risk Levels

| Level | Score Range | Action |
|-------|-------------|--------|
| 🟢 Low | 0-25 | Continue working normally |
| 🟡 Moderate | 26-50 | Consider a short break |
| 🟠 High | 51-75 | Take a break soon |
| 🔴 Critical | 76-100 | Immediate break recommended |

## Focus AI Mode

Dedicated deep work mode with:
- **Configurable timer** (default: 25 minutes)
- **Distraction detection** (YouTube, Facebook, Twitter, etc.)
- **Focus score** based on concentration quality
- **Session history** and statistics
- **Auto-start monitoring** when focus begins

## Posture Detection

Real-time webcam-based posture monitoring:
- **Head tilt** detection (>12° threshold)
- **Slouching** detection (>15° deviation)
- **Forward head posture** detection
- **Uneven shoulders** detection
- **Skeleton overlay** on video preview
- **Smoothed readings** to reduce jitter

## Dependencies

| Package | Purpose |
|---------|---------|
| pynput | Keyboard and mouse monitoring |
| psutil | System and process information |
| numpy | Numerical computations |
| pandas | Data manipulation |
| scikit-learn | Machine learning models |
| tkinter | Desktop GUI (included with Python) |
| mediapipe | Pose detection (optional) |
| opencv-python | Webcam capture (optional) |
| pillow | Image processing (optional) |

## Dynamic Paths

- All paths are resolved relative to the project root at runtime.
- To override the MediaPipe pose model location, set `MINDSHIELD_POSE_MODEL` to a custom path.

## Technical Documentation

For detailed technical information including:
- Algorithm explanations
- Code architecture
- Module-by-module breakdown
- Future enhancements
- Patent potential

See: **[Project_Documentation.md](Project_Documentation.md)**

## License

This project is developed for educational and research purposes.

## Authors

Mind-Shield+ Team
>>>>>>> c1412b8 (updated)
