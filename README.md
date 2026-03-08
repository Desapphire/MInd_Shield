# Mind-Shield+ Core Model

**AI-Powered Cognitive Fatigue Detection System - Prototype**

## Overview

Mind-Shield+ is an intelligent system that monitors behavioral interaction patterns to estimate cognitive load, predict mental fatigue, and detect behavioral drift. This prototype demonstrates the core artificial intelligence functionality without requiring a browser extension or mobile application.

## Features

- **Behavioral Data Simulation**: Generate synthetic interaction data representing normal and fatigued cognitive states
- **Feature Engineering**: Extract meaningful cognitive indicators from raw behavioral signals
- **Cognitive Load Estimation**: Calculate mental effort scores from behavioral patterns
- **Fatigue Prediction**: Machine learning models to predict fatigue probability
- **Behavioral Drift Detection**: Identify deviations from baseline interaction patterns
- **Risk Evaluation**: Integrated assessment combining all cognitive indicators
- **Visualization**: Comprehensive dashboards and charts for analysis

## Project Structure

```
EDI_Project/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── data_simulation.py       # Synthetic behavioral data generation
│   ├── feature_engineering.py   # Feature extraction and transformation
│   ├── cognitive_load.py        # Cognitive load scoring
│   ├── fatigue_model.py         # ML fatigue prediction models
│   ├── anomaly_detection.py     # Behavioral drift detection
│   ├── risk_evaluation.py       # Integrated risk assessment
│   ├── visualization.py         # Plotting and dashboards
│   └── main.py                  # Main demonstration script
├── output/                      # Generated visualizations
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.10 or newer

### Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 🌐 Run the Live Web Application (Recommended)

The easiest way to demonstrate Mind-Shield+ is through the web interface:

```bash
# Install dependencies
pip install -r requirements.txt

# Start the web server
python app.py
```

Then open **http://localhost:5000** in your browser.

**Features:**
- Real-time behavioral monitoring (typing, mouse, scrolling)
- Live fatigue probability display
- Dynamic cognitive load estimation
- Interactive testing area
- Permission/consent handling
- Visual dashboard with charts

### Run the Full Demonstration (Console)

```bash
cd src
python main.py
```

This will:
1. Generate synthetic behavioral data
2. Extract cognitive features
3. Train fatigue prediction models
4. Run behavioral drift detection
5. Estimate cognitive load
6. Evaluate integrated risk
7. Simulate a 90-minute work session
8. Generate visualization outputs

### Run Individual Modules

Each module can be run independently for testing:

```bash
# Data simulation
python src/data_simulation.py

# Feature engineering
python src/feature_engineering.py

# Cognitive load estimation
python src/cognitive_load.py

# Fatigue prediction
python src/fatigue_model.py

# Anomaly detection
python src/anomaly_detection.py

# Risk evaluation
python src/risk_evaluation.py

# Visualization
python src/visualization.py
```

## Behavioral Variables

The system monitors these behavioral signals:

| Variable | Description | Normal | Fatigued |
|----------|-------------|--------|----------|
| Typing Delay | Time between keystrokes (ms) | ~120ms | ~200ms |
| Dwell Time | Key press duration (ms) | ~80ms | ~120ms |
| Backspace Rate | Error correction frequency | ~5% | ~15% |
| Tab Switch Rate | Context switches per minute | ~2/min | ~5/min |
| Mouse Velocity | Navigation speed (px/s) | ~500 | ~300 |
| Idle Time | Pause duration (seconds) | ~3s | ~8s |

## Model Performance

The Random Forest classifier typically achieves:
- **Accuracy**: ~92%
- **Precision**: ~91%
- **Recall**: ~93%
- **ROC AUC**: ~97%

## Example Output

```
📊 CURRENT USER STATUS
========================================
  Cognitive Load Score:    72.5/100
  Fatigue Probability:     78.5%
  Behavioral Drift:        0.45
  Overall Risk Score:      68.3/100
  Risk Classification:     HIGH

  📋 RECOMMENDATIONS:
     • High fatigue risk detected - take a 10-minute break soon
     • Typing patterns suggest mental fatigue - slow down
     • Behavioral change detected: Review recent work for accuracy
========================================
```

## Risk Levels

| Level | Score Range | Action |
|-------|-------------|--------|
| Low | 0-25 | Continue working normally |
| Moderate | 25-50 | Consider a short break |
| High | 50-75 | Take a break soon |
| Critical | 75-100 | Immediate break recommended |

## Outputs

After running, check the `output/` directory for:
- `session_analysis.png` - Time-series fatigue analysis
- `feature_importance.png` - ML feature importance chart
- `dashboard.png` - Monitoring dashboard
- `behavioral_patterns.png` - Normal vs fatigued comparison

## Technical Details

### Algorithms Used

- **Feature Engineering**: Z-score normalization, composite indicators
- **Classification**: Logistic Regression, Random Forest, Gradient Boosting
- **Anomaly Detection**: Isolation Forest, Local Outlier Factor
- **Risk Scoring**: Weighted multi-factor integration

### Dependencies

- NumPy: Numerical computing
- Pandas: Data manipulation
- Scikit-learn: Machine learning
- Matplotlib: Visualization
- Seaborn: Statistical graphics

## Future Development

This prototype can be integrated into:
- Browser extensions for real-time monitoring
- Desktop applications for productivity tools
- Mobile apps for cognitive wellness tracking
- Workplace wellness platforms

## License

This project is developed for educational and research purposes.

## Authors

Mind-Shield+ Team
