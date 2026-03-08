"""
Mind-Shield+ Anomaly Detection Module

Detects behavioral drift from baseline patterns using
unsupervised anomaly detection algorithms.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class AnomalyResult:
    """Result of anomaly detection for a sample."""
    is_anomaly: bool
    anomaly_score: float
    drift_magnitude: float
    contributing_features: List[str]


class BehavioralDriftDetector:
    """
    Detects behavioral drift from user's baseline patterns.
    
    Uses unsupervised anomaly detection to identify when current
    behavior significantly deviates from historical baseline,
    indicating potential cognitive stress or fatigue.
    
    Methods:
    - Isolation Forest: Efficient anomaly detection in high dimensions
    - Local Outlier Factor: Density-based anomaly detection
    - Statistical: Z-score based deviation from baseline
    """
    
    def __init__(
        self,
        method: str = 'isolation_forest',
        contamination: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize the drift detector.
        
        Args:
            method: 'isolation_forest', 'lof', or 'statistical'
            contamination: Expected proportion of anomalies
            random_state: Random seed for reproducibility
        """
        self.method = method
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.baseline_stats: Optional[Dict] = None
        self.feature_names: List[str] = []
        self.is_fitted = False
    
    def _initialize_model(self) -> None:
        """Initialize the anomaly detection model."""
        if self.method == 'isolation_forest':
            self.model = IsolationForest(
                n_estimators=100,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.method == 'lof':
            self.model = LocalOutlierFactor(
                n_neighbors=20,
                contamination=self.contamination,
                novelty=True
            )
        elif self.method == 'statistical':
            self.model = None  # Uses statistical methods directly
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def fit_baseline(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> None:
        """
        Fit the detector to baseline (normal) behavioral data.
        
        Args:
            X: Baseline feature matrix (normal behavior samples)
            feature_names: Names of features
        """
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Compute baseline statistics
        self.baseline_stats = {
            'mean': np.mean(X_scaled, axis=0),
            'std': np.std(X_scaled, axis=0),
            'median': np.median(X_scaled, axis=0),
            'q25': np.percentile(X_scaled, 25, axis=0),
            'q75': np.percentile(X_scaled, 75, axis=0),
            'iqr': np.percentile(X_scaled, 75, axis=0) - np.percentile(X_scaled, 25, axis=0)
        }
        
        # Initialize and fit the anomaly detection model
        self._initialize_model()
        
        if self.model is not None:
            self.model.fit(X_scaled)
        
        self.is_fitted = True
    
    def detect_anomaly(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in new behavioral data.
        
        Args:
            X: Feature matrix to analyze
            
        Returns:
            Tuple of (anomaly labels, anomaly scores)
            Labels: -1 = anomaly, 1 = normal
            Scores: Lower = more anomalous
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detection")
        
        X_scaled = self.scaler.transform(X)
        
        if self.method == 'isolation_forest':
            labels = self.model.predict(X_scaled)
            scores = self.model.decision_function(X_scaled)
        elif self.method == 'lof':
            labels = self.model.predict(X_scaled)
            scores = self.model.decision_function(X_scaled)
        else:  # statistical
            labels, scores = self._statistical_detection(X_scaled)
        
        return labels, scores
    
    def _statistical_detection(self, X_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Statistical anomaly detection using z-scores."""
        z_scores = np.abs((X_scaled - self.baseline_stats['mean']) / 
                         (self.baseline_stats['std'] + 1e-6))
        
        # Maximum z-score across features
        max_z = np.max(z_scores, axis=1)
        
        # Mean z-score across features
        mean_z = np.mean(z_scores, axis=1)
        
        # Combined anomaly score (negative for anomalies)
        scores = -mean_z
        
        # Label as anomaly if max z-score > 2.5 or mean z-score > 2
        labels = np.where((max_z > 2.5) | (mean_z > 2), -1, 1)
        
        return labels, scores
    
    def get_drift_magnitude(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the magnitude of behavioral drift from baseline.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of drift magnitudes (0-1 scale)
        """
        X_scaled = self.scaler.transform(X)
        
        # Mahalanobis-like distance from baseline center
        deviations = np.abs(X_scaled - self.baseline_stats['mean'])
        normalized_deviations = deviations / (self.baseline_stats['std'] + 1e-6)
        
        # Mean normalized deviation
        drift = np.mean(normalized_deviations, axis=1)
        
        # Normalize to 0-1 scale (clip at 5 std deviations)
        drift_normalized = np.clip(drift / 5, 0, 1)
        
        return drift_normalized
    
    def get_contributing_features(
        self,
        X_single: np.ndarray,
        top_n: int = 3
    ) -> List[Dict[str, float]]:
        """
        Identify features contributing most to detected drift.
        
        Args:
            X_single: Single sample feature vector
            top_n: Number of top features to return
            
        Returns:
            List of dictionaries with feature name and deviation
        """
        if X_single.ndim == 1:
            X_single = X_single.reshape(1, -1)
        
        X_scaled = self.scaler.transform(X_single)[0]
        
        # Calculate deviation for each feature
        deviations = np.abs(X_scaled - self.baseline_stats['mean'])
        normalized_deviations = deviations / (self.baseline_stats['std'] + 1e-6)
        
        # Get top contributing features
        indices = np.argsort(normalized_deviations)[::-1][:top_n]
        
        contributions = []
        for idx in indices:
            contributions.append({
                'feature': self.feature_names[idx],
                'deviation': normalized_deviations[idx],
                'current_value': X_scaled[idx],
                'baseline_mean': self.baseline_stats['mean'][idx]
            })
        
        return contributions
    
    def analyze_sample(self, X_single: np.ndarray) -> AnomalyResult:
        """
        Comprehensive anomaly analysis for a single sample.
        
        Args:
            X_single: Single sample feature vector
            
        Returns:
            AnomalyResult with detection details
        """
        if X_single.ndim == 1:
            X_single = X_single.reshape(1, -1)
        
        labels, scores = self.detect_anomaly(X_single)
        drift_magnitude = self.get_drift_magnitude(X_single)[0]
        contributing = self.get_contributing_features(X_single)
        
        return AnomalyResult(
            is_anomaly=(labels[0] == -1),
            anomaly_score=float(scores[0]),
            drift_magnitude=drift_magnitude,
            contributing_features=[c['feature'] for c in contributing]
        )


class TemporalDriftAnalyzer:
    """
    Analyzes behavioral drift over time windows.
    
    Tracks how user behavior changes during a session
    and identifies emerging drift patterns.
    """
    
    def __init__(self, window_size: int = 10, drift_threshold: float = 0.3):
        """
        Initialize temporal drift analyzer.
        
        Args:
            window_size: Number of samples in analysis window
            drift_threshold: Threshold for significant drift
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.baseline_detector = BehavioralDriftDetector()
    
    def analyze_session(
        self,
        session_data: pd.DataFrame,
        feature_columns: List[str]
    ) -> pd.DataFrame:
        """
        Analyze behavioral drift throughout a session.
        
        Args:
            session_data: Time-ordered session data
            feature_columns: Columns to analyze
            
        Returns:
            DataFrame with drift analysis over time
        """
        X = session_data[feature_columns].values
        
        # Use first window as baseline
        baseline_data = X[:self.window_size]
        self.baseline_detector.fit_baseline(baseline_data, feature_columns)
        
        results = []
        
        for i in range(len(X)):
            sample = X[i:i+1]
            
            if i < self.window_size:
                # Still in baseline period
                drift_mag = 0.0
                is_drift = False
                anomaly_score = 0.0
            else:
                labels, scores = self.baseline_detector.detect_anomaly(sample)
                drift_mag = self.baseline_detector.get_drift_magnitude(sample)[0]
                is_drift = drift_mag > self.drift_threshold
                anomaly_score = float(scores[0])
            
            results.append({
                'index': i,
                'drift_magnitude': drift_mag,
                'is_significant_drift': is_drift,
                'anomaly_score': anomaly_score
            })
        
        return pd.DataFrame(results)


def main():
    """Demonstrate the anomaly detection module."""
    print("Mind-Shield+ Anomaly Detection Module")
    print("=" * 50)
    
    # Import required modules
    from data_simulation import BehavioralDataSimulator
    from feature_engineering import FeatureEngineer
    
    # Generate data
    print("\nGenerating behavioral data...")
    simulator = BehavioralDataSimulator(random_seed=42)
    df = simulator.generate_dataset(n_normal=500, n_fatigued=100)
    
    # Extract features
    engineer = FeatureEngineer()
    X, feature_names = engineer.fit_transform(df)
    y = df['fatigue_label'].values
    
    # Split into baseline (normal) and test (mixed) data
    X_normal = X[y == 0]
    
    print(f"Baseline samples (normal): {len(X_normal)}")
    print(f"Test samples: {len(X)}")
    
    # Test different methods
    print("\n" + "=" * 50)
    print("Testing Anomaly Detection Methods:")
    print("-" * 50)
    
    methods = ['isolation_forest', 'lof', 'statistical']
    
    for method in methods:
        detector = BehavioralDriftDetector(method=method, contamination=0.1)
        detector.fit_baseline(X_normal, feature_names)
        
        labels, scores = detector.detect_anomaly(X)
        drift = detector.get_drift_magnitude(X)
        
        # Calculate detection rates
        anomaly_detected = labels == -1
        true_fatigued = y == 1
        
        # True positive rate for fatigued samples
        tp_rate = np.sum(anomaly_detected & true_fatigued) / max(np.sum(true_fatigued), 1)
        # False positive rate for normal samples
        fp_rate = np.sum(anomaly_detected & ~true_fatigued) / max(np.sum(~true_fatigued), 1)
        
        print(f"\n{method.upper()}:")
        print(f"  Total anomalies detected: {np.sum(anomaly_detected)}")
        print(f"  True Positive Rate (fatigued): {tp_rate:.2%}")
        print(f"  False Positive Rate (normal): {fp_rate:.2%}")
        print(f"  Mean drift (normal): {drift[~true_fatigued].mean():.3f}")
        print(f"  Mean drift (fatigued): {drift[true_fatigued].mean():.3f}")
    
    # Detailed analysis example
    print("\n" + "=" * 50)
    print("Detailed Sample Analysis:")
    print("-" * 40)
    
    detector = BehavioralDriftDetector(method='isolation_forest')
    detector.fit_baseline(X_normal, feature_names)
    
    # Analyze a normal sample
    normal_idx = np.where(y == 0)[0][0]
    normal_result = detector.analyze_sample(X[normal_idx])
    
    print(f"\nNormal Sample Analysis:")
    print(f"  Is Anomaly: {normal_result.is_anomaly}")
    print(f"  Anomaly Score: {normal_result.anomaly_score:.3f}")
    print(f"  Drift Magnitude: {normal_result.drift_magnitude:.3f}")
    print(f"  Top Contributing Features: {normal_result.contributing_features}")
    
    # Analyze a fatigued sample
    fatigued_idx = np.where(y == 1)[0][0]
    fatigued_result = detector.analyze_sample(X[fatigued_idx])
    
    print(f"\nFatigued Sample Analysis:")
    print(f"  Is Anomaly: {fatigued_result.is_anomaly}")
    print(f"  Anomaly Score: {fatigued_result.anomaly_score:.3f}")
    print(f"  Drift Magnitude: {fatigued_result.drift_magnitude:.3f}")
    print(f"  Top Contributing Features: {fatigued_result.contributing_features}")
    
    # Time-series drift analysis
    print("\n" + "=" * 50)
    print("Temporal Drift Analysis:")
    print("-" * 40)
    
    ts_df = simulator.generate_time_series(duration_minutes=60, fatigue_onset_minute=30)
    feature_cols = ['typing_delay', 'backspace_rate', 'tab_switch_rate', 
                   'mouse_velocity', 'idle_time']
    
    temporal_analyzer = TemporalDriftAnalyzer(window_size=20, drift_threshold=0.3)
    drift_results = temporal_analyzer.analyze_session(ts_df, feature_cols)
    
    # Show drift progression
    print("\nDrift progression over session:")
    checkpoints = [0, 20, 40, 60, 80, 100, 119]
    for cp in checkpoints:
        if cp < len(drift_results):
            row = drift_results.iloc[cp]
            minute = ts_df.iloc[cp]['minute']
            status = "DRIFT" if row['is_significant_drift'] else "Normal"
            print(f"  Minute {minute:.0f}: Drift={row['drift_magnitude']:.3f} [{status}]")
    
    return detector


if __name__ == "__main__":
    detector = main()
