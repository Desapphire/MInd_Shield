"""
Mind-Shield+ Risk Evaluation Engine

Integrates cognitive load, fatigue prediction, and behavioral drift
to produce final risk assessments and recommendations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    """Risk level categories."""
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"


@dataclass
class RiskAssessment:
    """Complete risk assessment result."""
    risk_level: RiskLevel
    risk_score: float  # 0-100
    cognitive_load: float
    fatigue_probability: float
    behavioral_drift: float
    is_drift_detected: bool
    recommendations: List[str]
    contributing_factors: Dict[str, float]
    timestamp: Optional[str] = None


class RiskEvaluationEngine:
    """
    Integrates all cognitive indicators to produce risk assessments.
    
    Combines:
    - Cognitive load score (0-100)
    - Fatigue prediction probability (0-1)
    - Behavioral drift detection
    
    Produces:
    - Overall risk level classification
    - Specific recommendations
    - Contributing factor analysis
    """
    
    def __init__(
        self,
        cognitive_load_weight: float = 0.3,
        fatigue_weight: float = 0.4,
        drift_weight: float = 0.3,
        thresholds: Optional[Dict] = None
    ):
        """
        Initialize the risk evaluation engine.
        
        Args:
            cognitive_load_weight: Weight for cognitive load score
            fatigue_weight: Weight for fatigue probability
            drift_weight: Weight for behavioral drift
            thresholds: Custom thresholds for risk levels
        """
        self.weights = {
            'cognitive_load': cognitive_load_weight,
            'fatigue': fatigue_weight,
            'drift': drift_weight
        }
        
        # Ensure weights sum to 1
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
        
        # Default thresholds
        self.thresholds = thresholds or {
            'low': 25,
            'moderate': 50,
            'high': 75,
            'cognitive_load_alert': 60,
            'fatigue_probability_alert': 0.6,
            'drift_magnitude_alert': 0.4
        }
    
    def evaluate(
        self,
        cognitive_load: float,
        fatigue_probability: float,
        drift_magnitude: float,
        is_drift_detected: bool = False,
        timestamp: Optional[str] = None
    ) -> RiskAssessment:
        """
        Evaluate overall cognitive fatigue risk.
        
        Args:
            cognitive_load: Cognitive load score (0-100)
            fatigue_probability: ML-predicted fatigue probability (0-1)
            drift_magnitude: Behavioral drift magnitude (0-1)
            is_drift_detected: Whether anomaly detection flagged drift
            timestamp: Optional timestamp for the assessment
            
        Returns:
            Complete RiskAssessment
        """
        # Normalize inputs to 0-100 scale
        load_normalized = cognitive_load  # Already 0-100
        fatigue_normalized = fatigue_probability * 100  # Convert to 0-100
        drift_normalized = drift_magnitude * 100  # Convert to 0-100
        
        # Calculate weighted risk score
        risk_score = (
            self.weights['cognitive_load'] * load_normalized +
            self.weights['fatigue'] * fatigue_normalized +
            self.weights['drift'] * drift_normalized
        )
        
        # Apply drift detection boost if anomaly detected
        if is_drift_detected:
            risk_score = min(100, risk_score * 1.2)
        
        # Determine risk level
        risk_level = self._classify_risk(risk_score)
        
        # Generate contributing factors
        contributing_factors = {
            'cognitive_load': load_normalized,
            'fatigue_probability': fatigue_normalized,
            'behavioral_drift': drift_normalized
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            cognitive_load,
            fatigue_probability,
            drift_magnitude,
            is_drift_detected,
            risk_level
        )
        
        return RiskAssessment(
            risk_level=risk_level,
            risk_score=risk_score,
            cognitive_load=cognitive_load,
            fatigue_probability=fatigue_probability,
            behavioral_drift=drift_magnitude,
            is_drift_detected=is_drift_detected,
            recommendations=recommendations,
            contributing_factors=contributing_factors,
            timestamp=timestamp
        )
    
    def _classify_risk(self, risk_score: float) -> RiskLevel:
        """Classify risk score into risk level category."""
        if risk_score < self.thresholds['low']:
            return RiskLevel.LOW
        elif risk_score < self.thresholds['moderate']:
            return RiskLevel.MODERATE
        elif risk_score < self.thresholds['high']:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _generate_recommendations(
        self,
        cognitive_load: float,
        fatigue_probability: float,
        drift_magnitude: float,
        is_drift_detected: bool,
        risk_level: RiskLevel
    ) -> List[str]:
        """Generate specific recommendations based on risk factors."""
        recommendations = []
        
        # General recommendations based on risk level
        if risk_level == RiskLevel.CRITICAL:
            recommendations.append(
                "⚠️ CRITICAL: Take an immediate 15-20 minute break"
            )
            recommendations.append(
                "Consider ending intensive work for the session"
            )
        elif risk_level == RiskLevel.HIGH:
            recommendations.append(
                "High fatigue risk detected - take a 10-minute break soon"
            )
        elif risk_level == RiskLevel.MODERATE:
            recommendations.append(
                "Moderate fatigue indicators - consider a short break"
            )
        
        # Specific recommendations based on indicators
        if cognitive_load > self.thresholds['cognitive_load_alert']:
            recommendations.append(
                "High cognitive load: Try breaking tasks into smaller steps"
            )
        
        if fatigue_probability > self.thresholds['fatigue_probability_alert']:
            recommendations.append(
                "Fatigue patterns detected: Stay hydrated and stretch"
            )
        
        if drift_magnitude > self.thresholds['drift_magnitude_alert']:
            recommendations.append(
                "Behavioral change detected: Review recent work for accuracy"
            )
        
        if is_drift_detected:
            recommendations.append(
                "Unusual interaction patterns: Take a moment to refocus"
            )
        
        # Default message for low risk
        if not recommendations:
            recommendations.append(
                "✓ Cognitive state is normal - continue working"
            )
        
        return recommendations
    
    def evaluate_batch(
        self,
        cognitive_loads: np.ndarray,
        fatigue_probabilities: np.ndarray,
        drift_magnitudes: np.ndarray,
        drift_detected: np.ndarray
    ) -> List[RiskAssessment]:
        """
        Evaluate risk for multiple samples.
        
        Args:
            cognitive_loads: Array of cognitive load scores
            fatigue_probabilities: Array of fatigue probabilities
            drift_magnitudes: Array of drift magnitudes
            drift_detected: Array of drift detection flags
            
        Returns:
            List of RiskAssessment objects
        """
        assessments = []
        
        for i in range(len(cognitive_loads)):
            assessment = self.evaluate(
                cognitive_loads[i],
                fatigue_probabilities[i],
                drift_magnitudes[i],
                bool(drift_detected[i])
            )
            assessments.append(assessment)
        
        return assessments
    
    def get_summary_statistics(
        self,
        assessments: List[RiskAssessment]
    ) -> Dict:
        """
        Get summary statistics from multiple assessments.
        
        Args:
            assessments: List of RiskAssessment objects
            
        Returns:
            Dictionary with summary statistics
        """
        scores = [a.risk_score for a in assessments]
        levels = [a.risk_level for a in assessments]
        
        level_counts = {level: levels.count(level) for level in RiskLevel}
        
        return {
            'total_assessments': len(assessments),
            'mean_risk_score': np.mean(scores),
            'max_risk_score': np.max(scores),
            'min_risk_score': np.min(scores),
            'std_risk_score': np.std(scores),
            'risk_level_distribution': {k.value: v for k, v in level_counts.items()},
            'critical_count': level_counts[RiskLevel.CRITICAL],
            'high_risk_percentage': (level_counts[RiskLevel.HIGH] + 
                                     level_counts[RiskLevel.CRITICAL]) / len(assessments) * 100
        }


class SessionRiskTracker:
    """
    Tracks risk evolution throughout a work session.
    
    Maintains history and provides trend analysis.
    """
    
    def __init__(self, engine: Optional[RiskEvaluationEngine] = None):
        """Initialize the session tracker."""
        self.engine = engine or RiskEvaluationEngine()
        self.history: List[RiskAssessment] = []
        self.start_time: Optional[str] = None
    
    def add_assessment(self, assessment: RiskAssessment) -> None:
        """Add an assessment to the session history."""
        self.history.append(assessment)
        if self.start_time is None:
            self.start_time = assessment.timestamp
    
    def evaluate_and_track(
        self,
        cognitive_load: float,
        fatigue_probability: float,
        drift_magnitude: float,
        is_drift_detected: bool = False,
        timestamp: Optional[str] = None
    ) -> RiskAssessment:
        """Evaluate and add to tracking history."""
        assessment = self.engine.evaluate(
            cognitive_load,
            fatigue_probability,
            drift_magnitude,
            is_drift_detected,
            timestamp
        )
        self.add_assessment(assessment)
        return assessment
    
    def get_trend(self, window: int = 10) -> Dict:
        """
        Analyze risk trend over recent assessments.
        
        Args:
            window: Number of recent assessments to analyze
            
        Returns:
            Trend analysis dictionary
        """
        if len(self.history) < 2:
            return {'trend': 'insufficient_data'}
        
        recent = self.history[-window:]
        scores = [a.risk_score for a in recent]
        
        # Simple linear trend
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]
        
        if slope > 2:
            trend = 'increasing'
        elif slope < -2:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'slope': slope,
            'recent_mean': np.mean(scores),
            'recent_max': np.max(scores),
            'recent_min': np.min(scores)
        }
    
    def should_alert(self) -> Tuple[bool, Optional[str]]:
        """
        Determine if user should be alerted.
        
        Returns:
            Tuple of (should_alert, reason)
        """
        if not self.history:
            return False, None
        
        latest = self.history[-1]
        
        # Alert for critical or high risk
        if latest.risk_level == RiskLevel.CRITICAL:
            return True, "Critical fatigue risk level reached"
        
        # Alert for sustained high risk
        if len(self.history) >= 5:
            recent = self.history[-5:]
            high_count = sum(1 for a in recent if a.risk_level in 
                           [RiskLevel.HIGH, RiskLevel.CRITICAL])
            if high_count >= 3:
                return True, "Sustained elevated risk over multiple assessments"
        
        # Alert for rapid increase
        trend = self.get_trend()
        if trend['trend'] == 'increasing' and trend['slope'] > 5:
            return True, "Rapidly increasing fatigue indicators"
        
        return False, None
    
    def get_session_summary(self) -> Dict:
        """Get comprehensive session summary."""
        if not self.history:
            return {'status': 'no_data'}
        
        summary = self.engine.get_summary_statistics(self.history)
        summary['trend'] = self.get_trend()
        summary['session_duration'] = len(self.history)
        
        should_alert, alert_reason = self.should_alert()
        summary['alert_needed'] = should_alert
        summary['alert_reason'] = alert_reason
        
        return summary


def main():
    """Demonstrate the risk evaluation engine."""
    print("Mind-Shield+ Risk Evaluation Engine")
    print("=" * 50)
    
    # Initialize engine
    engine = RiskEvaluationEngine()
    
    # Test cases
    print("\nTest Case Evaluations:")
    print("-" * 50)
    
    test_cases = [
        {
            'name': 'Normal State',
            'cognitive_load': 25,
            'fatigue_probability': 0.15,
            'drift_magnitude': 0.1,
            'is_drift_detected': False
        },
        {
            'name': 'Moderate Fatigue',
            'cognitive_load': 55,
            'fatigue_probability': 0.45,
            'drift_magnitude': 0.25,
            'is_drift_detected': False
        },
        {
            'name': 'High Risk',
            'cognitive_load': 72,
            'fatigue_probability': 0.70,
            'drift_magnitude': 0.45,
            'is_drift_detected': True
        },
        {
            'name': 'Critical State',
            'cognitive_load': 85,
            'fatigue_probability': 0.88,
            'drift_magnitude': 0.65,
            'is_drift_detected': True
        }
    ]
    
    for tc in test_cases:
        assessment = engine.evaluate(
            cognitive_load=tc['cognitive_load'],
            fatigue_probability=tc['fatigue_probability'],
            drift_magnitude=tc['drift_magnitude'],
            is_drift_detected=tc['is_drift_detected']
        )
        
        print(f"\n{tc['name']}:")
        print(f"  Risk Level: {assessment.risk_level.value}")
        print(f"  Risk Score: {assessment.risk_score:.1f}/100")
        print(f"  Contributing Factors:")
        for factor, value in assessment.contributing_factors.items():
            print(f"    - {factor}: {value:.1f}")
        print(f"  Recommendations:")
        for rec in assessment.recommendations:
            print(f"    • {rec}")
    
    # Session tracking demonstration
    print("\n" + "=" * 50)
    print("Session Tracking Demonstration:")
    print("-" * 50)
    
    tracker = SessionRiskTracker()
    
    # Simulate a work session with increasing fatigue
    session_data = [
        (20, 0.1, 0.05, False),   # Start: normal
        (25, 0.15, 0.08, False),  # Normal
        (35, 0.25, 0.12, False),  # Slight increase
        (45, 0.40, 0.20, False),  # Moderate
        (55, 0.50, 0.28, False),  # Rising
        (65, 0.62, 0.35, True),   # High, drift detected
        (72, 0.70, 0.42, True),   # High
        (78, 0.78, 0.50, True),   # Critical approaching
        (82, 0.85, 0.58, True),   # Critical
        (88, 0.90, 0.65, True),   # Critical
    ]
    
    print("\nSession progression:")
    for i, (load, prob, drift, detected) in enumerate(session_data):
        assessment = tracker.evaluate_and_track(load, prob, drift, detected)
        print(f"  Assessment {i+1}: Risk={assessment.risk_score:.1f} ({assessment.risk_level.value})")
    
    # Session summary
    print("\nSession Summary:")
    summary = tracker.get_session_summary()
    print(f"  Total Assessments: {summary['total_assessments']}")
    print(f"  Mean Risk Score: {summary['mean_risk_score']:.1f}")
    print(f"  Max Risk Score: {summary['max_risk_score']:.1f}")
    print(f"  Trend: {summary['trend']['trend']} (slope: {summary['trend']['slope']:.2f})")
    print(f"  Alert Needed: {summary['alert_needed']}")
    if summary['alert_reason']:
        print(f"  Alert Reason: {summary['alert_reason']}")
    
    print(f"\n  Risk Level Distribution:")
    for level, count in summary['risk_level_distribution'].items():
        print(f"    {level}: {count}")
    
    return engine, tracker


if __name__ == "__main__":
    engine, tracker = main()
