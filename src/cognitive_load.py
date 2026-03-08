"""
Mind-Shield+ Cognitive Load Estimation Module

Estimates mental effort and cognitive workload based on
behavioral indicators using weighted scoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CognitiveLoadWeights:
    """Configuration for cognitive load calculation weights."""
    typing_delay: float = 0.25
    error_rate: float = 0.25
    attention_switching: float = 0.20
    idle_time: float = 0.15
    engagement: float = 0.15


class CognitiveLoadEstimator:
    """
    Estimates cognitive load score from behavioral indicators.
    
    Cognitive load represents the mental effort required to perform tasks.
    Higher load indicates increased mental strain and potential fatigue risk.
    
    The score is computed as a weighted combination of:
    - Typing delay: Slower typing indicates hesitation/mental fatigue
    - Error rate: More corrections indicate cognitive pressure
    - Attention switching: Fragmented attention increases load
    - Idle time: Variable pauses indicate engagement fluctuations
    """
    
    def __init__(
        self,
        weights: Optional[CognitiveLoadWeights] = None,
        baseline_calibration: bool = True
    ):
        """
        Initialize the cognitive load estimator.
        
        Args:
            weights: Custom weights for load components
            baseline_calibration: Whether to calibrate to user baseline
        """
        self.weights = weights or CognitiveLoadWeights()
        self.baseline_calibration = baseline_calibration
        self.baseline_stats: Optional[Dict] = None
        
        # Reference values for normal cognitive functioning
        self.reference_values = {
            'typing_delay_normal': 120,    # ms
            'typing_delay_high': 250,      # ms
            'backspace_rate_normal': 0.05,
            'backspace_rate_high': 0.20,
            'tab_switch_normal': 2.0,      # per minute
            'tab_switch_high': 8.0,
            'idle_time_normal': 3.0,       # seconds
            'idle_time_high': 10.0,
            'mouse_velocity_normal': 500,  # pixels/sec
            'mouse_velocity_low': 200,
        }
    
    def calibrate_baseline(self, baseline_df: pd.DataFrame) -> None:
        """
        Calibrate the estimator to a user's baseline behavior.
        
        Args:
            baseline_df: DataFrame with baseline behavioral samples
        """
        self.baseline_stats = {
            'typing_delay_mean': baseline_df['typing_delay'].mean(),
            'typing_delay_std': baseline_df['typing_delay'].std(),
            'backspace_rate_mean': baseline_df['backspace_rate'].mean(),
            'backspace_rate_std': baseline_df['backspace_rate'].std(),
            'tab_switch_mean': baseline_df['tab_switch_rate'].mean(),
            'tab_switch_std': baseline_df['tab_switch_rate'].std(),
            'idle_time_mean': baseline_df['idle_time'].mean(),
            'idle_time_std': baseline_df['idle_time'].std(),
            'mouse_velocity_mean': baseline_df['mouse_velocity'].mean(),
            'mouse_velocity_std': baseline_df['mouse_velocity'].std(),
        }
    
    def _normalize_to_score(
        self,
        value: float,
        normal_val: float,
        high_val: float,
        inverse: bool = False
    ) -> float:
        """
        Normalize a value to a 0-1 score.
        
        Args:
            value: Current value
            normal_val: Expected normal value
            high_val: High load threshold value
            inverse: True if lower values indicate higher load
            
        Returns:
            Normalized score between 0 and 1
        """
        if inverse:
            # For metrics where lower = higher load (e.g., mouse velocity)
            if value >= normal_val:
                return 0.0
            elif value <= high_val:
                return 1.0
            else:
                return (normal_val - value) / (normal_val - high_val)
        else:
            # For metrics where higher = higher load
            if value <= normal_val:
                return 0.0
            elif value >= high_val:
                return 1.0
            else:
                return (value - normal_val) / (high_val - normal_val)
    
    def estimate_load(self, sample: pd.Series) -> Tuple[float, Dict[str, float]]:
        """
        Estimate cognitive load for a single behavioral sample.
        
        Args:
            sample: Series with behavioral variables
            
        Returns:
            Tuple of (total load score, component scores dict)
        """
        components = {}
        
        # Typing delay component
        typing_delay = sample.get('typing_delay', self.reference_values['typing_delay_normal'])
        components['typing_load'] = self._normalize_to_score(
            typing_delay,
            self.reference_values['typing_delay_normal'],
            self.reference_values['typing_delay_high']
        )
        
        # Error rate component
        backspace_rate = sample.get('backspace_rate', self.reference_values['backspace_rate_normal'])
        components['error_load'] = self._normalize_to_score(
            backspace_rate,
            self.reference_values['backspace_rate_normal'],
            self.reference_values['backspace_rate_high']
        )
        
        # Attention switching component
        tab_switch = sample.get('tab_switch_rate', self.reference_values['tab_switch_normal'])
        components['attention_load'] = self._normalize_to_score(
            tab_switch,
            self.reference_values['tab_switch_normal'],
            self.reference_values['tab_switch_high']
        )
        
        # Idle time component
        idle_time = sample.get('idle_time', self.reference_values['idle_time_normal'])
        components['idle_load'] = self._normalize_to_score(
            idle_time,
            self.reference_values['idle_time_normal'],
            self.reference_values['idle_time_high']
        )
        
        # Engagement component (inverse - lower velocity = higher load)
        mouse_velocity = sample.get('mouse_velocity', self.reference_values['mouse_velocity_normal'])
        components['engagement_load'] = self._normalize_to_score(
            mouse_velocity,
            self.reference_values['mouse_velocity_normal'],
            self.reference_values['mouse_velocity_low'],
            inverse=True
        )
        
        # Calculate weighted total
        total_load = (
            self.weights.typing_delay * components['typing_load'] +
            self.weights.error_rate * components['error_load'] +
            self.weights.attention_switching * components['attention_load'] +
            self.weights.idle_time * components['idle_load'] +
            self.weights.engagement * components['engagement_load']
        )
        
        # Scale to 0-100
        total_load_score = min(100, max(0, total_load * 100))
        
        return total_load_score, components
    
    def estimate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate cognitive load for a batch of samples.
        
        Args:
            df: DataFrame with behavioral data
            
        Returns:
            DataFrame with load scores and components
        """
        results = []
        
        for idx in range(len(df)):
            sample = df.iloc[idx]
            total_load, components = self.estimate_load(sample)
            
            result = {'cognitive_load_score': total_load}
            result.update({f'load_{k}': v * 100 for k, v in components.items()})
            results.append(result)
        
        return pd.DataFrame(results, index=df.index)
    
    def get_load_category(self, load_score: float) -> str:
        """
        Categorize cognitive load level.
        
        Args:
            load_score: Load score (0-100)
            
        Returns:
            Category string
        """
        if load_score < 30:
            return "Low"
        elif load_score < 50:
            return "Moderate"
        elif load_score < 70:
            return "High"
        else:
            return "Critical"
    
    def get_recommendations(self, load_score: float, components: Dict[str, float]) -> list:
        """
        Generate recommendations based on load analysis.
        
        Args:
            load_score: Total cognitive load score
            components: Individual load component scores
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if load_score >= 70:
            recommendations.append("Consider taking a 10-15 minute break")
        elif load_score >= 50:
            recommendations.append("Brief 5-minute break recommended")
        
        # Component-specific recommendations
        if components.get('typing_load', 0) > 0.6:
            recommendations.append("Typing rhythm suggests mental fatigue - slow down")
        
        if components.get('error_load', 0) > 0.6:
            recommendations.append("High error rate detected - review work carefully")
        
        if components.get('attention_load', 0) > 0.6:
            recommendations.append("Frequent context switching - try to focus on one task")
        
        if components.get('idle_load', 0) > 0.6:
            recommendations.append("Extended pauses detected - maintain steady pace")
        
        if not recommendations:
            recommendations.append("Cognitive load is within normal range")
        
        return recommendations


def main():
    """Demonstrate the cognitive load estimation module."""
    print("Mind-Shield+ Cognitive Load Estimation Module")
    print("=" * 50)
    
    # Import data simulation
    from data_simulation import BehavioralDataSimulator
    
    # Generate sample data
    simulator = BehavioralDataSimulator(random_seed=42)
    df = simulator.generate_dataset(n_normal=50, n_fatigued=50)
    
    # Initialize estimator
    estimator = CognitiveLoadEstimator()
    
    # Estimate batch
    print("\nEstimating cognitive load for batch...")
    load_results = estimator.estimate_batch(df)
    
    # Combine results
    combined = pd.concat([df[['state', 'fatigue_label']], load_results], axis=1)
    
    # Summary statistics by state
    print("\nCognitive Load Statistics by State:")
    print("-" * 40)
    summary = combined.groupby('state')['cognitive_load_score'].agg(['mean', 'std', 'min', 'max'])
    print(summary.round(2))
    
    # Show examples
    print("\n" + "=" * 50)
    print("Example Estimations:")
    print("-" * 40)
    
    for state in ['normal', 'fatigued']:
        sample = df[df['state'] == state].iloc[0]
        load_score, components = estimator.estimate_load(sample)
        category = estimator.get_load_category(load_score)
        recommendations = estimator.get_recommendations(load_score, components)
        
        print(f"\n{state.upper()} State Sample:")
        print(f"  Cognitive Load Score: {load_score:.1f}/100 ({category})")
        print(f"  Components:")
        for comp, val in components.items():
            print(f"    - {comp}: {val*100:.1f}%")
        print(f"  Recommendations:")
        for rec in recommendations:
            print(f"    - {rec}")
    
    # Time series analysis
    print("\n" + "=" * 50)
    print("Time Series Cognitive Load Analysis:")
    print("-" * 40)
    
    ts_df = simulator.generate_time_series(duration_minutes=60, fatigue_onset_minute=30)
    ts_loads = estimator.estimate_batch(ts_df)
    
    # Show progression
    print("\nLoad progression over time:")
    checkpoints = [0, 15, 30, 45, 59]
    for cp in checkpoints:
        idx = min(cp * 2, len(ts_loads) - 1)  # Assuming 30-second intervals
        load = ts_loads.iloc[idx]['cognitive_load_score']
        category = estimator.get_load_category(load)
        print(f"  Minute {cp}: Load = {load:.1f} ({category})")
    
    return load_results


if __name__ == "__main__":
    load_results = main()
