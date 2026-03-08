"""
Mind-Shield+ Feature Engineering Module

Transforms raw behavioral signals into meaningful cognitive indicators
for machine learning models.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, List, Optional


class FeatureEngineer:
    """
    Extracts and transforms behavioral features for cognitive modeling.
    
    Feature categories:
    - Typing features: Motor coordination and hesitation patterns
    - Attention features: Context switching and focus indicators
    - Error features: Cognitive overload indicators
    - Temporal features: Engagement fluctuations
    """
    
    def __init__(self, normalization: str = 'standard'):
        """
        Initialize the feature engineer.
        
        Args:
            normalization: 'standard' (z-score) or 'minmax' (0-1 scaling)
        """
        self.normalization = normalization
        self.scaler = None
        self.feature_names: List[str] = []
        self._is_fitted = False
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract derived features from raw behavioral data.
        
        Args:
            df: DataFrame with raw behavioral variables
            
        Returns:
            DataFrame with extracted features
        """
        features = pd.DataFrame()
        
        # === Typing Features ===
        # Raw typing metrics
        if 'typing_delay' in df.columns:
            features['typing_delay'] = df['typing_delay']
            features['typing_speed'] = 1000 / df['typing_delay'].clip(lower=1)  # keystrokes per second
            
        if 'dwell_time' in df.columns:
            features['dwell_time'] = df['dwell_time']
            # Flight time approximation (time between key release and next key press)
            features['flight_time'] = (df['typing_delay'] - df['dwell_time']).clip(lower=0)
            
        # Typing rhythm variability (coefficient of variation proxy)
        if 'typing_delay' in df.columns:
            # For individual samples, use deviation from expected normal
            features['typing_variability'] = np.abs(df['typing_delay'] - 120) / 120
        
        # === Attention Features ===
        if 'tab_switch_rate' in df.columns:
            features['tab_switch_rate'] = df['tab_switch_rate']
            # High switching indicates fragmented attention
            features['attention_fragmentation'] = np.log1p(df['tab_switch_rate'])
            
        if 'scroll_velocity' in df.columns:
            features['scroll_velocity'] = df['scroll_velocity']
            # Scroll pattern irregularity
            features['scroll_irregularity'] = np.abs(df['scroll_velocity'] - 300) / 300
            
        if 'interaction_burst' in df.columns:
            features['interaction_burst'] = df['interaction_burst']
            # Low burst density indicates fatigue
            features['burst_intensity'] = df['interaction_burst'] / 15  # normalized to expected normal
        
        # === Error Features ===
        if 'backspace_rate' in df.columns:
            features['backspace_rate'] = df['backspace_rate']
            # Correction intensity
            features['correction_intensity'] = np.log1p(df['backspace_rate'] * 100)
            # Error ratio (proportion of corrections to normal typing)
            features['error_ratio'] = df['backspace_rate'] / (1 - df['backspace_rate']).clip(lower=0.01)
        
        # === Temporal/Engagement Features ===
        if 'idle_time' in df.columns:
            features['idle_time'] = df['idle_time']
            # Engagement score (inverse of idle time)
            features['engagement_score'] = 1 / (1 + df['idle_time'])
            # Idle pattern (deviation from expected)
            features['idle_deviation'] = np.abs(df['idle_time'] - 3) / 3
            
        if 'mouse_velocity' in df.columns:
            features['mouse_velocity'] = df['mouse_velocity']
            # Navigation confidence
            features['navigation_confidence'] = df['mouse_velocity'] / 500
        
        # === Composite Features ===
        # Overall motor coordination score
        if all(col in features.columns for col in ['typing_speed', 'mouse_velocity']):
            features['motor_coordination'] = (
                features['typing_speed'] / features['typing_speed'].mean() +
                features['mouse_velocity'] / features['mouse_velocity'].mean()
            ) / 2
        
        # Cognitive strain indicator
        strain_components = []
        if 'typing_variability' in features.columns:
            strain_components.append(features['typing_variability'])
        if 'attention_fragmentation' in features.columns:
            strain_components.append(features['attention_fragmentation'] / features['attention_fragmentation'].max())
        if 'error_ratio' in features.columns:
            strain_components.append(features['error_ratio'] / features['error_ratio'].max())
        if 'idle_deviation' in features.columns:
            strain_components.append(features['idle_deviation'])
            
        if strain_components:
            features['cognitive_strain'] = np.mean(strain_components, axis=0)
        
        # Focus quality score
        focus_components = []
        if 'engagement_score' in features.columns:
            focus_components.append(features['engagement_score'])
        if 'burst_intensity' in features.columns:
            focus_components.append(features['burst_intensity'].clip(upper=1))
        if 'attention_fragmentation' in features.columns:
            focus_components.append(1 - features['attention_fragmentation'] / features['attention_fragmentation'].max())
            
        if focus_components:
            features['focus_quality'] = np.mean(focus_components, axis=0)
        
        self.feature_names = list(features.columns)
        return features
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features and fit the normalizer.
        
        Args:
            df: Raw behavioral data
            
        Returns:
            Tuple of (normalized feature array, feature names)
        """
        features = self.extract_features(df)
        
        # Initialize scaler
        if self.normalization == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        # Fit and transform
        normalized = self.scaler.fit_transform(features)
        self._is_fitted = True
        
        return normalized, self.feature_names
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract and normalize features using fitted scaler.
        
        Args:
            df: Raw behavioral data
            
        Returns:
            Normalized feature array
        """
        if not self._is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform. Use fit_transform first.")
        
        features = self.extract_features(df)
        return self.scaler.transform(features)
    
    def get_feature_importance_names(self) -> List[str]:
        """Get list of feature names for interpretation."""
        return self.feature_names.copy()


class TemporalFeatureEngineer:
    """
    Extracts time-based features from sequential behavioral data.
    
    Computes rolling statistics and trend indicators for
    time-series cognitive modeling.
    """
    
    def __init__(self, window_sizes: List[int] = [5, 10, 20]):
        """
        Initialize temporal feature engineer.
        
        Args:
            window_sizes: Rolling window sizes for statistics
        """
        self.window_sizes = window_sizes
    
    def extract_temporal_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract temporal features from time-series data.
        
        Args:
            df: Time-series behavioral data
            columns: Columns to compute features for (default: all numeric)
            
        Returns:
            DataFrame with temporal features
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
            columns = [c for c in columns if c not in ['fatigue_label', 'fatigue_intensity', 'minute']]
        
        features = pd.DataFrame(index=df.index)
        
        for col in columns:
            if col not in df.columns:
                continue
                
            for window in self.window_sizes:
                # Rolling mean
                features[f'{col}_rolling_mean_{window}'] = (
                    df[col].rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling standard deviation
                features[f'{col}_rolling_std_{window}'] = (
                    df[col].rolling(window=window, min_periods=1).std().fillna(0)
                )
                
                # Rolling min/max range
                rolling_min = df[col].rolling(window=window, min_periods=1).min()
                rolling_max = df[col].rolling(window=window, min_periods=1).max()
                features[f'{col}_rolling_range_{window}'] = rolling_max - rolling_min
            
            # Trend features (difference from previous)
            features[f'{col}_diff'] = df[col].diff().fillna(0)
            features[f'{col}_diff_abs'] = features[f'{col}_diff'].abs()
            
            # Acceleration (second derivative)
            features[f'{col}_acceleration'] = features[f'{col}_diff'].diff().fillna(0)
            
            # Cumulative deviation from start
            features[f'{col}_cumulative_drift'] = (df[col] - df[col].iloc[0]).abs()
        
        return features


def main():
    """Demonstrate the feature engineering module."""
    print("Mind-Shield+ Feature Engineering Module")
    print("=" * 50)
    
    # Import data simulation
    from data_simulation import BehavioralDataSimulator
    
    # Generate sample data
    simulator = BehavioralDataSimulator(random_seed=42)
    df = simulator.generate_dataset(n_normal=100, n_fatigued=100)
    
    # Extract static features
    print("\nExtracting static behavioral features...")
    engineer = FeatureEngineer(normalization='standard')
    features, feature_names = engineer.fit_transform(df)
    
    print(f"Input shape: {df.shape}")
    print(f"Output shape: {features.shape}")
    print(f"\nExtracted features ({len(feature_names)}):")
    for i, name in enumerate(feature_names, 1):
        print(f"  {i}. {name}")
    
    # Feature statistics
    print(f"\nFeature statistics (normalized):")
    feature_df = pd.DataFrame(features, columns=feature_names)
    print(feature_df.describe().round(3))
    
    # Temporal features
    print("\n" + "=" * 50)
    print("Extracting temporal features from time-series...")
    ts_df = simulator.generate_time_series(duration_minutes=60)
    
    temporal_engineer = TemporalFeatureEngineer(window_sizes=[5, 10])
    temporal_features = temporal_engineer.extract_temporal_features(
        ts_df,
        columns=['typing_delay', 'backspace_rate', 'idle_time']
    )
    
    print(f"Time-series input shape: {ts_df.shape}")
    print(f"Temporal features shape: {temporal_features.shape}")
    print(f"\nTemporal feature columns ({len(temporal_features.columns)}):")
    for col in temporal_features.columns[:10]:
        print(f"  - {col}")
    if len(temporal_features.columns) > 10:
        print(f"  ... and {len(temporal_features.columns) - 10} more")
    
    return features, feature_names


if __name__ == "__main__":
    features, feature_names = main()
