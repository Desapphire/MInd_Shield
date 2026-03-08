"""
Mind-Shield+ Data Simulation Module

Generates synthetic behavioral interaction data representing
normal and fatigued cognitive states.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class BehavioralDataSimulator:
    """
    Simulates behavioral interaction data for cognitive fatigue modeling.
    
    Generates synthetic data representing:
    - Typing patterns (delay, dwell time)
    - Error rates (backspace frequency)
    - Attention patterns (tab switching)
    - Navigation behavior (mouse movement)
    - Engagement indicators (idle time)
    """
    
    def __init__(self, random_seed: Optional[int] = 42):
        """
        Initialize the simulator.
        
        Args:
            random_seed: Seed for reproducibility
        """
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Define statistical parameters for normal state
        self.normal_params = {
            'typing_delay_mean': 120,      # ms between keystrokes
            'typing_delay_std': 30,
            'dwell_time_mean': 80,         # ms key held down
            'dwell_time_std': 20,
            'backspace_rate_mean': 0.05,   # proportion of keystrokes
            'backspace_rate_std': 0.02,
            'tab_switch_rate_mean': 2.0,   # switches per minute
            'tab_switch_rate_std': 1.0,
            'mouse_velocity_mean': 500,    # pixels per second
            'mouse_velocity_std': 150,
            'idle_time_mean': 3.0,         # seconds
            'idle_time_std': 1.5,
            'scroll_velocity_mean': 300,   # pixels per second
            'scroll_velocity_std': 100,
            'interaction_burst_mean': 15,  # interactions per burst
            'interaction_burst_std': 5,
        }
        
        # Define statistical parameters for fatigued state
        self.fatigued_params = {
            'typing_delay_mean': 200,      # slower typing
            'typing_delay_std': 60,
            'dwell_time_mean': 120,        # longer key holds
            'dwell_time_std': 40,
            'backspace_rate_mean': 0.15,   # more errors
            'backspace_rate_std': 0.05,
            'tab_switch_rate_mean': 5.0,   # more context switching
            'tab_switch_rate_std': 2.0,
            'mouse_velocity_mean': 300,    # slower navigation
            'mouse_velocity_std': 100,
            'idle_time_mean': 8.0,         # longer pauses
            'idle_time_std': 4.0,
            'scroll_velocity_mean': 150,   # slower scrolling
            'scroll_velocity_std': 80,
            'interaction_burst_mean': 8,   # fewer interactions per burst
            'interaction_burst_std': 4,
        }
    
    def _generate_samples(self, params: dict, n_samples: int) -> dict:
        """
        Generate samples from given parameter distributions.
        
        Args:
            params: Dictionary of mean/std for each variable
            n_samples: Number of samples to generate
            
        Returns:
            Dictionary of generated sample arrays
        """
        samples = {}
        
        # Typing delay (gamma distribution for positive skew)
        shape = (params['typing_delay_mean'] / params['typing_delay_std']) ** 2
        scale = params['typing_delay_std'] ** 2 / params['typing_delay_mean']
        samples['typing_delay'] = np.random.gamma(shape, scale, n_samples)
        
        # Dwell time (gamma distribution)
        shape = (params['dwell_time_mean'] / params['dwell_time_std']) ** 2
        scale = params['dwell_time_std'] ** 2 / params['dwell_time_mean']
        samples['dwell_time'] = np.random.gamma(shape, scale, n_samples)
        
        # Backspace rate (beta distribution, bounded 0-1)
        # Convert mean/std to alpha/beta parameters
        mean = params['backspace_rate_mean']
        var = params['backspace_rate_std'] ** 2
        alpha = mean * (mean * (1 - mean) / var - 1)
        beta = (1 - mean) * (mean * (1 - mean) / var - 1)
        samples['backspace_rate'] = np.random.beta(max(alpha, 0.1), max(beta, 0.1), n_samples)
        
        # Tab switch rate (gamma distribution)
        shape = (params['tab_switch_rate_mean'] / params['tab_switch_rate_std']) ** 2
        scale = params['tab_switch_rate_std'] ** 2 / params['tab_switch_rate_mean']
        samples['tab_switch_rate'] = np.random.gamma(shape, scale, n_samples)
        
        # Mouse velocity (normal distribution, clipped to positive)
        samples['mouse_velocity'] = np.clip(
            np.random.normal(params['mouse_velocity_mean'], params['mouse_velocity_std'], n_samples),
            50, None
        )
        
        # Idle time (exponential-like via gamma)
        shape = (params['idle_time_mean'] / params['idle_time_std']) ** 2
        scale = params['idle_time_std'] ** 2 / params['idle_time_mean']
        samples['idle_time'] = np.random.gamma(shape, scale, n_samples)
        
        # Scroll velocity
        samples['scroll_velocity'] = np.clip(
            np.random.normal(params['scroll_velocity_mean'], params['scroll_velocity_std'], n_samples),
            20, None
        )
        
        # Interaction burst density
        shape = (params['interaction_burst_mean'] / params['interaction_burst_std']) ** 2
        scale = params['interaction_burst_std'] ** 2 / params['interaction_burst_mean']
        samples['interaction_burst'] = np.random.gamma(shape, scale, n_samples)
        
        return samples
    
    def generate_dataset(
        self,
        n_normal: int = 500,
        n_fatigued: int = 500,
        add_noise: bool = True,
        noise_level: float = 0.05
    ) -> pd.DataFrame:
        """
        Generate a complete synthetic behavioral dataset.
        
        Args:
            n_normal: Number of normal state samples
            n_fatigued: Number of fatigued state samples
            add_noise: Whether to add random noise for realism
            noise_level: Proportion of noise to add
            
        Returns:
            DataFrame with behavioral features and fatigue labels
        """
        # Generate normal samples
        normal_samples = self._generate_samples(self.normal_params, n_normal)
        normal_df = pd.DataFrame(normal_samples)
        normal_df['fatigue_label'] = 0
        normal_df['state'] = 'normal'
        
        # Generate fatigued samples
        fatigued_samples = self._generate_samples(self.fatigued_params, n_fatigued)
        fatigued_df = pd.DataFrame(fatigued_samples)
        fatigued_df['fatigue_label'] = 1
        fatigued_df['state'] = 'fatigued'
        
        # Combine datasets
        df = pd.concat([normal_df, fatigued_df], ignore_index=True)
        
        # Add noise for realism
        if add_noise:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [c for c in numeric_cols if c != 'fatigue_label']
            for col in numeric_cols:
                noise = np.random.normal(0, df[col].std() * noise_level, len(df))
                df[col] = df[col] + noise
                # Ensure non-negative values
                df[col] = df[col].clip(lower=0)
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        # Add session ID and timestamp simulation
        df['session_id'] = np.arange(len(df))
        df['timestamp'] = pd.date_range(
            start='2024-01-01',
            periods=len(df),
            freq='1min'
        )
        
        return df
    
    def generate_time_series(
        self,
        duration_minutes: int = 120,
        fatigue_onset_minute: int = 60,
        sampling_interval_seconds: int = 30
    ) -> pd.DataFrame:
        """
        Generate time-series behavioral data with gradual fatigue onset.
        
        This simulates a realistic work session where fatigue
        develops gradually over time.
        
        Args:
            duration_minutes: Total session duration
            fatigue_onset_minute: When fatigue starts developing
            sampling_interval_seconds: Time between samples
            
        Returns:
            DataFrame with time-series behavioral data
        """
        n_samples = (duration_minutes * 60) // sampling_interval_seconds
        timestamps = pd.date_range(
            start='2024-01-01 09:00:00',
            periods=n_samples,
            freq=f'{sampling_interval_seconds}s'
        )
        
        # Calculate fatigue progression (sigmoid curve)
        minutes = np.arange(n_samples) * sampling_interval_seconds / 60
        fatigue_intensity = 1 / (1 + np.exp(-0.1 * (minutes - fatigue_onset_minute)))
        
        # Generate interpolated parameters
        data = []
        for i, (ts, fi) in enumerate(zip(timestamps, fatigue_intensity)):
            # Interpolate between normal and fatigued parameters
            sample = {}
            for key in ['typing_delay', 'dwell_time', 'backspace_rate', 
                       'tab_switch_rate', 'mouse_velocity', 'idle_time',
                       'scroll_velocity', 'interaction_burst']:
                normal_val = self.normal_params[f'{key}_mean']
                fatigued_val = self.fatigued_params[f'{key}_mean']
                normal_std = self.normal_params[f'{key}_std']
                
                # Interpolate mean based on fatigue intensity
                mean_val = normal_val + fi * (fatigued_val - normal_val)
                # Add natural variation
                sample[key] = max(0, np.random.normal(mean_val, normal_std * (1 + fi)))
            
            sample['timestamp'] = ts
            sample['minute'] = minutes[i]
            sample['fatigue_intensity'] = fi
            sample['fatigue_label'] = 1 if fi > 0.5 else 0
            data.append(sample)
        
        return pd.DataFrame(data)


def main():
    """Demonstrate the data simulation module."""
    print("Mind-Shield+ Data Simulation Module")
    print("=" * 50)
    
    # Create simulator
    simulator = BehavioralDataSimulator(random_seed=42)
    
    # Generate static dataset
    print("\nGenerating static dataset...")
    df = simulator.generate_dataset(n_normal=500, n_fatigued=500)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nClass distribution:")
    print(df['state'].value_counts())
    print(f"\nSample statistics:")
    print(df.describe().round(2))
    
    # Generate time-series data
    print("\n" + "=" * 50)
    print("Generating time-series session data...")
    ts_df = simulator.generate_time_series(duration_minutes=120)
    print(f"Time-series shape: {ts_df.shape}")
    print(f"\nFatigue progression (first 5 and last 5 rows):")
    print(ts_df[['minute', 'fatigue_intensity', 'typing_delay', 'backspace_rate']].head())
    print("...")
    print(ts_df[['minute', 'fatigue_intensity', 'typing_delay', 'backspace_rate']].tail())
    
    return df, ts_df


if __name__ == "__main__":
    df, ts_df = main()
