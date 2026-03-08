"""
Mind-Shield+ Visualization Module

Provides visualization functions for cognitive fatigue monitoring
including dashboards, time-series plots, and summary charts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


class MindShieldVisualizer:
    """
    Visualization toolkit for Mind-Shield+ cognitive monitoring.
    
    Provides:
    - Time-series fatigue plots
    - Cognitive load distributions
    - Feature importance charts
    - Risk assessment dashboards
    - Behavioral drift visualizations
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size
            dpi: Figure resolution
        """
        self.figsize = figsize
        self.dpi = dpi
        
        # Color palette
        self.colors = {
            'normal': '#2ecc71',      # Green
            'moderate': '#f39c12',    # Orange
            'high': '#e74c3c',        # Red
            'critical': '#8e44ad',    # Purple
            'primary': '#3498db',     # Blue
            'secondary': '#95a5a6',   # Gray
            'background': '#ecf0f1'   # Light gray
        }
    
    def plot_time_series_fatigue(
        self,
        timestamps: np.ndarray,
        fatigue_probs: np.ndarray,
        cognitive_loads: np.ndarray,
        drift_magnitudes: Optional[np.ndarray] = None,
        title: str = "Cognitive Fatigue Over Time",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot fatigue indicators over time.
        
        Args:
            timestamps: Time points (minutes or datetime)
            fatigue_probs: Fatigue probabilities
            cognitive_loads: Cognitive load scores
            drift_magnitudes: Optional drift magnitude values
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib Figure object
        """
        fig, axes = plt.subplots(3 if drift_magnitudes is not None else 2, 
                                  1, figsize=(12, 8), sharex=True)
        
        # Fatigue probability
        ax1 = axes[0]
        ax1.fill_between(timestamps, fatigue_probs, alpha=0.3, color=self.colors['primary'])
        ax1.plot(timestamps, fatigue_probs, color=self.colors['primary'], linewidth=2)
        ax1.axhline(y=0.6, color=self.colors['high'], linestyle='--', 
                   label='Alert Threshold', alpha=0.7)
        ax1.set_ylabel('Fatigue Probability')
        ax1.set_ylim(0, 1)
        ax1.legend(loc='upper right')
        ax1.set_title('ML-Predicted Fatigue Probability')
        
        # Cognitive load
        ax2 = axes[1]
        colors = [self._get_load_color(load) for load in cognitive_loads]
        ax2.bar(timestamps, cognitive_loads, color=colors, alpha=0.7, width=0.8)
        ax2.axhline(y=60, color=self.colors['high'], linestyle='--', 
                   label='High Load Threshold', alpha=0.7)
        ax2.set_ylabel('Cognitive Load')
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper right')
        ax2.set_title('Cognitive Load Score')
        
        # Behavioral drift (if provided)
        if drift_magnitudes is not None:
            ax3 = axes[2]
            ax3.fill_between(timestamps, drift_magnitudes, alpha=0.3, color=self.colors['moderate'])
            ax3.plot(timestamps, drift_magnitudes, color=self.colors['moderate'], linewidth=2)
            ax3.axhline(y=0.4, color=self.colors['high'], linestyle='--', 
                       label='Drift Threshold', alpha=0.7)
            ax3.set_ylabel('Drift Magnitude')
            ax3.set_ylim(0, 1)
            ax3.legend(loc='upper right')
            ax3.set_title('Behavioral Drift from Baseline')
            ax3.set_xlabel('Time (minutes)')
        else:
            ax2.set_xlabel('Time (minutes)')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def _get_load_color(self, load: float) -> str:
        """Get color based on cognitive load level."""
        if load < 30:
            return self.colors['normal']
        elif load < 50:
            return self.colors['moderate']
        elif load < 70:
            return self.colors['high']
        else:
            return self.colors['critical']
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importances: List[float],
        title: str = "Feature Importance for Fatigue Prediction",
        top_n: int = 15,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot feature importance bar chart.
        
        Args:
            feature_names: Names of features
            importances: Importance scores
            title: Plot title
            top_n: Number of top features to show
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib Figure object
        """
        # Sort by importance
        sorted_indices = np.argsort(importances)[::-1][:top_n]
        sorted_names = [feature_names[i] for i in sorted_indices]
        sorted_importances = [importances[i] for i in sorted_indices]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y_pos = np.arange(len(sorted_names))
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(sorted_names)))[::-1]
        
        bars = ax.barh(y_pos, sorted_importances, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, sorted_importances)):
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_risk_distribution(
        self,
        risk_scores: np.ndarray,
        fatigue_labels: Optional[np.ndarray] = None,
        title: str = "Risk Score Distribution",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot distribution of risk scores.
        
        Args:
            risk_scores: Array of risk scores
            fatigue_labels: Optional true fatigue labels
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if fatigue_labels is not None:
            # Separate by true labels
            normal_scores = risk_scores[fatigue_labels == 0]
            fatigued_scores = risk_scores[fatigue_labels == 1]
            
            ax.hist(normal_scores, bins=30, alpha=0.6, 
                   color=self.colors['normal'], label='Normal State', density=True)
            ax.hist(fatigued_scores, bins=30, alpha=0.6, 
                   color=self.colors['high'], label='Fatigued State', density=True)
            ax.legend()
        else:
            ax.hist(risk_scores, bins=30, alpha=0.7, 
                   color=self.colors['primary'], density=True)
        
        # Add threshold lines
        ax.axvline(x=25, color=self.colors['normal'], linestyle='--', 
                  label='Low/Moderate', alpha=0.8)
        ax.axvline(x=50, color=self.colors['moderate'], linestyle='--', 
                  label='Moderate/High', alpha=0.8)
        ax.axvline(x=75, color=self.colors['high'], linestyle='--', 
                  label='High/Critical', alpha=0.8)
        
        ax.set_xlabel('Risk Score')
        ax.set_ylabel('Density')
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_dashboard(
        self,
        session_data: Dict,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comprehensive monitoring dashboard.
        
        Args:
            session_data: Dictionary containing session metrics
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib Figure object
        """
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Current Status Panel (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._draw_gauge(ax1, session_data.get('current_risk_score', 50),
                        'Current Risk Score')
        
        # Fatigue Probability (top-center)
        ax2 = fig.add_subplot(gs[0, 1])
        self._draw_gauge(ax2, session_data.get('fatigue_probability', 0.5) * 100,
                        'Fatigue Probability')
        
        # Cognitive Load (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._draw_gauge(ax3, session_data.get('cognitive_load', 50),
                        'Cognitive Load')
        
        # Time series (middle row)
        ax4 = fig.add_subplot(gs[1, :])
        if 'time_series' in session_data:
            ts = session_data['time_series']
            ax4.plot(ts.get('time', []), ts.get('fatigue_prob', []),
                    label='Fatigue Probability', color=self.colors['primary'], linewidth=2)
            ax4.plot(ts.get('time', []), np.array(ts.get('cognitive_load', [])) / 100,
                    label='Cognitive Load (normalized)', color=self.colors['moderate'], linewidth=2)
            ax4.set_xlabel('Time (minutes)')
            ax4.set_ylabel('Score')
            ax4.legend(loc='upper left')
            ax4.set_title('Session Progression', fontweight='bold')
            ax4.set_ylim(0, 1.1)
        
        # Risk Level Distribution (bottom-left)
        ax5 = fig.add_subplot(gs[2, 0])
        if 'risk_distribution' in session_data:
            dist = session_data['risk_distribution']
            labels = list(dist.keys())
            values = list(dist.values())
            colors = [self.colors['normal'], self.colors['moderate'], 
                     self.colors['high'], self.colors['critical']][:len(labels)]
            ax5.pie(values, labels=labels, colors=colors, autopct='%1.1f%%')
            ax5.set_title('Risk Level Distribution', fontweight='bold')
        
        # Recommendations (bottom-center/right)
        ax6 = fig.add_subplot(gs[2, 1:])
        ax6.axis('off')
        recommendations = session_data.get('recommendations', ['No recommendations'])
        rec_text = "RECOMMENDATIONS:\n\n" + "\n".join([f"• {r}" for r in recommendations])
        ax6.text(0.05, 0.95, rec_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=self.colors['background'], alpha=0.8))
        
        plt.suptitle('Mind-Shield+ Cognitive Monitoring Dashboard', 
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def _draw_gauge(self, ax: plt.Axes, value: float, title: str) -> None:
        """Draw a gauge-style indicator."""
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Draw background arc
        theta = np.linspace(0, np.pi, 100)
        x = np.cos(theta)
        y = np.sin(theta)
        ax.fill_between(x, 0, y, alpha=0.2, color=self.colors['secondary'])
        
        # Color zones
        for i, (start, end, color) in enumerate([
            (0, 0.25, self.colors['normal']),
            (0.25, 0.5, self.colors['moderate']),
            (0.5, 0.75, self.colors['high']),
            (0.75, 1.0, self.colors['critical'])
        ]):
            theta_start = np.pi * (1 - end)
            theta_end = np.pi * (1 - start)
            theta_zone = np.linspace(theta_start, theta_end, 20)
            x_zone = np.cos(theta_zone)
            y_zone = np.sin(theta_zone)
            ax.fill_between(x_zone, 0, y_zone, alpha=0.3, color=color)
        
        # Draw needle
        angle = np.pi * (1 - value / 100)
        ax.arrow(0, 0, 0.7 * np.cos(angle), 0.7 * np.sin(angle),
                head_width=0.1, head_length=0.05, fc='black', ec='black')
        ax.plot(0, 0, 'ko', markersize=8)
        
        # Add value and title
        ax.text(0, -0.3, f'{value:.0f}', ha='center', fontsize=18, fontweight='bold')
        ax.text(0, 1.3, title, ha='center', fontsize=11, fontweight='bold')
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str] = ['Normal', 'Fatigued'],
        title: str = "Fatigue Prediction Confusion Matrix",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot confusion matrix heatmap.
        
        Args:
            cm: Confusion matrix array
            class_names: Names of classes
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names, yticklabels=class_names,
               ylabel='True label',
               xlabel='Predicted label')
        
        # Rotate tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=14)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_behavioral_comparison(
        self,
        normal_data: pd.DataFrame,
        fatigued_data: pd.DataFrame,
        features: List[str],
        title: str = "Behavioral Patterns: Normal vs Fatigued",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare behavioral patterns between normal and fatigued states.
        
        Args:
            normal_data: DataFrame with normal state samples
            fatigued_data: DataFrame with fatigued state samples
            features: Features to compare
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib Figure object
        """
        n_features = len(features)
        fig, axes = plt.subplots(2, (n_features + 1) // 2, figsize=(14, 8))
        axes = axes.flatten()
        
        for i, feature in enumerate(features):
            ax = axes[i]
            
            # Box plots
            data = [normal_data[feature], fatigued_data[feature]]
            bp = ax.boxplot(data, labels=['Normal', 'Fatigued'], patch_artist=True)
            
            bp['boxes'][0].set_facecolor(self.colors['normal'])
            bp['boxes'][1].set_facecolor(self.colors['high'])
            
            ax.set_title(feature, fontsize=10)
            ax.set_ylabel('Value')
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig


def main():
    """Demonstrate the visualization module."""
    print("Mind-Shield+ Visualization Module")
    print("=" * 50)
    
    # Import required modules
    from data_simulation import BehavioralDataSimulator
    from feature_engineering import FeatureEngineer
    from fatigue_model import FatiguePredictionModel
    
    # Generate data
    print("\nGenerating data for visualization...")
    simulator = BehavioralDataSimulator(random_seed=42)
    df = simulator.generate_dataset(n_normal=300, n_fatigued=300)
    ts_df = simulator.generate_time_series(duration_minutes=90, fatigue_onset_minute=45)
    
    # Extract features and train model
    engineer = FeatureEngineer()
    X, feature_names = engineer.fit_transform(df)
    y = df['fatigue_label'].values
    
    model = FatiguePredictionModel(model_type='random_forest')
    metrics = model.train(X, y, feature_names)
    
    # Initialize visualizer
    viz = MindShieldVisualizer()
    
    # Create output directory
    import os
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Time series fatigue plot
    print("\n1. Creating time-series fatigue plot...")
    fig1 = viz.plot_time_series_fatigue(
        timestamps=ts_df['minute'].values,
        fatigue_probs=ts_df['fatigue_intensity'].values,
        cognitive_loads=np.clip(ts_df['fatigue_intensity'].values * 80 + 20, 0, 100),
        drift_magnitudes=ts_df['fatigue_intensity'].values * 0.7,
        save_path=os.path.join(output_dir, 'time_series_fatigue.png')
    )
    print("   Saved: output/time_series_fatigue.png")
    
    # 2. Feature importance plot
    print("\n2. Creating feature importance plot...")
    importance_df = model.get_feature_importance()
    fig2 = viz.plot_feature_importance(
        feature_names=importance_df['feature'].tolist(),
        importances=importance_df['importance'].tolist(),
        save_path=os.path.join(output_dir, 'feature_importance.png')
    )
    print("   Saved: output/feature_importance.png")
    
    # 3. Risk distribution plot
    print("\n3. Creating risk distribution plot...")
    risk_scores = model.predict_proba(X) * 100
    fig3 = viz.plot_risk_distribution(
        risk_scores=risk_scores,
        fatigue_labels=y,
        save_path=os.path.join(output_dir, 'risk_distribution.png')
    )
    print("   Saved: output/risk_distribution.png")
    
    # 4. Confusion matrix
    print("\n4. Creating confusion matrix plot...")
    cm = np.array(metrics['confusion_matrix'])
    fig4 = viz.plot_confusion_matrix(
        cm=cm,
        save_path=os.path.join(output_dir, 'confusion_matrix.png')
    )
    print("   Saved: output/confusion_matrix.png")
    
    # 5. Behavioral comparison
    print("\n5. Creating behavioral comparison plot...")
    features_to_compare = ['typing_delay', 'backspace_rate', 'tab_switch_rate', 
                          'idle_time', 'mouse_velocity', 'interaction_burst']
    fig5 = viz.plot_behavioral_comparison(
        normal_data=df[df['state'] == 'normal'],
        fatigued_data=df[df['state'] == 'fatigued'],
        features=features_to_compare,
        save_path=os.path.join(output_dir, 'behavioral_comparison.png')
    )
    print("   Saved: output/behavioral_comparison.png")
    
    # 6. Dashboard
    print("\n6. Creating monitoring dashboard...")
    session_data = {
        'current_risk_score': 65,
        'fatigue_probability': 0.72,
        'cognitive_load': 58,
        'time_series': {
            'time': ts_df['minute'].values[:60],
            'fatigue_prob': ts_df['fatigue_intensity'].values[:60],
            'cognitive_load': (ts_df['fatigue_intensity'].values[:60] * 80 + 20)
        },
        'risk_distribution': {'Low': 35, 'Moderate': 30, 'High': 25, 'Critical': 10},
        'recommendations': [
            'High fatigue risk detected - take a 10-minute break soon',
            'Typing patterns suggest mental fatigue - slow down',
            'Consider breaking tasks into smaller steps'
        ]
    }
    fig6 = viz.plot_dashboard(
        session_data=session_data,
        save_path=os.path.join(output_dir, 'dashboard.png')
    )
    print("   Saved: output/dashboard.png")
    
    print("\n" + "=" * 50)
    print("All visualizations created successfully!")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    
    # Close all figures to free memory
    plt.close('all')
    
    return viz


if __name__ == "__main__":
    viz = main()
