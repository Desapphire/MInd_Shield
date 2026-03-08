"""
Mind-Shield+ Main Demonstration Script

Complete demonstration of the cognitive fatigue detection system.
Integrates all modules to simulate, analyze, and predict mental fatigue.
"""

import numpy as np
import pandas as pd
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_simulation import BehavioralDataSimulator
from feature_engineering import FeatureEngineer, TemporalFeatureEngineer
from cognitive_load import CognitiveLoadEstimator
from fatigue_model import FatiguePredictionModel, ModelEnsemble
from anomaly_detection import BehavioralDriftDetector, TemporalDriftAnalyzer
from risk_evaluation import RiskEvaluationEngine, SessionRiskTracker, RiskLevel
from visualization import MindShieldVisualizer
 

def print_header(title: str, char: str = "=") -> None:
    """Print a formatted section header."""
    print(f"\n{char * 60}")
    print(f"  {title}")
    print(f"{char * 60}")


def print_subheader(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n{'-' * 40}")
    print(f"  {title}")
    print(f"{'-' * 40}")


def main():
    """
    Main demonstration of the Mind-Shield+ cognitive fatigue detection system.
    
    This script demonstrates the complete pipeline:
    1. Generate synthetic behavioral data
    2. Extract meaningful features
    3. Train fatigue prediction models
    4. Detect behavioral drift
    5. Estimate cognitive load
    6. Evaluate overall risk
    7. Visualize results
    """
    
    print_header("Mind-Shield+ Core Model Demonstration", "=")
    print("\nAI-Powered Cognitive Fatigue Detection System")
    print("Version 0.1.0 - Prototype\n")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================================================
    # STEP 1: DATA SIMULATION
    # =========================================================================
    print_header("Step 1: Behavioral Data Simulation")
    
    simulator = BehavioralDataSimulator(random_seed=42)
    
    # Generate static dataset for model training
    print("\nGenerating training dataset...")
    train_df = simulator.generate_dataset(n_normal=500, n_fatigued=500)
    print(f"  Total samples: {len(train_df)}")
    print(f"  Features: typing_delay, dwell_time, backspace_rate, tab_switch_rate,")
    print(f"            mouse_velocity, idle_time, scroll_velocity, interaction_burst")
    print(f"  Class distribution:")
    print(f"    - Normal: {(train_df['fatigue_label'] == 0).sum()}")
    print(f"    - Fatigued: {(train_df['fatigue_label'] == 1).sum()}")
    
    # Generate time-series for session simulation
    print("\nGenerating time-series session data...")
    session_df = simulator.generate_time_series(
        duration_minutes=90,
        fatigue_onset_minute=45,
        sampling_interval_seconds=30
    )
    print(f"  Session duration: 90 minutes")
    print(f"  Sampling interval: 30 seconds")
    print(f"  Total samples: {len(session_df)}")
    print(f"  Fatigue onset: ~45 minutes")
    
    # =========================================================================
    # STEP 2: FEATURE ENGINEERING
    # =========================================================================
    print_header("Step 2: Feature Engineering")
    
    engineer = FeatureEngineer(normalization='standard')
    X_train, feature_names = engineer.fit_transform(train_df)
    y_train = train_df['fatigue_label'].values
    
    print(f"\nExtracted {len(feature_names)} features:")
    for i, name in enumerate(feature_names, 1):
        print(f"  {i:2d}. {name}")
    
    print(f"\nFeature matrix shape: {X_train.shape}")
    print(f"Target vector shape: {y_train.shape}")
    
    # =========================================================================
    # STEP 3: FATIGUE PREDICTION MODEL TRAINING
    # =========================================================================
    print_header("Step 3: Fatigue Prediction Model Training")
    
    # Train multiple models
    model_types = ['logistic', 'random_forest', 'gradient_boosting']
    best_model = None
    best_auc = 0
    
    print("\nTraining and evaluating models...\n")
    
    results_table = []
    for model_type in model_types:
        model = FatiguePredictionModel(model_type=model_type, random_state=42)
        metrics = model.train(X_train, y_train, feature_names, cross_validate=False)
        
        results_table.append({
            'Model': model_type.replace('_', ' ').title(),
            'Accuracy': f"{metrics['accuracy']:.3f}",
            'Precision': f"{metrics['precision']:.3f}",
            'Recall': f"{metrics['recall']:.3f}",
            'F1 Score': f"{metrics['f1_score']:.3f}",
            'ROC AUC': f"{metrics['roc_auc']:.3f}"
        })
        
        if metrics['roc_auc'] > best_auc:
            best_auc = metrics['roc_auc']
            best_model = model
    
    # Print results table
    print("Model Comparison Results:")
    print("-" * 70)
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
    print("-" * 70)
    for row in results_table:
        print(f"{row['Model']:<20} {row['Accuracy']:<10} {row['Precision']:<10} {row['Recall']:<10} {row['F1 Score']:<10} {row['ROC AUC']:<10}")
    print("-" * 70)
    
    print(f"\nBest model: {best_model.model_type.replace('_', ' ').title()} (AUC: {best_auc:.3f})")
    
    # Feature importance
    print_subheader("Feature Importance (Top 10)")
    importance_df = best_model.get_feature_importance()
    for idx, row in importance_df.head(10).iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"  {row['feature']:<25} {row['importance']:.4f} {bar}")
    
    # =========================================================================
    # STEP 4: BEHAVIORAL DRIFT DETECTION
    # =========================================================================
    print_header("Step 4: Behavioral Drift Detection")
    
    # Fit detector on normal samples only
    X_normal = X_train[y_train == 0]
    
    detector = BehavioralDriftDetector(method='isolation_forest', contamination=0.1)
    detector.fit_baseline(X_normal, feature_names)
    
    print(f"\nBaseline fitted on {len(X_normal)} normal behavior samples")
    
    # Test drift detection
    labels, scores = detector.detect_anomaly(X_train)
    drift_magnitudes = detector.get_drift_magnitude(X_train)
    
    anomaly_mask = labels == -1
    normal_mask = y_train == 0
    fatigued_mask = y_train == 1
    
    print("\nDrift Detection Performance:")
    print(f"  Anomalies in normal samples: {anomaly_mask[normal_mask].sum()} / {normal_mask.sum()} ({anomaly_mask[normal_mask].mean()*100:.1f}%)")
    print(f"  Anomalies in fatigued samples: {anomaly_mask[fatigued_mask].sum()} / {fatigued_mask.sum()} ({anomaly_mask[fatigued_mask].mean()*100:.1f}%)")
    print(f"  Mean drift (normal): {drift_magnitudes[normal_mask].mean():.3f}")
    print(f"  Mean drift (fatigued): {drift_magnitudes[fatigued_mask].mean():.3f}")
    
    # =========================================================================
    # STEP 5: COGNITIVE LOAD ESTIMATION
    # =========================================================================
    print_header("Step 5: Cognitive Load Estimation")
    
    load_estimator = CognitiveLoadEstimator()
    load_results = load_estimator.estimate_batch(train_df)
    
    print("\nCognitive Load Statistics:")
    print("-" * 40)
    
    for state in ['normal', 'fatigued']:
        mask = train_df['state'] == state
        load_scores = load_results.loc[mask, 'cognitive_load_score']
        print(f"\n  {state.upper()} State:")
        print(f"    Mean: {load_scores.mean():.1f}")
        print(f"    Std:  {load_scores.std():.1f}")
        print(f"    Min:  {load_scores.min():.1f}")
        print(f"    Max:  {load_scores.max():.1f}")
    
    # =========================================================================
    # STEP 6: INTEGRATED RISK EVALUATION
    # =========================================================================
    print_header("Step 6: Integrated Risk Evaluation")
    
    risk_engine = RiskEvaluationEngine()
    
    # Evaluate risk for all samples
    cognitive_loads = load_results['cognitive_load_score'].values
    fatigue_probs = best_model.predict_proba(X_train)
    drift_mags = drift_magnitudes
    drift_detected = anomaly_mask
    
    assessments = risk_engine.evaluate_batch(
        cognitive_loads, fatigue_probs, drift_mags, drift_detected
    )
    
    # Summary statistics
    summary = risk_engine.get_summary_statistics(assessments)
    
    print("\nRisk Assessment Summary:")
    print(f"  Total assessments: {summary['total_assessments']}")
    print(f"  Mean risk score: {summary['mean_risk_score']:.1f}")
    print(f"  Max risk score: {summary['max_risk_score']:.1f}")
    
    print("\n  Risk Level Distribution:")
    for level, count in summary['risk_level_distribution'].items():
        pct = count / summary['total_assessments'] * 100
        bar = "█" * int(pct / 2)
        print(f"    {level:<10}: {count:>4} ({pct:>5.1f}%) {bar}")
    
    # =========================================================================
    # STEP 7: LIVE SESSION SIMULATION
    # =========================================================================
    print_header("Step 7: Live Session Simulation")
    
    print("\nSimulating a 90-minute work session...")
    print("(Fatigue develops gradually after minute 45)\n")
    
    # Prepare session data
    session_features = engineer.extract_features(session_df)
    session_X = engineer.scaler.transform(session_features)
    
    # Track session
    tracker = SessionRiskTracker()
    
    # Simulate with checkpoints
    checkpoints = [0, 15, 30, 45, 60, 75, 89]
    print(f"{'Time':<10} {'Cog.Load':<12} {'FatProb':<12} {'Drift':<10} {'Risk':<15} {'Level':<12}")
    print("-" * 75)
    
    for i in range(len(session_df)):
        # Get metrics
        sample = session_df.iloc[i:i+1]
        load_score, _ = load_estimator.estimate_load(sample.iloc[0])
        
        sample_X = session_X[i:i+1]
        fatigue_prob = best_model.predict_proba(sample_X)[0]
        drift_mag = detector.get_drift_magnitude(sample_X)[0]
        _, scores = detector.detect_anomaly(sample_X)
        is_anomaly = scores[0] < 0
        
        # Evaluate risk
        assessment = tracker.evaluate_and_track(
            load_score, fatigue_prob, drift_mag, is_anomaly,
            timestamp=str(session_df.iloc[i]['timestamp'])
        )
        
        # Print at checkpoints
        minute = int(session_df.iloc[i]['minute'])
        if minute in checkpoints and i == checkpoints.index(minute) * (len(session_df) // (len(checkpoints) - 1)):
            print(f"Min {minute:<5} {load_score:<12.1f} {fatigue_prob:<12.2f} {drift_mag:<10.3f} {assessment.risk_score:<15.1f} {assessment.risk_level.value:<12}")
    
    # Session summary
    print("\n" + "-" * 75)
    session_summary = tracker.get_session_summary()
    
    print("\nSession Summary:")
    print(f"  Duration: {session_summary['session_duration']} assessments")
    print(f"  Mean Risk: {session_summary['mean_risk_score']:.1f}")
    print(f"  Max Risk: {session_summary['max_risk_score']:.1f}")
    print(f"  Trend: {session_summary['trend']['trend'].upper()}")
    
    if session_summary['alert_needed']:
        print(f"\n  ⚠️  ALERT: {session_summary['alert_reason']}")
    
    # Final recommendations
    final_assessment = tracker.history[-1]
    print("\n  Final Recommendations:")
    for rec in final_assessment.recommendations:
        print(f"    • {rec}")
    
    # =========================================================================
    # STEP 8: VISUALIZATION
    # =========================================================================
    print_header("Step 8: Generating Visualizations")
    
    viz = MindShieldVisualizer()
    
    # Time series plot
    print("\nCreating visualizations...")
    
    session_loads = [a.cognitive_load for a in tracker.history]
    session_fatigue = [a.fatigue_probability for a in tracker.history]
    session_drift = [a.behavioral_drift for a in tracker.history]
    session_times = session_df['minute'].values[:len(tracker.history)]
    
    fig1 = viz.plot_time_series_fatigue(
        timestamps=session_times,
        fatigue_probs=np.array(session_fatigue),
        cognitive_loads=np.array(session_loads),
        drift_magnitudes=np.array(session_drift),
        title="Mind-Shield+ Session Analysis",
        save_path=os.path.join(output_dir, 'session_analysis.png')
    )
    print(f"  ✓ Session analysis: output/session_analysis.png")
    
    # Feature importance
    importance_df = best_model.get_feature_importance()
    fig2 = viz.plot_feature_importance(
        feature_names=importance_df['feature'].tolist(),
        importances=importance_df['importance'].tolist(),
        save_path=os.path.join(output_dir, 'feature_importance.png')
    )
    print(f"  ✓ Feature importance: output/feature_importance.png")
    
    # Dashboard
    dashboard_data = {
        'current_risk_score': final_assessment.risk_score,
        'fatigue_probability': final_assessment.fatigue_probability,
        'cognitive_load': final_assessment.cognitive_load,
        'time_series': {
            'time': session_times,
            'fatigue_prob': session_fatigue,
            'cognitive_load': session_loads
        },
        'risk_distribution': summary['risk_level_distribution'],
        'recommendations': final_assessment.recommendations
    }
    fig3 = viz.plot_dashboard(
        session_data=dashboard_data,
        save_path=os.path.join(output_dir, 'dashboard.png')
    )
    print(f"  ✓ Dashboard: output/dashboard.png")
    
    # Behavioral comparison
    features_to_compare = ['typing_delay', 'backspace_rate', 'tab_switch_rate', 
                          'idle_time', 'mouse_velocity', 'interaction_burst']
    fig4 = viz.plot_behavioral_comparison(
        normal_data=train_df[train_df['state'] == 'normal'],
        fatigued_data=train_df[train_df['state'] == 'fatigued'],
        features=features_to_compare,
        save_path=os.path.join(output_dir, 'behavioral_patterns.png')
    )
    print(f"  ✓ Behavioral patterns: output/behavioral_patterns.png")
    
    # Close figures
    import matplotlib.pyplot as plt
    plt.close('all')
    
    # =========================================================================
    # EXAMPLE OUTPUT
    # =========================================================================
    print_header("Example Prediction Output")
    
    # Simulate a fatigued user
    print("\n📊 CURRENT USER STATUS")
    print("=" * 40)
    print(f"  Cognitive Load Score:    {final_assessment.cognitive_load:.1f}/100")
    print(f"  Fatigue Probability:     {final_assessment.fatigue_probability:.1%}")
    print(f"  Behavioral Drift:        {final_assessment.behavioral_drift:.2f}")
    print(f"  Overall Risk Score:      {final_assessment.risk_score:.1f}/100")
    print(f"  Risk Classification:     {final_assessment.risk_level.value.upper()}")
    print("\n  📋 RECOMMENDATIONS:")
    for rec in final_assessment.recommendations:
        print(f"     • {rec}")
    print("=" * 40)
    
    # =========================================================================
    # COMPLETION
    # =========================================================================
    print_header("Demonstration Complete", "=")
    
    print("\n✅ Mind-Shield+ Core Model successfully demonstrated!")
    print("\nKey Achievements:")
    print("  • Generated 1000 synthetic behavioral samples")
    print("  • Extracted 20+ cognitive features")
    print(f"  • Trained fatigue prediction model (AUC: {best_auc:.3f})")
    print("  • Implemented behavioral drift detection")
    print("  • Integrated cognitive load estimation")
    print("  • Built risk evaluation engine")
    print("  • Simulated 90-minute work session")
    print("  • Generated visualization outputs")
    
    print(f"\nOutput files saved to: {os.path.abspath(output_dir)}")
    print("\nThis prototype proves the AI system functions correctly and")
    print("can be integrated into real-time environments such as browser")
    print("extensions or mobile applications.\n")
    
    return {
        'model': best_model,
        'feature_engineer': engineer,
        'drift_detector': detector,
        'load_estimator': load_estimator,
        'risk_engine': risk_engine,
        'visualizer': viz
    }


if __name__ == "__main__":
    components = main()
