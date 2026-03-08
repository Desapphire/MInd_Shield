"""
Mind-Shield+ Fatigue Prediction Model

Machine learning models for predicting cognitive fatigue probability
based on behavioral features.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class FatiguePredictionModel:
    """
    Machine learning model for predicting cognitive fatigue.
    
    Supports multiple algorithms:
    - Logistic Regression: Simple, interpretable baseline
    - Random Forest: Handles nonlinear relationships
    - Gradient Boosting: High accuracy ensemble method
    """
    
    def __init__(self, model_type: str = 'random_forest', random_state: int = 42):
        """
        Initialize the fatigue prediction model.
        
        Args:
            model_type: 'logistic', 'random_forest', or 'gradient_boosting'
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_names: list = []
        self.is_trained = False
        self.training_metrics: Dict[str, float] = {}
        
        # Initialize the selected model
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the ML model based on type."""
        if self.model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list] = None,
        test_size: float = 0.2,
        cross_validate: bool = True
    ) -> Dict[str, Any]:
        """
        Train the fatigue prediction model.
        
        Args:
            X: Feature matrix
            y: Fatigue labels (0 = normal, 1 = fatigued)
            feature_names: Names of features for interpretation
            test_size: Proportion of data for testing
            cross_validate: Whether to perform cross-validation
            
        Returns:
            Dictionary with training metrics
        """
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'test_samples': len(y_test),
            'train_samples': len(y_train)
        }
        
        # Cross-validation
        if cross_validate:
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='roc_auc')
            metrics['cv_roc_auc_mean'] = cv_scores.mean()
            metrics['cv_roc_auc_std'] = cv_scores.std()
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            metrics['feature_importances'] = dict(zip(self.feature_names, importances))
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
            metrics['feature_importances'] = dict(zip(self.feature_names, importances))
        
        self.training_metrics = metrics
        return metrics
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict fatigue probability for new samples.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of fatigue probabilities (0-1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict fatigue class for new samples.
        
        Args:
            X: Feature matrix
            threshold: Probability threshold for positive class
            
        Returns:
            Array of predicted labels (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance rankings.
        
        Returns:
            DataFrame with feature importance scores
        """
        if 'feature_importances' not in self.training_metrics:
            raise ValueError("Feature importances not available")
        
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in self.training_metrics['feature_importances'].items()
        ])
        return importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    def explain_prediction(self, X_single: np.ndarray) -> Dict[str, Any]:
        """
        Explain a single prediction.
        
        Args:
            X_single: Single sample feature vector
            
        Returns:
            Dictionary with prediction explanation
        """
        if X_single.ndim == 1:
            X_single = X_single.reshape(1, -1)
        
        prob = self.predict_proba(X_single)[0]
        prediction = self.predict(X_single)[0]
        
        explanation = {
            'fatigue_probability': prob,
            'prediction': 'Fatigued' if prediction == 1 else 'Normal',
            'confidence': prob if prediction == 1 else 1 - prob
        }
        
        # Add feature contributions if available
        if 'feature_importances' in self.training_metrics:
            importance = self.training_metrics['feature_importances']
            feature_values = dict(zip(self.feature_names, X_single[0]))
            
            # Top contributing features
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            explanation['top_factors'] = [
                {'feature': f, 'importance': imp, 'value': feature_values.get(f, None)}
                for f, imp in sorted_features[:5]
            ]
        
        return explanation


class ModelEnsemble:
    """
    Ensemble of multiple fatigue prediction models.
    
    Combines predictions from multiple algorithms for
    improved robustness and accuracy.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize the model ensemble."""
        self.random_state = random_state
        self.models: Dict[str, FatiguePredictionModel] = {}
        self.weights: Dict[str, float] = {}
        self.is_trained = False
    
    def add_model(self, name: str, model_type: str, weight: float = 1.0) -> None:
        """Add a model to the ensemble."""
        self.models[name] = FatiguePredictionModel(model_type, self.random_state)
        self.weights[name] = weight
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list = None) -> Dict[str, Any]:
        """Train all models in the ensemble."""
        all_metrics = {}
        
        for name, model in self.models.items():
            metrics = model.train(X, y, feature_names, cross_validate=True)
            all_metrics[name] = metrics
            # Update weight based on performance
            self.weights[name] = metrics['roc_auc']
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        self.is_trained = True
        return all_metrics
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Weighted ensemble probability prediction."""
        probabilities = np.zeros(len(X))
        
        for name, model in self.models.items():
            probabilities += self.weights[name] * model.predict_proba(X)
        
        return probabilities
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Ensemble class prediction."""
        return (self.predict_proba(X) >= threshold).astype(int)


def main():
    """Demonstrate the fatigue prediction module."""
    print("Mind-Shield+ Fatigue Prediction Model")
    print("=" * 50)
    
    # Import required modules
    from data_simulation import BehavioralDataSimulator
    from feature_engineering import FeatureEngineer
    
    # Generate data
    print("\nGenerating synthetic behavioral data...")
    simulator = BehavioralDataSimulator(random_seed=42)
    df = simulator.generate_dataset(n_normal=500, n_fatigued=500)
    
    # Extract features
    print("Extracting behavioral features...")
    engineer = FeatureEngineer()
    X, feature_names = engineer.fit_transform(df)
    y = df['fatigue_label'].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: Normal={sum(y==0)}, Fatigued={sum(y==1)}")
    
    # Train and compare models
    print("\n" + "=" * 50)
    print("Training and Evaluating Models:")
    print("-" * 50)
    
    model_types = ['logistic', 'random_forest', 'gradient_boosting']
    results = {}
    
    for model_type in model_types:
        print(f"\n{model_type.upper()}:")
        model = FatiguePredictionModel(model_type=model_type)
        metrics = model.train(X, y, feature_names)
        results[model_type] = metrics
        
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1 Score:  {metrics['f1_score']:.3f}")
        print(f"  ROC AUC:   {metrics['roc_auc']:.3f}")
        if 'cv_roc_auc_mean' in metrics:
            print(f"  CV ROC AUC: {metrics['cv_roc_auc_mean']:.3f} (+/- {metrics['cv_roc_auc_std']:.3f})")
    
    # Show best model feature importance
    print("\n" + "=" * 50)
    print("Feature Importance (Random Forest):")
    print("-" * 40)
    
    rf_model = FatiguePredictionModel(model_type='random_forest')
    rf_model.train(X, y, feature_names)
    importance_df = rf_model.get_feature_importance()
    
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")
    
    # Example prediction
    print("\n" + "=" * 50)
    print("Example Prediction:")
    print("-" * 40)
    
    sample_idx = 0
    sample = X[sample_idx:sample_idx+1]
    explanation = rf_model.explain_prediction(sample)
    
    print(f"  Fatigue Probability: {explanation['fatigue_probability']:.1%}")
    print(f"  Prediction: {explanation['prediction']}")
    print(f"  Confidence: {explanation['confidence']:.1%}")
    print(f"  True Label: {'Fatigued' if y[sample_idx] == 1 else 'Normal'}")
    
    # Ensemble demonstration
    print("\n" + "=" * 50)
    print("Model Ensemble:")
    print("-" * 40)
    
    ensemble = ModelEnsemble()
    ensemble.add_model('logistic', 'logistic')
    ensemble.add_model('rf', 'random_forest')
    ensemble.add_model('gb', 'gradient_boosting')
    
    ensemble_metrics = ensemble.train(X, y, feature_names)
    
    print("Final weights (based on ROC AUC):")
    for name, weight in ensemble.weights.items():
        print(f"  {name}: {weight:.3f}")
    
    # Ensemble predictions
    ensemble_probs = ensemble.predict_proba(X)
    ensemble_preds = ensemble.predict(X)
    ensemble_acc = accuracy_score(y, ensemble_preds)
    ensemble_auc = roc_auc_score(y, ensemble_probs)
    
    print(f"\nEnsemble Performance:")
    print(f"  Accuracy: {ensemble_acc:.3f}")
    print(f"  ROC AUC:  {ensemble_auc:.3f}")
    
    return rf_model, results


if __name__ == "__main__":
    model, results = main()
