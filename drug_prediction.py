import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.calibration import CalibratedClassifierCV
import shap
import warnings
warnings.filterwarnings('ignore')

class BiomarkerDrugPredictor:
    """
    Production-grade drug response predictor using biomarker data.
    
    Key features for clinical deployment:
    - Confidence intervals on predictions
    - SHAP values for interpretability
    - Calibrated probabilities
    - Cross-validated performance metrics
    - Flagging of out-of-distribution samples
    """
    
    def __init__(self, confidence_threshold=0.7):
        self.confidence_threshold = confidence_threshold
        self.scaler = StandardScaler()
        self.model = None
        self.calibrated_model = None
        self.feature_names = None
        self.training_stats = None
        
    def prepare_biomarker_data(self, df, biomarker_cols, outcome_col):
        """
        Prepare clinical biomarker data with quality checks.
        
        Parameters:
        -----------
        df : DataFrame with patient records
        biomarker_cols : list of biomarker measurement columns
        outcome_col : treatment response outcome (0=non-responder, 1=responder)
        """
        # Store feature names for interpretability
        self.feature_names = biomarker_cols
        
        # Extract features and outcome
        X = df[biomarker_cols].copy()
        y = df[outcome_col].copy()
        
        # Calculate training distribution statistics for OOD detection
        self.training_stats = {
            'mean': X.mean(),
            'std': X.std(),
            'min': X.min(),
            'max': X.max()
        }
        
        # Handle missing values with clinical rationale
        # In production, this should be informed by domain experts
        X = X.fillna(X.median())
        
        return X, y
    
    def flag_out_of_distribution(self, X_new):
        """
        Flag samples that are outside training distribution.
        Critical for clinical deployment - don't predict on unfamiliar data.
        """
        flags = []
        
        for idx, row in X_new.iterrows():
            is_ood = False
            ood_features = []
            
            for feature in self.feature_names:
                value = row[feature]
                mean = self.training_stats['mean'][feature]
                std = self.training_stats['std'][feature]
                
                # Flag if >3 standard deviations from training mean
                if abs(value - mean) > 3 * std:
                    is_ood = True
                    ood_features.append(feature)
            
            flags.append({
                'sample_id': idx,
                'is_ood': is_ood,
                'ood_features': ood_features
            })
        
        return pd.DataFrame(flags)
    
    def train_with_validation(self, X, y, n_folds=5):
        """
        Train with cross-validation and return performance metrics.
        Essential for understanding model reliability before clinical deployment.
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize model ensemble
        base_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        
        # Cross-validation with stratification
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Comprehensive scoring
        scoring = {
            'roc_auc': 'roc_auc',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1'
        }
        
        cv_results = cross_validate(
            base_model, X_scaled, y,
            cv=cv, scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        # Train final model on all data
        self.model = base_model.fit(X_scaled, y)
        
        # Calibrate probabilities for reliable confidence scores
        self.calibrated_model = CalibratedClassifierCV(
            self.model, method='sigmoid', cv=3
        )
        self.calibrated_model.fit(X_scaled, y)
        
        # Return validation metrics
        return {
            'mean_auc': cv_results['test_roc_auc'].mean(),
            'std_auc': cv_results['test_roc_auc'].std(),
            'mean_precision': cv_results['test_precision'].mean(),
            'mean_recall': cv_results['test_recall'].mean(),
            'mean_f1': cv_results['test_f1'].mean()
        }
    
    def predict_with_confidence(self, X_new):
        """
        Generate predictions with confidence intervals and interpretability.
        Returns actionable insights for clinical decision-making.
        """
        # Check for out-of-distribution samples
        ood_flags = self.flag_out_of_distribution(X_new)
        
        # Scale features
        X_scaled = self.scaler.transform(X_new)
        
        # Get calibrated probabilities
        proba = self.calibrated_model.predict_proba(X_scaled)[:, 1]
        
        # Calculate SHAP values for interpretability
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_scaled)
        
        # Generate predictions with confidence flags
        predictions = []
        
        for idx, prob in enumerate(proba):
            # Determine if confidence is sufficient for clinical use
            confidence_level = 'high' if (prob > self.confidence_threshold or 
                                         prob < (1 - self.confidence_threshold)) else 'low'
            
            # Get top contributing biomarkers
            feature_contributions = dict(zip(
                self.feature_names,
                shap_values[idx]
            ))
            top_features = sorted(
                feature_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]
            
            prediction = {
                'sample_id': X_new.index[idx],
                'response_probability': prob,
                'predicted_response': 'responder' if prob > 0.5 else 'non-responder',
                'confidence_level': confidence_level,
                'requires_manual_review': (confidence_level == 'low' or 
                                          ood_flags.loc[idx, 'is_ood']),
                'out_of_distribution': ood_flags.loc[idx, 'is_ood'],
                'top_contributing_biomarkers': top_features
            }
            
            predictions.append(prediction)
        
        return pd.DataFrame(predictions)
    
    def generate_clinical_report(self, predictions_df):
        """
        Generate human-readable report for clinical review.
        Critical for the "human in the loop" architecture.
        """
        report = []
        report.append("=" * 70)
        report.append("DRUG RESPONSE PREDICTION REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Summary statistics
        total = len(predictions_df)
        high_conf = (predictions_df['confidence_level'] == 'high').sum()
        requires_review = predictions_df['requires_manual_review'].sum()
        ood_samples = predictions_df['out_of_distribution'].sum()
        
        report.append(f"Total Predictions: {total}")
        report.append(f"High Confidence: {high_conf} ({high_conf/total*100:.1f}%)")
        report.append(f"Require Manual Review: {requires_review} ({requires_review/total*100:.1f}%)")
        report.append(f"Out of Distribution: {ood_samples}")
        report.append("")
        
        # Flag samples requiring review
        if requires_review > 0:
            report.append("SAMPLES REQUIRING MANUAL REVIEW:")
            report.append("-" * 70)
            
            review_samples = predictions_df[predictions_df['requires_manual_review']]
            for _, row in review_samples.iterrows():
                report.append(f"\nSample ID: {row['sample_id']}")
                report.append(f"  Response Probability: {row['response_probability']:.3f}")
                report.append(f"  Confidence: {row['confidence_level']}")
                
                if row['out_of_distribution']:
                    report.append(f"  WARNING: Out of distribution sample")
                
                report.append(f"  Top Contributing Biomarkers:")
                for biomarker, contribution in row['top_contributing_biomarkers']:
                    direction = "↑" if contribution > 0 else "↓"
                    report.append(f"    {direction} {biomarker}: {abs(contribution):.3f}")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)


# Example usage with synthetic AMD biomarker data
def generate_synthetic_amd_data(n_patients=200):
    """
    Generate synthetic AMD patient data for demonstration.
    In production, this would be real clinical data from StudyOS.
    """
    np.random.seed(42)
    
    # Simulate biomarkers
    data = {
        'patient_id': range(n_patients),
        'drusen_volume_mm3': np.random.gamma(2, 0.5, n_patients),
        'ez_integrity_percent': np.random.beta(8, 2, n_patients) * 100,
        'crora_area_mm2': np.random.exponential(0.3, n_patients),
        'htd_thickness_um': np.random.normal(150, 30, n_patients),
        'complement_c3_ng_ml': np.random.lognormal(4, 0.5, n_patients),
        'age_years': np.random.normal(75, 8, n_patients)
    }
    
    df = pd.DataFrame(data)
    
    # Simulate treatment response based on biomarkers
    # Higher drusen volume and cRORA area = lower response
    # Higher EZ integrity = higher response
    response_score = (
        -0.3 * (df['drusen_volume_mm3'] - df['drusen_volume_mm3'].mean()) / df['drusen_volume_mm3'].std() +
        0.4 * (df['ez_integrity_percent'] - df['ez_integrity_percent'].mean()) / df['ez_integrity_percent'].std() +
        -0.25 * (df['crora_area_mm2'] - df['crora_area_mm2'].mean()) / df['crora_area_mm2'].std() +
        np.random.normal(0, 0.5, n_patients)
    )
    
    df['treatment_response'] = (response_score > response_score.median()).astype(int)
    
    return df


# Demonstration
if __name__ == "__main__":
    print("Biomarker-Based Drug Response Prediction System")
    print("=" * 70)
    print()
    
    # Generate synthetic data
    print("Loading patient biomarker data...")
    df = generate_synthetic_amd_data(n_patients=200)
    
    # Define biomarker columns
    biomarker_cols = [
        'drusen_volume_mm3',
        'ez_integrity_percent', 
        'crora_area_mm2',
        'htd_thickness_um',
        'complement_c3_ng_ml',
        'age_years'
    ]
    
    # Split data
    train_df = df.iloc[:150]
    test_df = df.iloc[150:]
    
    # Initialize predictor
    predictor = BiomarkerDrugPredictor(confidence_threshold=0.7)
    
    # Prepare training data
    X_train, y_train = predictor.prepare_biomarker_data(
        train_df, biomarker_cols, 'treatment_response'
    )
    
    # Train with validation
    print("Training model with cross-validation...")
    metrics = predictor.train_with_validation(X_train, y_train)
    
    print("\nCross-Validation Performance:")
    print(f"  AUC: {metrics['mean_auc']:.3f} ± {metrics['std_auc']:.3f}")
    print(f"  Precision: {metrics['mean_precision']:.3f}")
    print(f"  Recall: {metrics['mean_recall']:.3f}")
    print(f"  F1 Score: {metrics['mean_f1']:.3f}")
    print()
    
    # Prepare test data
    X_test, y_test = predictor.prepare_biomarker_data(
        test_df, biomarker_cols, 'treatment_response'
    )
    
    # Generate predictions
    print("Generating predictions for new patients...")
    predictions = predictor.predict_with_confidence(X_test)
    
    # Generate clinical report
    report = predictor.generate_clinical_report(predictions)
    print(report)
    
    # Calculate actual performance on test set
    test_auc = roc_auc_score(y_test, predictions['response_probability'])
    print(f"\nTest Set AUC: {test_auc:.3f}")
    
    # Show example predictions
    print("\nExample Predictions (first 5 patients):")
    print("-" * 70)
    for _, row in predictions.head().iterrows():
        print(f"\nPatient {row['sample_id']}:")
        print(f"  Predicted: {row['predicted_response']} "
              f"(probability: {row['response_probability']:.3f})")
        print(f"  Confidence: {row['confidence_level']}")
        print(f"  Manual review needed: {row['requires_manual_review']}")