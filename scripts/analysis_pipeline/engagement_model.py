#!/usr/bin/env python3
"""
engagement_model.py - CatBoost model for predicting clip engagement performance

Trains a binary classifier:
- top25_eng - Will this clip have above-average engagement?

Uses SHAP for interpretability.

Usage:
    python engagement_model.py --input master_query_results.csv

Requirements:
    pip install catboost shap pandas numpy scikit-learn matplotlib
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, 
    precision_score, 
    recall_score, 
    classification_report,
    confusion_matrix
)

from catboost import CatBoostClassifier, Pool

import shap
import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURATION
# ============================================================================

# Feature columns from your schema
QUANT_FEATURES = [
    'duration_s',
    'word_count', 
    'wpm',
    'hook_word_count',
    'hook_wpm',
    'num_sentences',
    'question_start',
    'reading_level',
    'filler_count',
    'filler_density',
    'first_person_ratio',
    'second_person_ratio',
]

QUAL_FEATURES = [
    'hook_type',
    'hook_emotion', 
    'topic_primary',
    'has_examples',
    'has_payoff',
    'has_numbers',
    'insider_language',
    'technical_depth',
]

# Categorical features (CatBoost needs to know which ones)
CATEGORICAL_FEATURES = [
    'hook_type',
    'hook_emotion',
    'topic_primary', 
    'technical_depth',
]

# All features combined
ALL_FEATURES = QUANT_FEATURES + QUAL_FEATURES

# Random seed for reproducibility
RANDOM_SEED = 42


# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """
    Load CSV and compute normalized performance metrics.
    """
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} clips from {df['channel_id'].nunique()} channels")
    
    # Compute channel means for normalization
    channel_means = df.groupby('channel_id').agg({
        'view_count': 'mean',
        'like_count': 'mean', 
        'comment_count': 'mean'
    }).rename(columns={
        'view_count': 'mean_views',
        'like_count': 'mean_likes',
        'comment_count': 'mean_comments'
    })
    
    df = df.merge(channel_means, left_on='channel_id', right_index=True, how='left')
    
    # Calculate normalized performance indices
    df['VPI'] = (df['view_count'] / df['mean_views']).replace([np.inf, -np.inf], np.nan).fillna(0).round(3)
    df['LPI'] = (df['like_count'] / df['mean_likes']).replace([np.inf, -np.inf], np.nan).fillna(0).round(3)
    df['CPI'] = (df['comment_count'] / df['mean_comments']).replace([np.inf, -np.inf], np.nan).fillna(0).round(3)
    
    # Calculate engagement rate
    df['engagement_rate'] = ((df['like_count'] + 3 * df['comment_count']) / df['view_count']).fillna(0)
    
    print(f"\nVPI stats: mean={df['VPI'].mean():.3f}, median={df['VPI'].median():.3f}")
    print(f"Engagement stats: mean={df['engagement_rate'].mean():.4f}, median={df['engagement_rate'].median():.4f}")
    
    return df


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary classification target based on top 25% threshold for engagement.
    """
    print("\n" + "="*70)
    print("CREATING TARGET")
    print("="*70)
    
    eng_threshold = df['engagement_rate'].quantile(0.75)
    print(f"Engagement threshold (75th percentile): {eng_threshold:.4f}")
    
    df['top25_eng'] = (df['engagement_rate'] >= eng_threshold).astype(int)
    print(f"\nTop 25% Engagement clips: {df['top25_eng'].sum()} ({df['top25_eng'].mean()*100:.1f}%)")
    
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Extract feature matrix and identify categorical column indices.
    """
    print("\n" + "="*70)
    print("PREPARING FEATURES")
    print("="*70)
    
    # Extract feature columns
    X = df[ALL_FEATURES].copy()
    
    # Convert boolean columns to int
    bool_cols = ['has_examples', 'has_payoff', 'has_numbers', 'insider_language', 'question_start']
    for col in bool_cols:
        if col in X.columns:
            X[col] = X[col].astype(int)
    
    # Get indices of categorical features for CatBoost
    cat_feature_indices = [X.columns.get_loc(col) for col in CATEGORICAL_FEATURES if col in X.columns]
    
    print(f"Total features: {len(ALL_FEATURES)}")
    print(f"Quantitative: {len(QUANT_FEATURES)}")
    print(f"Qualitative: {len(QUAL_FEATURES)}")
    print(f"Categorical feature indices: {cat_feature_indices}")
    
    # Check for missing values
    missing = X.isnull().sum()
    if missing.any():
        print(f"\nWarning - Missing values:")
        print(missing[missing > 0])
        X = X.fillna(X.median(numeric_only=True))
    
    return X, cat_feature_indices


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cat_features: list,
    model_name: str
) -> tuple[CatBoostClassifier, dict]:
    """
    Train CatBoost classifier with cross-validation.
    """
    print(f"\n{'='*70}")
    print(f"TRAINING MODEL: {model_name}")
    print(f"{'='*70}")
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Positive class in train: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
    
    # Initialize model
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        cat_features=cat_features,
        random_seed=RANDOM_SEED,
        verbose=100,  # Print every 100 iterations
        early_stopping_rounds=50,
    )
    
    # Create CatBoost Pool objects
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    test_pool = Pool(X_test, y_test, cat_features=cat_features)
    
    # Train with validation
    model.fit(
        train_pool,
        eval_set=test_pool,
        use_best_model=True,
    )
    
    # Cross-validation on training set
    print("\nRunning 5-fold cross-validation...")
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
        scoring='roc_auc'
    )
    print(f"CV ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    
    # Evaluate on test set
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    metrics = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
    }
    
    print(f"\n--- TEST SET METRICS ---")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Bottom 75%', 'Top 25%']))
    
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, metrics


# ============================================================================
# SHAP ANALYSIS
# ============================================================================

def run_shap_analysis(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    model_name: str,
    output_dir: Path
) -> None:
    """
    Generate SHAP explanations and plots.
    """
    print(f"\n{'='*70}")
    print(f"SHAP ANALYSIS: {model_name}")
    print(f"{'='*70}")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # If binary classification, shap_values might be a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Take positive class
    
    # Global feature importance
    print("\n--- GLOBAL FEATURE IMPORTANCE (mean |SHAP|) ---")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:<25} {row['importance']:.4f}")
    
    # Save feature importance
    feature_importance.to_csv(output_dir / f'{model_name}_feature_importance.csv', index=False)
    
    # SHAP Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False, max_display=15)
    plt.title(f'SHAP Summary - {model_name}')
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {model_name}_shap_summary.png")
    
    # SHAP Bar Plot (simpler view)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=15)
    plt.title(f'Feature Importance - {model_name}')
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_shap_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_name}_shap_bar.png")
    
    return shap_values, feature_importance


def analyze_top_clips(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    df: pd.DataFrame,
    model_name: str,
    n_examples: int = 5
) -> None:
    """
    Show SHAP breakdown for top predicted clips.
    """
    print(f"\n--- TOP {n_examples} PREDICTED CLIPS ({model_name}) ---")
    
    # Get predictions
    proba = model.predict_proba(X)[:, 1]
    
    # Get top predicted
    top_indices = np.argsort(proba)[-n_examples:][::-1]
    
    explainer = shap.TreeExplainer(model)
    
    for rank, idx in enumerate(top_indices, 1):
        print(f"\n#{rank} - Predicted probability: {proba[idx]:.2%}")
        print(f"    Actual VPI: {df.iloc[idx]['VPI']:.2f}")
        print(f"    Hook: {df.iloc[idx]['hook_type']} + {df.iloc[idx]['hook_emotion']}")
        print(f"    Topic: {df.iloc[idx]['topic_primary']}")
        print(f"    Duration: {df.iloc[idx]['duration_s']:.0f}s")
        
        # Get SHAP values for this clip
        shap_vals = explainer.shap_values(X.iloc[[idx]])
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        
        # Top contributing features
        feature_contrib = pd.DataFrame({
            'feature': X.columns,
            'shap': shap_vals[0]
        }).sort_values('shap', key=abs, ascending=False)
        
        print(f"    Top SHAP contributors:")
        for _, row in feature_contrib.head(5).iterrows():
            direction = "+" if row['shap'] > 0 else ""
            print(f"      {row['feature']:<20} {direction}{row['shap']:.3f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train clip engagement prediction model')
    parser.add_argument('--input', type=str, default='master_query_results.csv',
                        help='Path to input CSV')
    parser.add_argument('--output-dir', type=str, default='data/tagging/model_output',
                        help='Directory for outputs')
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("CLIP ENGAGEMENT ML TRAINING")
    print("="*70)
    print(f"Input: {args.input}")
    print(f"Output: {output_dir}")
    
    # Load and prepare data
    df = load_and_prepare_data(args.input)
    df = create_targets(df)
    X, cat_feature_indices = prepare_features(df)
    
    # Train-test split for engagement target
    X_train, X_test, y_train_eng, y_test_eng = train_test_split(
        X, df['top25_eng'],
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=df['top25_eng']
    )
    
    # ========== MODEL: Engagement ==========
    model_eng, metrics_eng = train_model(
        X_train, y_train_eng,
        X_test, y_test_eng,
        cat_feature_indices,
        'top25_eng'
    )
    
    # Save model
    model_eng.save_model(output_dir / 'model_eng.cbm')
    print(f"\nSaved: model_eng.cbm")
    
    # SHAP analysis
    shap_eng, importance_eng = run_shap_analysis(model_eng, X, 'top25_eng', output_dir)
    
    # Analyze top clips
    analyze_top_clips(model_eng, X, df, 'top25_eng')
    
    # ========== SUMMARY ==========
    print("\n" + "="*70)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*70)
    
    print(f"\n{'Model':<20} {'ROC-AUC':<10} {'CV Mean':<10} {'Precision':<10} {'Recall':<10}")
    print("-"*60)
    print(f"{'Engagement':<20} {metrics_eng['roc_auc']:<10.3f} {metrics_eng['cv_mean']:<10.3f} {metrics_eng['precision']:<10.3f} {metrics_eng['recall']:<10.3f}")
    
    print(f"\n--- TOP 5 FEATURES BY MODEL ---")
    for _, row in importance_eng.head(5).iterrows():
        print(f"  {row['feature']:<25} {row['importance']:.4f}")
    
    print(f"\n{'='*70}")
    print(f"Outputs saved to: {output_dir}/")
    print(f"  - model_eng.cbm")
    print(f"  - top25_eng_shap_summary.png")
    print(f"  - top25_eng_shap_bar.png")
    print(f"  - top25_eng_feature_importance.csv")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
