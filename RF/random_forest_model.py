import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score
)


# Function to evaluate model and display results
def evaluate_and_display(X, y, group_name="All Data"):
    # Initialize the model
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    # Set up 5‑fold stratified CV and scoring metrics
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'roc_auc': make_scorer(roc_auc_score),
        'f1': make_scorer(f1_score),
        'recall': make_scorer(recall_score),
        'precision': make_scorer(precision_score)
    }

    # Run cross-validation
    cv_results = cross_validate(
        estimator=rf,
        X=X, y=y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    # Print CV results
    print(f"\n=== 5‑Fold CV results for {group_name} (n={len(X)}) ===")
    for metric in scoring:
        scores = cv_results[f'test_{metric}']
        print(f" - {metric:10s}: {scores.mean():.4f} ± {scores.std():.4f}")

    # Retrain on the full dataset and rank feature importances
    rf.fit(X, y)
    importances = rf.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

    print(f"\nFeature importance ranking for {group_name}:")
    for feature, imp in feat_imp.items():
        print(f"{feature:<8} : {imp:.4f}")

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    feat_imp.plot(kind='bar')
    plt.title(f'Feature Importance for {group_name}')
    plt.tight_layout()
    plt.show()

    return rf


# 1) Load the dataset
df = pd.read_csv('heart.csv')

df = df.drop_duplicates()

# 2) Define features (X) and target (y)
feature_cols = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]
X = df[feature_cols]
y = df['target']

# 3) First evaluate on the entire dataset
print("===== ANALYSIS FOR FULL DATASET =====")
full_model = evaluate_and_display(X, y, "Full Dataset")

# 4) Split dataset by age and evaluate each group
print("\n\n===== ANALYSIS BY AGE GROUPS =====")

# Under 55 group
young_mask = df['age'] < 55
X_young = X[young_mask]
y_young = y[young_mask]
young_model = evaluate_and_display(X_young, y_young, "Age < 55")

# 55 and over group
old_mask = df['age'] >= 55
X_old = X[old_mask]
y_old = y[old_mask]
old_model = evaluate_and_display(X_old, y_old, "Age >= 55")