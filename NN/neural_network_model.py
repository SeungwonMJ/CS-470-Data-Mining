import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, recall_score,
    precision_score, confusion_matrix
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def evaluate_model(X_data, y_data, feature_names):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {'accuracy': [], 'roc_auc': [], 'f1': [], 'recall': [], 'precision': []}
    confusion_matrices = []

    for train_idx, test_idx in kf.split(X_data):
        X_train, X_test = X_data[train_idx], X_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]

        model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]


        metrics['accuracy'].append(accuracy_score(y_test, preds))
        metrics['roc_auc'].append(roc_auc_score(y_test, probs))
        metrics['f1'].append(f1_score(y_test, preds))
        metrics['recall'].append(recall_score(y_test, preds))
        metrics['precision'].append(precision_score(y_test, preds))

        cm = confusion_matrix(y_test, preds)
        confusion_matrices.append(cm)

    metrics_summary = {k: f"{np.mean(v):.4f} ± {np.std(v):.4f}" for k, v in metrics.items()}
    avg_cm = np.mean(confusion_matrices, axis=0).astype(int)

    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X_data, y_data)
    feature_scores = dict(zip(feature_names, selector.scores_))
    top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:5]

    return metrics_summary, top_features, feature_scores, avg_cm

def print_results(title, metrics, top_features, full_scores, cm):
    print(f"\n=== {title.upper()} ===")
    print("Top 5 Features (with F-scores):")
    for feat, score in top_features:
        print(f"  {feat:<10}: {score:.4f}")

    print("\nAll Feature Scores:")
    for feat, score in sorted(full_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat:<10}: {score:.4f}")

    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"  {k.capitalize():<10}: {v}")

    print("\nAverage Confusion Matrix:")
    print(cm)

def run_all_classifiers(csv_path):
    df = pd.read_csv(csv_path).drop_duplicates()

    # Full dataset
    X_full = df.drop("target", axis=1)
    y_full = df["target"]
    scaler = StandardScaler()
    X_scaled_full = scaler.fit_transform(X_full)
    full_metrics, full_top, full_scores, full_cm = evaluate_model(X_scaled_full, y_full.values, X_full.columns)

    # Age < 55
    df_leq_55 = df[df["age"] < 55]
    X_leq_55 = df_leq_55.drop("target", axis=1)
    y_leq_55 = df_leq_55["target"]
    X_scaled_leq_55 = scaler.fit_transform(X_leq_55)
    metrics_leq_55, top_leq_55, scores_leq_55, cm_leq_55 = evaluate_model(X_scaled_leq_55, y_leq_55.values, X_leq_55.columns)

    # Age ≥ 55
    df_gt_55 = df[df["age"] >= 55]
    X_gt_55 = df_gt_55.drop("target", axis=1)
    y_gt_55 = df_gt_55["target"]
    X_scaled_gt_55 = scaler.fit_transform(X_gt_55)
    metrics_gt_55, top_gt_55, scores_gt_55, cm_gt_55 = evaluate_model(X_scaled_gt_55, y_gt_55.values, X_gt_55.columns)

    # Print results
    print_results("Full Dataset", full_metrics, full_top, full_scores, full_cm)
    print_results("Age < 55", metrics_leq_55, top_leq_55, scores_leq_55, cm_leq_55)
    print_results("Age ≥ 55", metrics_gt_55, top_gt_55, scores_gt_55, cm_gt_55)

if __name__ == "__main__":
    run_all_classifiers("../Data/heart.csv")