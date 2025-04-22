import sys
from contextlib import redirect_stdout
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

def run_and_save_output(file_path='logistic_regression_output.txt'):
    with open(file_path, 'w') as f:
        with redirect_stdout(f):
            main()

def main():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # ================================================
    # Import preprocessed DataFrame from preprocessing.py
    # ================================================
    from new_preprocessing import df

    # ================================================
    # Normalize only the 'Age' column for the general dataset model
    # ================================================
    from new_preprocessing import normalize_zscore

    X = df.drop(columns=['target'])
    y = df['target']

    # Create a new column 'Age_zscore' for the normalized age
    X['Age_zscore'] = normalize_zscore(X[['age']], columns=['age'])['age']

    # Drop the original 'Age' column (only for the general model's features)
    X = X.drop(columns=['age'])

    # ================================================
    # Split dataset into features and target
    # ================================================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ================================================
    # Train Logistic Regression Model for the whole dataset
    # ================================================
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)

    # ================================================
    # Evaluate the Model for the whole dataset
    # ================================================
    y_pred = model.predict(X_test)

    print("========== MODEL EVALUATION for the whole dataset ==========")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ================================================
    # ROC AUC Score and ROC Curve Visualization for the whole dataset
    # ================================================
    # Get predicted probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute ROC AUC
    auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC AUC Score: {auc:.4f}")

    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color='darkorange')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ================================================
    # Correlation Heatmap: Feature Relationships for the whole dataset
    # ================================================
    plt.figure(figsize=(14, 10))
    correlation_matrix = df.corr(numeric_only=True)

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )

    plt.title("Correlation Heatmap of Features for the whole dataset", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # ================================================
    # Feature Importance (Coefficients) for the whole dataset
    # ================================================
    print("========== FEATURE IMPORTANCE for the whole dataset ==========")
    importance = pd.Series(model.coef_[0], index=X.columns)
    importance_sorted = importance.reindex(importance.abs().sort_values(ascending=False).index)

    print("\nTop 10 Most Influential Features for the whole dataset:")
    print(importance_sorted.head(10))

    # ================================================
    # K-Fold Cross-Validation (k=5) with Full Metrics
    # ================================================
    from sklearn.metrics import precision_score, recall_score, f1_score

    print("\n========== 5-FOLD CROSS-VALIDATION (Full Metrics) for the whole dataset ==========")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    roc_auc_scores = []

    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
        y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]

        model_cv = LogisticRegression(class_weight='balanced', max_iter=1000)
        model_cv.fit(X_train_cv, y_train_cv)

        y_pred_cv = model_cv.predict(X_test_cv)
        y_prob_cv = model_cv.predict_proba(X_test_cv)[:, 1]

        acc = accuracy_score(y_test_cv, y_pred_cv)
        prec = precision_score(y_test_cv, y_pred_cv, average='weighted', zero_division=0)
        rec = recall_score(y_test_cv, y_pred_cv, average='weighted', zero_division=0)
        f1 = f1_score(y_test_cv, y_pred_cv, average='weighted', zero_division=0)
        auc = roc_auc_score(y_test_cv, y_prob_cv)

        accuracy_scores.append(acc)
        precision_scores.append(prec)
        recall_scores.append(rec)
        f1_scores.append(f1)
        roc_auc_scores.append(auc)

        print(f"Fold {fold}: Accuracy = {acc:.4f}, Precision = {prec:.4f}, Recall = {rec:.4f}, F1 = {f1:.4f}, ROC AUC = {auc:.4f}")

    print("\n===== AVERAGE METRICS ACROSS 5 FOLDS (± Std Dev) =====")
    print(f"Accuracy : {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
    print(f"Precision: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
    print(f"Recall   : {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
    print(f"F1-score : {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"ROC AUC  : {np.mean(roc_auc_scores):.4f} ± {np.std(roc_auc_scores):.4f}")

    # ================================================
    # Define Age Groups: Younger (<55) and Older (>=55)
    # ================================================ 
    bins = [0, 55, float('inf')]
    labels = ['Younger', 'Older']
    df['Age Group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    print("\n")
    print(df['Age Group'].value_counts())
    print(df['age'].min(), df['age'].max())

    # ================================================
    # Create a function to train and evaluate model per age group
    # ================================================
    def train_and_evaluate_for_age_group(age_group):
        # Filter dataset by the given age group
        df_age_group = df[df['Age Group'] == age_group]

        # Check if there is enough data in the age group
        if len(df_age_group) == 0:
            print(f"\nNo data available for {age_group} age group. Skipping evaluation.")
            return

        X_age_group = df_age_group.drop(columns=['target', 'Age Group'])
        y_age_group = df_age_group['target']

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_age_group, y_age_group, test_size=0.2, random_state=42, stratify=y_age_group)

        # Train Logistic Regression Model
        model = LogisticRegression(class_weight='balanced', max_iter=1000)
        model.fit(X_train, y_train)

        # Evaluate the Model
        y_pred = model.predict(X_test)
        print(f"\n========== EVALUATION FOR {age_group} AGE GROUP ==========")
        print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # ROC AUC Score and ROC Curve Visualization
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print(f"\nROC AUC Score ({age_group}): {auc:.4f}")

        # Plot ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color='darkorange')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for {age_group} Age Group")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Feature Importance (Coefficients)
        print("========== FEATURE IMPORTANCE ==========")
        importance = pd.Series(model.coef_[0], index=X_age_group.columns)
        importance_sorted = importance.reindex(importance.abs().sort_values(ascending=False).index)
        print("\nTop 10 Most Influential Features:")
        print(importance_sorted.head(10))

        # K-Fold CV for Age Group (with full metrics)
        print(f"\n========== 5-FOLD CROSS-VALIDATION for {age_group} AGE GROUP ==========")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        acc_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        auc_scores = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_age_group, y_age_group), 1):
            X_tr, X_te = X_age_group.iloc[train_idx], X_age_group.iloc[test_idx]
            y_tr, y_te = y_age_group.iloc[train_idx], y_age_group.iloc[test_idx]

            model_cv = LogisticRegression(class_weight='balanced', max_iter=1000)
            model_cv.fit(X_tr, y_tr)

            y_pred = model_cv.predict(X_te)
            y_prob = model_cv.predict_proba(X_te)[:, 1]

            acc = accuracy_score(y_te, y_pred)
            prec = precision_score(y_te, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_te, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_te, y_pred, average='weighted', zero_division=0)
            auc = roc_auc_score(y_te, y_prob)

            acc_scores.append(acc)
            precision_scores.append(prec)
            recall_scores.append(rec)
            f1_scores.append(f1)
            auc_scores.append(auc)

            print(f"Fold {fold}: Accuracy = {acc:.4f}, Precision = {prec:.4f}, Recall = {rec:.4f}, F1 = {f1:.4f}, ROC AUC = {auc:.4f}")

        print(f"\n===== AVERAGE METRICS for {age_group} AGE GROUP =====")
        print(f"Average Accuracy : {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
        print(f"Average Precision: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
        print(f"Average Recall   : {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
        print(f"Average F1-score : {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"Average ROC AUC  : {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")

    # ================================================
    # Train and Evaluate for Each Age Group
    # ================================================
    for age_group in labels:
        train_and_evaluate_for_age_group(age_group)

run_and_save_output('logistic_regression_output.txt')









