import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
)
from tqdm import tqdm
import time
import numpy as np

# ==============================================================================
# PREPROCESSING SECTION
# ==============================================================================

print("Starting data preprocessing...")
# Load Dataset
df = pd.read_csv("heart_attack_prediction_dataset.csv")

# 1. Drop unwanted columns
columns_to_drop = ['Patient ID', 'Country', 'Continent', 'Hemisphere']
df.drop(columns=columns_to_drop, inplace=True)

# Setup a preprocessing progress bar
preprocessing_steps = ['Splitting Blood Pressure', 'One-hot encoding', 'Mapping Diet', 'Converting booleans',
                       'Normalizing data']
pbar = tqdm(preprocessing_steps, desc="Preprocessing")


# 2. Split Blood Pressure column into SBP, DBP, MAP and drop original
def split_bp(bp_str):
    sbp, dbp = map(int, bp_str.strip("[]").split('/'))
    map_val = (sbp + 2 * dbp) / 3
    return pd.Series([sbp, dbp, round(map_val, 1)])


df[['SBP', 'DBP', 'MAP']] = df['Blood Pressure'].apply(split_bp)
df.drop(columns=['Blood Pressure'], inplace=True)
pbar.update(1)  # Update progress

# 3. One-hot encode 'Sex' column only
df = pd.get_dummies(df, columns=['Sex'], prefix='Sex')
pbar.update(1)  # Update progress

# 4. Map Diet: Unhealthy → 1, Average → 2, Healthy → 3
diet_map = {'Unhealthy': 1, 'Average': 2, 'Healthy': 3}
df['Diet'] = df['Diet'].map(diet_map)
pbar.update(1)  # Update progress

# 5. Convert all boolean columns to integers (False → 0, True → 1)
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)
pbar.update(1)  # Update progress


# Normalization Functions
def normalize_minmax(dataframe, columns):
    df_norm = dataframe.copy()
    for col in columns:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    return df_norm


def normalize_zscore(dataframe, columns):
    df_std = dataframe.copy()
    for col in columns:
        mean = df_std[col].mean()
        std = df_std[col].std()
        df_std[col] = (df_std[col] - mean) / std
    return df_std


# Apply Normalization Based on Model Type
def apply_normalization(df, model_type):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols.remove('Heart Attack Risk')

    if model_type == 'linear_regression':
        df = normalize_zscore(df, numeric_cols)
    elif model_type == 'neural_network':
        df = normalize_minmax(df, numeric_cols)
    return df


model_type = 'random_forest'
df = apply_normalization(df, model_type)
pbar.update(1)  # Update progress
pbar.close()  # Close preprocessing progress bar

# ==============================================================================
# TRAINING & TESTING SECTION
# ==============================================================================

print("\nPreparing training data...")
# Define Features and Target
X = df.drop(columns=['Heart Attack Risk'])
y = df['Heart Attack Risk']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

print("\nTraining model with default parameters...")

# Create and train model with default parameters
model = RandomForestClassifier(random_state=42, class_weight='balanced')

# Track training progress
t_start = time.time()
print("Training progress:")

# Display training progress with a progress bar
with tqdm(total=100, desc="Training Model") as pbar:
    # Function to update progress bar
    def progress_callback(progress):
        pbar.n = int(progress * 100)
        pbar.refresh()


    # Incrementally update progress during training
    # Training in 10% increments for progress visualization
    for i in range(10):
        model.fit(X_train.iloc[:int((i + 1) * 0.1 * len(X_train))],
                  y_train.iloc[:int((i + 1) * 0.1 * len(X_train))])
        progress_callback((i + 1) / 10)

    # Final fit with all data to ensure model is properly trained
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

t_end = time.time()
print(f"Model training completed in {t_end - t_start:.2f} seconds")

print("\nEvaluating model on test data...")
# Generate predictions and calculate feature importance
print("Calculating predictions...")
with tqdm(total=3, desc="Model Evaluation") as pbar:
    # Generate class predictions
    y_pred = model.predict(X_test)
    pbar.update(1)

    # Generate probability predictions
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    pbar.update(1)

    # Calculate feature importance
    feature_importances = pd.DataFrame(
        model.feature_importances_,
        index=X.columns,
        columns=['importance']
    ).sort_values('importance', ascending=False)
    pbar.update(1)

# ==============================================================================
# PREDICTION REPORT SECTION
# ==============================================================================

# Confusion Matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Check test set label distribution
print("Test Set Label Distribution:")
print(f"Label 0 count: {(y_test == 0).sum()}")
print(f"Label 1 count: {(y_test == 1).sum()}")

# Check prediction label distribution
print("\nPrediction Label Distribution:")
print(f"Label 0 count: {(y_pred == 0).sum()}")
print(f"Label 1 count: {(y_pred == 1).sum()}")

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auroc = roc_auc_score(y_test, y_pred_prob)

# Print performance results
print("\nModel Performance Metrics:")
print("True Positive: ", tp)
print("True Negative: ", tn)
print("False Positive: ", fp)
print("False Negative: ", fn)
print(f"\nAccuracy             : {accuracy:.4f}")
print(f"Recall (Sensitivity) : {recall:.4f}")
print(f"Specificity          : {specificity:.4f}")
print(f"False Positive Rate  : {fpr:.4f}")
print(f"Precision            : {precision:.4f}")
print(f"F1-score             : {f1:.4f}")
print(f"AUROC                : {auroc:.4f}")

# Display model parameters
default_params = {
    'n_estimators': 100,
    'max_depth': None,
    'max_features': 'sqrt',
    'criterion': 'gini',
    'min_samples_split': 2
}
print(f"\nUsed Default Parameters: {default_params}")

# Display top feature importances
print("\nTop 5 Most Important Features:")
print(feature_importances.head(5))