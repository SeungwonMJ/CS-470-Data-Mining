import pandas as pd
import numpy as np

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("heart_attack_prediction_dataset.csv")



# ===============================
# Preprocess the Dataset
# ===============================
# 1. Drop unwanted columns
columns_to_drop = ['Patient ID', 'Country', 'Continent', 'Hemisphere']
df.drop(columns=columns_to_drop, inplace=True)

# 2. Split Blood Pressure column into three different feature columns and drop original Blood Pressure column
def split_bp(bp_str):
    sbp, dbp = map(int, bp_str.strip("[]").split('/'))
    map_val = (sbp + 2 * dbp) / 3
    return pd.Series([sbp, dbp, round(map_val, 1)])
df[['SBP', 'DBP', 'MAP']] = df['Blood Pressure'].apply(split_bp)
df.drop(columns=['Blood Pressure'], inplace=True)

# 3. One-hot encode 'Sex' column only
df = pd.get_dummies(df, columns=['Sex'], prefix='Sex')

# 4. Map Diet: Unhealthy → 1, Average → 2, Healthy → 3
diet_map = {'Unhealthy': 1, 'Average': 2, 'Healthy': 3}
df['Diet'] = df['Diet'].map(diet_map)

# 5. Convert all boolean columns to integers (False → 0, True → 1)
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)



# ===============================
# Normalization Functions (Linear Regression = z-score, Neural Network = Min-Max, Ramdom Forest = None)
# ===============================
# Min-Max Normalization (0 to 1)
def normalize_minmax(dataframe, columns):
    df_norm = dataframe.copy()
    for col in columns:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    return df_norm

# Z-Score Standardization
def normalize_zscore(dataframe, columns):
    df_std = dataframe.copy()
    for col in columns:
        mean = df_std[col].mean()
        std = df_std[col].std()
        df_std[col] = (df_std[col] - mean) / std
    return df_std



# ===============================
# Apply Normalization Based on Model Type
# ===============================
def apply_normalization(df, model_type):
    # Identify numerical columns (excluding one-hot encoded or already categorical ones)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Don't normalize the label
    numeric_cols.remove('Heart Attack Risk')

    if model_type == 'linear_regression':
        print("Applying Z-score normalization for Linear Regression...")
        df = normalize_zscore(df, numeric_cols)
    elif model_type == 'neural_network':
        print("Applying Min-Max normalization for Neural Network...")
        df = normalize_minmax(df, numeric_cols)
    elif model_type == 'random_forest':
        print("No normalization needed for Random Forest.")
    else:
        print(f"Unknown model type: {model_type}. No normalization applied.")

    return df



# ===============================
# Example: Choose model type here
# ===============================
model_type = 'linear_regression'  # Change this to 'linear_regression' or 'neural_network' as needed

# Apply normalization accordingly
df = apply_normalization(df, model_type)

# ===============================
# Preview: Print First Row
# ===============================
print(df.iloc[0])
