import pandas as pd
import numpy as np

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("heart.csv")

# ===============================
# Preprocess the Dataset
# ===============================
# One-hot encode categorical columns
df = pd.get_dummies(df, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)
df = df.astype(int)

# ===============================
# Remove Duplicates
# ===============================
df = df.drop_duplicates()

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
    
    # Identify binary columns
    binary_cols = [
    'sex', 'fbs', 'exang', 'target'
    ]
    
    age_col = ['age']
    
    # Identify all numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Remove binary columns from numeric columns and Age for later processing
    numeric_cols = [col for col in numeric_cols if col not in binary_cols and col not in age_col]

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
# Check Missing Values
# ===============================
missing_values = df.isnull().sum().sum()
if missing_values == 0:
    print("No missing values in the dataset.")
else:
    print(f"There are {missing_values} missing values remaining.")

# ===============================
# Preview: Print First Row
# ===============================
# print(df.iloc[0])
