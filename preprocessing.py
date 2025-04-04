import pandas as pd
import numpy as np

# Load the dataset (update the file path as needed)
df = pd.read_csv("heart_attack_prediction_dataset.csv")

# 1. Drop unwanted columns
columns_to_drop = ['Patient ID', 'Country', 'Continent', 'Hemisphere']
df.drop(columns=columns_to_drop, inplace=True)

# 2. Split Blood Pressure column into three different feature columns
# Let's assume the column is named 'Blood Pressure' — update as needed
def split_bp(bp_str):
    sbp, dbp = map(int, bp_str.strip("[]").split('/'))
    map_val = (sbp + 2 * dbp) / 3
    return pd.Series([sbp, dbp, round(map_val, 1)])


df[['SBP', 'DBP', 'MAP']] = df['Blood Pressure'].apply(split_bp)

# Drop original Blood Pressure column
df.drop(columns=['Blood Pressure'], inplace=True)

# 3. One-hot encode 'Sex' column only
df = pd.get_dummies(df, columns=['Sex'], prefix='Sex')

# 4. Map Diet: Unhealthy → 1, Average → 2, Healthy → 3
diet_map = {'Unhealthy': 1, 'Average': 2, 'Healthy': 3}
df['Diet'] = df['Diet'].map(diet_map)

# 5. Convert all boolean columns to integers (False → 0, True → 1)
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# Show the resulting DataFrame's first object instance
print(df.iloc[0])
