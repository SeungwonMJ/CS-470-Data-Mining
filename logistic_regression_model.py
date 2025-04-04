import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ================================================
# Import preprocessed DataFrame from preprocessing.py
# ================================================
from preprocessing import df



# ================================================
# Split dataset into features and target
# ================================================
X = df.drop(columns=['Heart Attack Risk'])
y = df['Heart Attack Risk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



# ================================================
# Train Logistic Regression Model
# ================================================
model = LogisticRegression(class_weight='balanced', max_iter=1000)  # Increased max_iter for convergence safety
model.fit(X_train, y_train)



# ================================================
# Evaluate the Model
# ================================================
y_pred = model.predict(X_test)

print("========== MODEL EVALUATION ==========")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



# ================================================
# Feature Importance (Coefficients)
# ================================================
print("========== FEATURE IMPORTANCE ==========")
importance = pd.Series(model.coef_[0], index=X.columns)
importance_sorted = importance.abs().sort_values(ascending=False)

print("\nTop 10 Most Influential Features:")
print(importance_sorted.head(10))
