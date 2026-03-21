import pandas as pd
import numpy as np
import joblib
import os
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").values.flatten()
y_test = pd.read_csv("data/processed/y_test.csv").values.flatten()

print("Data Loaded")

# ✅ STEP 1: Apply SMOTE (before scaling)
smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

print("After SMOTE:")
print(pd.Series(y_train).value_counts())

# ✅ STEP 2: Scale
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")

print("Scaler Saved")

# ✅ STEP 3: Train model
model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1)
model.fit(X_train_scaled, y_train)

print("Model Trained")

# ✅ STEP 4: Evaluate
y_pred = model.predict(X_test_scaled)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ✅ STEP 5: Save model
joblib.dump(model, "models/obesity_model.pkl")

print("\nModel Saved Successfully")