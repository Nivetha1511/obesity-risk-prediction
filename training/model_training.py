import pandas as pd
import numpy as np
import joblib
import os
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ============================================
# STEP 1: Load Data
# ============================================

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").values.flatten()
y_test = pd.read_csv("data/processed/y_test.csv").values.flatten()

print("Data Loaded")

# ============================================
# STEP 2: Save Feature Names (CRITICAL)
# ============================================

feature_names = list(X_train.columns)

os.makedirs("models", exist_ok=True)
joblib.dump(feature_names, "models/feature_names.pkl")

print("Feature names saved")

# ============================================
# STEP 3: Check Class Distribution (Before SMOTE)
# ============================================

print("\nBefore SMOTE:")
print(pd.Series(y_train).value_counts())

# ============================================
# STEP 4: Apply SMOTE (Balance Data)
# ============================================

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(pd.Series(y_train).value_counts())

# ============================================
# STEP 5: Scaling
# ============================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")

print("Scaler Saved")

# ============================================
# STEP 6: Train Model
# ============================================

model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("Model Trained")

# ============================================
# STEP 7: Evaluate Model
# ============================================

y_pred = model.predict(X_test_scaled)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ============================================
# STEP 8: Save Model
# ============================================

joblib.dump(model, "models/obesity_model.pkl")

print("\nModel Saved Successfully")
print("\n🚀 TRAINING COMPLETED SUCCESSFULLY")