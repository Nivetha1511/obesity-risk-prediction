import pandas as pd
import numpy as np
import joblib
import os
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ============================================
# STEP 0: FIX PATH (VERY IMPORTANT)
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================
# STEP 1: Load Data
# ============================================

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").values.flatten()
y_test = pd.read_csv("data/processed/y_test.csv").values.flatten()

print("Data Loaded")

# ============================================
# STEP 2: FIX COLUMN NAMES
# ============================================

correct_columns = [
    "Gender", "Age", "Height", "Weight",
    "family_history_with_overweight", "FAVC", "FCVC", "NCP",
    "CAEC", "SMOKE", "CH2O", "SCC",
    "FAF", "TUE", "CALC", "MTRANS"
]

X_train.columns = correct_columns
X_test.columns = correct_columns

print("Column names fixed")

# ============================================
# STEP 3: SAVE FEATURE NAMES
# ============================================

feature_path = os.path.join(MODEL_DIR, "feature_names.pkl")
joblib.dump(correct_columns, feature_path)

print("Feature names saved at:", feature_path)

# ============================================
# STEP 4: CHECK DISTRIBUTION
# ============================================

print("\nBefore SMOTE:")
print(pd.Series(y_train).value_counts())

# ============================================
# STEP 5: APPLY SMOTE
# ============================================

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(pd.Series(y_train).value_counts())

# ============================================
# STEP 6: SCALING
# ============================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)

print("Scaler saved at:", scaler_path)

# ============================================
# STEP 7: TRAIN MODEL
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
# STEP 8: EVALUATE
# ============================================

y_pred = model.predict(X_test_scaled)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ============================================
# STEP 9: SAVE MODEL
# ============================================

model_path = os.path.join(MODEL_DIR, "obesity_model.pkl")
joblib.dump(model, model_path)

print("Model saved at:", model_path)

print("\n🚀 TRAINING COMPLETED SUCCESSFULLY")