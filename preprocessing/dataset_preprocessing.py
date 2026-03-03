# ============================================
# FINAL DATA PREPROCESSING FOR ANN PROJECT
# ============================================

# STEP 0: Import Libraries
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ============================================
# STEP 1: Load Dataset
# ============================================

df = pd.read_csv("data/obesity_dataset.csv")

print("Dataset Loaded Successfully")
print("Initial Shape:", df.shape)

# ============================================
# STEP 2: Remove Duplicates
# ============================================

print("Duplicate Rows:", df.duplicated().sum())
df = df.drop_duplicates()
print("Shape After Removing Duplicates:", df.shape)

# ============================================
# STEP 3: Handle Missing Values (Safe Handling)
# ============================================

print("\nMissing Values Before Handling:")
print(df.isnull().sum())

# Numerical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Categorical columns (fix for pandas warning)
cat_cols = df.select_dtypes(include=['object', 'string']).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

print("\nMissing Values After Handling:")
print(df.isnull().sum())

# ============================================
# STEP 4: Encode Binary Columns
# ============================================

binary_cols = [
    "family_history_with_overweight",
    "FAVC",
    "SMOKE",
    "SCC"
]

for col in binary_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})

print("Binary Columns Encoded")

# ============================================
# STEP 5: Encode Categorical Columns
# ============================================

categorical_cols = [
    "Gender",
    "CAEC",
    "CALC",
    "MTRANS"
]

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("Categorical Features Encoded")

# ============================================
# STEP 6: Encode Target Variable
# ============================================

target_encoder = LabelEncoder()
df["NObeyesdad"] = target_encoder.fit_transform(df["NObeyesdad"])

print("Target Variable Encoded")
print(df["NObeyesdad"].value_counts())

# ============================================
# STEP 7: Feature Scaling (VERY IMPORTANT FOR ANN)
# ============================================

X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Feature Scaling Completed")

# ============================================
# STEP 8: Train-Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# ============================================
# STEP 9: Save Preprocessing Objects
# ============================================

# Create models directory automatically
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")
joblib.dump(target_encoder, "models/target_encoder.pkl")

print("Preprocessing Objects Saved Successfully")

# ============================================
# STEP 10: Save Processed Data
# ============================================

pd.DataFrame(X_train).to_csv("data/processed/X_train.csv", index=False)
pd.DataFrame(X_test).to_csv("data/processed/X_test.csv", index=False)
pd.DataFrame(y_train).to_csv("data/processed/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("data/processed/y_test.csv", index=False)

print("\nPREPROCESSING COMPLETED SUCCESSFULLY")