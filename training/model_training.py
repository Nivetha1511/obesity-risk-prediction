# ============================================
# ANN MODEL TRAINING
# ============================================

import numpy as np
import pandas as pd
import os
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ============================================
# STEP 1: Load Processed Data
# ============================================

X_train_df = pd.read_csv("data/processed/X_train.csv")
X_test_df = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").values.flatten()
y_test = pd.read_csv("data/processed/y_test.csv").values.flatten()

print("Data Loaded Successfully")

# ============================================
# STEP 2: Save Feature Names
# ============================================

feature_names = list(X_train_df.columns)

os.makedirs("models", exist_ok=True)

joblib.dump(feature_names, "models/feature_names.pkl")

print("Feature names saved")

# ============================================
# STEP 3: Convert to Numpy Arrays
# ============================================

X_train = X_train_df.values
X_test = X_test_df.values

# ============================================
# STEP 4: Create and Save Scaler
# ============================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")

print("Scaler Saved Successfully")

# ============================================
# STEP 5: Encode Target Labels
# ============================================

encoder = LabelEncoder()

y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

joblib.dump(encoder, "models/target_encoder.pkl")

print("Target Encoder Saved Successfully")

# ============================================
# STEP 6: One-Hot Encode Target
# ============================================

num_classes = len(np.unique(y_train_encoded))

y_train_cat = to_categorical(y_train_encoded, num_classes)
y_test_cat = to_categorical(y_test_encoded, num_classes)

print("Target Converted to One-Hot Encoding")

# ============================================
# STEP 7: Build ANN Model
# ============================================

model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(num_classes, activation='softmax'))

print("ANN Model Architecture Created")

# ============================================
# STEP 8: Compile Model
# ============================================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model Compiled Successfully")

# ============================================
# STEP 9: Train Model
# ============================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train_cat,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

print("Model Training Completed")

# ============================================
# STEP 10: Evaluate Model
# ============================================

loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred_classes))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_encoded, y_pred_classes))

# ============================================
# STEP 11: Save Model
# ============================================

model.save("models/ann_obesity_model.keras")

print("\nModel Saved Successfully")

# ============================================
# STEP 12: Save Metadata for API
# ============================================

metadata = {
    "num_features": X_train.shape[1],
    "num_classes": num_classes
}

joblib.dump(metadata, "models/model_metadata.pkl")

print("Model metadata saved")

print("\nANN TRAINING COMPLETED SUCCESSFULLY")