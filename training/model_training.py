# ============================================
# ANN MODEL TRAINING
# ============================================

import numpy as np
import pandas as pd
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ============================================
# STEP 1: Load Processed Data
# ============================================

X_train = pd.read_csv("data/processed/X_train.csv").values
X_test = pd.read_csv("data/processed/X_test.csv").values
y_train = pd.read_csv("data/processed/y_train.csv").values.flatten()
y_test = pd.read_csv("data/processed/y_test.csv").values.flatten()

print("Data Loaded Successfully")

# ============================================
# STEP 2: One-Hot Encode Target
# ============================================

num_classes = 7

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print("Target Converted to One-Hot Encoding")

# ============================================
# STEP 3: Build ANN Model
# ============================================

model = Sequential()

# Input Layer + Hidden Layer 1
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3))

# Hidden Layer 2
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))

# Output Layer
model.add(Dense(num_classes, activation='softmax'))

print("ANN Model Architecture Created")

# ============================================
# STEP 4: Compile Model
# ============================================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model Compiled Successfully")

# ============================================
# STEP 5: Train Model
# ============================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

print("Model Training Completed")

# ============================================
# STEP 6: Evaluate Model
# ============================================

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))

# ============================================
# STEP 7: Save Trained Model
# ============================================

os.makedirs("models", exist_ok=True)

model.save("models/ann_obesity_model.keras")

print("\nModel Saved Successfully")
print("ANN TRAINING COMPLETED")