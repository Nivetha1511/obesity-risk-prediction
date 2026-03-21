# ============================================
# FINAL FIXED ANN MODEL TRAINING
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
from sklearn.utils.class_weight import compute_class_weight

# ============================================
# STEP 1: Load Data
# ============================================

X_train_df = pd.read_csv("data/processed/X_train.csv")
X_test_df = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").values.flatten()
y_test = pd.read_csv("data/processed/y_test.csv").values.flatten()

print("Data Loaded Successfully")

# ============================================
# STEP 2: Convert to Numpy
# ============================================

X_train = X_train_df.values
X_test = X_test_df.values

# ============================================
# STEP 3: Scale Features
# ============================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")

print("Scaler Saved")

# ============================================
# STEP 4: Encode Target
# ============================================

encoder = LabelEncoder()

y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

joblib.dump(encoder, "models/target_encoder.pkl")

print("Encoder Saved")

# ============================================
# STEP 5: One-hot Encoding
# ============================================

num_classes = len(np.unique(y_train_encoded))

y_train_cat = to_categorical(y_train_encoded, num_classes)
y_test_cat = to_categorical(y_test_encoded, num_classes)

# ============================================
# STEP 6: HANDLE IMBALANCE (VERY IMPORTANT)
# ============================================

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_encoded),
    y=y_train_encoded
)

class_weights = dict(enumerate(class_weights))

print("Class Weights:", class_weights)

# ============================================
# STEP 7: Build BETTER MODEL
# ============================================

model = Sequential()

model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.4))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))

print("Model Built")

# ============================================
# STEP 8: Compile
# ============================================

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================
# STEP 9: Train
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
    class_weight=class_weights,  # 🔥 KEY FIX
    callbacks=[early_stop],
    verbose=1
)

print("Training Completed")

# ============================================
# STEP 10: Evaluate
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
print("🔥 FINAL MODEL TRAINING COMPLETE")