# ============================================

# FLASK API FOR OBESITY PREDICTION

# ============================================

import numpy as np
import joblib
import os
from flask import Flask, request, jsonify

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Initialize Flask app

app = Flask(__name__)

# ============================================

# Recreate ANN Model Architecture

# (Same as training model)

# ============================================

model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(16,)))
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(7, activation='softmax'))

# Load trained weights

model.load_weights("models/ann_obesity_model.keras")

# Load preprocessing objects

scaler = joblib.load("models/scaler.pkl")
target_encoder = joblib.load("models/target_encoder.pkl")

print("Model and Preprocessing Objects Loaded Successfully")

# ============================================

# Prediction Endpoint

# ============================================

@app.route("/predict", methods=["POST"])
def predict():

```
try:
    data = request.json

    # Extract input features in EXACT same order as training
    input_data = [
        data["Gender"],
        data["Age"],
        data["Height"],
        data["Weight"],
        data["family_his"],
        data["FAVC"],
        data["FCVC"],
        data["NCP"],
        data["CAEC"],
        data["SMOKE"],
        data["CH2O"],
        data["SCC"],
        data["FAF"],
        data["TUE"],
        data["CALC"],
        data["MTRANS"]
    ]

    input_array = np.array(input_data).reshape(1, -1)

    # Scale features
    input_scaled = scaler.transform(input_array)

    # Predict
    prediction = model.predict(input_scaled)
    predicted_class = np.argmax(prediction, axis=1)

    # Decode class label
    risk_level = target_encoder.inverse_transform(predicted_class)[0]

    return jsonify({
        "predicted_class_index": int(predicted_class[0]),
        "predicted_risk_level": risk_level,
        "confidence": float(np.max(prediction))
    })

except Exception as e:
    return jsonify({"error": str(e)})
```

# ============================================

# Run App

# ============================================

if **name** == "**main**":
port = int(os.environ.get("PORT", 10000))
app.run(host="0.0.0.0", port=port)
