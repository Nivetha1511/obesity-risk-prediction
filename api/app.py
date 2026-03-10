# ============================================
# FLASK API FOR OBESITY PREDICTION
# ============================================

import numpy as np
import joblib
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# ============================================
# Load Saved Model and Preprocessing Objects
# ============================================

model = load_model("models/ann_obesity_model.keras")
scaler = joblib.load("models/scaler.pkl")
target_encoder = joblib.load("models/target_encoder.pkl")

print("Model and Preprocessing Objects Loaded Successfully")

# ============================================
# Prediction Endpoint
# ============================================

@app.route("/predict", methods=["POST"])
def predict():

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

# ============================================
# Run App
# ============================================

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)