import numpy as np
import joblib
import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ============================================
# Setup Base Directory (important for Render)
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "..", "models", "ann_obesity_model.keras")
scaler_path = os.path.join(BASE_DIR, "..", "models", "scaler.pkl")
encoder_path = os.path.join(BASE_DIR, "..", "models", "target_encoder.pkl")

# ============================================
# Load Model and Preprocessing Objects
# ============================================

model = load_model(model_path, compile=False)
scaler = joblib.load(scaler_path)
target_encoder = joblib.load(encoder_path)

print("Model and preprocessing objects loaded successfully")

# ============================================
# Home Route
# ============================================

@app.route("/")
def home():
    return "Obesity Risk Prediction API is running"

# ============================================
# Prediction Route
# ============================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        # Convert input values to float
        input_data = [
            float(data["Gender"]),
            float(data["Age"]),
            float(data["Height"]),
            float(data["Weight"]),
            float(data["family_history_with_overweight"]),
            float(data["FAVC"]),
            float(data["FCVC"]),
            float(data["NCP"]),
            float(data["CAEC"]),
            float(data["SMOKE"]),
            float(data["CH2O"]),
            float(data["SCC"]),
            float(data["FAF"]),
            float(data["TUE"]),
            float(data["CALC"]),
            float(data["MTRANS"])
        ]

        # Convert to numpy array
        input_array = np.array(input_data).reshape(1, -1)

        # Scale input
        input_scaled = scaler.transform(input_array)

        # Predict
        prediction = model.predict(input_scaled)

        predicted_class = np.argmax(prediction, axis=1)[0]

        risk_level = target_encoder.inverse_transform([predicted_class])[0]

        confidence = float(np.max(prediction))

        return jsonify({
            "predicted_class_index": int(predicted_class),
            "predicted_risk_level": risk_level,
            "confidence": confidence
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })

# ============================================
# Run Flask App
# ============================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(host="0.0.0.0", port=port)