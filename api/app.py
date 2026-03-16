import numpy as np
import joblib
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access


# Global objects (lazy loading)
model = None
scaler = None
target_encoder = None


# Human-readable obesity labels
obesity_labels = {
    0: "Insufficient Weight",
    1: "Normal Weight",
    2: "Overweight Level I",
    3: "Overweight Level II",
    4: "Obesity Type I",
    5: "Obesity Type II",
    6: "Obesity Type III"
}


@app.route("/")
def home():
    return "Obesity Risk Prediction API is running"


# Load model and preprocessing objects only when needed
def load_objects():
    global model, scaler, target_encoder

    if model is None:
        model = load_model("models/ann_obesity_model.keras", compile=False)

    if scaler is None:
        scaler = joblib.load("models/scaler.pkl")

    if target_encoder is None:
        target_encoder = joblib.load("models/target_encoder.pkl")

    print("Model and preprocessing objects loaded successfully")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        load_objects()

        data = request.get_json()

        input_data = [
            data["Gender"],
            data["Age"],
            data["Height"],
            data["Weight"],
            data["family_history_with_overweight"],
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

        # Scale input
        input_scaled = scaler.transform(input_array)

        # Model prediction
        prediction = model.predict(input_scaled)

        predicted_class = np.argmax(prediction, axis=1)
        class_index = int(predicted_class[0])

        risk_level = obesity_labels[class_index]
        confidence = float(np.max(prediction))

        return jsonify({
            "predicted_class_index": class_index,
            "predicted_risk_level": risk_level,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)