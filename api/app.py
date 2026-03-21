import numpy as np
import joblib
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Global objects
model = None
scaler = None
feature_names = None

# Labels
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

def load_objects():
    global model, scaler, feature_names

    if model is None:
        model = load_model("models/ann_obesity_model.keras", compile=False)

    if scaler is None:
        scaler = joblib.load("models/scaler.pkl")

    if feature_names is None:
        feature_names = joblib.load("models/feature_names.pkl")

    print("Objects loaded successfully")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        load_objects()

        data = request.get_json()

        # ✅ Create dictionary
        input_dict = {
            "Gender": data["Gender"],
            "Age": data["Age"],
            "Height": data["Height"],
            "Weight": data["Weight"],
            "family_history_with_overweight": data["family_history_with_overweight"],
            "FAVC": data["FAVC"],
            "FCVC": data["FCVC"],
            "NCP": data["NCP"],
            "CAEC": data["CAEC"],
            "SMOKE": data["SMOKE"],
            "CH2O": data["CH2O"],
            "SCC": data["SCC"],
            "FAF": data["FAF"],
            "TUE": data["TUE"],
            "CALC": data["CALC"],
            "MTRANS": data["MTRANS"]
        }

        # ✅ Align order with training
        ordered_input = [input_dict[col] for col in feature_names]

        input_array = np.array(ordered_input).reshape(1, -1)

        # Scale
        input_scaled = scaler.transform(input_array)

        # Predict
        prediction = model.predict(input_scaled)

        print("RAW PREDICTION:", prediction)

        predicted_class = np.argmax(prediction, axis=1)
        class_index = int(predicted_class[0])

        risk_level = obesity_labels[class_index]
        confidence = round(float(np.max(prediction)), 2)

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