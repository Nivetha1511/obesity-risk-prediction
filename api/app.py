import numpy as np
import joblib
import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained model
model = load_model("models/ann_obesity_model.keras", compile=False)

# Load preprocessing objects
scaler = joblib.load("models/scaler.pkl")
target_encoder = joblib.load("models/target_encoder.pkl")

print("Model and Preprocessing Objects Loaded Successfully")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

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
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)
        predicted_class = np.argmax(prediction, axis=1)

        risk_level = target_encoder.inverse_transform(predicted_class)[0]

        return jsonify({
            "predicted_class_index": int(predicted_class[0]),
            "predicted_risk_level": risk_level,
            "confidence": float(np.max(prediction))
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)