import requests

# Your Render API URL
url = "https://obesity-risk-prediction-cj25.onrender.com/predict"

# ✅ Test Case 1 (Healthy)
data = {
    "Gender": 1,
    "Age": 25,
    "Height": 1.60,
    "Weight": 95,
    "family_history_with_overweight": 1,
    "FAVC": 1,
    "FCVC": 1,
    "NCP": 4,
    "CAEC": 3,
    "SMOKE": 0,
    "CH2O": 1,
    "SCC": 0,
    "FAF": 0,
    "TUE": 2,
    "CALC": 3,
    "MTRANS": 3
}
# ✅ Send request
response = requests.post(url, json=data)

print("Status Code:", response.status_code)

try:
    result = response.json()
    print("\nAPI Response:")
    print("Predicted Class Index:", result.get("predicted_class_index"))
    print("Predicted Risk Level:", result.get("predicted_risk_level"))
    print("Confidence:", result.get("confidence"))
except:
    print("Raw Response:", response.text)