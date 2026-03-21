import requests

url = "https://obesity-risk-prediction-cj25.onrender.com/predict"

data = {
 "Gender": 1,
 "Age": 21,
 "Height": 1.72,
 "Weight": 80,
 "family_history_with_overweight": 1,
 "FAVC": 1,
 "FCVC": 2,
 "NCP": 3,
 "CAEC": 1,
 "SMOKE": 0,
 "CH2O": 2,
 "SCC": 0,
 "FAF": 1,
 "TUE": 2,
 "CALC": 1,
 "MTRANS": 2
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("\nRAW RESPONSE:")
print(response.text)