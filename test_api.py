import requests

url = "https://obesity-risk-prediction-cj25.onrender.com/predict"

data = {
 "Gender": 1,
 "Age": 30,
 "Height": 1.60,
 "Weight": 110,
 "family_history_with_overweight": 1,
 "FAVC": 1,
 "FCVC": 1,
 "NCP": 4,
 "CAEC": 3,
 "SMOKE": 1,
 "CH2O": 1,
 "SCC": 0,
 "FAF": 0,
 "TUE": 6,
 "CALC": 2,
 "MTRANS": 3
}
response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("\nRAW RESPONSE:")
print(response.text)