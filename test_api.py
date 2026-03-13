import requests

url = "http://127.0.0.1:10000/predict"

data = {
 "Gender":1,
 "Age":21,
 "Height":1.7,
 "Weight":70,
 "family_his":1,
 "FAVC":1,
 "FCVC":2,
 "NCP":3,
 "CAEC":1,
 "SMOKE":0,
 "CH2O":2,
 "SCC":0,
 "FAF":1,
 "TUE":1,
 "CALC":0,
 "MTRANS":2
}

response = requests.post(url, json=data)

print(response.json())