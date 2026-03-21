const API_URL = "https://obesity-risk-prediction-cj25.onrender.com/predict";


// Go from login page to questionnaire
function goToForm(){

let name = document.getElementById("name").value;
let mobile = document.getElementById("mobile").value;
let email = document.getElementById("email").value;

if(name === "" || mobile === "" || email === ""){
alert("Please fill all login details");
return;
}

localStorage.setItem("user_name", name);
localStorage.setItem("user_mobile", mobile);
localStorage.setItem("user_email", email);

window.location.href = "form.html";
}



// Send data to backend
function predictRisk(){

let data = {

Gender: parseInt(document.getElementById("Gender").value),
Age: parseInt(document.getElementById("Age").value),
Height: parseFloat(document.getElementById("Height").value),
Weight: parseFloat(document.getElementById("Weight").value),

family_history_with_overweight: parseInt(document.getElementById("family_history_with_overweight").value),

FAVC: parseInt(document.getElementById("FAVC").value),
FCVC: parseInt(document.getElementById("FCVC").value),

NCP: parseInt(document.getElementById("NCP").value),
CAEC: parseInt(document.getElementById("CAEC").value),

SMOKE: parseInt(document.getElementById("SMOKE").value),

CH2O: parseInt(document.getElementById("CH2O").value),

SCC: parseInt(document.getElementById("SCC").value),

FAF: parseInt(document.getElementById("FAF").value),

TUE: parseInt(document.getElementById("TUE").value),

CALC: parseInt(document.getElementById("CALC").value),

MTRANS: parseInt(document.getElementById("MTRANS").value)

};


fetch(API_URL,{
method:"POST",
headers:{
"Content-Type":"application/json"
},
body:JSON.stringify(data)
})
.then(response => response.json())
.then(result => {

console.log("API Response:", result);  // 👈 This will now be visible

localStorage.setItem("risk", result.predicted_risk_level);
localStorage.setItem("confidence", result.confidence);

// ⏳ Delay redirect by 5 seconds
setTimeout(() => {
    window.location.href = "result.html";
}, 5000);

})
.catch(error => {

console.error("API Error:", error);

alert("Prediction failed. Please check internet connection or API.");

});

}



// Display result when result page loads
window.onload = function(){

if(document.getElementById("risk")){

let risk = localStorage.getItem("risk");
let confidence = localStorage.getItem("confidence");

if(!risk){
document.getElementById("risk").innerText = "No prediction found.";
return;
}

document.getElementById("risk").innerText = "Risk Level: " + risk;

document.getElementById("confidence").innerText =
"Confidence: " + (confidence*100).toFixed(2) + "%";


let recommendations = document.getElementById("recommendations");

recommendations.innerHTML = `
<li>Increase daily physical activity</li>
<li>Maintain a balanced diet rich in vegetables</li>
<li>Reduce high calorie foods</li>
<li>Drink sufficient water daily</li>
<li>Maintain regular sleep schedule</li>
<li>Consult healthcare professionals if necessary</li>
`;

}

}