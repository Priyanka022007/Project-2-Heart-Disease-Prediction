import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model (assumes you have a model saved as 'heart_model.pkl')
with open("HeartDisease.pkl", "rb") as file:
    model = pickle.load(file)

st.title("❤️ Heart Disease Prediction App")

st.markdown("Enter the patient data below:")

# Input fields
age = st.number_input("Age", 18, 100, 45)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200, 120)
chol = st.number_input("Serum Cholesterol (chol)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved (thalach)", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, 0.1)
slope = st.selectbox("Slope of the Peak (slope)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored (ca)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# Prediction
if st.button("Predict"):
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(features)
    result = "Positive for Heart Disease 💔" if prediction[0] == 1 else "No Heart Disease ❤️"
    st.subheader(f"Prediction: {result}")

