import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("final_model.pkl")

st.sidebar.image("lung.png", width=200)

st.sidebar.header("About This App")

st.sidebar.write(
"""
This application predicts the **risk of lung cancer**
using a Machine Learning model trained on medical data.
"""
) 

st.title("Lung Cancer Risk Prediction")
st.write("Fill the details below to predict lung cancer risk.")

# Function to convert Yes/No → dataset encoding
def yes_no(value):
    return 2 if value == "Yes" else 1

# Gender input
gender = st.selectbox("Gender", ["Male", "Female"])
gender = 1 if gender == "Male" else 0

age = st.number_input("Age", min_value=1, max_value=100)

smoking = yes_no(st.selectbox("Smoking", ["No","Yes"]))
yellow_fingers = yes_no(st.selectbox("Yellow Fingers", ["No","Yes"]))
anxiety = yes_no(st.selectbox("Anxiety", ["No","Yes"]))
peer_pressure = yes_no(st.selectbox("Peer Pressure", ["No","Yes"]))
chronic_disease = yes_no(st.selectbox("Chronic Disease", ["No","Yes"]))
fatigue = yes_no(st.selectbox("Fatigue", ["No","Yes"]))
allergy = yes_no(st.selectbox("Allergy", ["No","Yes"]))
wheezing = yes_no(st.selectbox("Wheezing", ["No","Yes"]))
alcohol = yes_no(st.selectbox("Alcohol Consuming", ["No","Yes"]))
coughing = yes_no(st.selectbox("Coughing", ["No","Yes"]))
shortness_breath = yes_no(st.selectbox("Shortness of Breath", ["No","Yes"]))
swallowing = yes_no(st.selectbox("Difficulty Swallowing", ["No","Yes"]))
chest_pain = yes_no(st.selectbox("Chest Pain", ["No","Yes"]))

# Prediction button
if st.button("Predict"):

    features = np.array([[gender, age, smoking, yellow_fingers, anxiety,
                          peer_pressure, chronic_disease, fatigue,
                          allergy, wheezing, alcohol, coughing,
                          shortness_breath, swallowing, chest_pain]])

    prediction = model.predict(features)
    probability = model.predict_proba(features)

    if prediction[0] == 1:
        st.error("⚠ High Risk of Lung Cancer")
    else:
        st.success("✅ Low Risk of Lung Cancer")

    st.write("Prediction Probability:", probability)