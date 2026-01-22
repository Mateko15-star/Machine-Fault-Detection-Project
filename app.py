import streamlit as st
import numpy as np
import joblib

# Load saved objects
model = joblib.load("fault_detection_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("Electrical Motor Fault Detection")
st.subheader("Enter Motor Feature Values")

feature_names = [
    "max", "min", "mean", "sd", "rms",
    "skewness", "kurtosis", "crest", "form"
]

input_data = []

for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0)
    input_data.append(value)

# 🔽 PREDICTION BLOCK (ADD HERE)
if st.button("Predict Fault"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)

    result = label_encoder.inverse_transform(prediction)
    confidence = np.max(probabilities) * 100

    # ✅ ADD THIS DICTIONARY HERE
    fault_info = {
        "Normal": "Motor operating under healthy conditions.",
        "Ball_007_1": "Rolling element (ball) bearing defect causing impulsive vibrations.",
        "Inner_014_2": "Inner race bearing fault with higher severity.",
        "Outer_021_3": "Outer race bearing fault affecting load distribution."
    }

    # 🔽 DISPLAY RESULTS
    st.success(f"Predicted Fault Type: {result[0]}")
    st.info(f"Prediction Confidence: {confidence:.2f}%")
    st.info(f"Fault Description: {fault_info.get(result[0], 'No description available')}")
