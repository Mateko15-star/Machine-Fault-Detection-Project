import streamlit as st
import numpy as np
import joblib

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Electrical Motor Fault Detection",
    page_icon="⚙️",
    layout="centered"
)

# --------------------------------------------------
# Load trained model and preprocessing objects
# --------------------------------------------------
@st.cache_resource
def load_models():
    model = joblib.load("fault_detection_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder


model, scaler, label_encoder = load_models()

# --------------------------------------------------
# App title and description
# --------------------------------------------------
st.title("⚙️ Electrical Motor Fault Detection")
st.write(
    """
    This application predicts **bearing fault conditions** in electrical motors
    using machine learning based on statistical signal features.
    """
)

st.divider()

# --------------------------------------------------
# Feature input section
# --------------------------------------------------
st.subheader("🔢 Enter Motor Feature Values")

feature_names = [
    "max",
    "min",
    "mean",
    "sd",
    "rms",
    "skewness",
    "kurtosis",
    "crest",
    "form"
]

input_data = []

for feature in feature_names:
    value = st.number_input(
        label=f"{feature}",
        value=0.0,
        format="%.5f"
    )
    input_data.append(value)

st.divider()

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Predict Fault"):

    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)

    predicted_label = label_encoder.inverse_transform(prediction)[0]
    confidence = np.max(probabilities) * 100

    # --------------------------------------------------
    # Fault explanations
    # --------------------------------------------------
    fault_info = {
        "Normal": "Motor operating under healthy conditions.",
        "Ball_007_1": "Rolling element (ball) bearing defect causing impulsive vibrations.",
        "Inner_014_2": "Inner race bearing fault with increased severity.",
        "Outer_021_3": "Outer race bearing fault affecting load distribution."
    }

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------
    st.success(f"✅ Predicted Fault Type: **{predicted_label}**")
    st.info(f"📊 Prediction Confidence: **{confidence:.2f}%**")
    st.info(
        f"🛠 Fault Description: {fault_info.get(predicted_label, 'No description available.')}"
    )

    # --------------------------------------------------
    # Confidence interpretation
    # --------------------------------------------------
    if confidence < 60:
        st.warning(
            "⚠ Low prediction confidence. The fault may be in an early stage or overlap with other fault types."
        )
    elif confidence < 80:
        st.info(
            "ℹ Moderate confidence. Continuous monitoring is recommended."
        )
    else:
        st.success(
            "✅ High confidence prediction. Maintenance action may be required."
        )

    # --------------------------------------------------
    # Show probability distribution (optional but professional)
    # --------------------------------------------------
    st.subheader("📈 Fault Probability Distribution")

    prob_dict = {
        label_encoder.classes_[i]: round(probabilities[0][i] * 100, 2)
        for i in range(len(label_encoder.classes_))
    }

    st.json(prob_dict)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.caption(
    "Machine Learning–based Fault Diagnosis | Streamlit Deployment"
)
