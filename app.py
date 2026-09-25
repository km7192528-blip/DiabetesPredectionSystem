import streamlit as st
import pickle
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="AI Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>

.stApp {
    background-image: url("https://images.unsplash.com/photo-1576091160399-112ba8d25d1f");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

.main {
    background-color: rgba(0,0,0,0.6);
    padding: 20px;
    border-radius: 15px;
}

h1, h2, h3 {
    color: white;
    text-align: center;
}

label {
    color: white !important;
    font-weight: bold;
}

[data-testid="stMetricValue"] {
    color: #00ff99;
}

</style>
""", unsafe_allow_html=True)

# Load Model
model = pickle.load(
    open("model/diabetes_model.pkl", "rb")
)

# Sidebar
st.sidebar.title("🩺 Project Information")

st.sidebar.success("Model Accuracy: ~82%")

st.sidebar.info("""
AI-Based Diabetes Prediction System

Algorithm Used:
• Random Forest Classifier

Dataset:
• Pima Indians Diabetes Dataset

Purpose:
• Early Diabetes Risk Detection
""")

# Main Title
st.markdown(
    """
    <h1>🩺 AI-Based Diabetes Prediction System</h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h4 style='text-align:center;color:white;'>
    Predict diabetes risk using Machine Learning
    </h4>
    """,
    unsafe_allow_html=True
)

st.write("")

# Input Fields

preg = st.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=0
)

glucose = st.number_input(
    "Glucose",
    min_value=0,
    max_value=300,
    value=100
)

bp = st.number_input(
    "Blood Pressure",
    min_value=0,
    max_value=200,
    value=70
)

skin = st.number_input(
    "Skin Thickness",
    min_value=0,
    max_value=100,
    value=20
)

insulin = st.number_input(
    "Insulin",
    min_value=0,
    max_value=900,
    value=80
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

dpf = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)

st.write("")

# Predict Button
if st.button("🔍 Predict Diabetes Risk"):

    data = np.array([
        [
            preg,
            glucose,
            bp,
            skin,
            insulin,
            bmi,
            dpf,
            age
        ]
    ])

    prediction = model.predict(data)

    probability = model.predict_proba(data)

    risk = probability[0][1] * 100

    st.subheader("📊 Risk Analysis")

    st.progress(int(risk))

    st.metric(
        label="Diabetes Risk Percentage",
        value=f"{risk:.2f}%"
    )

    st.write("")

    if prediction[0] == 1:

        st.error(
            f"⚠️ High Risk of Diabetes\n\nRisk Score: {risk:.2f}%"
        )

        st.warning(
            """
            Recommendation:
            • Consult a healthcare professional.
            • Maintain a healthy diet.
            • Exercise regularly.
            • Monitor blood sugar levels.
            """
        )

    else:

        st.success(
            f"✅ Low Risk of Diabetes\n\nRisk Score: {risk:.2f}%"
        )

        st.info(
            """
            Recommendation:
            • Continue healthy lifestyle habits.
            • Maintain balanced nutrition.
            • Stay physically active.
            """
        )

