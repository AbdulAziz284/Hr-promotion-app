import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

st.set_page_config(page_title="HR Promotion Predictor", layout="wide")

# ---------- Load Model and Encoders ----------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        encoders = joblib.load("encoders.pkl")
        return model, encoders, True
    except Exception:
        st.warning("⚠️ Model loading failed. Running in demo mode.")
        return None, None, False

model, encoders, is_real_model = load_model()

# ---------- Input Fields ----------
def get_user_input():
    st.header("Enter Employee Details")
    col1, col2 = st.columns(2)

    with col1:
        department = st.selectbox("Department", ['Sales', 'Operations', 'Technology', 'Analytics', 'HR', 'Finance'])
        education = st.selectbox("Education Level", ['Bachelor’s', 'Master’s', 'PhD'])
        gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
        no_of_trainings = st.slider("Number of Trainings", 1, 10, 3)
        previous_year_rating = st.slider("Previous Year Rating", 1.0, 5.0, 3.0, 0.5)
    
    with col2:
        avg_training_score = st.slider("Average Training Score", 40, 100, 60)
        length_of_service = st.slider("Length of Service (years)", 1, 30, 5)
        age = st.slider("Age", 18, 60, 30)
        awards_won = st.selectbox("Awards Won?", [0, 1])
        KPIs_met = st.selectbox("KPIs Met (>80%)?", [0, 1])

    user_data = {
        'department': department,
        'education': education,
        'gender': gender,
        'no_of_trainings': no_of_trainings,
        'previous_year_rating': previous_year_rating,
        'avg_training_score': avg_training_score,
        'length_of_service': length_of_service,
        'age': age,
        'awards_won': awards_won,
        'KPIs_met': KPIs_met
    }
    return pd.DataFrame([user_data])

# ---------- Encode Inputs ----------
def preprocess_input(input_df, encoders):
    input_df = input_df.copy()
    for col in ['department', 'education', 'gender']:
        encoder = encoders.get(col)
        if encoder:
            input_df[col] = encoder.transform(input_df[[col]])
    return input_df

# ---------- Prediction ----------
def predict_promotion(model, input_df):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    return prediction, probability

# ---------- Main App Logic ----------
st.title("🧠 HR Promotion Prediction App")
st.markdown("Use AI to determine whether an employee is likely to be promoted based on past HR data.")

input_df = get_user_input()

if st.button("🚀 Predict Promotion"):
    if is_real_model:
        encoded_input = preprocess_input(input_df, encoders)
        result, prob = predict_promotion(model, encoded_input)
    else:
        result = np.random.choice([0, 1])
        prob = np.random.uniform(0.4, 0.95)

    st.subheader("🎯 Prediction Result")
    if result == 1:
        st.success(f"✅ This employee is **likely to be promoted** (Confidence: {prob:.2%})")
    else:
        st.error(f"❌ This employee is **not likely to be promoted** (Confidence: {1 - prob:.2%})")

# ---------- Insights ----------
st.markdown("---")
st.subheader("📊 Promotion Trends by Department (Simulated Data)")
sample_data = pd.DataFrame({
    "Department": ['Sales', 'Operations', 'Technology', 'Analytics', 'HR', 'Finance'],
    "Promotion Rate (%)": [25, 18, 30, 35, 20, 22]
})
fig = px.bar(sample_data, x="Department", y="Promotion Rate (%)", color="Promotion Rate (%)", text="Promotion Rate (%)")
st.plotly_chart(fig, use_container_width=True)

st.caption("Made with ❤️ by Abdul Aziz")
