# ==============================================================================
# Crop Recommendation System - Streamlit Web Application
# B.Tech 7th Sem - Advanced Data Science Laboratory
# ==============================================================================

import streamlit as st
import pandas as pd
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="centered"
)

# Application Header
st.title("🌱 Crop Recommendation System")
st.markdown("Enter soil nutrients and climatic parameters to predict the most suitable crop for cultivation.")
st.write("---")

# Check if trained model file exists
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("⚠️ Model file 'model.pkl' not found! Please run 'python train_model.py' first.")
else:
    # Load model
    model = joblib.load(MODEL_PATH)

    st.subheader("📊 Enter Soil & Weather Parameters")

    # Arrange input fields into two columns
    col1, col2 = st.columns(2)

    with col1:
        n = st.number_input("Nitrogen (N) ratio in soil:", min_value=0.0, max_value=150.0, value=90.0, step=1.0)
        p = st.number_input("Phosphorus (P) ratio in soil:", min_value=0.0, max_value=150.0, value=42.0, step=1.0)
        k = st.number_input("Potassium (K) ratio in soil:", min_value=0.0, max_value=210.0, value=43.0, step=1.0)
        ph = st.number_input("Soil pH value (0 - 14):", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

    with col2:
        temperature = st.number_input("Temperature (°C):", min_value=0.0, max_value=60.0, value=20.88, step=0.1)
        humidity = st.number_input("Humidity (%):", min_value=0.0, max_value=100.0, value=82.00, step=0.1)
        rainfall = st.number_input("Rainfall (mm):", min_value=0.0, max_value=500.0, value=202.94, step=1.0)

    st.write("")
    # Recommendation Button
    if st.button("🌾 Recommend Crop", type="primary", use_container_width=True):
        # Create DataFrame matching model training features
        feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        input_df = pd.DataFrame([[n, p, k, temperature, humidity, ph, rainfall]], columns=feature_names)
        
        # Prediction
        prediction = model.predict(input_df)[0]
        
        # Display Results
        st.success(f"### 🎉 Recommended Crop: **{prediction.upper()}**")
        st.info(f"Based on your soil nutrients (N: {n}, P: {p}, K: {k}), pH ({ph}), "
                f"temperature ({temperature}°C), humidity ({humidity}%), and rainfall ({rainfall} mm), "
                f"**{prediction.capitalize()}** is the best suited crop for optimal yield.")
