import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# --- 1. SETUP & LOADING ---
st.set_page_config(page_title="Smartphone Addiction Predictor", layout="centered")

@st.cache_resource
def load_assets():
    # Ensure these files are in your GitHub repo
    model = joblib.load('smartphone_model.pkl')
    # Note: If your scaler was fitted on the full 18 features, 
    # we need to be careful with the input shape.
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error("Error loading model files. Ensure 'smartphone_model.pkl' and 'scaler.pkl' are in the repository.")
    st.stop()

# --- 2. USER INTERFACE ---
st.title("📱 Smartphone Addiction Analyzer")
st.write("Fill in your usage details to get a risk assessment.")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        daily_screen_time = st.number_input("Daily Screen Time (Hours)", 3.0, 12.0, 7.5)
        social_media = st.number_input("Social Media Usage (Hours)", 0.5, 6.0, 3.0)
        weekend_time = st.number_input("Weekend Screen Time (Hours)", 3.5, 15.0, 9.0)
    
    with col2:
        gaming = st.number_input("Gaming Hours", 0.0, 4.0, 1.0)
        # These features were dropped in your final model but might be needed 
        # for calculation or if the scaler expects them.
        st.info("The model focuses on your screen time and entertainment habits.")

    submit = st.form_submit_button("Predict Addiction Risk")

# --- 3. PREDICTION LOGIC ---
if submit:
    # A. Feature Engineering (Must match your notebook exactly)
    total_screen_time = daily_screen_time + weekend_time
    entertainment_load = social_media + gaming
    social_ratio = social_media / (daily_screen_time + 1e-6)

    # B. Construct the feature set
    # Your 'Selected features' list:
    feature_names = [
        'daily_screen_time_hours', 
        'social_media_hours', 
        'total_screen_time', 
        'weekend_screen_time', 
        'entertainment_load', 
        'social_ratio'
    ]
    
    input_df = pd.DataFrame([[
        daily_screen_time, 
        social_media, 
        total_screen_time, 
        weekend_time, 
        entertainment_load, 
        social_ratio
    ]], columns=feature_names)

    try:
        # C. Transform and Predict
        # If your scaler expects 6 features (scaler_simple), this will work:
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)
        probability = model.predict_proba(scaled_input)[0][1]

        # --- 4. DISPLAY RESULTS ---
        st.divider()
        if prediction[0] == 1:
            st.error(f"### Result: High Risk of Addiction")
            st.progress(probability)
            st.write(f"Confidence: {probability:.1%}")
        else:
            st.success(f"### Result: Low Risk of Addiction")
            st.progress(probability)
            st.write(f"Risk Level: {probability:.1%}")

    except ValueError as e:
        st.error("Feature Mismatch: Your scaler expects more columns than provided.")
        st.info("To fix: Re-save your scaler in the notebook using ONLY the 6 selected features.")
