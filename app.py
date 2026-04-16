import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. SETTINGS & ASSETS ---
st.set_page_config(page_title="Smartphone Addiction Predictor", layout="centered")

@st.cache_resource
def load_assets():
    # Ensure these files are in the same folder as app.py
    model = joblib.load('smartphone_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error("Could not load model/scaler files. Make sure 'model.pkl' and 'scaler.pkl' are in your repository.")
    st.stop()

# --- 2. USER INTERFACE ---
st.title("📱 Smartphone Addiction Analyzer")
st.markdown("Enter your daily usage habits below to predict addiction risk level.")

with st.form("usage_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=35, value=25)
        daily_screen_time = st.number_input("Daily Screen Time (Hours)", 3.0, 12.0, 7.5)
        social_media = st.number_input("Social Media Usage (Hours)", 0.5, 6.0, 3.0)
        gaming = st.number_input("Gaming Hours", 0.0, 4.0, 2.0)
    
    with col2:
        work_study = st.number_input("Work/Study Hours", 0.5, 6.0, 3.0)
        sleep = st.number_input("Sleep Hours", 4.5, 9.0, 7.0)
        notifications = st.number_input("Notifications per Day", 20, 250, 130)
        weekend_time = st.number_input("Weekend Screen Time (Hours)", 3.5, 15.0, 9.0)

    submit = st.form_submit_with_button("Analyze Results")

# --- 3. LOGIC & PREDICTION ---
if submit:
    # A. Feature Engineering (Match your notebook logic)
    total_screen_time = daily_screen_time + weekend_time
    entertainment_load = social_media + gaming
    social_ratio = social_media / (daily_screen_time + 1e-6)

    # B. Create DataFrame with EXACT training names and order
    # Your simplified model uses these 6 columns specifically:
    feature_names = [
        'daily_screen_time_hours', 
        'social_media_hours', 
        'total_screen_time', 
        'weekend_screen_time', 
        'entertainment_load', 
        'social_ratio'
    ]
    
    input_data = pd.DataFrame([[
        daily_screen_time, 
        social_media, 
        total_screen_time, 
        weekend_time, 
        entertainment_load, 
        social_ratio
    ]], columns=feature_names)

    # C. Transform and Predict
    try:
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)
        probability = model.predict_proba(scaled_input)[0][1]

        # --- 4. DISPLAY RESULTS ---
        st.divider()
        if prediction[0] == 1:
            st.error(f"### Prediction: High Risk of Addiction")
        else:
            st.success(f"### Prediction: Low Risk of Addiction")
            
        st.write(f"**Confidence Level:** {probability:.2%}")
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.markdown("---")
st.caption("Model based on 2026 Smartphone Usage Analysis Dataset.")
