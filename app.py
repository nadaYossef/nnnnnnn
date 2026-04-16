import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    model = joblib.load('smartphone_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# --- UI ---
st.title("📱 Smartphone Addiction Analyzer")

with st.form("input_form"):
    daily_screen_time = st.number_input("Daily Screen Time (Hours)", 1.0, 15.0, 7.0)
    social_media = st.number_input("Social Media Usage (Hours)", 0.0, 10.0, 3.0)
    weekend_time = st.number_input("Weekend Screen Time (Hours)", 1.0, 20.0, 8.0)
    gaming = st.number_input("Gaming Hours", 0.0, 10.0, 1.0)
    
    submit = st.form_submit_button("Predict Risk")

# --- PREDICTION ---
if submit:
    # 1. Manual Feature Engineering
    total_screen_time = daily_screen_time + weekend_time
    entertainment_load = social_media + gaming
    social_ratio = social_media / (daily_screen_time + 1e-6)

    # 2. Match the 6 features in EXACT order
    input_df = pd.DataFrame([[
        daily_screen_time, 
        social_media, 
        total_screen_time, 
        weekend_time, 
        entertainment_load, 
        social_ratio
    ]], columns=['daily_screen_time_hours', 'social_media_hours', 'total_screen_time', 
                 'weekend_screen_time', 'entertainment_load', 'social_ratio'])

    # 3. Scale and Predict
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)
    
    if prediction[0] == 1:
        st.error("### Result: High Risk of Addiction")
    else:
        st.success("### Result: Low Risk of Addiction")
