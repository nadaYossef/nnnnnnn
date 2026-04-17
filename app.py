import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# 1. Load the pre-trained Gradient Boosting model
model = joblib.load('gbc_model.pkl')

st.title("📱 Smartphone Addiction Diagnostic Tool")
st.write("Enter your usage details to see your addiction risk level and personalized feedback.")

# 2. User Input Sidebar
st.sidebar.header("Usage Metrics")
daily_hours = st.sidebar.slider("Daily Screen Time (Hours)", 0.0, 24.0, 5.0)
social_hours = st.sidebar.slider("Social Media (Hours)", 0.0, 24.0, 2.0)
sleep_hours = st.sidebar.slider("Sleep Hours", 0.0, 12.0, 7.0)
notifications = st.sidebar.number_input("Notifications per Day", 0, 1000, 50)
app_opens = st.sidebar.number_input("App Opens per Day", 0, 500, 30)

# Derived Features (Ensure these match your training columns)
# Note: Adjust these calculation logic to match your 'model-and-eda.ipynb' feature engineering
avg_daily = daily_hours # Simplified for example
screen_sleep_ratio = daily_hours / (sleep_hours + 0.1)

# Prepare input data
input_data = pd.DataFrame([[
    daily_hours, social_hours, sleep_hours, 
    daily_hours * 1.2, # dummy for weekend_screen_time
    avg_daily, 
    (daily_hours - social_hours)/daily_hours if daily_hours > 0 else 0, # productivity_ratio
    social_hours / daily_hours if daily_hours > 0 else 0, # social_media_ratio
    0.1, # gaming_ratio
    notifications / daily_hours if daily_hours > 0 else 0,
    app_opens / daily_hours if daily_hours > 0 else 0,
    screen_sleep_ratio,
    1, 0, 0 # One-hot encoded gender (Male/Other/Female)
]], columns=model.feature_names_in_)

# 3. Prediction and Probability
if st.button("Analyze My Usage"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100
    
    # Define Risk Level
    if probability < 30:
        level = "Low"
        color = "green"
    elif 30 <= probability < 70:
        level = "Moderate"
        color = "orange"
    else:
        level = "High"
        color = "red"

    st.subheader(f"Risk Level: :{color}[{level}]")
    st.metric("Addiction Probability", f"{probability:.1f}%")

    # 4. Explanation (Feedback)
    st.write("### 🔍 Why did I get this score?")
    # Using simple logic for explanation (or you can use SHAP here)
    if daily_hours > 8:
        st.warning("- Your screen time is significantly above the healthy average of 6-7 hours.")
    if screen_sleep_ratio > 1.5:
        st.warning("- Your phone use is heavily cutting into your recovery/sleep time.")
    if notifications > 100:
        st.warning("- High notification frequency suggests frequent dopamine spikes and focus fragmentation.")

    # 5. Recommended Steps
    st.write("### 🚀 Recommended Steps")
    if level == "High":
        st.info("1. **Digital Detox**: Try a 24-hour break this weekend.\n"
                "2. **Greyscale Mode**: Turn your screen to black and white to make apps less stimulating.\n"
                "3. **Notification Cull**: Disable all non-human notifications (keep only calls/texts).")
    elif level == "Moderate":
        st.info("1. **App Timers**: Set a 30-min limit for your most-used social app.\n"
                "2. **No-Phone Zones**: Keep the phone out of the bedroom.")
    else:
        st.success("Great job! Keep maintaining your balance. Consider a weekly 'Screen-Free Sunday' to stay on track.")
