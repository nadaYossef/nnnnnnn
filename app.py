import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# --- Update this section in your app.py ---

# 1. Capture additional necessary inputs from sidebar
gaming_hours = st.sidebar.slider("Gaming (Hours)", 0.0, 24.0, 0.5)
is_weekend = st.sidebar.checkbox("Is today a weekend?")

# 2. Logic to handle "Gender" correctly for the model
gender = st.sidebar.selectbox("Gender", ["Female", "Male", "Other"])
g_male = 1 if gender == "Male" else 0
g_other = 1 if gender == "Other" else 0
# (Female is the reference/0-baseline, so it doesn't need its own column)

# 3. Data Preparation
if st.button("Analyze My Usage"):
    # Calculate derived features to match your model's training logic
    # Adding 0.01 to denominators to prevent division by zero errors
    safe_daily = daily_hours if daily_hours > 0 else 0.01
    
    # Matching the 13 features found in gbc_model.pkl
    data_row = [
        daily_hours,                                # daily_screen_time_hours
        social_hours,                               # social_media_hours
        sleep_hours,                                # sleep_hours
        daily_hours * 1.2 if is_weekend else daily_hours, # weekend_screen_time (estimate)
        daily_hours,                                # average_daily_screen_time
        (daily_hours - social_hours) / safe_daily,  # productivity_ratio
        social_hours / safe_daily,                  # social_media_ratio
        gaming_hours / safe_daily,                  # gaming_ratio
        notifications / safe_daily,                 # notifications_per_screen_hour
        app_opens / safe_daily,                     # app_opens_per_screen_hour
        daily_hours / (sleep_hours + 0.1),          # screen_time_sleep_ratio
        g_male,                                     # gender_Male
        g_other                                     # gender_Other
    ]

    # Create DataFrame with the exact columns the model expects
    input_df = pd.DataFrame([data_row], columns=model.feature_names_in_)
    
    # Now run prediction
    prediction = model.predict(input_df)[0]
    # ... rest of your prediction logic
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
