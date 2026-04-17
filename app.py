import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Addiction Diagnostic Tool", page_icon="📱", layout="centered")

# 2. Load the Pre-trained Model
@st.cache_resource
def load_model():
    # Make sure 'gbc_model.pkl' is in the same folder as this script
    return joblib.load('gbc_model.pkl')

model = load_model()

# 3. Header
st.title("📱 Smartphone Addiction Diagnostic Tool")
st.write("Enter your daily habits below to calculate your risk probability.")
st.markdown("---")

# 4. Input Sidebar
st.sidebar.header("Usage Metrics")

daily_hours = st.sidebar.slider("Total Daily Screen Time (Hours)", 0.0, 24.0, 5.0)
social_hours = st.sidebar.slider("Social Media Usage (Hours)", 0.0, 24.0, 2.0)
gaming_hours = st.sidebar.slider("Gaming Usage (Hours)", 0.0, 24.0, 0.5)
sleep_hours = st.sidebar.slider("Typical Sleep (Hours)", 0.0, 12.0, 7.0)

st.sidebar.subheader("Engagement")
notifications = st.sidebar.number_input("Notifications per Day", min_value=0, value=50)
app_opens = st.sidebar.number_input("App Opens per Day", min_value=0, value=30)
is_weekend = st.sidebar.checkbox("Is this a Weekend/Holiday?")

st.sidebar.subheader("Demographics")
gender = st.sidebar.selectbox("Gender", ["Female", "Male", "Other"])

# 5. Main Logic & Prediction
if st.button("Generate My Risk Report"):
    
    # Pre-calculations to avoid division by zero
    safe_daily = daily_hours if daily_hours > 0 else 0.01
    safe_sleep = sleep_hours if sleep_hours > 0 else 0.01

    # Mapping Gender to match your One-Hot Encoded training columns
    g_male = 1 if gender == "Male" else 0
    g_other = 1 if gender == "Other" else 0

    # Building the 13-feature array in EXACT order of model.feature_names_in_
    data_row = [
        float(daily_hours),                                # 1. daily_screen_time_hours
        float(social_hours),                               # 2. social_media_hours
        float(sleep_hours),                                # 3. sleep_hours
        float(daily_hours * 1.3 if is_weekend else daily_hours), # 4. weekend_screen_time
        float(daily_hours),                                # 5. average_daily_screen_time
        float((daily_hours - social_hours) / safe_daily),  # 6. productivity_ratio
        float(social_hours / safe_daily),                  # 7. social_media_ratio
        float(gaming_hours / safe_daily),                  # 8. gaming_ratio
        float(notifications / safe_daily),                 # 9. notifications_per_screen_hour
        float(app_opens / safe_daily),                     # 10. app_opens_per_screen_hour
        float(daily_hours / safe_sleep),                   # 11. screen_time_sleep_ratio
        int(g_male),                                       # 12. gender_Male
        int(g_other)                                       # 13. gender_Other
    ]

    # Create DataFrame and ensure it matches the model's expected column names
    try:
        input_df = pd.DataFrame([data_row], columns=model.feature_names_in_)
        
        # Get Probability
        prob_percent = model.predict_proba(input_df)[0][1] * 100

        # 6. Displaying Results
        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)
        
        with col1:
            if prob_percent < 35:
                st.success("**Risk Level: Low**")
            elif 35 <= prob_percent < 70:
                st.warning("**Risk Level: Moderate**")
            else:
                st.error("**Risk Level: High**")
                
        with col2:
            st.metric("Probability", f"{prob_percent:.1f}%")

        # Short Feedback
        st.markdown("---")
        if prob_percent > 70:
            st.info("💡 **Tip:** Try disabling non-essential notifications to reduce app opens.")
        elif prob_percent > 35:
            st.info("💡 **Tip:** Set a 30-minute timer for your most-used social media apps.")
        else:
            st.success("✨ Your habits look balanced!")

    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.write("Debug info: Expected 13 columns, got", len(data_row))
