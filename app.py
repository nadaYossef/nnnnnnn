import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Addiction Diagnostic Tool", page_icon="📱", layout="centered")

# 2. Load the Pre-trained Model
@st.cache_resource
def load_model():
    # Ensure 'gbc_model.pkl' is in the same directory as this file
    return joblib.load('gbc_model.pkl')

model = load_model()

# 3. Header
st.title("📱 Smartphone Addiction Diagnostic Tool")
st.write("Enter your daily habits below to calculate your risk probability and receive personalized feedback.")
st.markdown("---")

# 4. Input Sidebar
st.sidebar.header("Usage Metrics")

daily_hours = st.sidebar.slider("Total Daily Screen Time (Hours)", 0.0, 24.0, 5.0)
social_hours = st.sidebar.slider("Social Media Usage (Hours)", 0.0, 24.0, 2.0)
gaming_hours = st.sidebar.slider("Gaming Usage (Hours)", 0.0, 24.0, 0.5)
sleep_hours = st.sidebar.slider("Typical Sleep (Hours)", 0.0, 12.0, 7.0)

st.sidebar.subheader("Engagement")
notifications = st.sidebar.number_input("Notifications per Day", min_value=0, max_value=2000, value=50)
app_opens = st.sidebar.number_input("App Opens per Day", min_value=0, max_value=1000, value=30)
is_weekend = st.sidebar.checkbox("Is this a Weekend/Holiday?")

st.sidebar.subheader("Demographics")
gender = st.sidebar.selectbox("Gender", ["Female", "Male", "Other"])

# 5. Main Logic & Prediction
if st.button("Generate My Risk Report"):
    
    # --- Feature Engineering (Matching your Model's training logic) ---
    
    # Prevent division by zero
    safe_daily = daily_hours if daily_hours > 0 else 0.01
    safe_sleep = sleep_hours if sleep_hours > 0 else 0.01

    # One-hot encoding for gender (Model expects gender_Male and gender_Other)
    g_male = 1 if gender == "Male" else 0
    g_other = 1 if gender == "Other" else 0

    # Building the 13-feature array
    data_row = [
        daily_hours,                                        # daily_screen_time_hours
        social_hours,                                       # social_media_hours
        sleep_hours,                                        # sleep_hours
        daily_hours * 1.3 if is_weekend else daily_hours,   # weekend_screen_time
        daily_hours,                                        # average_daily_screen_time
        (daily_hours - social_hours) / safe_daily,          # productivity_ratio
        social_hours / safe_daily,                          # social_media_ratio
        gaming_hours / safe_daily,                          # gaming_ratio
        notifications / safe_daily,                         # notifications_per_screen_hour
        app_opens / safe_daily,                             # app_opens_per_screen_hour
        daily_hours / safe_sleep,                           # screen_time_sleep_ratio
        g_male,                                             # gender_Male
        g_other                                             # gender_Other
    ]

    # Create DataFrame with exact column names from the model
    input_df = pd.DataFrame([data_row], columns=model.feature_names_in_)

    # Get Prediction Probabilities
    # Class 1 is "Addicted", so we take index [1]
    prob_percent = model.predict_proba(input_df)[0][1] * 100

    # 6. Displaying Results
    st.subheader("Analysis Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prob_percent < 35:
            st.success(f"**Risk Level: Low**")
        elif 35 <= prob_percent < 70:
            st.warning(f"**Risk Level: Moderate**")
        else:
            st.error(f"**Risk Level: High**")
            
    with col2:
        st.metric("Addiction Probability", f"{prob_percent:.1f}%")

    st.markdown("---")
    
    # 7. Personalized Feedback & Explanations
    st.write("### 🔍 Key Factors in Your Score")
    
    recs = []
    
    if daily_hours > 7:
        st.write("❌ **Screen Time:** Your usage is above the recommended threshold for digital wellbeing.")
        recs.append("Try a 'Digital Sunset'—no screens 60 minutes before bed.")
    else:
        st.write("✅ **Screen Time:** Your daily usage is within a healthy range.")

    if notifications / safe_daily > 15:
        st.write("❌ **Fragmentation:** You are interrupted by notifications very frequently, which spikes dopamine and reduces focus.")
        recs.append("Turn off non-human notifications (apps, news, stores).")

    if daily_hours / safe_sleep > 1.2:
        st.write("❌ **Sleep Interference:** Your phone time is high relative to your sleep, which may affect your circadian rhythm.")
        recs.append("Charge your phone in a different room tonight.")

    # 8. Action Plan
    if recs:
        st.write("### 🚀 Recommended Steps")
        for r in recs:
            st.info(r)
    else:
        st.success("Your habits look great! Keep maintaining your current balance.")

else:
    st.info("Adjust the sliders in the sidebar and click the button to see your results.")
