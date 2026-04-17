import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Addiction Diagnostic Tool", page_icon="📱", layout="centered")

# 2. Load the Pre-trained Model
@st.cache_resource
def load_model():
    # Ensure 'gbc_model.pkl' is in the same folder as this script
    return joblib.load('gbc_model.pkl')

model = load_model()

# 3. Header
st.title("📱 Smartphone Addiction Diagnostic Tool")
st.write("Calculate your addiction risk probability based on your usage habits.")
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

# 5. Prediction Logic
if st.button("Generate My Risk Report"):
    
    # Pre-calculations to avoid division by zero
    safe_daily = daily_hours if daily_hours > 0 else 0.01
    safe_sleep = sleep_hours if sleep_hours > 0 else 0.01

    # Define our known features in a dictionary
    # The keys here MUST match the column names used during your model training
    input_dict = {
        'daily_screen_time_hours': float(daily_hours),
        'social_media_hours': float(social_hours),
        'sleep_hours': float(sleep_hours),
        'weekend_screen_time': float(daily_hours * 1.3 if is_weekend else daily_hours),
        'average_daily_screen_time': float(daily_hours),
        'productivity_ratio': float((daily_hours - social_hours) / safe_daily),
        'social_media_ratio': float(social_hours / safe_daily),
        'gaming_ratio': float(gaming_hours / safe_daily),
        'notifications_per_screen_hour': float(notifications / safe_daily),
        'app_opens_per_screen_hour': float(app_opens / safe_daily),
        'screen_time_sleep_ratio': float(daily_hours / safe_sleep),
        'gender_Male': 1 if gender == "Male" else 0,
        'gender_Other': 1 if gender == "Other" else 0,
        'gender_Female': 1 if gender == "Female" else 0 # Just in case it's in the 16
    }

    # --- THE FIX: DYNAMIC COLUMN ALIGNMENT ---
    # We create a dataframe with the EXACT 16 columns the model expects
    # We fill it by looking up the keys in our dictionary. If a key is missing, it puts 0.
    try:
        input_df = pd.DataFrame(columns=model.feature_names_in_)
        input_df.loc[0] = [input_dict.get(col, 0) for col in model.feature_names_in_]
        
        # Ensure all data is numeric for the model
        input_df = input_df.apply(pd.to_numeric)

        # Get Probability (Class 1 is Addicted)
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

        st.markdown("---")
        # Simple Logic-based Recommendations
        if prob_percent > 60:
            st.info("💡 **Recommendation:** Your probability is high. Try using 'Grayscale mode' to make your phone less stimulating.")
        elif prob_percent > 30:
            st.info("💡 **Recommendation:** Consider setting a daily limit for Social Media apps.")
        else:
            st.success("✨ Your usage patterns appear healthy and balanced!")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.write("Missing features found in model:", [c for c in model.feature_names_in_ if c not in input_dict])

else:
    st.info("Please adjust your metrics and click the button above.")
