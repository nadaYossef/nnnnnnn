import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Smartphone Addiction Predictor",
    page_icon="📱",
    layout="centered"
)

# --- LOAD ASSETS ---
@st.cache_resource
def load_model():
    # Ensure these files are in the same directory as app.py
    model = joblib.load('smartphone_model.pkl')
    # Note: Your training script used X_simple for the final fit, 
    # which didn't require scaling for the Gradient Boosting/Random Forest model.
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- APP UI ---
st.title("📱 Smartphone Addiction Analysis")
st.markdown("""
Enter your usage habits below to see the prediction. 
The model uses key behavioral indicators to estimate addiction probability.
""")

st.sidebar.header("User Usage Metrics")

def user_input_features():
    # These represent the 'selected_features' from your training:
    # ['daily_screen_time_hours', 'social_media_hours', 'total_screen_time', 
    #  'weekend_screen_time', 'entertainment_load', 'social_ratio']
    
    daily_screen = st.sidebar.slider("Daily Screen Time (Hours)", 1.0, 15.0, 7.5)
    weekend_screen = st.sidebar.slider("Weekend Screen Time (Hours)", 1.0, 20.0, 9.0)
    social_media = st.sidebar.slider("Social Media Usage (Hours)", 0.0, 10.0, 3.0)
    gaming = st.sidebar.slider("Gaming Usage (Hours)", 0.0, 10.0, 2.0)
    
    # Feature Engineering (Mirroring your training script logic)
    total_screen_time = daily_screen + weekend_screen
    social_ratio = social_media / (daily_screen + 1e-6)
    entertainment_load = social_media + gaming
    
    data = {
        'daily_screen_time_hours': daily_screen,
        'social_media_hours': social_media,
        'total_screen_time': total_screen_time,
        'weekend_screen_time': weekend_screen,
        'entertainment_load': entertainment_load,
        'social_ratio': social_ratio
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- PREDICTION ---
st.subheader("Analysis Results")

# Display the engineered features to the user
cols = st.columns(3)
cols[0].metric("Total Weekly/Day Load", f"{input_df['total_screen_time'].iloc[0]:.1f}h")
cols[1].metric("Social Ratio", f"{input_df['social_ratio'].iloc[0]*100:.1f}%")
cols[2].metric("Ent. Load", f"{input_df['entertainment_load'].iloc[0]:.1f}h")

prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# --- DISPLAY RESULTS ---
if prediction[0] == 1:
    st.warning("### Result: High Risk of Addiction")
else:
    st.success("### Result: Healthy Usage Level")

# Probability Gauge/Bar
prob_val = prediction_proba[0][1]
st.write(f"**Confidence Level:** {prob_val*100:.2f}%")
st.progress(prob_val)

st.markdown("---")
st.info("""
**Note:** This tool is for educational purposes based on a machine learning model. 
High AUC (0.98) suggests high reliability on the provided dataset, 
but it is not a clinical diagnosis.
""")
