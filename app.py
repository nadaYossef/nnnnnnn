import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. SETTINGS & ASSETS ---
st.set_page_config(page_title="Smartphone Addiction Predictor", layout="centered")
# In your notebook, after you define X_simple
scaler_simple = StandardScaler()
X_simple_scaled = scaler_simple.fit_transform(X_simple)
@st.cache_resource
def load_assets():
    # Make sure 'model.pkl' and 'scaler.pkl' are uploaded to your GitHub repo
    model = joblib.load('smartphone_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error("Model files not found. Please upload 'model.pkl' and 'scaler.pkl' to your GitHub repository.")
    st.stop()

# --- 2. USER INTERFACE ---
st.title("📱 Smartphone Addiction Analyzer")
st.write("Enter your habits to see if your usage patterns suggest addiction.")

# Using a form to group inputs
with st.form("my_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 18, 35, 25)
        daily_screen_time = st.number_input("Daily Screen Time (Hours)", 3.0, 12.0, 7.0)
        social_media = st.number_input("Social Media Usage (Hours)", 0.5, 6.0, 2.0)
        gaming = st.number_input("Gaming Hours", 0.0, 4.0, 1.0)
    
    with col2:
        work_study = st.number_input("Work/Study Hours", 0.5, 6.0, 3.0)
        sleep = st.number_input("Sleep Hours", 4.5, 9.0, 7.0)
        notifications = st.number_input("Notifications per Day", 20, 250, 100)
        weekend_time = st.number_input("Weekend Screen Time (Hours)", 3.5, 15.0, 8.0)

    # FIXED: The correct function name is form_submit_button
    submit = st.form_submit_button("Analyze Results")

# --- 3. LOGIC & PREDICTION ---
if submit:
    # A. Feature Engineering (Calculating the 3 extra columns your model needs)
    total_screen_time = daily_screen_time + weekend_time
    entertainment_load = social_media + gaming
    social_ratio = social_media / (daily_screen_time + 1e-6)

    # B. Create the DataFrame with the 6 specific features your model was trained on
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

    # C. Transform and Predict
    try:
        # Scale the data using your saved scaler
        scaled_input = scaler.transform(input_df)
        
        # Get prediction and probability
        prediction = model.predict(scaled_input)
        probability = model.predict_proba(scaled_input)[0][1]

        # --- 4. DISPLAY RESULTS ---
        st.divider()
        if prediction[0] == 1:
            st.error(f"### Result: High Risk of Addiction")
            st.write(f"The model is **{probability:.1%}** confident in this result.")
        else:
            st.success(f"### Result: Low Risk of Addiction")
            st.write(f"The model is **{(1-probability):.1%}** confident in this result.")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
