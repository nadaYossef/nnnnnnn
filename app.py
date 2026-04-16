import streamlit as st
import pandas as pd
import joblib

# 1. Load your saved assets
model = joblib.load('smartphone_model.pkl')
scaler = joblib.load('scaler.pkl')

# 2. Collect ALL 9 inputs required by your model
age = st.number_input("Age", 18, 100)
screen_time = st.number_input("Daily Screen Time (Hours)", 0, 24)
social_media = st.number_input("Social Media Hours", 0, 24)
gaming = st.number_input("Gaming Hours", 0, 24)
work_study = st.number_input("Work/Study Hours", 0, 24)
sleep = st.number_input("Sleep Hours", 0, 24)
notifications = st.number_input("Notifications per Day", 0, 1000)
app_opens = st.number_input("App Opens per Day", 0, 1000)
weekend_time = st.number_input("Weekend Screen Time", 0, 48)

if st.button("Predict"):
    # --- ADD THE CODE HERE ---
    # Create a DataFrame with all 9 columns in the EXACT order from your notebook
    input_df = pd.DataFrame([[
        age, screen_time, social_media, gaming, 
        work_study, sleep, notifications, app_opens, weekend_time
    ]], columns=[
        'age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 
        'work_study_hours', 'sleep_hours', 'notifications_per_day', 
        'app_opens_per_day', 'weekend_screen_time'
    ])
    
    # Scale the features
    scaled_features = scaler.transform(input_df)
    
    # Make the prediction
    prediction = model.predict(scaled_features)
    # -------------------------
    
    st.write(f"Result: {'Addicted' if prediction[0] == 1 else 'Not Addicted'}")
