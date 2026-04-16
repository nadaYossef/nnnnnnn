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
    # 1. Calculate the 'Engineered Features' used in your notebook
    total_screen_time = daily_screen_time_hours + weekend_screen_time
    entertainment_load = social_media_hours + gaming_hours
    social_ratio = social_media_hours / (daily_screen_time_hours + 1e-6)

    # 2. Create the DataFrame with the EXACT names and ORDER from your training
    # These 6 features were identified as the 'Selected features' in your notebook
    input_data = [[
        daily_screen_time_hours, 
        social_media_hours, 
        total_screen_time, 
        weekend_screen_time, 
        entertainment_load, 
        social_ratio
    ]]
    
    column_names = [
        'daily_screen_time_hours', 
        'social_media_hours', 
        'total_screen_time', 
        'weekend_screen_time', 
        'entertainment_load', 
        'social_ratio'
    ]
    
    input_df = pd.DataFrame(input_data, columns=column_names)

    # 3. Scale and Predict
    scaled_features = scaler.transform(input_df)
    prediction = model.predict(scaled_features)
    
    # Display results
    if prediction[0] == 1:
        st.error("High Risk of Addiction")
    else:
        st.success("Low Risk of Addiction")
