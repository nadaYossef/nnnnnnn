import streamlit as st
import joblib
import numpy as np

# Load the frozen assets
model = joblib.load('smartphone_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Smartphone Addiction Predictor")
st.write("Enter your usage stats to see your addiction risk level.")

# Create input fields for the user
age = st.number_input("Age", min_value=18, max_value=100)
screen_time = st.slider("Daily Screen Time (Hours)", 0.0, 24.0, 5.0)
notifications = st.number_input("Notifications per Day", 0, 1000)

if st.button("Predict"):
    features = np.array([[age, screen_time, notifications, sleep_hours, social_media]])

    scaled_features = scaler.transform(features)    
    prediction = model.predict(scaled_features)
    
    if prediction[0] == 1:
        st.error("High Risk of Addiction")
    else:
        st.success("Low Risk of Addiction")
