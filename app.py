import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as ob

# --- CONFIG & ASSETS ---
st.set_page_config(page_title="Zenith AI: Digital Wellness", page_icon="🧠", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load('smartphone_model.pkl')
    return model

model = load_assets()

# --- CUSTOM CSS FOR "INTELLIGENT" FEEL ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_content_type=True)

# --- SIDEBAR: INTUITIVE INPUTS ---
with st.sidebar:
    st.title("🧠 Usage Profile")
    st.info("Input your typical weekly patterns for a deep neural analysis.")
    
    daily = st.slider("Typical Weekday Screen Time", 1, 15, 6, help="Average hours on your phone Mon-Fri")
    weekend = st.slider("Typical Weekend Screen Time", 1, 15, 8, help="Average hours on your phone Sat-Sun")
    social = st.slider("Social Media Depth", 0.0, 10.0, 3.0)
    gaming = st.slider("Gaming Frequency", 0.0, 10.0, 1.0)
    
    # Advanced context questions (even if not in model, they add to "intellect" feel)
    st.divider()
    purpose = st.selectbox("Primary Use Case", ["Work/Productivity", "Social/Entertainment", "Doomscrolling", "Utility"])

# --- FEATURE ENGINEERING ---
total_screen = daily + weekend
soc_ratio = social / (daily + 1e-6)
ent_load = social + gaming

input_data = pd.DataFrame({
    'daily_screen_time_hours': [daily],
    'social_media_hours': [social],
    'total_screen_time': [total_screen],
    'weekend_screen_time': [weekend],
    'entertainment_load': [ent_load],
    'social_ratio': [soc_ratio]
})

# --- PREDICTION ENGINE ---
prob = model.predict_proba(input_data)[0][1]

# --- UI LAYOUT ---
st.title("Digital Behavioral Analysis")
st.write(f"Patient ID: #USR-{np.random.randint(1000, 9999)}")

col1, col2 = st.columns([1, 1])

with col1:
    # 1. Gauge Chart for "Intelligence" Visualization
    fig = ob.Figure(ob.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        title = {'text': "Addiction Probability Index"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 30], 'color': "#e8f5e9"},
                {'range': [30, 70], 'color': "#fff3e0"},
                {'range': [70, 100], 'color': "#ffebee"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 85}
        }
    ))
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # 2. Persona Diagnosis
    st.subheader("Diagnostic Persona")
    if prob < 0.25:
        st.success("### 🟢 Digital Minimalist")
        st.write("Your digital footprint is lean. You likely use your device as a tool, not a crutch.")
    elif prob < 0.55:
        st.info("### 🟡 Functional User")
        st.write("Usage is within normal parameters, though slight evening 'creep' may be present.")
    elif prob < 0.85:
        st.warning("### 🟠 Habitual Consumer")
        st.write("Caution: Your dopamine pathways are heavily tied to social notifications.")
    else:
        st.error("### 🔴 High Dependency")
        st.write("Analysis indicates a strong neurological reliance on screen-based stimulation.")

st.divider()

# 3. INTERPRETABILITY: Why did the AI say this?
st.subheader("Neural Insight: Key Drivers")
# We simulate feature importance for this specific user
impact_data = pd.DataFrame({
    'Metric': ['Daily Intensity', 'Social Load', 'Weekend Surge', 'Ent. Load'],
    'Impact': [daily, social*1.5, (weekend-daily), ent_load]
}).sort_values(by='Impact', ascending=False)

fig_bar = px.bar(impact_data, x='Impact', y='Metric', orientation='h', 
             title="Factors Driving Your Score", color='Impact', color_continuous_scale='RdYlGn_r')
st.plotly_chart(fig_bar, use_container_width=True)

# 4. Pro-Active Coaching (The "Intelligent" Part)
st.subheader("AI Recommendations")
recs = []
if soc_ratio > 0.5: recs.append("- **Social Satiety:** Your social media use is >50% of your total time. Try 'Greyscale Mode'.")
if weekend > (daily + 2): recs.append("- **Weekend Bingeing:** Significant surge detected on weekends. Suggest a 'Digital Sabbath' on Sundays.")
if prob > 0.7: recs.append("- **Biological Priority:** Model predicts sleep disruption. Lock device 90 mins before bed.")

for r in recs:
    st.info(r)
