# app.py
import streamlit as st
from utils.track import generate_track, compute_features, compute_acceleration
from utils.visualize import plot_track
from utils.model import predict_thrill


st.set_page_config(page_title="AI Roller Coaster Designer", layout="wide")


st.title("🎢 AI Roller Coaster Designer with Acceleration Coloring")
st.sidebar.header("Track Parameters")


# --- User Inputs ---
height_peak = st.sidebar.slider("Peak Height (m)", 20, 100, 50)
drop_angle = st.sidebar.slider("Drop Angle (°)", 30, 85, 70)
loop_radius = st.sidebar.slider("Loop Radius (m)", 5, 20, 10)
num_hills = st.sidebar.slider("Number of Hills", 1, 5, 3)
hill_amplitude = st.sidebar.slider("Hill Amplitude (m)", 3, 10, 8)


# --- Generate Track ---
track_df = generate_track(height_peak, drop_angle, loop_radius, num_hills, hill_amplitude)


# --- Compute Acceleration ---
track_df = compute_acceleration(track_df, height_peak)


# --- Compute Features ---
features = compute_features(track_df, loop_radius, num_hills)


# --- Predict Thrill ---
predicted_thrill = predict_thrill(features)

# --- Layout ---
col1, col2 = st.columns([2, 1])
with col1:
    st.plotly_chart(plot_track(track_df, color_by="acceleration"), use_container_width=True)
with col2:
    st.metric("Predicted Thrill", f"{predicted_thrill:.2f}/10")
    st.json(features)