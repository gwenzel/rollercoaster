# app.py
import streamlit as st
import numpy as np
import pandas as pd
from utils.track import build_modular_track, compute_features_modular, compute_acceleration
from utils.visualize import plot_track
from utils.model import predict_thrill
from utils.bigru_predictor import predict_score_bigru
import os


st.set_page_config(page_title="AI Roller Coaster Designer", layout="wide")

st.title("🎢 AI Roller Coaster Designer - Modular Track Builder")

# Initialize session state for track elements
if 'track_elements' not in st.session_state:
    st.session_state.track_elements = [
        {'type': 'climb', 'params': {'length': 30, 'height': 50}},
        {'type': 'drop', 'params': {'length': 60, 'angle': 70}},
        {'type': 'loop', 'params': {'radius': 10}},
        {'type': 'hills', 'params': {'num_hills': 3, 'amplitude': 8, 'wavelength': 30}},
    ]

# Sidebar - Add/Remove Elements
st.sidebar.header("🔧 Track Elements")

# Add new element
st.sidebar.subheader("Add Element")
element_type = st.sidebar.selectbox(
    "Element Type",
    ["climb", "drop", "loop", "clothoid_loop", "hills", "gaussian_curve", "parabolic_curve", "rotation"],
    key="new_element_type"
)

if st.sidebar.button("➕ Add Element"):
    # Default parameters for each type
    default_params = {
        'climb': {'length': 30, 'height': 50},
        'drop': {'length': 60, 'angle': 70},
        'loop': {'radius': 10},
        'clothoid_loop': {'radius': 12, 'transition_length': 15},
        'hills': {'num_hills': 3, 'amplitude': 8, 'wavelength': 30},
        'gaussian_curve': {'amplitude': 15, 'width': 20, 'length': 60},
        'parabolic_curve': {'amplitude': 20, 'length': 50},
        'rotation': {'angle': 180, 'radius': 8, 'axis': 'roll'}
    }
    st.session_state.track_elements.append({
        'type': element_type,
        'params': default_params[element_type]
    })
    st.rerun()

st.sidebar.markdown("---")

# Display and edit each element
st.sidebar.subheader("Current Track Elements")

for idx, element in enumerate(st.session_state.track_elements):
    with st.sidebar.expander(f"{idx+1}. {element['type'].upper()}", expanded=False):
        
        # Element-specific parameter sliders
        if element['type'] == 'climb':
            element['params']['length'] = st.slider(f"Length (m)", 10, 100, element['params']['length'], key=f"climb_len_{idx}")
            element['params']['height'] = st.slider(f"Height (m)", 10, 100, element['params']['height'], key=f"climb_h_{idx}")
        
        elif element['type'] == 'drop':
            element['params']['length'] = st.slider(f"Length (m)", 20, 100, element['params']['length'], key=f"drop_len_{idx}")
            element['params']['angle'] = st.slider(f"Angle (°)", 30, 89, element['params']['angle'], key=f"drop_ang_{idx}")
        
        elif element['type'] == 'loop':
            element['params']['radius'] = st.slider(f"Radius (m)", 5, 25, element['params']['radius'], key=f"loop_r_{idx}")
        
        elif element['type'] == 'clothoid_loop':
            element['params']['radius'] = st.slider(f"Radius (m)", 8, 30, element['params']['radius'], key=f"cloth_r_{idx}")
            element['params']['transition_length'] = st.slider(f"Transition (m)", 10, 40, element['params']['transition_length'], key=f"cloth_t_{idx}")
        
        elif element['type'] == 'hills':
            element['params']['num_hills'] = st.slider(f"Number", 1, 8, element['params']['num_hills'], key=f"hills_n_{idx}")
            element['params']['amplitude'] = st.slider(f"Amplitude (m)", 3, 15, element['params']['amplitude'], key=f"hills_a_{idx}")
            element['params']['wavelength'] = st.slider(f"Wavelength (m)", 15, 50, element['params']['wavelength'], key=f"hills_w_{idx}")
        
        elif element['type'] == 'gaussian_curve':
            element['params']['amplitude'] = st.slider(f"Amplitude (m)", 5, 30, element['params']['amplitude'], key=f"gauss_a_{idx}")
            element['params']['width'] = st.slider(f"Width (m)", 10, 50, element['params']['width'], key=f"gauss_w_{idx}")
            element['params']['length'] = st.slider(f"Length (m)", 30, 100, element['params']['length'], key=f"gauss_l_{idx}")
        
        elif element['type'] == 'parabolic_curve':
            element['params']['amplitude'] = st.slider(f"Amplitude (m)", 10, 40, element['params']['amplitude'], key=f"para_a_{idx}")
            element['params']['length'] = st.slider(f"Length (m)", 30, 100, element['params']['length'], key=f"para_l_{idx}")
        
        elif element['type'] == 'rotation':
            element['params']['angle'] = st.slider(f"Angle (°)", 90, 360, element['params']['angle'], key=f"rot_ang_{idx}")
            element['params']['radius'] = st.slider(f"Radius (m)", 5, 20, element['params']['radius'], key=f"rot_r_{idx}")
            element['params']['axis'] = st.selectbox(f"Axis", ['roll', 'barrel'], key=f"rot_axis_{idx}")
        
        # Remove button
        if st.button(f"🗑️ Remove", key=f"remove_{idx}"):
            st.session_state.track_elements.pop(idx)
            st.rerun()

# --- Generate Track ---
track_df = build_modular_track(st.session_state.track_elements)

# --- Compute Acceleration ---
max_height = track_df['y'].max()
track_df = compute_acceleration(track_df, max_height)

# --- Compute Features ---
features = compute_features_modular(track_df, st.session_state.track_elements)

# --- Predict Thrill (rule-based) ---
predicted_thrill = predict_thrill(features)

# --- Predict Score with BiGRU (if model exists) ---
model_path = "models/bigru_score_model.pth"
bigru_score = None
bigru_error = None

if os.path.exists(model_path):
    try:
        bigru_score = predict_score_bigru(track_df, model_path)
    except Exception as e:
        bigru_error = str(e)

# --- Main Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.plotly_chart(plot_track(track_df, color_by="acceleration"), width='stretch')

with col2:
    # Display predictions
    st.subheader("🎯 Predictions")
    
    col2a, col2b = st.columns(2)
    
    with col2a:
        st.metric("Rule-Based Thrill", f"{predicted_thrill:.2f}/10")
    
    with col2b:
        if bigru_score is not None:
            st.metric("🧠 BiGRU Score", f"{bigru_score:.2f}/5.0")
        elif bigru_error:
            st.error(f"BiGRU Error: {bigru_error}")
        else:
            st.info("No BiGRU model found")
    
    # Model info
    if bigru_score is not None:
        with st.expander("ℹ️ About BiGRU Model"):
            st.write("""
            This score is predicted by a **Bidirectional GRU neural network** 
            trained on real rollercoaster acceleration data and ratings.
            
            - **Input**: 3-axis accelerometer data (Lateral, Vertical, Longitudinal)
            - **Transformation**: Track coordinates → Rider's reference frame
            - **Output**: Predicted rating (1.0 - 5.0)
            - **Model**: 2-layer BiGRU with attention mechanism
            
            The model analyzes how a rider would experience the forces, 
            using data from wearable accelerometers mounted on real coasters.
            """)
    
    st.subheader("📊 Track Statistics")
    st.json(features)
    
    st.subheader("📋 Element Sequence")
    for idx, elem in enumerate(st.session_state.track_elements):
        st.text(f"{idx+1}. {elem['type'].replace('_', ' ').title()}")