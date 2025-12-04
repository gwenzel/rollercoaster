# app.py
import streamlit as st
import numpy as np
import pandas as pd
from utils.track import build_modular_track, compute_features_modular, compute_acceleration
from utils.acceleration import compute_acc_profile
from utils.track_library import ensure_library, pick_random_entry, load_entry
from utils.visualize import plot_track
from utils.model import predict_thrill
from utils.bigru_predictor import predict_score_bigru
import os
import plotly.express as px


st.set_page_config(page_title="AI Roller Coaster Designer", layout="wide")

st.title("🎢 AI Roller Coaster Designer - Modular Track Builder")

# Initialize session state for track elements (simple coaster with one loop)
if 'track_elements' not in st.session_state:
    st.session_state.track_elements = [
        {'type': 'climb', 'params': {'length': 40, 'height': 35}},
        {'type': 'drop', 'params': {'length': 50, 'angle': 65}},
        {'type': 'clothoid_loop', 'params': {'radius': 12, 'transition_length': 15}},
        {'type': 'parabolic_curve', 'params': {'amplitude': 6, 'length': 40}},
    ]

# Sidebar - Add/Remove Elements
st.sidebar.header("🔧 Track Elements")
st.sidebar.subheader("🎲 Library Random Start")
use_library = st.sidebar.checkbox("Pick random from library", value=True)
library_entries = ensure_library(dt=0.02)
if use_library and library_entries:
    if st.sidebar.button("🎲 Load random precomputed track"):
        entry = pick_random_entry(library_entries)
        geo, phys = load_entry(entry)
        # build a DataFrame from points
        pts = geo['points']
        track_df = pd.DataFrame(pts, columns=['x','y','z'])
        # attach physics
        track_df['f_lat_g'] = phys['f_lat_g']
        track_df['f_vert_g'] = phys['f_vert_g']
        track_df['f_long_g'] = phys['f_long_g']
        track_df['f_lat'] = phys['f_lat']
        track_df['f_vert'] = phys['f_vert']
        track_df['f_long'] = phys['f_long']
        # stash for rendering (skip build path)
        st.session_state._precomputed_track_df = track_df
        st.session_state._precomputed_entry_name = entry['name']
        st.experimental_rerun()

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

# Performance / Preview Options
st.sidebar.subheader("⚡ Performance")
fast_preview = st.sidebar.checkbox("Fast preview (skip heavy compute)", value=True)
plot_acc = st.sidebar.checkbox("Show acceleration chart", value=not fast_preview)
compute_forces = st.sidebar.checkbox("Compute rider-frame forces", value=not fast_preview)
preview_points = st.sidebar.slider("Preview resolution (points)", 100, 3000, 600, step=50)
downsample_plot = st.sidebar.checkbox("Downsample plot for speed", value=True)
enable_ai = st.sidebar.checkbox("Enable AI rating (on demand)", value=False)
rate_now = st.sidebar.button("Rate this track")

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
if '_precomputed_track_df' in st.session_state:
    track_df = st.session_state._precomputed_track_df.copy()
else:
    track_df = build_modular_track(st.session_state.track_elements)
if preview_points and len(track_df) > preview_points:
    # uniform downsample for preview speed
    idx = np.linspace(0, len(track_df)-1, preview_points).astype(int)
    track_df = track_df.iloc[idx].reset_index(drop=True)

# --- Compute Acceleration ---
if not fast_preview:
    max_height = track_df['y'].max()
    track_df = compute_acceleration(track_df, max_height)

@st.cache_data
def _cached_acc_profile(points_array, dt):
    return compute_acc_profile(points_array, dt=dt)

# --- Rider-Frame Accelerations (for AI) ---
rider_acc_ok = False
rider_acc_err = None
if compute_forces:
    try:
        pts = track_df[['x','y','z']].to_numpy(dtype=float)
        acc = _cached_acc_profile(pts.round(3), dt=0.02)
        # attach normalized-by-g components expected by the AI later
        track_df['f_lat_g'] = acc['f_lat_g']
        track_df['f_vert_g'] = acc['f_vert_g']
        track_df['f_long_g'] = acc['f_long_g']
        # raw units too (optional)
        track_df['f_lat'] = acc['f_lat']
        track_df['f_vert'] = acc['f_vert']
        track_df['f_long'] = acc['f_long']
        rider_acc_ok = True
    except Exception as _e:
        rider_acc_err = str(_e)

# --- Compute Features ---
features = compute_features_modular(track_df, st.session_state.track_elements)

# --- Predict Thrill (rule-based) ---
predicted_thrill = predict_thrill(features)

# --- Predict Score with BiGRU (if model exists) ---
model_path = "models/bigru_score_model.pth"
bigru_score = None
bigru_error = None

if enable_ai and rate_now and os.path.exists(model_path):
    try:
        bigru_score = predict_score_bigru(track_df, model_path)
    except Exception as e:
        bigru_error = str(e)

# --- Main Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    # Optionally downsample for plotting only
    plot_df = track_df
    if downsample_plot and len(plot_df) > 2000:
        idxp = np.linspace(0, len(plot_df)-1, 2000).astype(int)
        plot_df = plot_df.iloc[idxp].reset_index(drop=True)
    st.plotly_chart(plot_track(plot_df, color_by="acceleration"), width='stretch')
    # Optional: acceleration components over path index
    if plot_acc and 'f_lat_g' in track_df and 'f_vert_g' in track_df and 'f_long_g' in track_df:
        acc_plot_df = pd.DataFrame({
            'index': np.arange(len(track_df)),
            'Lateral (g)': track_df['f_lat_g'],
            'Vertical (g)': track_df['f_vert_g'],
            'Longitudinal (g)': track_df['f_long_g'],
        })
        acc_long = acc_plot_df.melt(id_vars=['index'], var_name='Component', value_name='g')
        fig_acc = px.line(acc_long, x='index', y='g', color='Component', title='Rider-Frame Accelerations (g)')
        fig_acc.update_layout(margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_acc, use_container_width=True)

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
            st.info("No BiGRU model found or not requested")

    st.subheader("🎛️ Rider-Frame Accelerations")
    if rider_acc_ok:
        # show quick stats for validation
        stats = {
            'f_lat_g_max': float(np.max(np.abs(track_df['f_lat_g']))),
            'f_vert_g_max': float(np.max(np.abs(track_df['f_vert_g']))),
            'f_long_g_max': float(np.max(np.abs(track_df['f_long_g']))),
        }
        st.json(stats)
        # Export for AI ingestion
        export_cols = ['f_lat_g','f_vert_g','f_long_g']
        export_df = track_df[export_cols].copy()
        csv_bytes = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Accelerations (g) CSV",
            data=csv_bytes,
            file_name="rider_accelerations_g.csv",
            mime="text/csv"
        )
    else:
        st.error(f"Acceleration transform error: {rider_acc_err}")
    
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