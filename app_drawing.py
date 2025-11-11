"""
Interactive Roller Coaster Builder App
Build your roller coaster using pre-defined building blocks!
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import splprep, splev
import sys
import os

# Import utilities
from utils.accelerometer_transform import track_to_accelerometer_data
from utils.bigru_predictor import predict_score_bigru

st.set_page_config(
    page_title="Roller Coaster Builder",
    page_icon="🎢",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF4B4B;
        margin-bottom: 1rem;
    }
    .block-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #dee2e6;
        margin-bottom: 1rem;
    }
    .rating-display {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        color: #FFD700;
        margin: 2rem 0;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎢 Draw Your Roller Coaster!</div>', unsafe_allow_html=True)

# Initialize session state
if 'drawn_points' not in st.session_state:
    st.session_state.drawn_points = []
if 'track_generated' not in st.session_state:
    st.session_state.track_generated = False

# Sidebar controls
with st.sidebar:
    st.header("🎨 Drawing Controls")
    
    st.markdown("""
    <div class="instruction-box">
    <h4>How to Use:</h4>
    <ol>
        <li>Click "Start Drawing Mode" below</li>
        <li>Use the canvas to sketch your track profile</li>
        <li>Click "Generate Ride!" to convert to 3D track</li>
        <li>Get instant AI rating prediction!</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    drawing_mode = st.selectbox(
        "Input Method",
        ["Simple Template", "Manual Point Entry", "Canvas Drawing (coming soon)"]
    )
    
    if drawing_mode == "Simple Template":
        st.subheader("Quick Design Template")
        
        template = st.selectbox(
            "Choose Template",
            ["Custom", "Classic Loop", "Mega Drop", "Twister", "Family Friendly"]
        )
        
        if template == "Custom":
            num_points = st.slider("Number of Points", 5, 20, 10)
            smoothness = st.slider("Track Smoothness", 0.5, 5.0, 2.0, 0.5)
        elif template == "Classic Loop":
            st.info("Traditional loop coaster with vertical loop")
            num_points = 12
            smoothness = 2.0
        elif template == "Mega Drop":
            st.info("Extreme drop with high speed sections")
            num_points = 8
            smoothness = 1.5
        elif template == "Twister":
            st.info("Multiple curves and banking turns")
            num_points = 15
            smoothness = 3.0
        else:  # Family Friendly
            st.info("Gentle hills and smooth transitions")
            num_points = 10
            smoothness = 3.5
    
    elif drawing_mode == "Manual Point Entry":
        st.subheader("Enter Track Points")
        num_manual_points = st.number_input("Number of control points", 3, 15, 6)
        
        manual_points = []
        for i in range(num_manual_points):
            col1, col2 = st.columns(2)
            with col1:
                x = st.number_input(f"Point {i+1} - Distance (m)", 0, 500, i*50, key=f"x_{i}")
            with col2:
                y = st.number_input(f"Point {i+1} - Height (m)", 0, 100, 20 if i==0 else 50-i*8, key=f"y_{i}")
            manual_points.append((x, y))
        
        smoothness = st.slider("Smoothness", 0.5, 5.0, 2.0, 0.5)
    
    st.divider()
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        track_length_multiplier = st.slider(
            "Track Length Multiplier", 
            0.5, 2.0, 1.0, 0.1,
            help="Stretch or compress the track horizontally"
        )
        height_multiplier = st.slider(
            "Height Multiplier", 
            0.5, 2.0, 1.0, 0.1,
            help="Scale the track height"
        )
        initial_speed = st.slider(
            "Initial Speed (m/s)", 
            0, 20, 5, 1,
            help="Starting velocity at the first point"
        )

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Track Profile View")
    
    # Generate track based on mode
    if drawing_mode == "Simple Template":
        if st.button("🎢 Generate Ride!", type="primary", use_container_width=True):
            with st.spinner("Building your roller coaster..."):
                # Generate template-specific track
                if template == "Classic Loop":
                    # Create a profile with a loop
                    x_points = np.array([0, 30, 60, 100, 140, 180, 220, 260, 300, 340, 360, 380])
                    y_points = np.array([5, 50, 10, 15, 35, 55, 35, 15, 10, 20, 10, 5])
                elif template == "Mega Drop":
                    x_points = np.array([0, 40, 80, 100, 150, 200, 250, 300])
                    y_points = np.array([5, 80, 85, 5, 10, 30, 15, 5])
                elif template == "Twister":
                    x_points = np.linspace(0, 400, 15)
                    y_points = 30 + 20*np.sin(x_points/50) + 10*np.cos(x_points/30)
                elif template == "Family Friendly":
                    x_points = np.linspace(0, 300, 10)
                    y_points = 25 + 15*np.sin(x_points/60)
                else:  # Custom
                    x_points = np.linspace(0, 300, num_points)
                    y_points = np.random.uniform(10, 60, num_points)
                    y_points[0] = 5  # Start low
                    y_points[-1] = 5  # End low
                
                # Apply multipliers
                x_points = x_points * track_length_multiplier
                y_points = y_points * height_multiplier
                
                # Smooth the track
                s_smooth = len(x_points) * smoothness
                tck, u = splprep([x_points, y_points], s=s_smooth, k=min(3, len(x_points)-1))
                u_new = np.linspace(0, 1, 500)
                x_smooth, y_smooth = splev(u_new, tck)
                
                # Store in session state
                st.session_state.track_x = x_smooth
                st.session_state.track_y = y_smooth
                st.session_state.track_generated = True
                st.rerun()
    
    elif drawing_mode == "Manual Point Entry":
        if st.button("🎢 Generate Ride!", type="primary", use_container_width=True):
            with st.spinner("Building your roller coaster..."):
                x_points = np.array([p[0] for p in manual_points]) * track_length_multiplier
                y_points = np.array([p[1] for p in manual_points]) * height_multiplier
                
                # Smooth the track
                s_smooth = len(x_points) * smoothness
                tck, u = splprep([x_points, y_points], s=s_smooth, k=min(3, len(x_points)-1))
                u_new = np.linspace(0, 1, 500)
                x_smooth, y_smooth = splev(u_new, tck)
                
                st.session_state.track_x = x_smooth
                st.session_state.track_y = y_smooth
                st.session_state.track_generated = True
                st.rerun()
    
    # Display track if generated
    if st.session_state.track_generated:
        fig = go.Figure()
        
        # Add track line
        fig.add_trace(go.Scatter(
            x=st.session_state.track_x,
            y=st.session_state.track_y,
            mode='lines',
            line=dict(color='rgb(255, 75, 75)', width=4),
            name='Track Profile',
            hovertemplate='Distance: %{x:.1f}m<br>Height: %{y:.1f}m<extra></extra>'
        ))
        
        # Add ground line
        fig.add_shape(
            type="line",
            x0=0, x1=max(st.session_state.track_x),
            y0=0, y1=0,
            line=dict(color="green", width=2, dash="dash")
        )
        
        fig.update_layout(
            title="Your Roller Coaster Profile",
            xaxis_title="Distance (m)",
            yaxis_title="Height (m)",
            showlegend=False,
            hovermode='closest',
            height=400,
            plot_bgcolor='rgba(240, 242, 246, 0.5)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate statistics
        total_length = np.sum(np.sqrt(np.diff(st.session_state.track_x)**2 + 
                                       np.diff(st.session_state.track_y)**2))
        max_height = np.max(st.session_state.track_y)
        max_drop = np.max(np.diff(st.session_state.track_y[st.session_state.track_y.argsort()]))
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Track Length", f"{total_length:.0f} m")
        col_b.metric("Max Height", f"{max_height:.1f} m")
        col_c.metric("Max Drop", f"{abs(max_drop):.1f} m")
    
    else:
        # Placeholder
        st.info("👈 Configure your track and click 'Generate Ride!' to start")
        
        # Show example image
        example_fig = go.Figure()
        example_x = np.linspace(0, 300, 100)
        example_y = 30 + 20*np.sin(example_x/40)
        example_fig.add_trace(go.Scatter(
            x=example_x, y=example_y,
            mode='lines',
            line=dict(color='lightgray', width=3, dash='dash'),
            name='Example'
        ))
        example_fig.update_layout(
            title="Example Track Profile",
            xaxis_title="Distance (m)",
            yaxis_title="Height (m)",
            showlegend=False,
            height=400,
            plot_bgcolor='rgba(240, 242, 246, 0.3)'
        )
        st.plotly_chart(example_fig, use_container_width=True)

with col2:
    st.subheader("AI Rating Prediction")
    
    if st.session_state.track_generated:
        with st.spinner("🤖 Analyzing your design..."):
            try:
                # Convert 2D profile to 3D track
                x = st.session_state.track_x
                y = st.session_state.track_y
                z = np.zeros_like(x)  # Flat for now, could add banking later
                
                # Create DataFrame
                track_df = pd.DataFrame({
                    'X': x,
                    'Y': z,  # Lateral (side-to-side) - flat for profile view
                    'Z': y   # Height is Z in 3D coordinates
                })
                
                # Add time column (assume constant horizontal speed for simplicity)
                dt = 0.1  # 10 Hz sampling
                track_df['Time'] = np.arange(len(track_df)) * dt
                
                # Convert to accelerometer data
                accel_df = track_to_accelerometer_data(
                    track_df,
                    initial_speed=initial_speed,
                    dt=dt
                )
                
                if accel_df is not None and len(accel_df) > 10:
                    # Predict rating using BiGRU model
                    predicted_rating = predict_score_bigru(accel_df)
                    
                    # Display rating
                    st.markdown(f'<div class="rating-display">⭐ {predicted_rating:.2f} / 5.00</div>', 
                               unsafe_allow_html=True)
                    
                    # Rating interpretation
                    if predicted_rating >= 4.5:
                        st.success("🔥 Outstanding! This would be a world-class coaster!")
                        st.balloons()
                    elif predicted_rating >= 4.0:
                        st.success("🎉 Excellent! Riders would love this!")
                    elif predicted_rating >= 3.5:
                        st.info("👍 Great design! Good balance of thrills.")
                    elif predicted_rating >= 3.0:
                        st.info("😊 Solid ride! Some improvements possible.")
                    elif predicted_rating >= 2.5:
                        st.warning("🤔 Decent, but could use more excitement.")
                    else:
                        st.warning("💡 Try adding more dramatic elements!")
                    
                    # Show acceleration plot
                    st.subheader("G-Force Analysis")
                    
                    fig_accel = go.Figure()
                    
                    fig_accel.add_trace(go.Scatter(
                        x=accel_df['Time'],
                        y=accel_df['Lateral'],
                        name='Lateral (Side)',
                        line=dict(color='blue')
                    ))
                    
                    fig_accel.add_trace(go.Scatter(
                        x=accel_df['Time'],
                        y=accel_df['Vertical'],
                        name='Vertical (Up/Down)',
                        line=dict(color='red')
                    ))
                    
                    fig_accel.add_trace(go.Scatter(
                        x=accel_df['Time'],
                        y=accel_df['Longitudinal'],
                        name='Longitudinal (Forward)',
                        line=dict(color='green')
                    ))
                    
                    fig_accel.update_layout(
                        title="Rider G-Forces Over Time",
                        xaxis_title="Time (s)",
                        yaxis_title="Acceleration (g)",
                        height=300,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig_accel, use_container_width=True)
                    
                    # Statistics
                    st.subheader("Ride Statistics")
                    
                    max_g_vertical = accel_df['Vertical'].max()
                    max_g_lateral = accel_df['Lateral'].abs().max()
                    max_g_long = accel_df['Longitudinal'].abs().max()
                    
                    col_1, col_2 = st.columns(2)
                    col_1.metric("Max Vertical G", f"{max_g_vertical:.2f}g")
                    col_1.metric("Max Lateral G", f"{max_g_lateral:.2f}g")
                    col_2.metric("Max Longitudinal G", f"{max_g_long:.2f}g")
                    col_2.metric("Ride Duration", f"{accel_df['Time'].max():.1f}s")
                    
                    # Tips
                    st.subheader("💡 Design Tips")
                    if max_g_vertical < 2.0:
                        st.info("• Try adding a steeper drop for more vertical g-forces!")
                    if max_g_lateral < 1.0:
                        st.info("• Consider adding curves or banks for lateral excitement!")
                    if predicted_rating < 3.5:
                        st.info("• Add more height variation for better ratings")
                        st.info("• Try templates like 'Mega Drop' or 'Classic Loop'")
                    
                else:
                    st.error("Could not generate acceleration data. Try adjusting your track.")
                    
            except Exception as e:
                st.error(f"Error analyzing track: {str(e)}")
                st.info("Try adjusting smoothness or track points.")
    
    else:
        st.info("Generate a track to see the AI rating prediction!")
        
        # Show some example ratings
        st.subheader("Example Ratings")
        st.write("Real coasters from our database:")
        
        examples = [
            ("Zadra", 4.90, "⭐⭐⭐⭐⭐"),
            ("VelociCoaster", 4.86, "⭐⭐⭐⭐⭐"),
            ("Steel Vengeance", 4.82, "⭐⭐⭐⭐⭐"),
            ("Intimidator 305", 4.20, "⭐⭐⭐⭐"),
            ("Generic Kiddie Coaster", 1.50, "⭐"),
        ]
        
        for name, rating, stars in examples:
            st.markdown(f"**{name}**: {stars} ({rating:.2f})")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9rem;">
    🎢 Powered by BiGRU Neural Network trained on 1,299 real roller coasters<br>
    Using accelerometer data from RideForcesDB and ratings from Captain Coaster
</div>
""", unsafe_allow_html=True)

# Add clear button
if st.session_state.track_generated:
    if st.sidebar.button("🗑️ Clear and Start Over", use_container_width=True):
        st.session_state.track_generated = False
        st.rerun()
