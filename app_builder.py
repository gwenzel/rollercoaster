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
from utils.track_library import ensure_library, pick_random_entry, load_entry, add_entry

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
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        color: #FFD700;
        margin: 1rem 0;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎢 Roller Coaster Builder</div>', unsafe_allow_html=True)

# ============================================================================
# BUILDING BLOCK DEFINITIONS
# ============================================================================

class TrackBlock:
    """A building block for roller coaster tracks"""
    def __init__(self, name, description, icon, base_profile_func):
        self.name = name
        self.description = description
        self.icon = icon
        self.base_profile_func = base_profile_func
        
    def generate_profile(self, **params):
        """Generate track profile with given parameters"""
        return self.base_profile_func(**params)

from utils.track_blocks import (
    lift_hill_profile,
    vertical_drop_profile,
    loop_profile,
    airtime_hill_profile,
    spiral_profile,
    banked_turn_profile,
    bunny_hop_profile,
    flat_section_profile,
)

 

# Define all available blocks
BLOCK_LIBRARY = {
    "lift_hill": TrackBlock("Lift Hill", "Chain lift to initial height", "⛰️", lift_hill_profile),
    "drop": TrackBlock("Vertical Drop", "Steep initial drop", "⬇️", vertical_drop_profile),
    "loop": TrackBlock("Clothoid Loop", "Smooth clothoid loop element", "🔄", loop_profile),
    "airtime_hill": TrackBlock("Airtime Hill", "Floater airtime moment", "🎈", airtime_hill_profile),
    "spiral": TrackBlock("Spiral", "Helix/corkscrew element", "🌀", spiral_profile),
    "bunny_hop": TrackBlock("Bunny Hop", "Quick airtime bump", "🐰", bunny_hop_profile),
    "banked_turn": TrackBlock("Banked Turn", "High-speed turn", "↪️", banked_turn_profile),
    "flat_section": TrackBlock("Flat Section", "Brake/station run", "➡️", flat_section_profile),
}

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

# Always initialize with default template
if 'track_sequence' not in st.session_state:
    st.session_state.track_sequence = []

if 'track_generated' not in st.session_state:
    st.session_state.track_generated = False

# Always start with default curve on first load
if 'initialized' not in st.session_state:
    st.session_state.track_sequence = [
        {
            'type': 'drop',
            'block': BLOCK_LIBRARY['drop'],
            'params': {'height': 30, 'steepness': 0.85}
        },
        {
            'type': 'flat_section',
            'block': BLOCK_LIBRARY['flat_section'],
            'params': {'length': 30}
        },
        {
            'type': 'loop',
            'block': BLOCK_LIBRARY['loop'],
            'params': {'diameter': 24}
        },
        {
            'type': 'flat_section',
            'block': BLOCK_LIBRARY['flat_section'],
            'params': {'length': 25}
        }
    ]
    st.session_state.initialized = True
    st.session_state.track_generated = False

# ============================================================================
# SIDEBAR: BUILDING BLOCK PALETTE
# ============================================================================

with st.sidebar:
    st.header("🧱 Building Blocks")
    
    # Quick Start section at the top
    st.subheader("🎲 Quick Start")
    
    col_rand1_top, col_rand2_top = st.columns(2)
    
    with col_rand1_top:
        if st.button("🎲 Random Template", key="random_top", use_container_width=True, help="Generate a random coaster with 4-8 blocks"):
            import random
            
            # Random number of blocks (4-8)
            num_blocks = random.randint(4, 8)
            
            # Always start with a lift hill followed by a vertical drop and a flat section
            new_sequence = [
                {
                    'type': 'lift_hill',
                    'block': BLOCK_LIBRARY['lift_hill'],
                    'params': {
                        'length': random.randint(30, 80),
                        'height': random.randint(20, 50)
                    }
                },
                {
                    'type': 'drop',
                    'block': BLOCK_LIBRARY['drop'],
                    'params': {
                        'height': random.randint(25, 45),
                        'steepness': random.uniform(0.7, 0.9)
                    }
                },
                {
                    'type': 'flat_section',
                    'block': BLOCK_LIBRARY['flat_section'],
                    'params': {'length': random.randint(20, 40)}
                }
            ]
            
            # Available blocks (after initial drop + flat)
            available_blocks = ['drop', 'loop', 'airtime_hill', 'spiral', 'bunny_hop', 'banked_turn']
            
            for i in range(num_blocks - 3):  # -3 because we added lift + drop + flat
                block_type = random.choice(available_blocks)
                
                if block_type == 'drop':
                    params = {
                        'height': random.randint(20, 50),
                        'steepness': random.uniform(0.6, 0.85)  # Max 30° angle
                    }
                elif block_type == 'loop':
                    params = {'diameter': random.randint(20, 40)}
                elif block_type == 'airtime_hill':
                    params = {
                        'length': random.randint(25, 50),
                        'height': random.randint(8, 20)
                    }
                elif block_type == 'spiral':
                    params = {
                        'diameter': random.randint(20, 35),
                        'turns': random.uniform(1.0, 2.5)
                    }
                elif block_type == 'bunny_hop':
                    params = {
                        'length': random.randint(15, 25),
                        'height': random.randint(5, 12)
                    }
                elif block_type == 'banked_turn':
                    params = {
                        'radius': random.randint(20, 40),
                        'angle': random.randint(60, 120)
                    }
                
                new_sequence.append({
                    'type': block_type,
                    'block': BLOCK_LIBRARY[block_type],
                    'params': params
                })
            
            # Always end with a flat section (brake run)
            new_sequence.append({
                'type': 'flat_section',
                'block': BLOCK_LIBRARY['flat_section'],
                'params': {'length': random.randint(25, 40)}
            })
            
            st.session_state.track_sequence = new_sequence
            st.session_state.track_generated = False
            # Enforce end level equals start for random generation
            st.session_state.force_end_level = True
            st.session_state.start_level = 0.0
            # Enable 3D for random designs
            st.session_state.random_3d = True
            st.success(f"🎲 Generated random coaster with {num_blocks} blocks!")
            st.rerun()
    
    with col_rand2_top:
        if st.button("🔄 \n Reset to Default", key="reset_top", use_container_width=True, help="Reset to the starter template"):
            st.session_state.track_sequence = [
                {
                    'type': 'lift_hill',
                    'block': BLOCK_LIBRARY['lift_hill'],
                    'params': {'length': 50, 'height': 35}
                },
                {
                    'type': 'drop',
                    'block': BLOCK_LIBRARY['drop'],
                    'params': {'height': 30, 'steepness': 0.85}
                },
                {
                    'type': 'flat_section',
                    'block': BLOCK_LIBRARY['flat_section'],
                    'params': {'length': 30}
                },
                {
                    'type': 'loop',
                    'block': BLOCK_LIBRARY['loop'],
                    'params': {'diameter': 24}
                },
                {
                    'type': 'flat_section',
                    'block': BLOCK_LIBRARY['flat_section'],
                    'params': {'length': 25}
                }
            ]
            st.session_state.track_generated = False
            st.session_state.force_end_level = False
            st.success("🔄 Reset to default template!")
            st.rerun()

    # Precomputed safe library
    st.subheader("📚 Precomputed Safe Library")
    try:
        # Refresh control
        col_refresh = st.columns([1])[0]
        if col_refresh.button("🔄 Refresh Library", use_container_width=True):
            # Re-scan and rebuild metadata if needed, then rerun UI
            ensure_library(dt=0.02)
            st.success("Library refreshed")
            st.rerun()

        library_entries = ensure_library(dt=0.02)
        names = [e['name'] for e in library_entries] if library_entries else []
        selected_name = st.selectbox("Select saved design", options=names, index=0 if names else None)
        if st.button("📦 Load + Apply Selected", use_container_width=True) and selected_name:
            entry = next((e for e in library_entries if e['name'] == selected_name), None)
            if entry:
                # Load geometry + physics
                geo, phys = load_entry(entry)
                pts = geo['points']
                st.session_state.track_x = pts[:,0]
                st.session_state.track_y = pts[:,1]
                if pts.shape[1] >= 3:
                    st.session_state.track_z = pts[:,2].tolist()
                st.session_state.track_generated = True
                n = pts.shape[0]
                dt = 0.02
                time = np.arange(n) * dt
                accel_df = pd.DataFrame({
                    'Time': time,
                    'Vertical': phys['f_vert_g'],
                    'Lateral': phys['f_lat_g'],
                    'Longitudinal': phys['f_long_g'],
                })
                st.session_state.accel_df = accel_df

                # Apply elements to builder for editing
                st.session_state.track_sequence = [
                    {
                        'type': e['type'],
                        'block': BLOCK_LIBRARY[e['type'] if e['type'] in BLOCK_LIBRARY else 'flat_section'],
                        'params': e['params']
                    }
                    for e in entry.get('elements', [])
                ]
                st.success(f"Loaded and applied: {entry['name']}")
                st.rerun()
    except Exception as _lib_e:
        st.caption("Library unavailable; continuing without precomputed tracks.")
    
    st.divider()
    
    # Save current design to precomputed library
    st.subheader("💾 Save Design")
    new_lib_name = st.text_input("Entry name", value="my_design")
    if st.button("📚 Add Design to Library", use_container_width=True, help="Save current track with physics to the library"):
        if st.session_state.get('track_generated'):
            track_df = pd.DataFrame({
                'x': st.session_state.track_x,
                'y': st.session_state.track_y,
                'z': np.array(st.session_state.get('track_z', np.zeros_like(st.session_state.track_x)))
            })
            elements = st.session_state.get('track_sequence', [])
            meta = add_entry(new_lib_name, elements, track_df)
            if meta:
                st.success(f"Saved to library as '{new_lib_name}'")
            else:
                st.error("Failed to save to library")
        else:
            st.warning("Generate a track first, then save.")

    st.markdown("""
    <div style="background-color: #e3f2fd; padding: 0.8rem; border-radius: 0.3rem; margin-bottom: 1rem; color: #1a1a1a;">
    <b style="color: #0d47a1;">Instructions:</b><br>
    <span style="color: #1a1a1a;">
    1. Select a block below<br>
    2. Adjust its parameters<br>
    3. Click "Add to Track"<br>
    4. Build your complete ride!<br>
    5. AI rating updates automatically
    </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Block selection
    selected_block_key = st.selectbox(
        "Choose Block Type",
        options=list(BLOCK_LIBRARY.keys()),
        format_func=lambda x: f"{BLOCK_LIBRARY[x].icon} {BLOCK_LIBRARY[x].name}"
    )
    
    selected_block = BLOCK_LIBRARY[selected_block_key]
    
    st.markdown(f"**{selected_block.icon} {selected_block.name}**")
    st.caption(selected_block.description)
    
    st.divider()
    
    # Block-specific parameters
    st.subheader("⚙️ Parameters")
    
    params = {}
    
    if selected_block_key == "lift_hill":
        params['length'] = st.slider("Length (m)", 20, 100, 50, 5)
        params['height'] = st.slider("Height (m)", 20, 80, 40, 5)
        
    elif selected_block_key == "drop":
        params['height'] = st.slider("Drop Height (m)", 20, 90, 40, 5)
        params['steepness'] = st.slider("Steepness (max 30°)", 0.5, 1.0, 0.8, 0.05)
        st.caption("💡 Steepness limited to 30° for realism")
        
    elif selected_block_key == "loop":
        params['diameter'] = st.slider("Loop Diameter (m)", 15, 45, 30, 5)
        
    elif selected_block_key == "airtime_hill":
        params['length'] = st.slider("Hill Length (m)", 20, 60, 40, 5)
        params['height'] = st.slider("Hill Height (m)", 5, 25, 15, 2)
        
    elif selected_block_key == "spiral":
        params['diameter'] = st.slider("Spiral Diameter (m)", 15, 40, 25, 5)
        params['turns'] = st.slider("Number of Turns", 0.5, 3.0, 1.5, 0.5)
        
    elif selected_block_key == "bunny_hop":
        params['length'] = st.slider("Hop Length (m)", 10, 30, 20, 5)
        params['height'] = st.slider("Hop Height (m)", 3, 15, 8, 1)
        
    elif selected_block_key == "banked_turn":
        params['radius'] = st.slider("Turn Radius (m)", 15, 50, 30, 5)
        params['angle'] = st.slider("Turn Angle (°)", 30, 180, 90, 15)
        
    elif selected_block_key == "flat_section":
        params['length'] = st.slider("Section Length (m)", 10, 50, 30, 5)
    
    # Add block button
    if st.button("➕ Add to Track", type="primary", use_container_width=True):
        # Get current height for drop validation
        if selected_block_key == "drop" and len(st.session_state.track_sequence) > 0:
            # Calculate current height from existing track
            all_x, all_y = [], []
            current_x_offset, current_y_offset = 0, 0
            for block_info in st.session_state.track_sequence:
                x, y = block_info['block'].generate_profile(**block_info['params'])
                all_x.extend(x + current_x_offset)
                all_y.extend(y + current_y_offset)
                current_x_offset = all_x[-1]
                current_y_offset = all_y[-1]
            
            params['current_height'] = current_y_offset
            
            if current_y_offset < 5:
                st.error(f"⚠️ Not enough altitude! Current height: {current_y_offset:.1f}m. Add a lift hill first.")
                st.stop()
        
        st.session_state.track_sequence.append({
            'type': selected_block_key,
            'block': selected_block,
            'params': params.copy()
        })
        st.session_state.track_generated = False
        st.success(f"✅ Added {selected_block.name}")
        st.rerun()
    
    st.divider()
    
    # Track sequence management
    st.subheader("📋 Current Track Sequence")
    
    if len(st.session_state.track_sequence) == 0:
        st.info("No blocks added yet. Start building!")
    else:
        for idx, block_info in enumerate(st.session_state.track_sequence):
            with st.expander(f"{idx+1}. {block_info['block'].icon} {block_info['block'].name}", expanded=False):
                # Show parameters
                st.caption("**Current Parameters:**")
                for param_name, param_value in block_info['params'].items():
                    if param_name != 'current_height':  # Don't show internal params
                        st.text(f"• {param_name}: {param_value:.1f}" if isinstance(param_value, float) else f"• {param_name}: {param_value}")
                
                col_del, col_up, col_down = st.columns(3)
                with col_del:
                    if st.button("🗑️ Remove", key=f"del_{idx}", use_container_width=True):
                        st.session_state.track_sequence.pop(idx)
                        st.session_state.track_generated = False
                        st.rerun()
                with col_up:
                    if idx > 0 and st.button("⬆️ Move Up", key=f"up_{idx}", use_container_width=True):
                        st.session_state.track_sequence[idx], st.session_state.track_sequence[idx-1] = \
                            st.session_state.track_sequence[idx-1], st.session_state.track_sequence[idx]
                        st.session_state.track_generated = False
                        st.rerun()
                with col_down:
                    if idx < len(st.session_state.track_sequence) - 1 and st.button("⬇️ Move Down", key=f"down_{idx}", use_container_width=True):
                        st.session_state.track_sequence[idx], st.session_state.track_sequence[idx+1] = \
                            st.session_state.track_sequence[idx+1], st.session_state.track_sequence[idx]
                        st.session_state.track_generated = False
                        st.rerun()
        
        st.divider()
        
        # Clear all button
        if st.button("🗑️ Clear All Blocks", use_container_width=True):
            st.session_state.track_sequence = []
            st.session_state.track_generated = False
            st.rerun()

# ============================================================================
# MAIN AREA: TRACK VISUALIZATION AND RATING
# ============================================================================

def check_track_smoothness(x, y, max_angle_deg=30, allow_steep_start=True):
    """
    Check if track has any vertical walls or impossible sections.
    
    Args:
        x, y: Track coordinates
        max_angle_deg: Maximum allowed track angle from horizontal (default 30°)
        allow_steep_start: If True, allow steeper angles at the start (lift hill)
    
    Returns:
        (is_valid, max_angle, problem_indices)
    """
    dx = np.diff(x)
    dy = np.diff(y)
    
    # Calculate angle from horizontal in degrees
    angles = np.abs(np.arctan2(dy, dx) * 180 / np.pi)
    
    if allow_steep_start and len(angles) > 0:
        # Allow steeper angles for the first 20% of track (lift hill)
        lift_section_end = int(len(angles) * 0.2)
        
        # Check if first section is going up (lift hill)
        if lift_section_end > 0 and np.mean(dy[:lift_section_end]) > 0:
            # Allow up to 60° for lift hill section only
            problem_mask = np.zeros(len(angles), dtype=bool)
            problem_mask[:lift_section_end] = angles[:lift_section_end] > 60
            problem_mask[lift_section_end:] = angles[lift_section_end:] > max_angle_deg
        else:
            problem_mask = angles > max_angle_deg
    else:
        problem_mask = angles > max_angle_deg
    
    problem_indices = np.where(problem_mask)[0]
    max_angle = np.max(angles) if len(angles) > 0 else 0
    
    is_valid = len(problem_indices) == 0
    
    return is_valid, max_angle, problem_indices

def smooth_track_profile(x, y, smoothness_factor=0.5):
    """
    Apply light smoothing to track profile to remove sharp edges.
    
    Args:
        x, y: Track coordinates
        smoothness_factor: Higher = more smoothing (default 0.5 for light smoothing)
    
    Returns:
        x_smooth, y_smooth: Smoothed coordinates
    """
    if len(x) < 4:
        return x, y
    
    try:
        # Use spline interpolation with light smoothing
        # s parameter is key: lower = preserves shape better
        tck, u = splprep([x, y], s=len(x) * smoothness_factor, k=min(3, len(x)-1))
        u_new = np.linspace(0, 1, len(x) * 2)  # Reduced from 3x to 2x interpolation
        x_smooth, y_smooth = splev(u_new, tck)
        
        # Verify smoothness improved
        is_valid, max_angle, _ = check_track_smoothness(x_smooth, y_smooth)
        
        if is_valid or max_angle < 30:
            return np.array(x_smooth), np.array(y_smooth)
        else:
            # If still too steep, apply moderate smoothing (not aggressive)
            tck, u = splprep([x, y], s=len(x) * smoothness_factor * 3, k=min(3, len(x)-1))
            u_new = np.linspace(0, 1, len(x) * 2)
            x_smooth, y_smooth = splev(u_new, tck)
            return np.array(x_smooth), np.array(y_smooth)
            
    except Exception as e:
        # Fallback: return original if smoothing fails
        return np.array(x), np.array(y)

def smooth_joints(x, y, block_boundaries, window_size=5):
    """
    Apply moving average smoothing at block joints to eliminate spikes.
    
    Args:
        x, y: Track coordinates
        block_boundaries: List of indices where blocks join
        window_size: Size of moving average window (must be odd)
    
    Returns:
        x_smooth, y_smooth: Smoothed coordinates
    """
    x_smooth = x.copy()
    y_smooth = y.copy()
    
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    half_window = window_size // 2
    
    # Apply smoothing around each joint
    for joint_idx in block_boundaries:
        # Define smoothing region around the joint
        start_idx = max(0, joint_idx - half_window)
        end_idx = min(len(x), joint_idx + half_window + 1)
        
        if end_idx - start_idx < 3:
            continue  # Skip if region too small
        
        # Apply moving average
        for i in range(start_idx, end_idx):
            # Define local window
            local_start = max(0, i - half_window)
            local_end = min(len(x), i + half_window + 1)
            
            # Compute moving average
            x_smooth[i] = np.mean(x[local_start:local_end])
            y_smooth[i] = np.mean(y[local_start:local_end])
    
    return x_smooth, y_smooth

def detect_curvature_spikes(x, y, threshold=2.0):
    """
    Detect spikes in curvature (potential problem areas).
    
    Args:
        x, y: Track coordinates
        threshold: Curvature change threshold (default 2.0)
    
    Returns:
        spike_indices: Indices where curvature spikes occur
    """
    # Calculate first and second derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Calculate curvature
    ds = np.sqrt(dx**2 + dy**2)
    curvature = np.abs(ddx * dy - dx * ddy) / (ds**3 + 1e-6)
    
    # Detect sudden changes in curvature
    curvature_change = np.abs(np.gradient(curvature))
    
    # Normalize by median to get relative spikes
    median_change = np.median(curvature_change)
    if median_change > 0:
        relative_change = curvature_change / median_change
    else:
        relative_change = curvature_change
    
    spike_indices = np.where(relative_change > threshold)[0]
    
    return spike_indices, curvature

def generate_track_from_blocks():
    """Generate complete track from block sequence with improved C1 joint blending.
    Ensures continuity of position and first derivative in both x and y.
    """

    def endpoint_tangent(x_arr, y_arr, at_start=False):
        # Compute tangent vector using local finite difference
        if at_start:
            i0, i1 = 0, min(1, len(x_arr)-1)
        else:
            i1 = len(x_arr)-1
            i0 = max(0, i1-1)
        tx = x_arr[i1] - x_arr[i0]
        ty = y_arr[i1] - y_arr[i0]
        return tx, ty

    def hermite_blend(P0, P1, T0, T1, steps=24):
        t = np.linspace(0.0, 1.0, steps)
        h00 = 2*t**3 - 3*t**2 + 1
        h10 = t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 = t**3 - t**2
        dx = h00*P0[0] + h10*T0[0] + h01*P1[0] + h11*T1[0]
        dy = h00*P0[1] + h10*T0[1] + h01*P1[1] + h11*T1[1]
        return dx, dy

    def blend_joint(prev_x, prev_y, next_x_rel, next_y_rel, steps=24):
        # Absolute endpoints
        x0, y0 = prev_x[-1], prev_y[-1]
        x1, y1 = x0 + next_x_rel[0], y0 + next_y_rel[0]
        # Tangents at endpoints (absolute for prev, relative mapped for next)
        t0x, t0y = endpoint_tangent(prev_x, prev_y, at_start=False)
        n_tx, n_ty = endpoint_tangent(next_x_rel, next_y_rel, at_start=True)
        # Normalize and scale tangents to local segment length for stability
        seg_len = max(np.hypot(x1 - x0, y1 - y0), 1e-6)
        def scaled(tvx, tvy):
            n = max(np.hypot(tvx, tvy), 1e-6)
            s = seg_len * 0.5  # scale factor controls elbow size
            return (tvx / n * s, tvy / n * s)
        T0 = scaled(t0x, t0y)
        T1 = scaled(n_tx, n_ty)

        # Heuristic: try flipping the vertical orientation of the next tangent
        # and pick the blend with lower max curvature to avoid downward elbows.
        bx1, by1 = hermite_blend((x0, y0), (x1, y1), T0, T1, steps=steps)
        T1_flip = (T1[0], -T1[1])
        bx2, by2 = hermite_blend((x0, y0), (x1, y1), T0, T1_flip, steps=steps)

        def max_curv(xa, ya):
            dx = np.gradient(xa)
            dy = np.gradient(ya)
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            ds = np.sqrt(dx**2 + dy**2) + 1e-9
            k = np.abs(ddx * dy - dx * ddy) / (ds**3)
            return float(np.nanmax(k))

        if max_curv(bx2, by2) < max_curv(bx1, by1):
            return bx2, by2
        return bx1, by1

    all_x = []
    all_y = []
    current_x_offset = 0.0
    current_y_offset = 0.0
    joints_count = 0

    for idx, block_info in enumerate(st.session_state.track_sequence):
        x_rel, y_rel = block_info['block'].generate_profile(**block_info['params'])

        if idx == 0:
            # First block: add a short introductory blend from origin to avoid downward elbow
            x0, y0 = 0.0, 0.0
            x1, y1 = x_rel[0], y_rel[0]
            # Tangent at start and a gentle initial horizontal tangent
            n_tx, n_ty = endpoint_tangent(x_rel, y_rel, at_start=True)
            seg_len = max(np.hypot(x1 - x0, y1 - y0), 1e-6)
            init_scale = seg_len * 0.5
            T0 = (init_scale, 0.0)
            def scaled(vx, vy):
                n = max(np.hypot(vx, vy), 1e-6)
                return (vx / n * init_scale, vy / n * init_scale)
            T1 = scaled(n_tx, n_ty)
            bx1, by1 = hermite_blend((x0, y0), (x1, y1), T0, T1, steps=24)
            T1_flip = (T1[0], -T1[1])
            bx2, by2 = hermite_blend((x0, y0), (x1, y1), T0, T1_flip, steps=24)
            def max_curv(xa, ya):
                dx = np.gradient(xa)
                dy = np.gradient(ya)
                ddx = np.gradient(dx)
                ddy = np.gradient(dy)
                ds = np.sqrt(dx**2 + dy**2) + 1e-9
                k = np.abs(ddx * dy - dx * ddy) / (ds**3)
                return float(np.nanmax(k))
            bx, by = (bx2, by2) if max_curv(bx2, by2) < max_curv(bx1, by1) else (bx1, by1)
            # Append blend (skip origin to avoid duplicate)
            all_x.extend(bx[1:].tolist())
            all_y.extend(by[1:].tolist())
            # Append rest of first block relative to last blend point
            x_abs = x_rel + all_x[-1]
            y_abs = y_rel + all_y[-1]
            all_x.extend(x_abs.tolist())
            all_y.extend(y_abs.tolist())
            current_x_offset = all_x[-1]
            current_y_offset = all_y[-1]
            joints_count += 1
            continue

        # Before appending next block, insert a blend segment to match slopes
        x_blend, y_blend = blend_joint(np.array(all_x), np.array(all_y), np.array(x_rel), np.array(y_rel), steps=32)

        # Append blend (avoid duplicating endpoint)
        all_x.extend(x_blend[1:].tolist())
        all_y.extend(y_blend[1:].tolist())
        joints_count += 1

        # Now append the next block offset from last absolute point
        x_abs = x_rel + all_x[-1]
        y_abs = y_rel + all_y[-1]
        all_x.extend(x_abs.tolist())
        all_y.extend(y_abs.tolist())

        current_x_offset = all_x[-1]
        current_y_offset = all_y[-1]

    all_x = np.array(all_x)
    all_y = np.array(all_y)

    st.session_state.joint_smoothing_applied = f"✓ Blended {joints_count} joints (cubic)"
    st.session_state.smoothness_warning = None
    _, final_curvature = detect_curvature_spikes(all_x, all_y)
    st.session_state.track_curvature = final_curvature
    # If requested, enforce ending level equals starting level (y-coordinate)
    if st.session_state.get('force_end_level'):
        y_target = float(st.session_state.get('start_level', 0.0))
        y_end = float(all_y[-1])
        if abs(y_end - y_target) > 1e-3:
            # Create a gentle leveling segment
            x0, y0 = all_x[-1], all_y[-1]
            x1 = x0 + 40.0
            y1 = y_target
            # Tangents: continue current direction, land horizontally
            t0x, t0y = endpoint_tangent(all_x, all_y, at_start=False)
            seg_len = max(np.hypot(x1 - x0, y1 - y0), 1e-6)
            scale = seg_len * 0.5
            def sc(vx, vy):
                n = max(np.hypot(vx, vy), 1e-6)
                return (vx / n * scale, vy / n * scale)
            T0 = sc(t0x, t0y)
            T1 = (scale, 0.0)
            bx, by = hermite_blend((x0, y0), (x1, y1), T0, T1, steps=48)
            all_x = np.concatenate([all_x, bx[1:]])
            all_y = np.concatenate([all_y, by[1:]])
    return all_x, all_y

def simple_gforce_analysis(x, y, initial_speed=15.0, dt=0.1):
    """Simple physics-based g-force calculation"""
    g = 9.81
    
    # Calculate velocities using energy conservation
    h_max = np.max(y)
    v = np.sqrt(initial_speed**2 + 2 * g * (h_max - y))
    v = np.maximum(v, 0.1)  # Minimum velocity
    
    # Calculate accelerations
    dx = np.gradient(x)
    dy = np.gradient(y)
    ds = np.sqrt(dx**2 + dy**2)
    
    # Tangential acceleration
    dv = np.gradient(v) / dt
    a_tangential = dv / np.maximum(ds, 0.001)
    
    # Normal acceleration (centripetal)
    # Radius of curvature
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(ddx * dy - dx * ddy) / (ds**3 + 1e-6)
    a_normal = v**2 * curvature
    
    # Vertical component (includes gravity)
    theta = np.arctan2(dy, dx)
    a_vertical = (a_normal * np.cos(theta) - g + a_tangential * np.sin(theta)) / g
    
    # Lateral (simplified - assumes no banking)
    a_lateral = a_normal * np.sin(theta) / g
    
    # Longitudinal
    a_longitudinal = a_tangential * np.cos(theta) / g
    
    # Create time array
    time = np.arange(len(x)) * dt
    
    return pd.DataFrame({
        'Time': time,
        'Vertical': a_vertical,
        'Lateral': a_lateral,
        'Longitudinal': a_longitudinal
    })

def check_gforce_safety(accel_df):
    """
    Check if g-forces are within safe human tolerance limits.
    
    Human G-force tolerances:
    - Positive (downward): 5g dangerous, 9g+ fatal
    - Negative (upward): -2g to -3g dangerous
    - Lateral: ±2g uncomfortable, ±5g dangerous
    
    Returns:
        dict with safety analysis
    """
    max_vertical = accel_df['Vertical'].max()
    min_vertical = accel_df['Vertical'].min()
    max_lateral = accel_df['Lateral'].abs().max()
    max_longitudinal = accel_df['Longitudinal'].abs().max()
    
    warnings = []
    dangers = []
    
    # Vertical G-forces (positive)
    if max_vertical > 5.0:
        dangers.append(f"🚨 DANGEROUS: {max_vertical:.1f}g positive vertical (>5g can cause blackout)")
    elif max_vertical > 4.0:
        warnings.append(f"⚠️ HIGH: {max_vertical:.1f}g positive vertical (uncomfortable, >4g)")
    elif max_vertical > 3.0:
        warnings.append(f"⚠️ Intense: {max_vertical:.1f}g positive vertical (>3g)")
    
    # Vertical G-forces (negative/airtime)
    if min_vertical < -3.0:
        dangers.append(f"🚨 DANGEROUS: {min_vertical:.1f}g negative vertical (< -3g unsafe)")
    elif min_vertical < -2.0:
        warnings.append(f"⚠️ HIGH: {min_vertical:.1f}g negative airtime (< -2g)")
    
    # Lateral G-forces
    if max_lateral > 5.0:
        dangers.append(f"🚨 DANGEROUS: {max_lateral:.1f}g lateral (>5g can cause injury)")
    elif max_lateral > 2.0:
        warnings.append(f"⚠️ Uncomfortable: {max_lateral:.1f}g lateral (>2g)")
    
    # Longitudinal G-forces
    if max_longitudinal > 3.0:
        warnings.append(f"⚠️ Intense: {max_longitudinal:.1f}g longitudinal (>3g)")
    
    # Overall safety rating
    if len(dangers) > 0:
        safety_level = "DANGEROUS"
        safety_emoji = "🚨"
        safety_color = "error"
    elif len(warnings) > 0:
        safety_level = "Intense/Uncomfortable"
        safety_emoji = "⚠️"
        safety_color = "warning"
    else:
        safety_level = "Safe"
        safety_emoji = "✅"
        safety_color = "success"
    
    return {
        'level': safety_level,
        'emoji': safety_emoji,
        'color': safety_color,
        'warnings': warnings,
        'dangers': dangers,
        'max_vertical': max_vertical,
        'min_vertical': min_vertical,
        'max_lateral': max_lateral,
        'max_longitudinal': max_longitudinal
    }

# Auto-generate preview when blocks are added
if len(st.session_state.track_sequence) > 0:
    st.session_state.track_x, st.session_state.track_y = generate_track_from_blocks()
    st.session_state.track_generated = True
    # If random 3D is enabled, synthesize a gentle lateral profile (z) while keeping 2D plots unfolded
    if st.session_state.get('random_3d'):
        x = np.array(st.session_state.track_x)
        n = len(x)
        # Create low-amplitude lateral variations using a few harmonics
        t = np.linspace(0, 1, n)
        z = (
            2.0 * np.sin(2*np.pi * (0.5*t)) +
            1.2 * np.sin(2*np.pi * (1.0*t + 0.3)) +
            0.8 * np.sin(2*np.pi * (1.8*t + 0.7))
        )
        # Smooth and scale to safe lateral displacement
        # Smooth via moving average to avoid external dependencies
        win = max(5, n // 100)
        kernel = np.ones(win) / win
        z = np.convolve(z, kernel, mode='same')
        z *= 0.5  # meters
        st.session_state.track_z = z.tolist()
    
    # Always get AI rating automatically
    st.session_state.get_ai_rating = True

# Create layout with multiple plots
if st.session_state.track_generated:
    # Create subplots layout
    st.subheader("🎢 Your Roller Coaster Design")
    
    # Row 0: AI Rating Prediction (COMPACT) + Library add button
    st.markdown("**🤖 AI Rating Prediction**")
    
    # AI rating runs automatically (button removed)

    try:
        # Convert to 3D track format for the AI model
        track_df = pd.DataFrame({
            'x': st.session_state.track_x,
            'y': st.session_state.track_y,
            'z': np.zeros_like(st.session_state.track_x)  # No lateral banking yet
        })
        
        # Get accelerometer data using the existing function, unless precomputed is already present
        accel_df = st.session_state.get('accel_df')
        if accel_df is None or len(accel_df) < 10:
            accel_df = track_to_accelerometer_data(track_df)
        
        if accel_df is not None and len(accel_df) > 10:
            # Store for g-force plot
            st.session_state.accel_df = accel_df
            
            # Check safety FIRST before showing rating
            safety = check_gforce_safety(accel_df)
            
            # Predict rating automatically
            predicted_rating = predict_score_bigru(accel_df)
            st.session_state.predicted_rating = predicted_rating
            
            # (Moved to sidebar)

            if safety['dangers']:
                # DANGEROUS - Show warning AND rating side by side
                col_danger, col_rate_danger = st.columns([1, 1])
                with col_danger:
                    st.error("🚨 **UNSAFE DESIGN**")
                    for danger in safety['dangers']:
                        st.caption(danger)
                    st.caption("⚠️ Fix safety issues")
                
                with col_rate_danger:
                    if predicted_rating is not None:
                        st.markdown(f'<div style="font-size: 2rem; font-weight: bold; text-align: center; color: #FFD700;">⭐ {predicted_rating:.2f}</div>', 
                                   unsafe_allow_html=True)
                        st.caption("Rating (if rideable)")
                    else:
                        st.info("Click 'Rate this track' to compute AI score")
                
            elif safety['warnings']:
                # Has warnings - compact format with rating
                col_warn, col_rate = st.columns([1, 1])
                with col_warn:
                    st.warning("**⚠️ SAFETY CONCERNS**")
                    for warning in safety['warnings']:
                        st.caption(warning)
                
                with col_rate:
                    if predicted_rating is not None:
                        st.markdown(f'<div style="font-size: 2rem; font-weight: bold; text-align: center; color: #FFD700;">⭐ {predicted_rating:.2f}</div>', 
                                   unsafe_allow_html=True)
                        st.caption("⚠️ Has comfort issues")
                    else:
                        st.info("Click 'Rate this track' to compute AI score")
                
            else:
                # SAFE - Compact two-column layout
                col_bucket, col_rating = st.columns([1, 1])
                
                with col_bucket:
                    # Interpretation
                    if predicted_rating >= 4.5:
                        st.success("🔥 **World-Class!**")
                        st.balloons()
                    elif predicted_rating >= 4.0:
                        st.success("🎉 **Excellent!**")
                    elif predicted_rating >= 3.5:
                        st.info("👍 **Great Ride!**")
                    elif predicted_rating >= 3.0:
                        st.info("😊 **Solid Design**")
                    else:
                        st.warning("💡 **Needs More Excitement**")
                
                with col_rating:
                    # Numerical rating with smaller display
                    if predicted_rating is not None:
                        st.markdown(f'<div style="font-size: 2.5rem; font-weight: bold; text-align: center; color: #FFD700; margin-top: 0;">⭐ {predicted_rating:.2f}</div>', 
                                   unsafe_allow_html=True)
                        st.progress(predicted_rating / 5.0)
                    else:
                        st.info("Click 'Rate this track' to compute AI score")
            
            # On-demand 3D view (resource heavy)
            st.divider()
            st.markdown("**🧭 On-Demand 3D View**")
            import plotly.graph_objects as go
            downsample = st.slider("Preview resolution", 200, 2000, 1200, 100,
                                    help="Fewer points = faster rendering")
            if st.button("🎥 Generate 3D View", help="Render a 3D preview of the current track"):
                x = np.array(st.session_state.track_x)
                # Map vertical profile to Z axis for correct orientation
                z = np.array(st.session_state.track_y)
                # Lateral Y axis: use provided track_z if any, else zeros
                y = np.array(st.session_state.get('track_z', np.zeros_like(x)))
                n = len(x)
                if n > downsample:
                    idx = np.linspace(0, n-1, downsample).astype(int)
                    x, y, z = x[idx], y[idx], z[idx]
                fig3d = go.Figure(data=[
                    go.Scatter3d(x=x, y=y, z=z,
                                 mode='lines', line=dict(width=6, color='#1f77b4'),
                                 name='Track')
                ])
                fig3d.update_layout(
                    scene=dict(
                        xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
                        aspectmode='data',
                        camera=dict(up=dict(x=0, y=0, z=1))
                    ),
                    height=600, margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig3d, use_container_width=True)

        else:
            st.error("Track too short for AI analysis")
            # Fallback to simple g-force analysis
            initial_speed = st.session_state.get('initial_speed', 15.0)
            st.session_state.accel_df = simple_gforce_analysis(
                st.session_state.track_x, 
                st.session_state.track_y, 
                initial_speed=initial_speed
            )
            
    except Exception as e:
        st.error(f"AI Error: {str(e)}")
        st.caption("Using simple physics model instead...")
        # Fallback to simple g-force analysis
        initial_speed = st.session_state.get('initial_speed', 15.0)
        st.session_state.accel_df = simple_gforce_analysis(
            st.session_state.track_x, 
            st.session_state.track_y, 
            initial_speed=initial_speed
        )
    
    # Ensure we always have g-force data for the plots
    if not hasattr(st.session_state, 'accel_df'):
        initial_speed = st.session_state.get('initial_speed', 15.0)
        st.session_state.accel_df = simple_gforce_analysis(
            st.session_state.track_x, 
            st.session_state.track_y, 
            initial_speed=initial_speed
        )
    
    st.divider()
    
    # Row 1: Track profile and Statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Track Profile (Side View)**")
        fig_profile = go.Figure()
        
        fig_profile.add_trace(go.Scatter(
            x=st.session_state.track_x,
            y=st.session_state.track_y,
            mode='lines',
            line=dict(color='rgb(255, 75, 75)', width=4),
            name='Track',
            hovertemplate='Distance: %{x:.1f}m<br>Height: %{y:.1f}m<extra></extra>'
        ))
        
        # Ground line
        fig_profile.add_shape(
            type="line",
            x0=0, x1=max(st.session_state.track_x),
            y0=0, y1=0,
            line=dict(color="green", width=2, dash="dash")
        )
        
        fig_profile.update_layout(
            xaxis_title="Distance (m)",
            yaxis_title="Height (m)",
            showlegend=False,
            height=350,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        
        st.plotly_chart(fig_profile, use_container_width=True)
    
    with col2:
        st.markdown("**Track Statistics**")
        
        # Calculate stats
        x = st.session_state.track_x
        y = st.session_state.track_y
        
        total_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        max_height = np.max(y)
        min_height = np.min(y)
        max_drop = max_height - min_height
        
        # Check track smoothness (allow steep start for lift hill)
        is_smooth, max_angle, _ = check_track_smoothness(x, y, max_angle_deg=30, allow_steep_start=True)
        
        # Get angle after lift section for display
        dx = np.diff(x)
        dy = np.diff(y)
        angles = np.abs(np.arctan2(dy, dx) * 180 / np.pi)
        lift_section_end = int(len(angles) * 0.2)
        max_angle_after_lift = np.max(angles[lift_section_end:]) if len(angles) > lift_section_end else max_angle
        
        smoothness_emoji = "✅" if max_angle_after_lift <= 30 else "⚠️"
        
        # Display metrics
        col_a, col_b = st.columns(2)
        col_a.metric("Track Length", f"{total_length:.0f} m")
        col_a.metric("Max Height", f"{max_height:.1f} m")
        col_b.metric("Total Drop", f"{max_drop:.1f} m")
        col_b.metric("Max Angle (excl. lift)", f"{max_angle_after_lift:.1f}° {smoothness_emoji}")
        
        if max_angle_after_lift > 30:
            st.caption("⚠️ Track has steep sections (>30°) - smoothing applied")
        else:
            st.caption("✅ Track angles within safe limits (<30°)")
        
        # Curvature smoothness indicator
        if hasattr(st.session_state, 'track_curvature'):
            curvature = st.session_state.track_curvature
            max_curvature = np.max(curvature)
            avg_curvature = np.mean(curvature)
            
            st.markdown("**Track Smoothness:**")
            st.text(f"Avg: {avg_curvature:.4f} | Max: {max_curvature:.4f}")
            
            if max_curvature < 0.01:
                st.caption("✅ Very smooth track")
            elif max_curvature < 0.05:
                st.caption("✅ Smooth track")
            else:
                st.caption("⚠️ Some tight curves present")
        
        # Block breakdown - compact format
        st.markdown("**Block Sequence:**")
        block_summary = " → ".join([f"{block_info['block'].icon}" for block_info in st.session_state.track_sequence])
        st.markdown(f"<div style='font-size: 1.5rem; line-height: 2rem;'>{block_summary}</div>", unsafe_allow_html=True)
        
        # Count by type
        block_counts = {}
        for block_info in st.session_state.track_sequence:
            name = block_info['block'].name
            block_counts[name] = block_counts.get(name, 0) + 1
        
        count_text = " | ".join([f"{count}× {name}" for name, count in block_counts.items()])
        st.caption(count_text)
    
    st.divider()
    
    # Show joint smoothing info
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        if hasattr(st.session_state, 'joint_smoothing_applied'):
            st.info(st.session_state.joint_smoothing_applied)
    with info_col2:
        if hasattr(st.session_state, 'smoothness_warning') and st.session_state.smoothness_warning:
            st.warning(st.session_state.smoothness_warning)
    
    # Ride configuration
    st.subheader("⚙️ Ride Configuration")
    initial_speed = st.slider("Initial Speed (m/s)", 5.0, 30.0, 15.0, 1.0, key="speed_slider")
    st.caption("Launch speed or chain lift speed")
    st.session_state.initial_speed = initial_speed
    
    st.divider()
    
    # Row 1.5: Curvature Analysis
    if hasattr(st.session_state, 'track_curvature'):
        st.markdown("**Track Curvature Profile**")
        
        fig_curve = go.Figure()
        
        # Calculate distance along track
        x = st.session_state.track_x
        y = st.session_state.track_y
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        distance = np.cumsum(np.sqrt(dx**2 + dy**2))
        
        curvature = st.session_state.track_curvature
        
        # Plot curvature
        fig_curve.add_trace(go.Scatter(
            x=distance,
            y=curvature,
            mode='lines',
            line=dict(color='purple', width=2),
            fill='tozeroy',
            fillcolor='rgba(128, 0, 128, 0.1)',
            name='Curvature',
            hovertemplate='Distance: %{x:.1f}m<br>Curvature: %{y:.4f}<extra></extra>'
        ))
        
        # Add threshold lines
        fig_curve.add_hline(y=0.01, line_dash="dash", line_color="green", 
                           annotation_text="Smooth", annotation_position="right")
        fig_curve.add_hline(y=0.05, line_dash="dash", line_color="orange",
                           annotation_text="Tight", annotation_position="right")
        
        fig_curve.update_layout(
            xaxis_title="Distance Along Track (m)",
            yaxis_title="Curvature (1/m)",
            showlegend=False,
            height=200,
            margin=dict(l=20, r=20, t=10, b=40)
        )
        
        st.plotly_chart(fig_curve, use_container_width=True)
        st.caption("💡 Lower curvature = smoother ride. Spikes indicate sharp transitions.")
    
    st.divider()
    
    # Row 2: G-Force Analysis
    st.markdown("**G-Force Analysis**")
    
    if 'accel_df' in st.session_state:
        accel_df = st.session_state.accel_df
        
        # Create subplots for each G-force
        fig_g = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Vertical G-Forces", "Lateral G-Forces", "Longitudinal G-Forces"),
            vertical_spacing=0.12
        )
        
        time_max = accel_df['Time'].max()
        
        # Vertical G-forces with safety zones
        fig_g.add_trace(
            go.Scatter(x=accel_df['Time'], y=accel_df['Vertical'],
                      name='Vertical', line=dict(color='red', width=2)),
            row=1, col=1
        )
        # Safety zones for vertical
        fig_g.add_hrect(y0=5, y1=10, fillcolor="red", opacity=0.1, 
                       annotation_text="Dangerous", annotation_position="top left",
                       row=1, col=1)
        fig_g.add_hrect(y0=3, y1=5, fillcolor="orange", opacity=0.1,
                       annotation_text="Intense", annotation_position="top left",
                       row=1, col=1)
        fig_g.add_hrect(y0=-3, y1=-10, fillcolor="red", opacity=0.1,
                       annotation_text="Dangerous", annotation_position="bottom left",
                       row=1, col=1)
        fig_g.add_hrect(y0=-2, y1=-3, fillcolor="orange", opacity=0.1,
                       annotation_text="High", annotation_position="bottom left",
                       row=1, col=1)
        
        # Lateral G-forces with safety zones
        fig_g.add_trace(
            go.Scatter(x=accel_df['Time'], y=accel_df['Lateral'],
                      name='Lateral', line=dict(color='blue', width=2)),
            row=2, col=1
        )
        fig_g.add_hrect(y0=2, y1=10, fillcolor="orange", opacity=0.1,
                       annotation_text="Uncomfortable", annotation_position="top left",
                       row=2, col=1)
        fig_g.add_hrect(y0=-2, y1=-10, fillcolor="orange", opacity=0.1,
                       annotation_text="Uncomfortable", annotation_position="bottom left",
                       row=2, col=1)
        fig_g.add_hrect(y0=5, y1=10, fillcolor="red", opacity=0.15,
                       row=2, col=1)
        fig_g.add_hrect(y0=-5, y1=-10, fillcolor="red", opacity=0.15,
                       row=2, col=1)
        
        # Longitudinal G-forces
        fig_g.add_trace(
            go.Scatter(x=accel_df['Time'], y=accel_df['Longitudinal'],
                      name='Longitudinal', line=dict(color='green', width=2)),
            row=3, col=1
        )
        fig_g.add_hrect(y0=3, y1=10, fillcolor="orange", opacity=0.1,
                       annotation_text="Intense", annotation_position="top left",
                       row=3, col=1)
        fig_g.add_hrect(y0=-3, y1=-10, fillcolor="orange", opacity=0.1,
                       annotation_text="Intense", annotation_position="bottom left",
                       row=3, col=1)
        
        # Add zero lines
        for i in range(1, 4):
            fig_g.add_hline(y=0, line_dash="dash", line_color="gray", row=i, col=1)
        
        fig_g.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig_g.update_yaxes(title_text="G", row=2, col=1)
        
        fig_g.update_layout(
            height=500,
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig_g, use_container_width=True)

    # Egg Plot Visualization (comfort envelopes)
    if 'accel_df' in st.session_state:
        accel_df = st.session_state.accel_df
        st.divider()
        st.markdown("**Egg Plots (Comfort Envelopes)**")

        import plotly.graph_objects as go

        # Helper to make an ellipse path
        def ellipse_points(rx, ry, n=180):
            t = np.linspace(0, 2*np.pi, n)
            return rx*np.cos(t), ry*np.sin(t)

        col_egg1, col_egg2, col_egg3 = st.columns(3)

        # Lateral vs Vertical egg
        with col_egg1:
            fig_egg_lv = go.Figure()
            fig_egg_lv.add_trace(go.Scatter(
                x=accel_df['Lateral'], y=accel_df['Vertical'],
                mode='markers', marker=dict(size=4, color=accel_df['Time'], colorscale='Viridis'),
                name='Samples',
                hovertemplate='Lat: %{x:.2f} g<br>Vert: %{y:.2f} g<extra></extra>'
            ))
            ex, ey = ellipse_points(2.0, 3.0)
            fig_egg_lv.add_trace(go.Scatter(x=ex, y=ey, mode='lines', line=dict(color='green'), name='Comfort'))
            ex2, ey2 = ellipse_points(3.5, 4.5)
            fig_egg_lv.add_trace(go.Scatter(x=ex2, y=ey2, mode='lines', line=dict(color='orange', dash='dash'), name='Intense'))
            ex3, ey3 = ellipse_points(5.0, 6.0)
            fig_egg_lv.add_trace(go.Scatter(x=ex3, y=ey3, mode='lines', line=dict(color='red', dash='dot'), name='Danger'))
            # Set equal aspect ratio and fixed symmetric ranges for circular look
            fixed_r = 6.0
            fig_egg_lv.update_layout(
                title='Lat vs Vert (g)', xaxis_title='Lateral (g)', yaxis_title='Vertical (g)',
                height=360, width=360, margin=dict(l=10, r=10, t=30, b=10),
                xaxis=dict(range=[-fixed_r, fixed_r], scaleanchor='y', scaleratio=1),
                yaxis=dict(range=[-fixed_r, fixed_r])
            )
            st.plotly_chart(fig_egg_lv, use_container_width=False)

        # Longitudinal vs Vertical egg
        with col_egg2:
            fig_egg_lv2 = go.Figure()
            fig_egg_lv2.add_trace(go.Scatter(
                x=accel_df['Longitudinal'], y=accel_df['Vertical'],
                mode='markers', marker=dict(size=4, color=accel_df['Time'], colorscale='Viridis'),
                name='Samples',
                hovertemplate='Long: %{x:.2f} g<br>Vert: %{y:.2f} g<extra></extra>'
            ))
            exa, eya = ellipse_points(3.0, 3.0)
            fig_egg_lv2.add_trace(go.Scatter(x=exa, y=eya, mode='lines', line=dict(color='green'), name='Comfort'))
            exa2, eya2 = ellipse_points(4.0, 4.5)
            fig_egg_lv2.add_trace(go.Scatter(x=exa2, y=eya2, mode='lines', line=dict(color='orange', dash='dash'), name='Intense'))
            exa3, eya3 = ellipse_points(6.0, 6.0)
            fig_egg_lv2.add_trace(go.Scatter(x=exa3, y=eya3, mode='lines', line=dict(color='red', dash='dot'), name='Danger'))
            # Equal aspect ratio and fixed symmetric ranges
            fixed_r2 = 6.0
            fig_egg_lv2.update_layout(
                title='Long vs Vert (g)', xaxis_title='Longitudinal (g)', yaxis_title='Vertical (g)',
                height=360, width=360, margin=dict(l=10, r=10, t=30, b=10),
                xaxis=dict(range=[-fixed_r2, fixed_r2], scaleanchor='y', scaleratio=1),
                yaxis=dict(range=[-fixed_r2, fixed_r2])
            )
            st.plotly_chart(fig_egg_lv2, use_container_width=False)

        # Lateral vs Longitudinal egg
        with col_egg3:
            fig_egg_ll = go.Figure()
            fig_egg_ll.add_trace(go.Scatter(
                x=accel_df['Lateral'], y=accel_df['Longitudinal'],
                mode='markers', marker=dict(size=4, color=accel_df['Time'], colorscale='Viridis'),
                name='Samples',
                hovertemplate='Lat: %{x:.2f} g<br>Long: %{y:.2f} g<extra></extra>'
            ))
            exb, eyb = ellipse_points(2.0, 3.0)
            fig_egg_ll.add_trace(go.Scatter(x=exb, y=eyb, mode='lines', line=dict(color='green'), name='Comfort'))
            exb2, eyb2 = ellipse_points(3.5, 4.0)
            fig_egg_ll.add_trace(go.Scatter(x=exb2, y=eyb2, mode='lines', line=dict(color='orange', dash='dash'), name='Intense'))
            exb3, eyb3 = ellipse_points(5.0, 6.0)
            fig_egg_ll.add_trace(go.Scatter(x=exb3, y=eyb3, mode='lines', line=dict(color='red', dash='dot'), name='Danger'))
            # Equal aspect ratio and fixed symmetric ranges
            fixed_r3 = 6.0
            fig_egg_ll.update_layout(
                title='Lat vs Long (g)', xaxis_title='Lateral (g)', yaxis_title='Longitudinal (g)',
                height=360, width=360, margin=dict(l=10, r=10, t=30, b=10),
                xaxis=dict(range=[-fixed_r3, fixed_r3], scaleanchor='y', scaleratio=1),
                yaxis=dict(range=[-fixed_r3, fixed_r3])
            )
            st.plotly_chart(fig_egg_ll, use_container_width=False)
    
    st.divider()
    
    # Row 3: G-Force Safety Analysis
    if 'accel_df' in st.session_state:
        accel_df = st.session_state.accel_df
        safety = check_gforce_safety(accel_df)
        
        st.subheader(f"{safety['emoji']} G-Force Safety Analysis")
        
        # Safety status banner
        if safety['color'] == 'error':
            st.error(f"**Safety Level: {safety['level']}**")
        elif safety['color'] == 'warning':
            st.warning(f"**Safety Level: {safety['level']}**")
        else:
            st.success(f"**Safety Level: {safety['level']}**")
        
        # Show all dangers first
        if safety['dangers']:
            st.error("**⚠️ CRITICAL SAFETY ISSUES:**")
            for danger in safety['dangers']:
                st.markdown(f"- {danger}")
        
        # Show warnings
        if safety['warnings']:
            if not safety['dangers']:  # Only show warning box if no dangers
                st.warning("**⚠️ Safety Warnings:**")
            for warning in safety['warnings']:
                st.markdown(f"- {warning}")
        
        # If no issues, show positive message
        if not safety['warnings'] and not safety['dangers']:
            st.info("✅ All g-forces within comfortable limits for riders")
    
    st.divider()
    
    # Row 4: Detailed Ride Statistics
    st.subheader("📊 Detailed Ride Statistics")
    
    col5, col6, col7, col8 = st.columns(4)
    
    if 'accel_df' in st.session_state:
        accel_df = st.session_state.accel_df
        safety = check_gforce_safety(accel_df)
        
        # Color-code metrics based on safety
        max_g_vert = safety['max_vertical']
        min_g_vert = safety['min_vertical']
        max_g_lat = safety['max_lateral']
        max_g_long = safety['max_longitudinal']
        
        # Determine colors
        vert_color = "🚨" if max_g_vert > 5 else ("⚠️" if max_g_vert > 3 else "✅")
        neg_color = "🚨" if min_g_vert < -3 else ("⚠️" if min_g_vert < -2 else "✅")
        lat_color = "🚨" if max_g_lat > 5 else ("⚠️" if max_g_lat > 2 else "✅")
        
        col5.metric("Max Vertical G", f"{max_g_vert:.2f}g {vert_color}")
        col6.metric("Min Vertical G", f"{min_g_vert:.2f}g {neg_color}")
        col7.metric("Max Lateral G", f"{max_g_lat:.2f}g {lat_color}")
        col8.metric("Ride Duration", f"{accel_df['Time'].max():.1f}s")
        
        # Airtime detection
        airtime_mask = accel_df['Vertical'] < -0.1
        airtime_seconds = np.sum(airtime_mask) * 0.1
        
        if airtime_seconds > 0:
            st.success(f"🎈 **Airtime Detected:** {airtime_seconds:.1f} seconds of negative G-forces!")

else:
    # Welcome screen
    st.info("👈 Add building blocks from the sidebar to start building your roller coaster!")
    
    # Show example blocks
    st.subheader("Available Building Blocks")
    
    cols = st.columns(4)
    for idx, (key, block) in enumerate(BLOCK_LIBRARY.items()):
        with cols[idx % 4]:
            st.markdown(f"""
            <div class="block-card">
            <h3 style="text-align: center; margin: 0;">{block.icon}</h3>
            <p style="text-align: center; margin: 0.5rem 0 0 0;"><b>{block.name}</b></p>
            <p style="text-align: center; font-size: 0.8rem; color: gray;">{block.description}</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9rem;">
    🎢 Powered by BiGRU Neural Network trained on 359 real roller coasters<br>
    Using accelerometer data from RideForcesDB and ratings from Captain Coaster
</div>
""", unsafe_allow_html=True)
