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
from scipy import stats
import sys
import os

# Import utilities
from utils.accelerometer_transform import track_to_accelerometer_data
from utils.lgbm_predictor import predict_score_lgb, compute_lightgbm_features
from utils.track_library import ensure_library, pick_random_entry, load_entry, add_entry


def _is_local_debug_mode():
    """Check if running in local debug mode (not deployment)."""
    # In deployment, Streamlit Cloud sets these environment variables
    is_deployment = (
        os.getenv('STREAMLIT_SERVER_ENV') is not None or 
        os.getenv('STREAMLIT_SHARING') is not None
    )
    return not is_deployment
from utils.track_blocks import (
    lift_hill_profile,
    vertical_drop_profile,
    loop_profile,
    airtime_hill_profile,
    spiral_profile,
    banked_turn_profile,
    bunny_hop_profile,
    launch_profile,
    flat_section_profile,
    brake_run_profile,
)

# Set page layout and main header
st.set_page_config(page_title="Roller Coaster Builder", page_icon="🎢", layout="wide")
st.title("🎢 Roller Coaster Builder")

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

# Define all available blocks
BLOCK_LIBRARY = {
        "lift_hill": TrackBlock("Lift Hill", "Chain lift to initial height", "⛰️", lift_hill_profile),
        "drop": TrackBlock("Vertical Drop", "Steep initial drop", "⬇️", vertical_drop_profile),
        "loop": TrackBlock("Clothoid Loop", "Smooth clothoid loop element", "🔄", loop_profile),
        "airtime_hill": TrackBlock("Airtime Hill", "Floater airtime moment", "🎈", airtime_hill_profile),
        "spiral": TrackBlock("Spiral", "Helix/corkscrew element", "🌀", spiral_profile),
        "bunny_hop": TrackBlock("Bunny Hop", "Quick airtime bump", "🐰", bunny_hop_profile),
        "banked_turn": TrackBlock("Banked Turn", "High-speed turn", "↪️", banked_turn_profile),
        "launch": TrackBlock("Launch Section", "Magnetic acceleration boost (LSM/LIM)", "🚀", launch_profile),
        "brake_run": TrackBlock("Brake Run", "Final braking section to stop", "🛑", brake_run_profile),
        "flat_section": TrackBlock("Flat Section", "Straight section", "➡️", flat_section_profile),
    }

# Always start with a complete starter track on first load
if 'initialized' not in st.session_state:
    st.session_state.track_sequence = [
        {
            'type': 'launch',
            'block': BLOCK_LIBRARY['launch'],
            'params': {'length': 60, 'speed_boost': 30}  # Longer launch, higher speed for more energy
        },
        {
            'type': 'lift_hill',
            'block': BLOCK_LIBRARY['lift_hill'],
            'params': {'length': 120, 'height': 120}  # Much higher lift hill for more potential energy
        },
        {
            'type': 'drop',
            'block': BLOCK_LIBRARY['drop'],
            'params': {'height': 120, 'steepness': 1.0}  # Higher drop = more speed = higher G-forces
        },
        {
            'type': 'loop',
            'block': BLOCK_LIBRARY['loop'],
            'params': {'diameter': 35}  # Larger loop for higher G-forces
        },
        {
            'type': 'flat_section',
            'block': BLOCK_LIBRARY['flat_section'],
            'params': {'length': 40}  # Flat section between loop and airtime hill
        },
        {
            'type': 'airtime_hill',
            'block': BLOCK_LIBRARY['airtime_hill'],
            'params': {'length': 80, 'height': 25}  # Larger airtime hill
        },
        {
            'type': 'flat_section',
            'block': BLOCK_LIBRARY['flat_section'],
            'params': {'length': 60}  # Flat section (replaced second drop)
        },
        {
            'type': 'loop',
            'block': BLOCK_LIBRARY['loop'],
            'params': {'diameter': 30}  # Second loop
        },
        {
            'type': 'flat_section',
            'block': BLOCK_LIBRARY['flat_section'],
            'params': {'length': 40}  # Flat section between loop and airtime hill
        },
        {
            'type': 'airtime_hill',
            'block': BLOCK_LIBRARY['airtime_hill'],
            'params': {'length': 70, 'height': 20}  # Second airtime hill
        },
        {
            'type': 'flat_section',
            'block': BLOCK_LIBRARY['flat_section'],
            'params': {'length': 50}  # Transition section
        },
        {
            'type': 'flat_section',
            'block': BLOCK_LIBRARY['flat_section'],
            'params': {'length': 50}  # Flat section (replaced third drop)
        },
        {
            'type': 'airtime_hill',
            'block': BLOCK_LIBRARY['airtime_hill'],
            'params': {'length': 60, 'height': 18}  # Third airtime hill
        },
        {
            'type': 'flat_section',
            'block': BLOCK_LIBRARY['flat_section'],
            'params': {'length': 40}  # Transition
        },
        {
            'type': 'brake_run',
            'block': BLOCK_LIBRARY['brake_run'],
            'params': {'length': 50}  # Longer brake run
        }
    ]
    st.session_state.initialized = True

# The rest of the Builder UI continues below (unchanged code)...

# ============================================================================
# SIDEBAR: BUILDING BLOCK PALETTE
# ============================================================================

with st.sidebar:
    st.header("🧱 Building Blocks")
    
    # Show success message from random generation if it exists
    if 'random_gen_success' in st.session_state and st.session_state.random_gen_success:
        st.success(st.session_state.random_gen_success)
        # Clear the message after showing it
        st.session_state.random_gen_success = None
    
    # Quick Start section at the top
    st.subheader("🎲 Quick Start")
    
    col_rand1_top, col_rand2_top = st.columns(2)
    

    
    with col_rand1_top:
        if st.button("🎲 Random Template", key=f"btn_random_template_quickstart", use_container_width=True, help="Generate a random coaster with 5-10 blocks"):
            import random
            
            try:
                # Random number of blocks (targeting ~500m)
                num_blocks = random.randint(6, 10)
                
                # Always start with launch, then lift hill followed by a vertical drop and a flat section
                new_sequence = [
                {
                    'type': 'launch',
                    'block': BLOCK_LIBRARY['launch'],
                    'params': {
                        'length': random.randint(30, 50),
                        'speed_boost': random.randint(20, 28)
                    }
                },
                {
                    'type': 'lift_hill',
                    'block': BLOCK_LIBRARY['lift_hill'],
                    'params': {
                        'length': random.randint(50, 80),
                        'height': random.randint(40, 70)
                    }
                },
                {
                    'type': 'drop',
                    'block': BLOCK_LIBRARY['drop'],
                    'params': {
                        'height': random.randint(40, 70),
                        'steepness': random.uniform(0.8, 1.0)
                    }
                },
                {
                    'type': 'flat_section',
                    'block': BLOCK_LIBRARY['flat_section'],
                    'params': {'length': random.randint(20, 40)}
                }
                ]
                
                # Available blocks (after initial drop + flat)
                # Exclude banked_turn and spiral to avoid lateral forces in random designs
                available_blocks = ['drop', 'loop', 'airtime_hill', 'bunny_hop', 'launch']

                # Record the initial drop height to cap later drops to <= 1/3
                first_drop_height = next((b['params']['height'] for b in new_sequence if b['type'] == 'drop'), None)
                
                for i in range(num_blocks - 4):  # -4 because we added launch + lift + drop + flat
                    block_type = random.choice(available_blocks)
                    # Enforce: at least one flat section in every 3 blocks
                    # Check last two blocks (after the initial lift+drop+flat)
                    recent_types = [b['type'] for b in new_sequence[-2:]] if len(new_sequence) >= 2 else []
                    # If this would make three consecutive without a flat, force a flat_section
                    if (i % 3 == 2) and ('flat_section' not in recent_types):
                        block_type = 'flat_section'
                    
                    # Initialize params to None to catch any unhandled block types
                    params = None
                    
                    if block_type == 'drop':
                        # Cap height to at most 1/3 of the original first drop
                        if first_drop_height is not None:
                            max_drop = max(10, int(first_drop_height / 3))
                            params = {
                                'height': random.randint(10, max_drop),
                                'steepness': random.uniform(0.7, 0.9)
                            }
                        else:
                            params = {
                                'height': random.randint(25, 50),
                                'steepness': random.uniform(0.7, 0.9)
                            }
                    elif block_type == 'loop':
                        params = {'diameter': random.randint(20, 35)}
                    elif block_type == 'airtime_hill':
                        params = {
                            'length': random.randint(30, 60),
                            'height': random.randint(10, 20)
                        }
                    elif block_type == 'spiral':
                        params = {
                            'diameter': random.randint(20, 30),
                            'turns': random.uniform(1.0, 2.0)
                        }
                    elif block_type == 'bunny_hop':
                        params = {
                            'length': random.randint(15, 30),
                            'height': random.randint(5, 12)
                        }
                    elif block_type == 'banked_turn':
                        params = {
                            'radius': random.randint(20, 35),
                            'angle': random.randint(60, 120)
                        }
                    elif block_type == 'launch':
                        params = {
                            'length': random.randint(30, 50),
                            'speed_boost': random.randint(20, 28)
                        }
                    elif block_type == 'flat_section':
                        params = {
                            'length': random.randint(20, 40)
                        }
                    elif block_type == 'brake_run':
                        params = {
                            'length': random.randint(25, 40)
                        }
                    else:
                        # Fallback for any unhandled block types
                        st.error(f"Unknown block type in random generation: {block_type}")
                        params = {'length': 30}  # Default params
                    
                    # Only append if params were successfully set
                    if params is not None:
                        new_sequence.append({
                            'type': block_type,
                            'block': BLOCK_LIBRARY[block_type],
                            'params': params
                        })
                    else:
                        st.error(f"Failed to generate params for block type: {block_type}")
                
                # Always end with a brake run
                new_sequence.append({
                    'type': 'brake_run',
                    'block': BLOCK_LIBRARY['brake_run'],
                    'params': {'length': random.randint(25, 40)}
                })
                
                st.session_state.track_sequence = new_sequence
                # Clear cached metrics to force recomputation on rerun
                st.session_state.predicted_rating = None
                st.session_state.accel_df = None
                st.session_state.airtime_metrics = None
                st.session_state.track_generated = False
                # Enforce end level equals start for random generation
                st.session_state.force_end_level = True
                st.session_state.start_level = 0.0
                # Disable 3D for random designs (2D tracks should have no lateral forces)
                st.session_state.random_3d = False
                # Store success message in session state to show after rerun
                st.session_state.random_gen_success = f"🎲 Generated random coaster with {num_blocks} blocks!"
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error generating random coaster: {str(e)}")
                import traceback
                st.exception(e)
    
    with col_rand2_top:
        if st.button("🔄 \n Reset to Default", key=f"btn_reset_default_quickstart", use_container_width=True, help="Reset to the starter template"):
            st.session_state.track_sequence = [
                {
                    'type': 'launch',
                    'block': BLOCK_LIBRARY['launch'],
                    'params': {'length': 60, 'speed_boost': 30}
                },
                {
                    'type': 'lift_hill',
                    'block': BLOCK_LIBRARY['lift_hill'],
                    'params': {'length': 120, 'height': 120}
                },
                {
                    'type': 'drop',
                    'block': BLOCK_LIBRARY['drop'],
                    'params': {'height': 120, 'steepness': 1.0}
                },
                {
                    'type': 'loop',
                    'block': BLOCK_LIBRARY['loop'],
                    'params': {'diameter': 35}
                },
                {
                    'type': 'flat_section',
                    'block': BLOCK_LIBRARY['flat_section'],
                    'params': {'length': 40}
                },
                {
                    'type': 'airtime_hill',
                    'block': BLOCK_LIBRARY['airtime_hill'],
                    'params': {'length': 80, 'height': 25}
                },
                {
                    'type': 'flat_section',
                    'block': BLOCK_LIBRARY['flat_section'],
                    'params': {'length': 60}
                },
                {
                    'type': 'loop',
                    'block': BLOCK_LIBRARY['loop'],
                    'params': {'diameter': 30}
                },
                {
                    'type': 'flat_section',
                    'block': BLOCK_LIBRARY['flat_section'],
                    'params': {'length': 40}
                },
                {
                    'type': 'airtime_hill',
                    'block': BLOCK_LIBRARY['airtime_hill'],
                    'params': {'length': 70, 'height': 20}
                },
                {
                    'type': 'flat_section',
                    'block': BLOCK_LIBRARY['flat_section'],
                    'params': {'length': 50}
                },
                {
                    'type': 'flat_section',
                    'block': BLOCK_LIBRARY['flat_section'],
                    'params': {'length': 50}
                },
                {
                    'type': 'airtime_hill',
                    'block': BLOCK_LIBRARY['airtime_hill'],
                    'params': {'length': 60, 'height': 18}
                },
                {
                    'type': 'brake_run',
                    'block': BLOCK_LIBRARY['brake_run'],
                    'params': {'length': 50}
                }
            ]
            st.session_state.track_generated = False
            st.session_state.force_end_level = False
            st.success("🔄 Reset to default template!")
            st.rerun()

    # Precomputed safe library (hidden)
    # st.subheader("📚 Precomputed Safe Library")
    # Hidden per request. Library UI and loading disabled.
    
    st.divider()
    
    # Save current design to precomputed library (hidden per request)
    # st.subheader("💾 Save Design")
    # new_lib_name = st.text_input("Entry name", value="my_design")
    # if st.button("📚 Add Design to Library", use_container_width=True, help="Save current track with physics to the library"):
    #     if st.session_state.get('track_generated'):
    #         track_df = pd.DataFrame({
    #             'x': st.session_state.track_x,
    #             'y': st.session_state.track_y,
    #             'z': np.array(st.session_state.get('track_z', np.zeros_like(st.session_state.track_x)))
    #         })
    #         elements = st.session_state.get('track_sequence', [])
    #         meta = add_entry(new_lib_name, elements, track_df)
    #         if meta:
    #             st.success(f"Saved to library as '{new_lib_name}'")
    #         else:
    #             st.error("Failed to save to library")
    #     else:
    #         st.warning("Generate a track first, then save.")

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
    
    # Physics engine selection (hidden, default to Advanced)
    # st.divider()
    # st.subheader("⚙️ Physics Engine")
    # physics_mode = st.radio(
    #     "G-Force Calculation Method",
    #     options=["Advanced (Realistic)", "Simple (Geometric)"],
    #     index=0,
    #     key="physics_mode_selector",
    #     help="Both use pure energy conservation from track geometry. Advanced: Full 3D physics with detailed curvature. Simple: Frenet-Serret frame calculation. Use Launch blocks to add initial speed!"
    # )
    # Store in session state - default to Advanced
    if 'physics_mode' not in st.session_state:
        st.session_state.physics_mode = "Advanced (Realistic)"
    physics_mode = st.session_state.physics_mode  # Use stored value (default: Advanced)
    
    # Physics Parameters Section (hidden by default, accessible via expander)
    st.divider()
    with st.expander("⚙️ Physics Parameters", expanded=False):
        # Initialize physics parameters in session state if not present
        if 'physics_mass' not in st.session_state:
            st.session_state.physics_mass = 500.0
        if 'physics_A' not in st.session_state:
            st.session_state.physics_A = 2.0
        if 'physics_Cd' not in st.session_state:
            st.session_state.physics_Cd = 0.1
        if 'physics_mu' not in st.session_state:
            st.session_state.physics_mu = 0.001
        if 'physics_rho' not in st.session_state:
            st.session_state.physics_rho = 1.2
        
        # Store previous values to detect changes
        prev_mass = st.session_state.physics_mass
        prev_A = st.session_state.physics_A
        prev_Cd = st.session_state.physics_Cd
        prev_mu = st.session_state.physics_mu
        prev_rho = st.session_state.physics_rho
        
        # Mass (kg)
        st.session_state.physics_mass = st.number_input(
            "Mass (kg)",
            min_value=10.0,
            max_value=10000.0,
            value=st.session_state.physics_mass,
            step=50.0,
            help="Cart mass. Lower = more acceleration, more sensitive"
        )
        
        # Frontal Area (m²)
        st.session_state.physics_A = st.number_input(
            "Frontal Area (m²)",
            min_value=0.5,
            max_value=10.0,
            value=st.session_state.physics_A,
            step=0.1,
            help="Cross-sectional area. Smaller = less drag"
        )
        
        # Drag Coefficient
        st.session_state.physics_Cd = st.number_input(
            "Drag Coefficient",
            min_value=0.01,
            max_value=2.0,
            value=st.session_state.physics_Cd,
            step=0.01,
            help="Aerodynamic drag coefficient. Lower = less air resistance"
        )
        
        # Friction Coefficient
        st.session_state.physics_mu = st.number_input(
            "Friction Coefficient",
            min_value=0.0,
            max_value=0.1,
            value=st.session_state.physics_mu,
            step=0.0001,
            format="%.4f",
            help="Rolling friction. Lower = smoother rails, less energy loss"
        )
        
        # Air Density (kg/m³)
        st.session_state.physics_rho = st.number_input(
            "Air Density (kg/m³)",
            min_value=0.5,
            max_value=2.0,
            value=st.session_state.physics_rho,
            step=0.1,
            help="Air density. Lower = less air resistance"
        )
        
        # Detect if any parameter changed and clear cached data to force recalculation
        if (prev_mass != st.session_state.physics_mass or
            prev_A != st.session_state.physics_A or
            prev_Cd != st.session_state.physics_Cd or
            prev_mu != st.session_state.physics_mu or
            prev_rho != st.session_state.physics_rho):
            # Clear cached acceleration data to force recalculation
            if 'accel_df' in st.session_state:
                del st.session_state['accel_df']
            if 'predicted_rating' in st.session_state:
                del st.session_state['predicted_rating']
            if 'airtime_metrics' in st.session_state:
                del st.session_state['airtime_metrics']
            st.session_state.track_generated = False
        
        st.caption("💡 Adjust these parameters to fine-tune the physics simulation")
    
    st.divider()
    
    # Block selection
    selected_block_key = st.selectbox(
        "Choose Block Type",
        options=list(BLOCK_LIBRARY.keys()),
        format_func=lambda x: f"{BLOCK_LIBRARY[x].icon} {BLOCK_LIBRARY[x].name}"
    )
    # RFDB analysis moved to multipage app under pages/02_RFDB_Data.py
    
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
        
    elif selected_block_key == "launch":
        params['length'] = st.slider("Launch Length (m)", 20, 80, 40, 5)
        params['speed_boost'] = st.slider("Speed Boost (m/s)", 10, 40, 20, 5)
        st.caption(f"💡 Target speed: {params['speed_boost']:.1f} m/s")
        st.info("🚀 **TIP:** Start your track with a Launch block! Without initial speed, the train won't have energy to climb. Launch provides the kinetic energy needed for hills and loops.")
        
    elif selected_block_key == "flat_section":
        params['length'] = st.slider("Section Length (m)", 10, 50, 30, 5)
    
    elif selected_block_key == "brake_run":
        params['length'] = st.slider("Brake Length (m)", 20, 50, 30, 5)
        st.info("🛑 **TIP:** End your track with a Brake Run for a safe, comfortable stop. This provides the final deceleration zone.")
    
    # Add block button
    if st.button("➕ Add to Track", type="primary", use_container_width=True):
        # Get current height for drop validation
        if selected_block_key == "drop" and len(st.session_state.track_sequence) > 0:
            # Calculate current height from existing track
            all_x, all_y, all_z = [], [], []
            current_x_offset, current_y_offset, current_z_offset = 0, 0, 0
            for block_info in st.session_state.track_sequence:
                x, y, z = block_info['block'].generate_profile(**block_info['params'])
                all_x.extend(x + current_x_offset)
                all_y.extend(y + current_y_offset)
                all_z.extend(z + current_z_offset)
                current_x_offset = all_x[-1]
                current_y_offset = all_y[-1]
                current_z_offset = all_z[-1]
            
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
        # Clear cached physics so plots recompute after adding a block
        st.session_state.pop('accel_df', None)
        st.session_state.pop('airtime_metrics', None)
        st.session_state.pop('ride_features', None)
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
                st.caption("**Edit Parameters:**")
                edited_params = block_info['params'].copy()
                btype = block_info.get('type')
                # Render appropriate controls per block type
                if btype == 'lift_hill':
                    edited_params['length'] = st.slider("Length (m)", 20, 100, int(edited_params.get('length', 50)), 5, key=f"edit_len_{idx}")
                    edited_params['height'] = st.slider("Height (m)", 20, 80, int(edited_params.get('height', 40)), 5, key=f"edit_h_{idx}")
                elif btype == 'drop':
                    edited_params['height'] = st.slider("Drop Height (m)", 20, 90, int(edited_params.get('height', 40)), 5, key=f"edit_dh_{idx}")
                    edited_params['steepness'] = st.slider("Steepness (max 30°)", 0.5, 1.0, float(edited_params.get('steepness', 0.8)), 0.05, key=f"edit_ds_{idx}")
                elif btype == 'loop':
                    edited_params['diameter'] = st.slider("Loop Diameter (m)", 15, 45, int(edited_params.get('diameter', 30)), 5, key=f"edit_ld_{idx}")
                elif btype == 'airtime_hill':
                    edited_params['length'] = st.slider("Hill Length (m)", 20, 60, int(edited_params.get('length', 40)), 5, key=f"edit_ahl_{idx}")
                    edited_params['height'] = st.slider("Hill Height (m)", 5, 25, int(edited_params.get('height', 15)), 1, key=f"edit_ahh_{idx}")
                elif btype == 'spiral':
                    edited_params['diameter'] = st.slider("Spiral Diameter (m)", 15, 40, int(edited_params.get('diameter', 25)), 5, key=f"edit_spd_{idx}")
                    edited_params['turns'] = st.slider("Number of Turns", 0.5, 3.0, float(edited_params.get('turns', 1.5)), 0.5, key=f"edit_spt_{idx}")
                elif btype == 'bunny_hop':
                    edited_params['length'] = st.slider("Hop Length (m)", 10, 30, int(edited_params.get('length', 20)), 5, key=f"edit_bhl_{idx}")
                    edited_params['height'] = st.slider("Hop Height (m)", 3, 15, int(edited_params.get('height', 8)), 1, key=f"edit_bhh_{idx}")
                elif btype == 'banked_turn':
                    edited_params['radius'] = st.slider("Turn Radius (m)", 15, 50, int(edited_params.get('radius', 30)), 5, key=f"edit_btr_{idx}")
                    edited_params['angle'] = st.slider("Turn Angle (°)", 30, 180, int(edited_params.get('angle', 90)), 15, key=f"edit_bta_{idx}")
                elif btype == 'launch':
                    edited_params['length'] = st.slider("Launch Length (m)", 20, 80, int(edited_params.get('length', 40)), 5, key=f"edit_ll_{idx}")
                    edited_params['speed_boost'] = st.slider("Speed Boost (m/s)", 10, 40, int(edited_params.get('speed_boost', 20)), 5, key=f"edit_lsb_{idx}")
                    st.caption(f"Target speed: {edited_params['speed_boost']:.1f} m/s")
                elif btype == 'flat_section':
                    edited_params['length'] = st.slider("Section Length (m)", 10, 50, int(edited_params.get('length', 30)), 5, key=f"edit_fsl_{idx}")
                elif btype == 'brake_run':
                    edited_params['length'] = st.slider("Brake Length (m)", 20, 50, int(edited_params.get('length', 30)), 5, key=f"edit_brl_{idx}")
                else:
                    # Fallback generic editors
                    for param_name, param_value in edited_params.items():
                        if param_name == 'current_height':
                            continue
                        if isinstance(param_value, (int, float)):
                            edited_params[param_name] = st.number_input(param_name, value=float(param_value), key=f"edit_gen_{param_name}_{idx}")
                        else:
                            edited_params[param_name] = st.text_input(param_name, value=str(param_value), key=f"edit_gen_txt_{param_name}_{idx}")

                if st.button("💾 Save Changes", key=f"save_{idx}", use_container_width=True):
                    st.session_state.track_sequence[idx]['params'] = edited_params
                    st.session_state.track_generated = False
                    st.success("Saved block changes")
                    st.rerun()
                
                col_del, col_up, col_down = st.columns(3)
                with col_del:
                    if st.button("🗑️ Remove", key=f"del_{idx}", use_container_width=True):
                        st.session_state.track_sequence.pop(idx)
                        st.session_state.track_generated = False
                        # Clear cached physics so plots recompute
                        st.session_state.pop('accel_df', None)
                        st.session_state.pop('airtime_metrics', None)
                        st.session_state.pop('ride_features', None)
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
            # Clear cached physics so plots recompute
            st.session_state.pop('accel_df', None)
            st.session_state.pop('airtime_metrics', None)
            st.session_state.pop('ride_features', None)
            st.rerun()

# ============================================================================
# MAIN AREA: TRACK VISUALIZATION AND RATING
# ============================================================================

def check_lateral_smoothness(x, z, max_angle_deg=20):
    """
    Check for sharp lateral transitions (z-coordinate changes).
    
    Args:
        x, z: Track coordinates (forward and lateral)
        max_angle_deg: Maximum allowed lateral banking angle change (default 20°)
    
    Returns:
        (is_valid, max_angle, problem_indices)
    """
    # If z is essentially flat (max deviation < 0.1m), skip check
    if np.abs(z).max() < 0.1:
        return True, 0.0, np.array([])
    
    dx = np.diff(x)
    dz = np.diff(z)
    
    # Only check where there's meaningful lateral change (> 1cm)
    meaningful_dz = np.abs(dz) > 0.01
    
    if not meaningful_dz.any():
        return True, 0.0, np.array([])
    
    # Calculate lateral angle from centerline in degrees
    angles = np.abs(np.arctan2(dz, dx) * 180 / np.pi)
    
    # Only check angles where dz is meaningful
    problem_mask = meaningful_dz & (angles > max_angle_deg)
    problem_indices = np.where(problem_mask)[0]
    max_angle = np.max(angles[meaningful_dz]) if meaningful_dz.any() else 0
    
    is_valid = len(problem_indices) == 0
    
    return is_valid, max_angle, problem_indices

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
        """Hermite blend supporting 2D or 3D points."""
        t = np.linspace(0.0, 1.0, steps)
        h00 = 2*t**3 - 3*t**2 + 1
        h10 = t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 = t**3 - t**2
        
        # Support 2D or 3D
        if len(P0) == 2:
            dx = h00*P0[0] + h10*T0[0] + h01*P1[0] + h11*T1[0]
            dy = h00*P0[1] + h10*T0[1] + h01*P1[1] + h11*T1[1]
            return dx, dy
        else:  # 3D
            dx = h00*P0[0] + h10*T0[0] + h01*P1[0] + h11*T1[0]
            dy = h00*P0[1] + h10*T0[1] + h01*P1[1] + h11*T1[1]
            dz = h00*P0[2] + h10*T0[2] + h01*P1[2] + h11*T1[2]
            return dx, dy, dz

    def blend_joint(prev_x, prev_y, next_x_rel, next_y_rel, steps=24):
        """2D blend for x,y coordinates (preserves original logic)."""
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

    def blend_z_coordinate(prev_z, next_z_rel, blend_length, z_tangent_prev=None, z_tangent_next=None):
        """Smooth z-coordinate blending using Hermite interpolation.
        
        Args:
            prev_z: Previous z-coordinates array
            next_z_rel: Next z-coordinates (relative)
            blend_length: Number of points in blend segment
            z_tangent_prev: Optional tangent at end of previous segment
            z_tangent_next: Optional tangent at start of next segment
        """
        z0 = prev_z[-1] if len(prev_z) > 0 else 0.0
        z1 = z0 + next_z_rel[0]
        
        # Compute tangents if not provided
        if z_tangent_prev is None:
            if len(prev_z) >= 2:
                z_tangent_prev = prev_z[-1] - prev_z[-2]
            else:
                z_tangent_prev = 0.0
                
        if z_tangent_next is None:
            if len(next_z_rel) >= 2:
                z_tangent_next = next_z_rel[1] - next_z_rel[0]
            else:
                z_tangent_next = 0.0
        
        # Use 1D Hermite interpolation for z
        t = np.linspace(0.0, 1.0, blend_length)
        h00 = 2*t**3 - 3*t**2 + 1
        h10 = t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 = t**3 - t**2
        
        z_blend = h00*z0 + h10*z_tangent_prev + h01*z1 + h11*z_tangent_next
        return z_blend

    all_x = []
    all_y = []
    all_z = []
    current_x_offset = 0.0
    current_y_offset = 0.0
    current_z_offset = 0.0
    joints_count = 0

    for idx, block_info in enumerate(st.session_state.track_sequence):
        x_rel, y_rel, z_rel = block_info['block'].generate_profile(**block_info['params'])

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
            # Add z-coordinates for the initial blend (start at 0, end at first block's z[0])
            z_blend_init = np.linspace(0.0, z_rel[0], len(bx))
            all_z.extend(z_blend_init[1:].tolist())
            # Append rest of first block relative to last blend point
            x_abs = x_rel + all_x[-1]
            y_abs = y_rel + all_y[-1]
            z_abs = z_rel + all_z[-1]
            all_x.extend(x_abs.tolist())
            all_y.extend(y_abs.tolist())
            all_z.extend(z_abs.tolist())
            current_x_offset = all_x[-1]
            current_y_offset = all_y[-1]
            current_z_offset = all_z[-1]
            joints_count += 1
            continue

        # Before appending next block, insert a blend segment to match slopes
        x_blend, y_blend = blend_joint(np.array(all_x), np.array(all_y), np.array(x_rel), np.array(y_rel), steps=32)

        # Append blend (avoid duplicating endpoint)
        all_x.extend(x_blend[1:].tolist())
        all_y.extend(y_blend[1:].tolist())
        
        # For z, use smooth Hermite interpolation instead of linear
        z_blend = blend_z_coordinate(np.array(all_z), z_rel, blend_length=len(x_blend))
        all_z.extend(z_blend[1:].tolist())
        joints_count += 1

        # Now append the next block offset from last absolute point
        x_abs = x_rel + all_x[-1]
        y_abs = y_rel + all_y[-1]
        z_abs = z_rel + all_z[-1]
        all_x.extend(x_abs.tolist())
        all_y.extend(y_abs.tolist())
        all_z.extend(z_abs.tolist())

        current_x_offset = all_x[-1]
        current_y_offset = all_y[-1]
        current_z_offset = all_z[-1]

    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_z = np.array(all_z)

    # Hide blended joints message
    st.session_state.joint_smoothing_applied = None
    st.session_state.smoothness_warning = None
    _, final_curvature = detect_curvature_spikes(all_x, all_y)
    st.session_state.track_curvature = final_curvature
    # If requested, enforce ending level equals starting level (y-coordinate)
    if st.session_state.get('force_end_level'):
        y_target = float(st.session_state.get('start_level', 0.0))
        y_end = float(all_y[-1])
        if abs(y_end - y_target) > 1e-3:
            # Create a gentle leveling segment
            x0, y0, z0 = all_x[-1], all_y[-1], all_z[-1]
            x1 = x0 + 40.0
            y1 = y_target
            z1 = 0.0  # Return to center laterally
            
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
            
            # Smoothly blend z back to center using Hermite interpolation
            z_tangent_prev = all_z[-1] - all_z[-2] if len(all_z) >= 2 else 0.0
            z_tangent_next = 0.0  # Land flat laterally
            t = np.linspace(0.0, 1.0, len(bx))
            h00 = 2*t**3 - 3*t**2 + 1
            h10 = t**3 - 2*t**2 + t
            h01 = -2*t**3 + 3*t**2
            h11 = t**3 - t**2
            bz = h00*z0 + h10*z_tangent_prev + h01*z1 + h11*z_tangent_next
            
            all_x = np.concatenate([all_x, bx[1:]])
            all_y = np.concatenate([all_y, by[1:]])
            all_z = np.concatenate([all_z, bz[1:]])
    return all_x, all_y, all_z

def simple_gforce_analysis(x, y, z=None, dt=0.02):
    """Simple geometric g-force calculation - direct call to compute_rider_accelerations"""
    
    # Just use the working compute_rider_accelerations from accelerometer_transform
    from utils.accelerometer_transform import compute_rider_accelerations
    
    # Create DataFrame
    track_df = pd.DataFrame({
        'x': x,
        'y': y,
        'z': z if z is not None else np.zeros_like(x)
    })
    
    # Call the working function
    result_df = compute_rider_accelerations(track_df)
    
    # Return in the expected format
    return result_df[['Time', 'Lateral', 'Vertical', 'Longitudinal']]
    
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
    
    # Calculate safety score (0-5 stars) with continuous penalties
    # Start with 5 stars and deduct for safety violations
    # More continuous and stronger penalties than before
    safety_score = 5.0
    
    # Continuous penalty for vertical g-forces (positive)
    # Penalty increases smoothly from 2g onwards
    if max_vertical > 2.0:
        # Gradual penalty: 0 at 2g, increases to -2.5 at 5g, -4.0 at 7g
        vertical_penalty = 0.0
        if max_vertical > 5.0:
            # Critical: -2.5 base, then -0.5 per g above 5g
            vertical_penalty = -2.5 - 0.5 * (max_vertical - 5.0)
        elif max_vertical > 4.0:
            # High: -1.5 base, then -0.33 per g above 4g
            vertical_penalty = -1.5 - 0.33 * (max_vertical - 4.0)
        elif max_vertical > 3.0:
            # Moderate: -0.5 base, then -0.33 per g above 3g
            vertical_penalty = -0.5 - 0.33 * (max_vertical - 3.0)
        elif max_vertical > 2.0:
            # Low: -0.1 per g above 2g
            vertical_penalty = -0.1 * (max_vertical - 2.0)
        safety_score += vertical_penalty
    
    # Continuous penalty for negative g-forces (airtime)
    # Penalty increases smoothly from -1g onwards
    if min_vertical < -1.0:
        vertical_neg_penalty = 0.0
        if min_vertical < -3.0:
            # Critical: -2.5 base, then -0.5 per g below -3g
            vertical_neg_penalty = -2.5 - 0.5 * abs(min_vertical + 3.0)
        elif min_vertical < -2.0:
            # High: -1.5 base, then -0.5 per g below -2g
            vertical_neg_penalty = -1.5 - 0.5 * abs(min_vertical + 2.0)
        elif min_vertical < -1.5:
            # Moderate: -0.5 base, then -0.33 per g below -1.5g
            vertical_neg_penalty = -0.5 - 0.33 * abs(min_vertical + 1.5)
        elif min_vertical < -1.0:
            # Low: -0.2 per g below -1g
            vertical_neg_penalty = -0.2 * abs(min_vertical + 1.0)
        safety_score += vertical_neg_penalty
    
    # Continuous penalty for lateral g-forces
    # Penalty increases smoothly from 1g onwards
    if max_lateral > 1.0:
        lateral_penalty = 0.0
        if max_lateral > 5.0:
            # Critical: -2.5 base, then -0.4 per g above 5g
            lateral_penalty = -2.5 - 0.4 * (max_lateral - 5.0)
        elif max_lateral > 3.0:
            # High: -1.2 base, then -0.43 per g above 3g
            lateral_penalty = -1.2 - 0.43 * (max_lateral - 3.0)
        elif max_lateral > 2.0:
            # Moderate: -0.5 base, then -0.35 per g above 2g
            lateral_penalty = -0.5 - 0.35 * (max_lateral - 2.0)
        elif max_lateral > 1.5:
            # Low: -0.2 base, then -0.6 per g above 1.5g
            lateral_penalty = -0.2 - 0.6 * (max_lateral - 1.5)
        elif max_lateral > 1.0:
            # Very low: -0.1 per g above 1g
            lateral_penalty = -0.1 * (max_lateral - 1.0)
        safety_score += lateral_penalty
    
    # Continuous penalty for longitudinal g-forces
    # Penalty increases smoothly from 2g onwards
    if max_longitudinal > 2.0:
        longitudinal_penalty = 0.0
        if max_longitudinal > 4.0:
            # High: -1.0 base, then -0.25 per g above 4g
            longitudinal_penalty = -1.0 - 0.25 * (max_longitudinal - 4.0)
        elif max_longitudinal > 3.0:
            # Moderate: -0.5 base, then -0.5 per g above 3g
            longitudinal_penalty = -0.5 - 0.5 * (max_longitudinal - 3.0)
        elif max_longitudinal > 2.5:
            # Low: -0.2 base, then -0.6 per g above 2.5g
            longitudinal_penalty = -0.2 - 0.6 * (max_longitudinal - 2.5)
        elif max_longitudinal > 2.0:
            # Very low: -0.1 per g above 2g
            longitudinal_penalty = -0.1 * (max_longitudinal - 2.0)
        safety_score += longitudinal_penalty
    
    # Clamp to 0-5 range
    safety_score = max(0.0, min(5.0, safety_score))
    
    return {
        'level': safety_level,
        'emoji': safety_emoji,
        'color': safety_color,
        'warnings': warnings,
        'dangers': dangers,
        'max_vertical': max_vertical,
        'min_vertical': min_vertical,
        'max_lateral': max_lateral,
        'max_longitudinal': max_longitudinal,
        'safety_score': safety_score  # 0-5 stars
    }

def compute_airtime_metrics(accel_df):
    """Compute airtime metrics from vertical g data.
    Categories match notebook definition (by vertical g in g-units):
    - Floater Airtime: -0.5g <= Vertical < 0.0g
    - Flojector Airtime: -1.0g <= Vertical < -0.5g
    - Ejector Airtime: Vertical < -1.0g (not used in notebook, but kept for display)

    Returns seconds for each category and total airtime.
    Uses time spacing from 'Time' column.
    """
    if accel_df is None or 'Vertical' not in accel_df or 'Time' not in accel_df:
        return {'floater': 0.0, 'flojector': 0.0, 'ejector': 0.0, 'total_airtime': 0.0}

    t = accel_df['Time'].to_numpy()
    g = accel_df['Vertical'].to_numpy()
    if len(t) < 2:
        dt = 0.1
    else:
        dt = float(np.median(np.diff(t)))

    # Notebook definitions
    floater_mask = (g < 0.0) & (g >= -0.5)
    flojector_mask = (g < -0.5) & (g >= -1.0)
    ejector_mask = (g < -1.0)

    floater_time = float(floater_mask.sum() * dt)
    flojector_time = float(flojector_mask.sum() * dt)
    ejector_time = float(ejector_mask.sum() * dt)
    total_airtime = floater_time + flojector_time + ejector_time

    return {
        'floater': floater_time,
        'flojector': flojector_time,
        'ejector': ejector_time,
        'total_airtime': total_airtime,
        # Additional metrics for feature calculation
        'floater_proportion': float(floater_mask.sum() / len(g)) if len(g) > 0 else 0.0,
        'flojector_proportion': float(flojector_mask.sum() / len(g)) if len(g) > 0 else 0.0,
        'total_length_seconds': float(len(g) * dt),
    }

def calculate_ride_features(accel_df):
    """Calculate features consistent with the LightGBM extreme model."""
    if accel_df is None or len(accel_df) == 0:
        return {}
    if not all(col in accel_df.columns for col in ['Vertical', 'Lateral', 'Longitudinal']):
        return {}

    feature_vector, feature_map = compute_lightgbm_features(accel_df, return_dict=True)

    # Backward-compatible keys for the advanced panel
    alias = {
        'num_positive_g_peaks': "Num Positive G (>3.0g)",
        'max_negative_vertical_g': "Max Negative Vertical G",
        'max_positive_vertical_g': "Max Positive Vertical G",
        'max_lateral_g': "Max Lateral G",
        'max_longitudinal_g': "Max Longitudinal G",
        'vertical_variance': "Vertical Variance",
        'lateral_variance': "Lateral Variance",
        'vertical_jerk': "Vertical Jerk",
        'avg_total_g': "Avg Total G",
        'airtime_gforce_interaction': "Airtime×G-Force Interaction",
        'g_force_range': "G-Force Range",
        'lateral_jerk': "Lateral Jerk",
        'g_force_skewness': "G-Force Skewness",
        'intensity_pacing': "Intensity Pacing",
        'force_transitions': "Force Transitions",
        'peak_density': "Peak Density",
        'rhythm_score': "Rhythm Score",
        'lateral_vibration': "Lateral Vibration",
        'vertical_vibration': "Vertical Vibration",
        'longitudinal_vibration': "Longitudinal Vibration",
    }

    ride_features = {k: float(feature_map.get(v, 0.0)) for k, v in alias.items()}
    # Include airtime + metadata pieces for completeness
    ride_features.update({
        'total_length_log_sec': float(feature_map.get("Total Length (log-sec)", 0.0)),
        'floater_airtime_pct': float(feature_map.get("Floater Airtime %", 0.0)),
        'flojector_airtime_pct': float(feature_map.get("Flojector Airtime %", 0.0)),
        'height_m': float(feature_map.get("Height (m)", 0.0)),
        'speed_kmh': float(feature_map.get("Speed (km/h)", 0.0)),
        'track_length_m': float(feature_map.get("Track Length (m)", 0.0)),
        'feature_vector': feature_vector,
    })

    return ride_features

# Auto-generate preview when blocks are added
if len(st.session_state.track_sequence) > 0:
    st.session_state.track_x, st.session_state.track_y, st.session_state.track_z = generate_track_from_blocks()
    st.session_state.track_generated = True
    
    # Calculate block boundaries for visualization
    block_boundaries = [0]
    cumulative_x = 0
    for block_info in st.session_state.track_sequence:
        x_block, y_block, z_block = block_info['block'].generate_profile(**block_info['params'])
        cumulative_x += x_block[-1]  # Add length of this block
        block_boundaries.append(cumulative_x)
    st.session_state.block_boundaries = block_boundaries
    st.session_state.block_names = [block_info['block'].name for block_info in st.session_state.track_sequence]
    st.session_state.block_icons = [block_info['block'].icon for block_info in st.session_state.track_sequence]
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
            'z': st.session_state.get('track_z', np.zeros_like(st.session_state.track_x))
        })
        
        # Get accelerometer data based on physics mode
        physics_mode = st.session_state.get('physics_mode', 'Advanced (Realistic)')
        
        if physics_mode == "Simple (Geometric)":
            # Use simple geometric calculation
            accel_df = simple_gforce_analysis(
                st.session_state.track_x,
                st.session_state.track_y,
                st.session_state.get('track_z', np.zeros_like(st.session_state.track_x))
            )
        else:
            # Use advanced physics with full 3D acceleration computation
            accel_df = st.session_state.get('accel_df')
            if accel_df is None or len(accel_df) < 10:
                accel_df = track_to_accelerometer_data(
                    track_df,
                    mass=st.session_state.get('physics_mass', 500.0),
                    rho=st.session_state.get('physics_rho', 1.2),
                    Cd=st.session_state.get('physics_Cd', 0.1),
                    A=st.session_state.get('physics_A', 2.0),
                    mu=st.session_state.get('physics_mu', 0.001)
                )
        
        if accel_df is not None and len(accel_df) > 10:
            # Store for g-force plot
            st.session_state.accel_df = accel_df
            
            # Check safety FIRST before showing rating
            safety = check_gforce_safety(accel_df)
            # Compute airtime metrics and store
            airtime = compute_airtime_metrics(accel_df)
            st.session_state.airtime_metrics = airtime
            
            # Calculate comprehensive ride features
            ride_features = calculate_ride_features(accel_df)
            st.session_state.ride_features = ride_features
            
            # Compute metadata from track geometry for better predictions
            x_track = np.array(st.session_state.track_x)
            y_track = np.array(st.session_state.track_y)
            z_track = np.array(st.session_state.get('track_z', np.zeros_like(x_track)))
            
            # Track length: 3D arc length
            dx = np.diff(x_track, prepend=x_track[0])
            dy = np.diff(y_track, prepend=y_track[0])
            dz = np.diff(z_track, prepend=z_track[0])
            track_length_m = float(np.sum(np.sqrt(dx**2 + dy**2 + dz**2)))
            
            # Max height
            height_m = float(np.max(y_track))
            
            # Max speed: estimate from energy conservation (v = sqrt(2*g*h))
            # Use max height drop as proxy for max speed
            max_height_drop = float(np.max(y_track) - np.min(y_track))
            g = 9.81
            energy_efficiency = 0.95  # Match accelerometer_transform
            max_speed_ms = np.sqrt(2 * g * max_height_drop * energy_efficiency)
            speed_kmh = float(max_speed_ms * 3.6)  # Convert to km/h
            
            metadata = {
                'height_m': height_m,
                'speed_kmh': speed_kmh,
                'track_length_m': track_length_m
            }
            
            # Predict rating automatically
            with st.spinner('🤖 AI analyzing your design...'):
                predicted_rating = predict_score_lgb(accel_df, metadata=metadata)
                st.session_state.predicted_rating = predicted_rating
            
            # Compact rating display with scores and airtime in one row
            safety_score = safety['safety_score']
            
            # Determine color based on safety
            if safety['dangers']:
                safety_color = "#FF0000"
                safety_emoji = "🚨"
            elif safety['warnings']:
                safety_color = "#FFA500"
                safety_emoji = "⚠️"
            else:
                safety_color = "#00C853"
                safety_emoji = "✅"
            
            # Compact rating display - two scores side by side with more space
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if predicted_rating is not None:
                    st.markdown(f'<div style="text-align: center;"><span style="font-size: 2.5rem; font-weight: bold; color: #FFD700;">⭐ {predicted_rating:.2f}</span><br><span style="font-size: 0.85rem; color: gray;">Fun Rating</span></div>', unsafe_allow_html=True)
                    st.progress(min(1.0, max(0.0, predicted_rating / 5.0)))
            
            with col2:
                st.markdown(f'<div style="text-align: center;"><span style="font-size: 2.5rem; font-weight: bold; color: {safety_color};">{safety_emoji} {safety_score:.1f}</span><br><span style="font-size: 0.85rem; color: gray;">Safety Score</span></div>', unsafe_allow_html=True)
                st.progress(min(1.0, max(0.0, safety_score / 5.0)))
            
            # Compact warnings/dangers
            if safety['dangers']:
                with st.expander("🚨 Safety Issues (DANGEROUS)", expanded=False):
                    for danger in safety['dangers']:
                        st.caption(danger)
            elif safety['warnings']:
                with st.expander("⚠️ Safety Warnings", expanded=False):
                    for warning in safety['warnings']:
                        st.caption(warning)
            
            # Feature importance explanation (after all safety cases, for ALL tracks)
            if predicted_rating is not None:
                with st.expander("🧠 Why this rating? (AI Explanation)", expanded=False):
                    st.caption("**What the AI considers:**")
                    
                    # Calculate feature contributions based on actual track stats
                    total_airtime = airtime['total_airtime']
                    ejector_time = airtime['ejector']
                    
                    # G-force variety (from accel_df)
                    vertical_range = accel_df['Vertical'].max() - accel_df['Vertical'].min()
                    
                    # Simple heuristic scoring for display
                    airtime_score = min(100, (total_airtime / 10.0) * 100)  # ~10s is excellent
                    variety_score = min(100, (vertical_range / 6.0) * 100)  # 6g range is excellent
                    intensity_score = min(100, (ejector_time / 3.0) * 100)  # 3s ejector is great
                    
                    # Display as progress bars
                    col_exp1, col_exp2 = st.columns(2)
                    with col_exp1:
                        st.metric("Airtime Impact", f"{airtime_score:.0f}%", 
                                 help=f"{total_airtime:.1f}s total airtime")
                        st.progress(min(1.0, max(0.0, airtime_score / 100)))
                        
                        st.metric("G-Force Variety", f"{variety_score:.0f}%",
                                 help=f"{vertical_range:.1f}g vertical range")
                        st.progress(min(1.0, max(0.0, variety_score / 100)))
                    
                    with col_exp2:
                        st.metric("Intensity/Thrills", f"{intensity_score:.0f}%",
                                 help=f"{ejector_time:.1f}s ejector airtime")
                        st.progress(min(1.0, max(0.0, intensity_score / 100)))
                        
                        # Smoothness (inverse of max g-forces)
                        max_vertical = abs(accel_df['Vertical']).max()
                        smoothness_score = max(0, 100 - (max_vertical - 4) * 20)  # Penalty above 4g
                        st.metric("Comfort/Smoothness", f"{smoothness_score:.0f}%",
                                 help=f"Max {max_vertical:.1f}g vertical")
                        st.progress(min(1.0, max(0.0, smoothness_score / 100)))
                    
                    st.caption("⚠️ This is a simplified explanation. The actual AI model uses complex sequential patterns in acceleration data.")
            
            # Safety score explanation
            with st.expander("🛡️ Safety Score Breakdown", expanded=False):
                max_vertical = safety['max_vertical']
                min_vertical = safety['min_vertical']
                max_lateral = safety['max_lateral']
                max_longitudinal = safety['max_longitudinal']
                
                # Calculate deductions
                vertical_pos_deduction = 0
                if max_vertical > 5.0:
                    vertical_pos_deduction = 2.0
                elif max_vertical > 4.0:
                    vertical_pos_deduction = 0.8
                elif max_vertical > 3.0:
                    vertical_pos_deduction = 0.3
                
                vertical_neg_deduction = 0
                if min_vertical < -3.0:
                    vertical_neg_deduction = 2.0
                elif min_vertical < -2.0:
                    vertical_neg_deduction = 0.8
                elif min_vertical < -1.5:
                    vertical_neg_deduction = 0.2
                
                lateral_deduction = 0
                if max_lateral > 5.0:
                    lateral_deduction = 2.0
                elif max_lateral > 2.0:
                    lateral_deduction = 0.5
                elif max_lateral > 1.5:
                    lateral_deduction = 0.2
                
                longitudinal_deduction = 0
                if max_longitudinal > 3.0:
                    longitudinal_deduction = 0.4
                elif max_longitudinal > 2.5:
                    longitudinal_deduction = 0.2
                
                total_deduction = vertical_pos_deduction + vertical_neg_deduction + lateral_deduction + longitudinal_deduction
                
                # Compact 4-column layout
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    ded_text = f"-{vertical_pos_deduction:.1f}★" if vertical_pos_deduction > 0 else "✓"
                    st.markdown(f'<div style="font-size: 0.85rem;"><span style="color: #00C853;">↑ {max_vertical:.2f}g</span><br><small>{ded_text}</small></div>', unsafe_allow_html=True)
                
                with col2:
                    ded_text = f"-{vertical_neg_deduction:.1f}★" if vertical_neg_deduction > 0 else "✓"
                    st.markdown(f'<div style="font-size: 0.85rem;"><span style="color: #FF0000;">↓ {min_vertical:.2f}g</span><br><small>{ded_text}</small></div>', unsafe_allow_html=True)
                
                with col3:
                    ded_text = f"-{lateral_deduction:.1f}★" if lateral_deduction > 0 else "✓"
                    st.markdown(f'<div style="font-size: 0.85rem;">↔ {max_lateral:.2f}g<br><small>{ded_text}</small></div>', unsafe_allow_html=True)
                
                with col4:
                    ded_text = f"-{longitudinal_deduction:.1f}★" if longitudinal_deduction > 0 else "✓"
                    st.markdown(f'<div style="font-size: 0.85rem;">↕ {max_longitudinal:.2f}g<br><small>{ded_text}</small></div>', unsafe_allow_html=True)
                
                # Compact summary
                final_score = max(0.0, 5.0 - total_deduction)
                st.markdown(f'<div style="margin-top: 0.5rem; font-size: 0.9rem;"><strong>5.0 - {total_deduction:.1f} = {final_score:.1f}★</strong></div>', unsafe_allow_html=True)
                
                if final_score >= 4.5:
                    st.caption("✅ Excellent safety")
                elif final_score >= 3.5:
                    st.caption("⚠️ Acceptable")
                elif final_score >= 2.0:
                    st.caption("⚠️ High g-forces")
                else:
                    st.caption("🚨 Dangerous")
            
            # Submit to Leaderboard Section
            with st.expander("🏆 Submit to Leaderboard", expanded=False):
                with st.form("submit_form"):
                    submitter_name = st.text_input("Your Name", placeholder="Enter your name", key="submitter_name")
                    col_submit1, col_submit2 = st.columns([1, 1])
                    
                    with col_submit1:
                        submitted = st.form_submit_button("🚀 Submit Rollercoaster", use_container_width=True, type="primary")
                    
                    with col_submit2:
                        view_leaderboard = st.form_submit_button("📊 View Leaderboard", use_container_width=True)
                        if view_leaderboard:
                            st.switch_page("pages/03_Leaderboard.py")
                    
                    if submitted:
                        if not submitter_name or submitter_name.strip() == "":
                            st.error("Please enter your name before submitting.")
                        elif predicted_rating is None:
                            st.error("Please wait for the AI rating to complete before submitting.")
                        else:
                            # Import submission manager
                            from utils.submission_manager import save_submission_to_s3
                            
                            # Prepare geometry data
                            geometry = {
                                'x': st.session_state.track_x,
                                'y': st.session_state.track_y,
                                'z': st.session_state.get('track_z', np.zeros_like(st.session_state.track_x))
                            }
                            
                            # Save to local storage
                            with st.spinner("Saving submission to leaderboard..."):
                                success = save_submission_to_s3(
                                    submitter_name=submitter_name.strip(),
                                    geometry=geometry,
                                    score=predicted_rating,
                                    safety_score=safety_score
                                )
                            
                            if success:
                                st.success(f"✅ Successfully submitted! Your rollercoaster has been added to the leaderboard.")
                                st.balloons()
                            else:
                                st.error("❌ Failed to save submission. Please try again later.")
            
            # 3D view moved to bottom section

        else:
            st.error("Track too short for AI analysis")
            # Fallback to simple g-force analysis
            st.session_state.accel_df = simple_gforce_analysis(
                st.session_state.track_x, 
                st.session_state.track_y,
                st.session_state.get('track_z', np.zeros_like(st.session_state.track_x))
            )
            
    except Exception as e:
        st.error(f"AI Error: {str(e)}")
        st.caption("Using simple physics model instead...")
        # Fallback to simple g-force analysis
        st.session_state.accel_df = simple_gforce_analysis(
            st.session_state.track_x, 
            st.session_state.track_y,
            st.session_state.get('track_z', np.zeros_like(st.session_state.track_x))
        )
    
    # Ensure we always have g-force data for the plots
    if not hasattr(st.session_state, 'accel_df'):
        st.session_state.accel_df = simple_gforce_analysis(
            st.session_state.track_x, 
            st.session_state.track_y,
            st.session_state.get('track_z', np.zeros_like(st.session_state.track_x))
        )
    
    st.divider()
    
    # Row 1: Track profile and Statistics (2:1 ratio)
    col1, col2 = st.columns([2, 1])
    
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
        
        # Add block boundaries as vertical lines
        if hasattr(st.session_state, 'block_boundaries') and hasattr(st.session_state, 'block_icons'):
            y_max = max(st.session_state.track_y)
            y_min = min(st.session_state.track_y)
            for i, boundary in enumerate(st.session_state.block_boundaries[1:-1], start=1):  # Skip first and last
                fig_profile.add_shape(
                    type="line",
                    x0=boundary, x1=boundary,
                    y0=y_min, y1=y_max,
                    line=dict(color="rgba(128, 128, 128, 0.3)", width=1, dash="dot")
                )
                # Add block icon annotation at the midpoint
                if i < len(st.session_state.block_icons):
                    midpoint_x = (st.session_state.block_boundaries[i-1] + st.session_state.block_boundaries[i]) / 2
                    fig_profile.add_annotation(
                        x=midpoint_x,
                        y=y_max * 1.08,  # Slightly above the track
                        text=st.session_state.block_icons[i-1],
                        showarrow=False,
                        font=dict(size=14),
                        xanchor="center"
                    )
            # Add last block icon
            if len(st.session_state.block_icons) > 0:
                midpoint_x = (st.session_state.block_boundaries[-2] + st.session_state.block_boundaries[-1]) / 2
                fig_profile.add_annotation(
                    x=midpoint_x,
                    y=y_max * 1.08,
                    text=st.session_state.block_icons[-1],
                    showarrow=False,
                    font=dict(size=14),
                    xanchor="center"
                )
        
        fig_profile.update_layout(
            xaxis_title="Distance (m)",
            yaxis_title="Height (m)",
            showlegend=False,
            height=350,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        
        st.plotly_chart(fig_profile, use_container_width=True)
        # Airtime Timeline directly below main plot
        if 'accel_df' in st.session_state:
            accel_df_tl = st.session_state.accel_df.copy()
            t = accel_df_tl['Time'].values
            g = accel_df_tl['Vertical'].values
            
            # Convert time to distance along track to match the profile plot above
            # Use track x-coordinates which represent distance along the track
            x_track = st.session_state.track_x
            # Interpolate to match acceleration dataframe length
            if len(x_track) != len(t):
                distance = np.linspace(0, x_track[-1], len(t))
            else:
                distance = x_track
            
            # Updated definitions to match notebook:
            # Floater: -0.5g <= Vertical < 0.0g
            flo_mask = (g < 0.0) & (g >= -0.5)
            
            # Flojector: -1.0g <= Vertical < -0.5g
            flj_mask = (g < -0.5) & (g >= -1.0)
            
            # Ejector: Vertical < -1.0g (extreme airtime)
            ej_mask = (g < -1.0)

            def mask_to_intervals(mask, position_array):
                intervals = []
                if len(position_array) == 0:
                    return intervals
                prev = False
                start = None
                for i, m in enumerate(mask):
                    if m and not prev:
                        start = position_array[i]
                    if prev and not m:
                        end = position_array[i]
                        intervals.append((start, end))
                        start = None
                    prev = m
                if prev and start is not None:
                    intervals.append((start, position_array[-1]))
                return intervals

            flo_int = mask_to_intervals(flo_mask, distance)
            flj_int = mask_to_intervals(flj_mask, distance)
            ej_int = mask_to_intervals(ej_mask, distance)

            import plotly.graph_objects as go
            fig_tl = go.Figure()
            if len(distance):
                fig_tl.add_trace(go.Scatter(x=[distance[0], distance[-1]], y=[3,3], mode='lines', line=dict(color='#e0e0e0', width=1), showlegend=False))
                fig_tl.add_trace(go.Scatter(x=[distance[0], distance[-1]], y=[2,2], mode='lines', line=dict(color='#e0e0e0', width=1), showlegend=False))
                fig_tl.add_trace(go.Scatter(x=[distance[0], distance[-1]], y=[1,1], mode='lines', line=dict(color='#e0e0e0', width=1), showlegend=False))

            def add_band(intervals, y_top, y_bottom, color, name):
                for (a, b) in intervals:
                    fig_tl.add_shape(type='rect', x0=a, x1=b, y0=y_bottom, y1=y_top, fillcolor=color, opacity=0.6, line=dict(width=0))
                fig_tl.add_trace(go.Scatter(x=[], y=[], name=name, mode='markers', marker=dict(color=color)))

            add_band(ej_int, 3.2, 2.8, '#F44336', 'Ejector')
            add_band(flj_int, 2.2, 1.8, '#FF9800', 'Flojector')
            add_band(flo_int, 1.2, 0.8, '#4CAF50', 'Floater')
            
            # Add block boundaries as vertical lines
            if hasattr(st.session_state, 'block_boundaries'):
                for boundary in st.session_state.block_boundaries[1:-1]:  # Skip first and last
                    fig_tl.add_shape(
                        type="line",
                        x0=boundary, x1=boundary,
                        y0=0.5, y1=3.5,
                        line=dict(color="rgba(128, 128, 128, 0.3)", width=1, dash="dot")
                    )

            fig_tl.update_layout(
                height=180,
                margin=dict(l=10, r=10, t=5, b=10),
                yaxis=dict(showticklabels=True, tickmode='array', tickvals=[1,2,3], ticktext=['Floater','Flojector','Ejector'], range=[0.5,3.5]),
                xaxis_title='Distance (m)'
            )
            st.plotly_chart(fig_tl, use_container_width=True)
    
    with col2:
        st.markdown("**Track Statistics**")
        
        # Calculate stats
        x = st.session_state.track_x
        y = st.session_state.track_y
        z = st.session_state.get('track_z', np.zeros_like(x))
        
        total_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        max_height = np.max(y)
        min_height = np.min(y)
        max_drop = max_height - min_height
        
        # Check track smoothness (allow steep start for lift hill)
        is_smooth, max_angle, _ = check_track_smoothness(x, y, max_angle_deg=30, allow_steep_start=True)
        
        # Check lateral smoothness (z-coordinate transitions)
        try:
            is_lateral_smooth, max_lateral_angle, lateral_problem_indices = check_lateral_smoothness(x, z, max_angle_deg=20)
        except Exception as e:
            st.error(f"Lateral smoothness check error: {str(e)}")
            is_lateral_smooth, max_lateral_angle, lateral_problem_indices = True, 0.0, np.array([])
        
        # Get angle after lift section for display
        dx = np.diff(x)
        dy = np.diff(y)
        angles = np.abs(np.arctan2(dy, dx) * 180 / np.pi)
        lift_section_end = int(len(angles) * 0.2)
        max_angle_after_lift = np.max(angles[lift_section_end:]) if len(angles) > lift_section_end else max_angle
        
        smoothness_emoji = "✅" if (max_angle_after_lift <= 30 and is_lateral_smooth) else "⚠️"
        
        # Warn about lateral sharpness if detected
        if not is_lateral_smooth:
            st.warning(f"⚠️ Sharp lateral transition detected! Max lateral angle: {max_lateral_angle:.1f}° (should be < 20°). Check banking transitions between blocks.")
        
        # Display metrics
        col_a, col_b = st.columns(2)
        col_a.metric("Track Length", f"{total_length:.0f} m")
        col_a.metric("Max Height", f"{max_height:.1f} m")
        col_b.metric("Total Drop", f"{max_drop:.1f} m")
        col_b.metric("Max Angle (excl. lift)", f"{max_angle_after_lift:.1f}° {smoothness_emoji}")

        # Airtime metrics if available (large-number display)
        airtime = st.session_state.get('airtime_metrics')
        if airtime:
            st.markdown("**🎈 Airtime**")
            col_floater, col_flojector, col_ejector, col_total = st.columns([1, 1, 1, 1])
            with col_floater:
                st.markdown(
                    f"<div style='font-size: 1.4rem; font-weight: bold; text-align: center;'>⛅ {airtime['floater']:.2f}s</div>",
                    unsafe_allow_html=True
                )
                st.caption("Floater")
            with col_flojector:
                st.markdown(
                    f"<div style='font-size: 1.4rem; font-weight: bold; text-align: center;'>💨 {airtime['flojector']:.2f}s</div>",
                    unsafe_allow_html=True
                )
                st.caption("Flojector")
            with col_ejector:
                st.markdown(
                    f"<div style='font-size: 1.4rem; font-weight: bold; text-align: center;'>🚀 {airtime['ejector']:.2f}s</div>",
                    unsafe_allow_html=True
                )
                st.caption("Ejector")
            with col_total:
                st.markdown(
                    f"<div style='font-size: 1.4rem; font-weight: bold; text-align: center; color: #FFD700;'>Σ {airtime['total_airtime']:.2f}s</div>",
                    unsafe_allow_html=True
                )
                st.caption("Total")
        
        # Hidden panel with comprehensive features
        with st.expander("🔬 Advanced Ride Features", expanded=False):
            ride_features = st.session_state.get('ride_features')
            if ride_features:
                st.markdown("**📊 Comprehensive Ride Analysis**")
                
                # Basic Features
                st.markdown("**🎯 Basic Features**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Positive G Peaks (>3g)", f"{ride_features.get('num_positive_g_peaks', 0):.0f}")
                with col2:
                    st.metric("Max Negative G", f"{ride_features.get('max_negative_vertical_g', 0):.2f}g")
                with col3:
                    st.metric("Max Positive G", f"{ride_features.get('max_positive_vertical_g', 0):.2f}g")
                with col4:
                    st.metric("Max Lateral G", f"{ride_features.get('max_lateral_g', 0):.2f}g")
                
                # Smoothness Features
                st.markdown("**🌊 Smoothness & Rhythm**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Vertical Variance", f"{ride_features.get('vertical_variance', 0):.3f}")
                with col2:
                    st.metric("Vertical Jerk", f"{ride_features.get('vertical_jerk', 0):.3f}")
                with col3:
                    st.metric("Avg Total G", f"{ride_features.get('avg_total_g', 0):.2f}g")
                with col4:
                    st.metric("Rhythm Score", f"{ride_features.get('rhythm_score', 0):.3f}")
                
                # Advanced Features
                st.markdown("**⚡ Advanced Dynamics**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Airtime×G-Force", f"{ride_features.get('airtime_gforce_interaction', 0):.2f}")
                with col2:
                    st.metric("G-Force Range", f"{ride_features.get('g_force_range', 0):.2f}g")
                with col3:
                    st.metric("Lateral Jerk", f"{ride_features.get('lateral_jerk', 0):.3f}")
                with col4:
                    st.metric("G-Force Skewness", f"{ride_features.get('g_force_skewness', 0):.2f}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Intensity Pacing", f"{ride_features.get('intensity_pacing', 0):.2f}")
                with col2:
                    st.metric("Force Transitions", f"{ride_features.get('force_transitions', 0):.3f}")
                with col3:
                    st.metric("Peak Density", f"{ride_features.get('peak_density', 0):.1f}")
                with col4:
                    st.metric("Max Longitudinal G", f"{ride_features.get('max_longitudinal_g', 0):.2f}g")
                
                # Vibration Features
                st.markdown("**📳 Vibration Analysis**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Lateral Vibration", f"{ride_features.get('lateral_vibration', 0):.3f}")
                with col2:
                    st.metric("Vertical Vibration", f"{ride_features.get('vertical_vibration', 0):.3f}")
                with col3:
                    st.metric("Longitudinal Vibration", f"{ride_features.get('longitudinal_vibration', 0):.3f}")
            else:
                st.info("Generate a track and run physics simulation to see advanced features.")
        
        if max_angle_after_lift > 30:
            st.caption("⚠️ Track has steep sections (>30°) - smoothing applied")
        else:
            st.caption("✅ Track angles within safe limits (<30°)")
        
        # Curvature smoothness indicator (commented out)
        # if hasattr(st.session_state, 'track_curvature'):
        #     curvature = st.session_state.track_curvature
        #     max_curvature = np.max(curvature)
        #     avg_curvature = np.mean(curvature)
        #     
        #     st.markdown("**Track Smoothness:**")
        #     st.text(f"Avg: {avg_curvature:.4f} | Max: {max_curvature:.4f}")
        #     
        #     if max_curvature < 0.01:
        #         st.caption("✅ Very smooth track")
        #     elif max_curvature < 0.05:
        #         st.caption("✅ Smooth track")
        #     else:
        #         st.caption("⚠️ Some tight curves present")
        
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
    
    # Show smoothness warnings (joint info hidden)
    info_col1, info_col2 = st.columns(2)
    with info_col2:
        if hasattr(st.session_state, 'smoothness_warning') and st.session_state.smoothness_warning:
            st.warning(st.session_state.smoothness_warning)
    
    # Ride configuration
    #st.subheader("⚙️ Ride Configuration")
    #initial_speed = st.slider("Initial Speed (m/s)", 5.0, 30.0, 15.0, 1.0, key="speed_slider")
    #st.caption("Launch speed or chain lift speed")
    #st.session_state.initial_speed = initial_speed
    
    #st.divider()
    
    # Row 2: G-Force Analysis
    st.markdown("**G-Force Analysis**")
    
    if 'accel_df' in st.session_state:
        accel_df = st.session_state.accel_df
        
        # Build distance axis to plot G-forces over track distance (arc length), not time
        if (
            'track_x' in st.session_state and
            'track_y' in st.session_state and
            'track_z' in st.session_state and
            len(st.session_state.track_x) > 1
        ):
            x_track = np.array(st.session_state.track_x, dtype=float)
            y_track = np.array(st.session_state.track_y, dtype=float)
            z_track = np.array(st.session_state.track_z, dtype=float)
            # Compute cumulative distance along the 3D track
            dx = np.diff(x_track, prepend=x_track[0])
            dy = np.diff(y_track, prepend=y_track[0])
            dz = np.diff(z_track, prepend=z_track[0])
            distance_raw = np.cumsum(np.sqrt(dx**2 + dy**2 + dz**2))
            # Interpolate to match acceleration dataframe length if needed
            if len(distance_raw) != len(accel_df):
                distance_axis = np.linspace(0, distance_raw[-1], len(accel_df))
            else:
                distance_axis = distance_raw
        else:
            # Fallback: use index as distance surrogate
            distance_axis = np.linspace(0, len(accel_df), len(accel_df))
        
        # Create subplots for each G-force
        fig_g = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Vertical G-Forces", "Lateral G-Forces", "Longitudinal G-Forces"),
            vertical_spacing=0.12
        )

        # Vertical G-forces with safety zones
        fig_g.add_trace(
            go.Scatter(x=distance_axis, y=accel_df['Vertical'],
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
            go.Scatter(x=distance_axis, y=accel_df['Lateral'],
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
            go.Scatter(x=distance_axis, y=accel_df['Longitudinal'],
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
        
        # Calculate y-axis ranges: default -5 to 5, extend to -10 to 10 if needed
        vertical_max = accel_df['Vertical'].max()
        vertical_min = accel_df['Vertical'].min()
        lateral_max = accel_df['Lateral'].max()
        lateral_min = accel_df['Lateral'].min()
        longitudinal_max = accel_df['Longitudinal'].max()
        longitudinal_min = accel_df['Longitudinal'].min()
        
        # Determine y-axis range for each subplot
        def get_y_range(data_min, data_max):
            """Get y-axis range: default -5 to 5, extend to -10 to 10 if data exceeds ±5"""
            if data_max > 5 or data_min < -5:
                return [-10, 10]
            else:
                return [-5, 5]
        
        y_range_vertical = get_y_range(vertical_min, vertical_max)
        y_range_lateral = get_y_range(lateral_min, lateral_max)
        y_range_longitudinal = get_y_range(longitudinal_min, longitudinal_max)
        
        # Set y-axis ranges for each subplot
        fig_g.update_yaxes(range=y_range_vertical, row=1, col=1)
        fig_g.update_yaxes(range=y_range_lateral, row=2, col=1)
        fig_g.update_yaxes(range=y_range_longitudinal, row=3, col=1)
        
        # Set x-axis titles to distance for all subplots
        for i in range(1, 4):
            fig_g.update_xaxes(title_text="Distance (m)", row=i, col=1)
        fig_g.update_yaxes(title_text="G", row=2, col=1)
        
        fig_g.update_layout(
            height=700,  # Increased from 500 to 700 for better y-axis visibility
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig_g, use_container_width=True)
    
    # Egg Plot Visualization (comfort envelopes)
    if 'accel_df' in st.session_state:
        accel_df = st.session_state.accel_df
        st.divider()
    # Bottom: On-demand 3D view (resource heavy)
    st.divider()
    st.subheader("🧭 On-Demand 3D View")
    import plotly.graph_objects as go
    downsample = st.slider("Preview resolution", 200, 2000, 1200, 100,
                            help="Fewer points = faster rendering")
    if st.button("🎥 Generate 3D View", help="Render a 3D preview of the current track") and 'track_x' in st.session_state:
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
    🎢 Powered by LightGBM Extreme Features model (26 engineered features)<br>
    Using accelerometer data from RideForcesDB and ratings from Captain Coaster
</div>
""", unsafe_allow_html=True)
