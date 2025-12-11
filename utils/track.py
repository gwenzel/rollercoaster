# utils/track.py
import numpy as np
import pandas as pd


def generate_track(height_peak, drop_angle, loop_radius, num_hills, hill_amplitude):
    height_start = 0
    drop_length = 60
    hill_wavelength = 30
    final_length = 40


    # --- SECTION 1: Climb ---
    x1 = np.linspace(0, 30, 100)
    y1 = np.linspace(height_start, height_peak, 100)


    # --- SECTION 2: Drop ---
    x2 = np.linspace(x1[-1], x1[-1] + drop_length, 100)
    y2 = height_peak - (x2 - x1[-1]) * np.tan(np.radians(drop_angle))


    # --- SECTION 3: Loop ---
    loop_length = 2 * np.pi * loop_radius
    x3 = np.linspace(x2[-1], x2[-1] + loop_length, 200)
    theta = np.linspace(0, 2*np.pi, 200)
    loop_center_y = y2[-1] - loop_radius
    y3 = loop_center_y + loop_radius * np.cos(theta)


    # --- SECTION 4: Hills ---
    x4 = np.linspace(x3[-1], x3[-1] + num_hills * hill_wavelength, 300)
    y4 = y3[-1] + hill_amplitude * np.sin(2 * np.pi * (x4 - x3[-1]) / hill_wavelength)


    # --- SECTION 5: Final Flat ---
    x5 = np.linspace(x4[-1], x4[-1] + final_length, 50)
    y5 = np.ones_like(x5) * y4[-1]


    x = np.concatenate([x1, x2, x3, x4, x5])
    y = np.concatenate([y1, y2, y3, y4, y5])


    return pd.DataFrame({"x": x, "y": y})


# ============ MODULAR TRACK BUILDING ============

def build_element(element_type, params, start_x, start_y):
    """Generate a single track element."""
    
    if element_type == 'climb':
        length = params['length']
        height = params['height']
        num_points = max(50, int(length * 2))
        x = np.linspace(start_x, start_x + length, num_points)
        y = np.linspace(start_y, start_y + height, num_points)
    
    elif element_type == 'drop':
        # Support both legacy (length+angle) and app-style (height+steepness) params
        if 'height' in params:
            height = float(params.get('height', 30))
            # Map 'steepness' (0.5..1.0) to angle up to 30° if angle not provided
            angle = float(params.get('angle', 30.0 * float(params.get('steepness', 0.8))))
            # Compute length to achieve the desired vertical drop height at given angle
            tan_ang = np.tan(np.radians(angle))
            length = float(params.get('length', max(20.0, height / max(tan_ang, 1e-3))))
        else:
            length = float(params['length'])
            angle = float(params['angle'])
        num_points = max(50, int(length * 2))
        x = np.linspace(start_x, start_x + length, num_points)
        y = start_y - (x - start_x) * np.tan(np.radians(angle))
    
    elif element_type == 'loop':
        radius = params['radius']
        theta = np.linspace(0, 2*np.pi, 200)
        loop_length = 2 * np.pi * radius
        x = np.linspace(start_x, start_x + loop_length, 200)
        loop_center_y = start_y - radius
        y = loop_center_y + radius * np.cos(theta)
    
    elif element_type == 'clothoid_loop':
        radius = params['radius']
        transition_length = params['transition_length']
        
        n_trans = 50
        n_loop = 150
        
        # Entry clothoid: gradually increase curvature from straight to circular
        # Clothoid parameter for smooth transition
        s_entry = np.linspace(0, transition_length, n_trans)
        # Curvature increases quadratically (clothoid property)
        k_max = 1.0 / radius  # Maximum curvature (at circle)
        k_entry = k_max * (s_entry / transition_length)**2
        
        # Integrate to get angles and positions
        angles_entry = np.cumsum(k_entry) * (transition_length / n_trans)
        
        x1 = [start_x]
        y1 = [start_y]
        ds = transition_length / n_trans
        
        for angle in angles_entry[:-1]:
            x1.append(x1[-1] + ds * np.cos(angle))
            y1.append(y1[-1] + ds * np.sin(angle))
        
        x1 = np.array(x1)
        y1 = np.array(y1)
        
        # Main circular loop - connect smoothly to clothoid
        entry_angle = angles_entry[-2] if len(angles_entry) > 1 else 0
        
        # Calculate loop center to connect smoothly
        # At end of clothoid, we're moving at entry_angle with curvature 1/radius
        loop_center_x = x1[-1] - radius * np.sin(entry_angle)
        loop_center_y = y1[-1] + radius * np.cos(entry_angle)
        
        # Full circular loop starting from entry angle
        theta_loop = np.linspace(entry_angle, entry_angle + 2*np.pi, n_loop)
        x2 = loop_center_x + radius * np.sin(theta_loop)
        y2 = loop_center_y - radius * np.cos(theta_loop)
        
        # Exit clothoid: gradually decrease curvature from circular to straight
        exit_angle = theta_loop[-1]
        s_exit = np.linspace(transition_length, 0, n_trans)
        k_exit = k_max * (s_exit / transition_length)**2
        
        # Angles decrease as curvature decreases
        angle_changes = k_exit * (transition_length / n_trans)
        angles_exit = exit_angle - np.cumsum(angle_changes)
        
        x3 = [x2[-1]]
        y3 = [y2[-1]]
        
        for angle in angles_exit[:-1]:
            x3.append(x3[-1] + ds * np.cos(angle))
            y3.append(y3[-1] + ds * np.sin(angle))
        
        x3 = np.array(x3)
        y3 = np.array(y3)
        
        x = np.concatenate([x1, x2, x3])
        y = np.concatenate([y1, y2, y3])
    
    elif element_type == 'hills':
        num_hills = params['num_hills']
        amplitude = params['amplitude']
        wavelength = params['wavelength']
        total_length = num_hills * wavelength
        num_points = max(100, num_hills * 50)
        x = np.linspace(start_x, start_x + total_length, num_points)
        y = start_y + amplitude * np.sin(2 * np.pi * (x - start_x) / wavelength)
    
    elif element_type == 'gaussian_curve':
        amplitude = params['amplitude']
        width = params['width']
        length = params['length']
        num_points = max(100, int(length * 2))
        x = np.linspace(start_x, start_x + length, num_points)
        # Gaussian bump: y = A * exp(-(x-center)^2 / (2*width^2))
        center = start_x + length / 2
        y = start_y + amplitude * np.exp(-((x - center)**2) / (2 * width**2))
    
    elif element_type == 'parabolic_curve':
        amplitude = params['amplitude']
        length = params['length']
        num_points = max(100, int(length * 2))
        x = np.linspace(start_x, start_x + length, num_points)
        # Parabolic arc: y = A * (4 * (x-start)*(end-x) / L^2)
        t = (x - start_x) / length  # normalize 0 to 1
        y = start_y + amplitude * 4 * t * (1 - t)
    
    elif element_type == 'rotation':
        angle_deg = params['angle']
        radius = params['radius']
        axis = params.get('axis', 'roll')
        
        # For 2D visualization, show as a spiral/helix projection
        angle_rad = np.radians(angle_deg)
        num_points = max(100, int(angle_deg))
        
        if axis == 'roll':
            # Barrel roll - maintain height, spiral horizontally
            theta = np.linspace(0, angle_rad, num_points)
            arc_length = radius * angle_rad
            x = np.linspace(start_x, start_x + arc_length, num_points)
            # Sinusoidal variation to show rotation
            y = start_y + radius * 0.3 * np.sin(theta * 2)
        else:  # barrel
            # Similar but with different profile
            theta = np.linspace(0, angle_rad, num_points)
            arc_length = radius * angle_rad
            x = np.linspace(start_x, start_x + arc_length, num_points)
            y = start_y + radius * 0.5 * np.cos(theta)
    
    else:
        # Flat section fallback
        x = np.array([start_x, start_x + 20])
        y = np.array([start_y, start_y])
    
    return x, y


def build_modular_track(track_elements):
    """Build complete track from list of elements."""
    all_x = []
    all_y = []
    
    current_x = 0
    current_y = 0
    
    for element in track_elements:
        x, y = build_element(element['type'], element['params'], current_x, current_y)
        all_x.append(x)
        all_y.append(y)
        
        # Update position for next element
        current_x = x[-1]
        current_y = y[-1]
    
    # Concatenate all segments
    x_full = np.concatenate(all_x)
    y_full = np.concatenate(all_y)
    
    return pd.DataFrame({"x": x_full, "y": y_full})


def compute_features_modular(track_df, track_elements):
    """Compute features for modular track."""
    x, y = track_df["x"].values, track_df["y"].values
    slope = np.gradient(y, x)
    curvature = np.gradient(slope, x)
    total_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    
    # Count element types
    element_counts = {}
    for elem in track_elements:
        elem_type = elem['type']
        element_counts[elem_type] = element_counts.get(elem_type, 0) + 1
    
    return {
        "total_length": float(total_length),
        "max_height": float(np.max(y)),
        "min_height": float(np.min(y)),
        "max_slope": float(np.max(np.abs(slope))),
        "mean_curvature": float(np.mean(np.abs(curvature))),
        "num_elements": len(track_elements),
        "element_types": element_counts,
    }

def compute_features(track_df, loop_radius, num_hills):
    x, y = track_df["x"].values, track_df["y"].values
    slope = np.gradient(y, x)
    curvature = np.gradient(slope, x)
    total_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))


    return {
    "total_length": float(total_length),
    "max_height": float(np.max(y)),
    "max_slope": float(np.max(np.abs(slope))),
    "mean_curvature": float(np.mean(np.abs(curvature))),
    "loop_radius": loop_radius,
    "num_hills": num_hills,
    }


def compute_acceleration(track_df, height_peak, g=9.81):
    # Approximate particle acceleration assuming conservation of energy from the top
    y = track_df["y"].values
    v = np.sqrt(2 * g * (height_peak - y).clip(min=0)) # velocity based on drop height
    x = track_df["x"].values
    
    # Add small epsilon to avoid divide by zero in gradient calculation
    x_diff = np.diff(x)
    x_diff = np.where(np.abs(x_diff) < 1e-10, 1e-10, x_diff)
    
    # Use gradient with edge_order=1 to be more stable
    with np.errstate(divide='ignore', invalid='ignore'):
        a_tangent = np.gradient(v, x, edge_order=1)
        a_tangent = np.nan_to_num(a_tangent, nan=0.0, posinf=0.0, neginf=0.0)
    
    track_df["velocity"] = v
    track_df["acceleration"] = a_tangent
    return track_df