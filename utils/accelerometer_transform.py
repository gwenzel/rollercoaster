"""
Transform track coordinates into relative accelerometer readings.

This module converts absolute track geometry (x, y, z coordinates) into 
relative acceleration measurements as they would be recorded by a wearable 
accelerometer on a rollercoaster rider.

The transformation accounts for:
- Gravitational acceleration (9.81 m/s²)
- Centripetal acceleration in curves
- Tangential acceleration along the track
- Reference frame rotation (track-relative to rider-relative)
"""

import numpy as np
import pandas as pd
from utils.acceleration import compute_acc_profile


def compute_track_derivatives(track_df):
    """
    Compute velocity, acceleration, and curvature from track coordinates.
    
    Args:
        track_df: DataFrame with columns ['x', 'y'] (and optionally 'z')
        
    Returns:
        DataFrame with added columns for derivatives and motion parameters
    """
    df = track_df.copy()
    
    # Get coordinates
    x = df['x'].values
    y = df['y'].values
    z = df.get('z', pd.Series(np.zeros_like(x))).values
    
    # Compute arc length (approximate)
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    dz = np.diff(z, prepend=z[0])
    ds = np.sqrt(dx**2 + dy**2 + dz**2)
    s = np.cumsum(ds)
    
    # Compute velocity using pure energy conservation
    # Speed is determined by height changes and initial energy
    # No hardcoded lift mechanics - use launch/booster blocks for initial speed
    g = 9.81
    h_initial = y[0]  # Starting height
    
    # Energy conservation: (1/2)mv² = (1/2)mv0² + mg(h_initial - h)
    # Solving for v: v² = v0² + 2g(h_initial - h)
    # Start from rest (v0 = 0) - use launch blocks to add initial energy
    v0 = 0.0  # Start from rest
    energy_efficiency = 0.95  # 95% efficiency (minimal friction for modern coasters)
    
    v = np.sqrt(np.maximum(0, v0**2 + 2 * g * (h_initial - y) * energy_efficiency))
    
    # Add small minimum velocity to avoid division by zero
    v = np.maximum(v, 0.1)
    
    # Compute tangent vector (direction of motion)
    # Smooth derivatives to avoid numerical noise
    from scipy.ndimage import gaussian_filter1d
    
    dx_smooth = gaussian_filter1d(x, sigma=2, mode='nearest')
    dy_smooth = gaussian_filter1d(y, sigma=2, mode='nearest')
    dz_smooth = gaussian_filter1d(z, sigma=2, mode='nearest')
    
    dx_ds = np.gradient(dx_smooth)
    dy_ds = np.gradient(dy_smooth)
    dz_ds = np.gradient(dz_smooth)
    
    # Normalize tangent vector
    tangent_mag = np.sqrt(dx_ds**2 + dy_ds**2 + dz_ds**2)
    tangent_mag = np.maximum(tangent_mag, 1e-10)
    
    tangent_x = dx_ds / tangent_mag
    tangent_y = dy_ds / tangent_mag
    tangent_z = dz_ds / tangent_mag
    
    # Store results
    df['arc_length'] = s
    df['velocity'] = v
    df['tangent_x'] = tangent_x
    df['tangent_y'] = tangent_y
    df['tangent_z'] = tangent_z
    
    return df


def compute_rider_accelerations(track_df):
    """
    Compute accelerations in the rider's reference frame.
    
    The rider's reference frame:
    - Longitudinal: Forward/backward (along track direction)
    - Lateral: Left/right (perpendicular to track, horizontal)
    - Vertical: Up/down (perpendicular to track, includes gravity)
    
    Args:
        track_df: DataFrame with x, y coordinates and derivatives
        
    Returns:
        DataFrame with columns ['Lateral', 'Vertical', 'Longitudinal']
    """
    df = compute_track_derivatives(track_df)
    
    # Get track parameters
    x = df['x'].values
    y = df['y'].values
    z = df.get('z', pd.Series(np.zeros_like(x))).values
    v = df['velocity'].values
    
    tangent_x = df['tangent_x'].values
    tangent_y = df['tangent_y'].values
    tangent_z = df['tangent_z'].values
    
    g = 9.81
    
    # Compute tangential acceleration (rate of speed change)
    dv_ds = np.gradient(v)
    a_tangential = dv_ds * v  # dv/dt = dv/ds * ds/dt = dv/ds * v
    
    # Compute curvature and centripetal acceleration
    # Curvature κ = |dT/ds| where T is the unit tangent
    # Apply smoothing to reduce noise spikes at block joints
    from scipy.ndimage import gaussian_filter1d
    
    tangent_x_smooth = gaussian_filter1d(tangent_x, sigma=1.5, mode='nearest')
    tangent_y_smooth = gaussian_filter1d(tangent_y, sigma=1.5, mode='nearest')
    tangent_z_smooth = gaussian_filter1d(tangent_z, sigma=1.5, mode='nearest')
    
    dt_x = np.gradient(tangent_x_smooth)
    dt_y = np.gradient(tangent_y_smooth)
    dt_z = np.gradient(tangent_z_smooth)
    
    curvature = np.sqrt(dt_x**2 + dt_y**2 + dt_z**2)
    curvature = np.maximum(curvature, 1e-10)
    
    # Smooth the curvature itself to eliminate remaining spikes
    curvature = gaussian_filter1d(curvature, sigma=1.0, mode='nearest')
    
    # Centripetal acceleration magnitude
    a_centripetal = v**2 * curvature
    
    # Normal vector (points toward center of curvature)
    # N = dT/ds / |dT/ds|
    normal_x = dt_x / curvature
    normal_y = dt_y / curvature
    normal_z = dt_z / curvature
    
    # Smooth normal vectors to reduce discontinuities
    normal_x = gaussian_filter1d(normal_x, sigma=1.0, mode='nearest')
    normal_y = gaussian_filter1d(normal_y, sigma=1.0, mode='nearest')
    normal_z = gaussian_filter1d(normal_z, sigma=1.0, mode='nearest')
    
    # Re-normalize after smoothing
    normal_mag = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_mag = np.maximum(normal_mag, 1e-10)
    normal_x /= normal_mag
    normal_y /= normal_mag
    normal_z /= normal_mag
    
    # Binormal vector (perpendicular to both tangent and normal)
    # B = T × N (use smoothed tangents)
    binormal_x = tangent_y_smooth * normal_z - tangent_z_smooth * normal_y
    binormal_y = tangent_z_smooth * normal_x - tangent_x_smooth * normal_z
    binormal_z = tangent_x_smooth * normal_y - tangent_y_smooth * normal_x
    
    # Normalize binormal
    binormal_mag = np.sqrt(binormal_x**2 + binormal_y**2 + binormal_z**2)
    binormal_mag = np.maximum(binormal_mag, 1e-10)
    
    binormal_x /= binormal_mag
    binormal_y /= binormal_mag
    binormal_z /= binormal_mag
    
    # Transform to Z-up coordinate system (matching app_builder and compute_acc_profile)
    # Original coordinates: x=forward, y=vertical, z=lateral
    # Z-up coordinates: x=forward, y=lateral, z=vertical
    # Transformation: (x, y, z) -> (x, z, y)
    
    # Transform tangent vectors to Z-up
    tangent_x_up = tangent_x  # Forward stays same
    tangent_y_up = tangent_z  # Lateral (old z) becomes y
    tangent_z_up = tangent_y  # Vertical (old y) becomes z
    
    # Transform normal vectors to Z-up
    normal_x_up = normal_x
    normal_y_up = normal_z
    normal_z_up = normal_y
    
    # Transform binormal vectors to Z-up
    # Original: binormal = [bx, by, bz] where by is lateral (Y in original = lateral)
    # Z-up: we want binormal to point in Y direction (lateral in Z-up)
    # Transformation: (x, y, z) -> (x, z, y)
    # So: binormal_up = [bx, bz, by]
    # But wait - in original coords, for 2D track:
    #   - binormal points in Y direction (lateral) = [0, by, 0]
    #   - After transform: [0, 0, by] which is Z direction (wrong!)
    # We need binormal to point in Y direction (lateral) in Z-up coords
    # So: binormal_up = [bx, by, bz] where by (original lateral) becomes Y (Z-up lateral)
    # Actually, the correct transformation is:
    #   Original binormal [bx, by, bz] -> Z-up [bx, bz, by]
    #   But for 2D track, binormal = [0, by, 0], so we get [0, 0, by] which is wrong
    # The issue is that binormal in original coords points in Y (lateral)
    # In Z-up coords, lateral is Y, so binormal should be [0, by, 0] in Z-up
    # So we need: binormal_up = [bx, by, bz] (no swap, because by is already lateral)
    # Wait, let me think again...
    # Original: (x, y, z) = (forward, vertical, lateral)
    # Z-up: (x, y, z) = (forward, lateral, vertical)
    # Binormal in original: points in Y direction (which is vertical in original, but lateral in Z-up? No!)
    # Actually, in original coords, Y is vertical, Z is lateral
    # Binormal = T × N, where T and N are in XZ plane (forward-vertical plane)
    # So binormal points in Y direction (perpendicular to XZ plane)
    # But Y in original is vertical, not lateral!
    # For a 2D track in XZ plane, binormal should point in the direction perpendicular to the plane
    # In original coords, that's Y (vertical direction)
    # But we want lateral, which is Z in original coords
    # So the binormal calculation is wrong for 2D tracks!
    # Actually, for a 2D track in the XZ plane:
    #   - T = [tx, ty, tz] but ty=0 (no vertical component in original? No, ty is vertical!)
    #   - For track in XZ plane: T = [tx, 0, tz] where tx is forward, tz is... wait
    # Let me reconsider: track is in XY plane (forward-vertical), so Z is lateral
    # T = [tx, ty, 0] where tx is forward, ty is vertical
    # N is also in XY plane
    # Binormal = T × N = [0, 0, something] which points in Z direction (lateral) - correct!
    # After transform to Z-up: (x, y, z) -> (x, z, y)
    # Binormal [0, 0, bz] -> [0, bz, 0] which is Y direction (lateral) in Z-up - correct!
    # So the transformation should be: [bx, by, bz] -> [bx, bz, by]
    binormal_x_up = binormal_x
    binormal_y_up = binormal_z  # Original Z (lateral) becomes Y (lateral in Z-up)
    binormal_z_up = binormal_y  # Original Y (vertical) becomes Z (vertical in Z-up)
    
    # Gravity vector in Z-up coordinate system (Z-down convention, matching compute_acc_profile)
    gravity = np.array([0, 0, -g])
    
    # Total acceleration in world frame (Z-up coordinates)
    # a_total = a_tangential * T + a_centripetal * N
    a_world = np.stack([
        a_tangential * tangent_x_up + a_centripetal * normal_x_up,
        a_tangential * tangent_y_up + a_centripetal * normal_y_up,
        a_tangential * tangent_z_up + a_centripetal * normal_z_up
    ], axis=1)

    # Subtract gravity vector to get specific force (what the rider feels)
    # This matches the advanced model convention (Z-up, Z-down gravity)
    a_spec = a_world - gravity

    # Project specific force onto rider axes (in Z-up coordinates)
    # Use same lateral axis calculation as advanced model: cross(ez, tangent)
    # This ensures consistency and correct handling of 2D tracks
    tangent = np.stack([tangent_x_up, tangent_y_up, tangent_z_up], axis=1)
    ez = np.array([0.0, 0.0, 1.0])  # Z-up unit vector
    lat_vec = np.cross(ez, tangent)  # Lateral axis (same as advanced model)
    lat_norm = np.linalg.norm(lat_vec, axis=1, keepdims=True)
    lat_vec = lat_vec / np.where(lat_norm < 1e-9, 1.0, lat_norm)
    
    normal = np.stack([normal_x_up, normal_y_up, normal_z_up], axis=1)

    a_longitudinal = np.einsum('ij,ij->i', a_spec, tangent)
    a_lateral = np.einsum('ij,ij->i', a_spec, lat_vec)  # Use cross(ez, tangent) like advanced model
    # Vertical is the Z component of specific force (Z-up convention, matching compute_acc_profile)
    a_vertical = a_spec[:, 2]

    # Normalize to g-forces (divide by 9.81)
    a_lateral_g = a_lateral / g
    a_vertical_g = a_vertical / g
    a_longitudinal_g = a_longitudinal / g

    # Create output DataFrame matching wearable format
    result_df = pd.DataFrame({
        'Time': np.linspace(0, len(df) * 0.02, len(df)),  # Assume 50Hz sampling
        'Lateral': a_lateral_g,
        'Vertical': a_vertical_g,
        'Longitudinal': a_longitudinal_g
    })
    
    # Handle NaN and infinite values
    result_df = result_df.replace([np.inf, -np.inf], np.nan)
    result_df = result_df.ffill().bfill().fillna(0)
    
    # Clip extreme values to realistic range (-10g to +10g)
    for col in ['Lateral', 'Vertical', 'Longitudinal']:
        result_df[col] = np.clip(result_df[col], -10, 10)
    
    return result_df


def track_to_accelerometer_data(track_df, mass=1200.0, rho=1.0, Cd=0.08, A=2.5, mu=0.001):
    """
    Convert track coordinates to accelerometer readings for BiGRU model.
    
    This is the main function to use for Streamlit integration.
    Uses the realistic physics engine (with friction/drag) as default,
    with fallback to energy conservation if it fails.
    
    Args:
        track_df: DataFrame with columns ['x', 'y'] from build_modular_track()
        
    Returns:
        DataFrame with columns ['Time', 'Lateral', 'Vertical', 'Longitudinal']
        matching the format of real wearable accelerometer data
    """
    try:
        # Prepare 3D points array for acceleration.py (expects Nx3 numpy array)
        x = track_df['x'].values
        y = track_df['y'].values
        z = track_df.get('z', pd.Series(np.zeros_like(x))).values
        
        # Stack into Nx3 array with z-up convention (acceleration.py uses z as vertical)
        # Our coordinates: x=forward, y=vertical, z=lateral
        # acceleration.py expects: x=forward, y=lateral, z=vertical
        # So we map: (x, y, z) -> (x, z, y)
        points = np.column_stack([x, z, y])  # (forward, lateral, vertical)
        
        # Initial speed: let energy conservation handle it naturally
        # Start at near-zero speed (3 m/s ~ walking pace at station)
        # Energy will build from lift hill height automatically
        v0 = 3.0  # m/s (station/lift start speed)
        
        # Compute acceleration profile using realistic physics
        # Use provided physics parameters (or defaults)
        acc_result = compute_acc_profile(
            points,
            dt=0.02,           # 50Hz sampling
            mass=mass,         # Train mass (kg) - parameterized
            rho=rho,           # Air density (kg/m³) - parameterized
            Cd=Cd,             # Drag coefficient - parameterized
            A=A,               # Frontal area (m²) - parameterized
            mu=mu,             # Rolling friction - parameterized
            v0=v0,             # Initial speed (m/s) - calculated above
            use_energy_conservation=True  # Use energy-based speeds for reliability
        )
        
        # Extract specific force in g-units (what accelerometer measures)
        # acceleration.py returns f_long_g, f_lat_g, f_vert_g
        n = len(x)
        
        # Clip extreme values to prevent physics simulation artifacts
        # Set to ±10g to allow visibility of intense forces while filtering numerical spikes
        lateral = np.clip(acc_result['f_lat_g'], -10.0, 10.0)
        vertical = np.clip(acc_result['f_vert_g'], -10.0, 10.0)
        longitudinal = np.clip(acc_result['f_long_g'], -10.0, 10.0)
        
        # Apply moderate Gaussian smoothing to reduce numerical oscillations
        # Balances smoothness with peak force preservation for realistic coaster feel
        from scipy.ndimage import gaussian_filter1d
        sigma_smooth = 2.0  # Moderate smoothing (~100ms window at 50Hz)
        # Similar to real accelerometer response time
        lateral = gaussian_filter1d(lateral, sigma=sigma_smooth, mode='nearest')
        vertical = gaussian_filter1d(vertical, sigma=sigma_smooth, mode='nearest')
        longitudinal = gaussian_filter1d(longitudinal, sigma=sigma_smooth, mode='nearest')
        
        accel_df = pd.DataFrame({
            'Time': np.linspace(0, n * 0.02, n),
            'Lateral': lateral,          # Side-to-side
            'Vertical': vertical,        # Up-down (includes gravity effect)
            'Longitudinal': longitudinal # Forward-backward
        })
        
        return accel_df
    
    except Exception as e:
        print(f"Warning: Realistic physics failed ({e}), falling back to energy conservation...")
        
        # Fallback to the original energy conservation method
        try:
            accel_df = compute_rider_accelerations(track_df)
            accel_df = accel_df[['Time', 'Lateral', 'Vertical', 'Longitudinal']]
            return accel_df
        except Exception as e2:
            print(f"Error in fallback method: {e2}")
            
            # Last resort: return simple approximation
            n = len(track_df)
            return pd.DataFrame({
                'Time': np.linspace(0, n * 0.02, n),
                'Lateral': np.zeros(n),
                'Vertical': np.ones(n),  # 1g vertical (gravity)
                'Longitudinal': np.zeros(n)
            })


if __name__ == "__main__":
    # Test with a simple track
    from utils.track import build_modular_track
    
    track_elements = [
        {'type': 'climb', 'params': {'length': 30, 'height': 50}},
        {'type': 'drop', 'params': {'length': 60, 'angle': 70}},
        {'type': 'loop', 'params': {'radius': 10}},
    ]
    
    print("Building track...")
    track_df = build_modular_track(track_elements)
    
    print(f"Track has {len(track_df)} points")
    
    print("\nConverting to accelerometer data...")
    accel_df = track_to_accelerometer_data(track_df)
    
    print("\nAccelerometer data shape:", accel_df.shape)
    print("\nFirst few rows:")
    print(accel_df.head(10))
    
    print("\nStatistics:")
    print(accel_df.describe())
    
    print("\nData ranges:")
    print(f"Lateral: [{accel_df['Lateral'].min():.2f}, {accel_df['Lateral'].max():.2f}] g")
    print(f"Vertical: [{accel_df['Vertical'].min():.2f}, {accel_df['Vertical'].max():.2f}] g")
    print(f"Longitudinal: [{accel_df['Longitudinal'].min():.2f}, {accel_df['Longitudinal'].max():.2f}] g")
