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
    
    # Compute velocity using energy conservation
    # v = sqrt(2*g*(h_max - h))
    g = 9.81
    h_max = y.max()
    v = np.sqrt(2 * g * np.maximum(0, h_max - y))
    
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
    dt_x = np.gradient(tangent_x)
    dt_y = np.gradient(tangent_y)
    dt_z = np.gradient(tangent_z)
    
    curvature = np.sqrt(dt_x**2 + dt_y**2 + dt_z**2)
    curvature = np.maximum(curvature, 1e-10)
    
    # Centripetal acceleration magnitude
    a_centripetal = v**2 * curvature
    
    # Normal vector (points toward center of curvature)
    # N = dT/ds / |dT/ds|
    normal_x = dt_x / curvature
    normal_y = dt_y / curvature
    normal_z = dt_z / curvature
    
    # Binormal vector (perpendicular to both tangent and normal)
    # B = T × N
    binormal_x = tangent_y * normal_z - tangent_z * normal_y
    binormal_y = tangent_z * normal_x - tangent_x * normal_z
    binormal_z = tangent_x * normal_y - tangent_y * normal_x
    
    # Normalize binormal
    binormal_mag = np.sqrt(binormal_x**2 + binormal_y**2 + binormal_z**2)
    binormal_mag = np.maximum(binormal_mag, 1e-10)
    
    binormal_x /= binormal_mag
    binormal_y /= binormal_mag
    binormal_z /= binormal_mag
    
    # Gravity vector (in world frame)
    gravity = np.array([0, -g, 0])
    
    # Total acceleration in world frame (exclude gravity here; add it via vertical projection below)
    # a_total = a_tangential * T + a_centripetal * N
    a_world_x = a_tangential * tangent_x + a_centripetal * normal_x
    a_world_y = a_tangential * tangent_y + a_centripetal * normal_y
    a_world_z = a_tangential * tangent_z + a_centripetal * normal_z
    
    # Transform to rider's reference frame
    # Longitudinal = projection onto tangent (forward direction)
    a_longitudinal = (a_world_x * tangent_x + 
                     a_world_y * tangent_y + 
                     a_world_z * tangent_z)
    
    # Lateral = projection onto binormal (side direction)
    a_lateral = (a_world_x * binormal_x + 
                a_world_y * binormal_y + 
                a_world_z * binormal_z)
    
    # Vertical = projection onto normal (perpendicular to track)
    a_vertical = (a_world_x * normal_x + 
                 a_world_y * normal_y + 
                 a_world_z * normal_z)
    
    # Add gravity component in vertical direction only
    # The rider feels gravity as part of the vertical acceleration
    gravity_vertical = -(gravity[0] * normal_x + 
                        gravity[1] * normal_y + 
                        gravity[2] * normal_z)
    a_vertical += gravity_vertical
    
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


def track_to_accelerometer_data(track_df):
    """
    Convert track coordinates to accelerometer readings for BiGRU model.
    
    This is the main function to use for Streamlit integration.
    
    Args:
        track_df: DataFrame with columns ['x', 'y'] from build_modular_track()
        
    Returns:
        DataFrame with columns ['Time', 'Lateral', 'Vertical', 'Longitudinal']
        matching the format of real wearable accelerometer data
    """
    try:
        accel_df = compute_rider_accelerations(track_df)
        
        # Ensure output is in correct format
        accel_df = accel_df[['Time', 'Lateral', 'Vertical', 'Longitudinal']]
        
        return accel_df
    
    except Exception as e:
        print(f"Error in track_to_accelerometer_data: {e}")
        
        # Fallback: return simple approximation
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
