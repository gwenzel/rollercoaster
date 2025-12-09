"""
Debug lateral axis calculation
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.track import build_modular_track
from utils.accelerometer_transform import compute_rider_accelerations, compute_track_derivatives
from utils.acceleration import compute_acc_profile, G_VEC

def debug_lateral_axis():
    """Debug why lateral forces aren't zero for 2D track"""
    track_df = build_modular_track([
        {'type': 'lift_hill', 'params': {'length': 40, 'height': 80}},
        {'type': 'drop', 'params': {'length': 50, 'angle': 60}},
    ])
    
    x = track_df['x'].values
    y = track_df['y'].values
    z = track_df.get('z', pd.Series(np.zeros_like(x))).values
    points = np.column_stack([x, z, y])
    
    print("="*80)
    print("LATERAL AXIS DEBUG")
    print("="*80)
    
    print("\n1. TRACK GEOMETRY:")
    print(f"   x (forward): [{x.min():.2f}, {x.max():.2f}] m")
    print(f"   y (vertical): [{y.min():.2f}, {y.max():.2f}] m")
    print(f"   z (lateral): [{z.min():.2f}, {z.max():.2f}] m")
    print(f"   Track is 2D: z is all zeros = {np.allclose(z, 0)}")
    
    # Advanced model
    print("\n2. ADVANCED MODEL LATERAL AXIS:")
    acc_result = compute_acc_profile(points, dt=0.02, v0=0.0, use_energy_conservation=True)
    e_tan = acc_result['e_tan']
    ez = np.array([0.0, 0.0, 1.0])
    
    # Lateral axis calculation
    lat_vec_advanced = np.cross(ez, e_tan)
    lat_norm_advanced = np.linalg.norm(lat_vec_advanced, axis=1, keepdims=True)
    lat_vec_advanced = lat_vec_advanced / np.where(lat_norm_advanced < 1e-9, 1.0, lat_norm_advanced)
    
    print(f"   Tangent (first 3): {e_tan[:3]}")
    print(f"   Tangent Z component: [{e_tan[:, 2].min():.3f}, {e_tan[:, 2].max():.3f}]")
    print(f"   Lateral vector (first 3): {lat_vec_advanced[:3]}")
    print(f"   Lateral vector Y component: [{lat_vec_advanced[:, 1].min():.3f}, {lat_vec_advanced[:, 1].max():.3f}]")
    print(f"   Lateral G-forces: [{acc_result['f_lat_g'].min():.6f}, {acc_result['f_lat_g'].max():.6f}] g")
    print(f"   Should be zero: {np.allclose(acc_result['f_lat_g'], 0, atol=1e-6)}")
    
    # Simple model
    print("\n3. SIMPLE MODEL LATERAL AXIS:")
    df = compute_track_derivatives(track_df)
    result_simple = compute_rider_accelerations(track_df)
    
    # Get tangent vectors (in original coords)
    tangent_x = df['tangent_x'].values
    tangent_y = df['tangent_y'].values
    tangent_z = df['tangent_z'].values
    
    print(f"   Tangent (original, first 3):")
    print(f"     X: {tangent_x[:3]}")
    print(f"     Y: {tangent_y[:3]}")
    print(f"     Z: {tangent_z[:3]}")
    
    # After transformation to Z-up
    tangent_x_up = tangent_x
    tangent_y_up = tangent_z  # Lateral (old z) becomes y
    tangent_z_up = tangent_y  # Vertical (old y) becomes z
    
    print(f"   Tangent (Z-up, first 3):")
    print(f"     X: {tangent_x_up[:3]}")
    print(f"     Y: {tangent_y_up[:3]} (should be ~0 for 2D track)")
    print(f"     Z: {tangent_z_up[:3]}")
    
    # Check if tangent Y is zero
    print(f"   Tangent Y component (lateral): [{tangent_y_up.min():.6f}, {tangent_y_up.max():.6f}]")
    print(f"   Should be zero: {np.allclose(tangent_y_up, 0, atol=1e-6)}")
    
    # Binormal calculation
    # In original coords: binormal = T × N
    # After transform: binormal_up = T_up × N_up
    # But we need to check what the binormal actually is
    
    print(f"   Lateral G-forces: [{result_simple['Lateral'].min():.6f}, {result_simple['Lateral'].max():.6f}] g")
    print(f"   Should be zero: {np.allclose(result_simple['Lateral'], 0, atol=1e-6)}")
    
    # Check what cross(ez, tangent) gives for simple model
    ez = np.array([0.0, 0.0, 1.0])
    tangent_up = np.column_stack([tangent_x_up, tangent_y_up, tangent_z_up])
    lat_vec_simple_cross = np.cross(ez, tangent_up)
    lat_norm_simple_cross = np.linalg.norm(lat_vec_simple_cross, axis=1, keepdims=True)
    lat_vec_simple_cross = lat_vec_simple_cross / np.where(lat_norm_simple_cross < 1e-9, 1.0, lat_norm_simple_cross)
    
    print(f"\n4. COMPARISON:")
    print(f"   Advanced lateral vector (cross(ez, tan)): {lat_vec_advanced[0]}")
    print(f"   Simple lateral vector (cross(ez, tan_up)): {lat_vec_simple_cross[0]}")
    print(f"   Match: {np.allclose(lat_vec_advanced, lat_vec_simple_cross, atol=1e-6)}")
    
    # Check if binormal matches cross(ez, tangent)
    # We need to get the binormal from the simple model
    # It's computed as T × N, then transformed
    
    print(f"\n5. CHECKING IF BINORMAL = CROSS(EZ, TANGENT):")
    print(f"   For 2D track in XZ plane, binormal should be in Y direction")
    print(f"   cross(ez, tangent) should also be in Y direction")
    print(f"   They should match!")

if __name__ == "__main__":
    debug_lateral_axis()

