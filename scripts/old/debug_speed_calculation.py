"""
Debug speed calculation differences
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.track import build_modular_track
from utils.accelerometer_transform import compute_track_derivatives
from utils.acceleration import compute_acc_profile, G_NORM

def debug_speed_calculation():
    """Debug why speeds don't match"""
    track_df = build_modular_track([
        {'type': 'lift_hill', 'params': {'length': 40, 'height': 80}},
        {'type': 'drop', 'params': {'length': 50, 'angle': 60}},
    ])
    
    x = track_df['x'].values
    y = track_df['y'].values
    z = track_df.get('z', pd.Series(np.zeros_like(x))).values
    points = np.column_stack([x, z, y])
    
    print("="*80)
    print("SPEED CALCULATION DEBUG")
    print("="*80)
    
    # Simple model calculation
    print("\n1. SIMPLE MODEL CALCULATION:")
    h_initial_simple = y[0]
    print(f"   h_initial = {h_initial_simple:.2f} m")
    print(f"   y range: [{y.min():.2f}, {y.max():.2f}] m")
    
    g = 9.81
    v0_simple = 0.0
    energy_efficiency = 0.95
    
    # Manual calculation
    v_simple_manual = np.sqrt(np.maximum(0, v0_simple**2 + 2 * g * (h_initial_simple - y) * energy_efficiency))
    print(f"   Manual calculation:")
    print(f"     v² = v0² + 2*g*(h_initial - y) * efficiency")
    print(f"     v² = 0² + 2*9.81*({h_initial_simple} - y) * 0.95")
    print(f"     First point: v² = 2*9.81*({h_initial_simple} - {y[0]}) * 0.95 = {2 * g * (h_initial_simple - y[0]) * energy_efficiency:.4f}")
    print(f"     First point: v = {v_simple_manual[0]:.4f} m/s")
    print(f"     Last point: v² = 2*9.81*({h_initial_simple} - {y[-1]}) * 0.95 = {2 * g * (h_initial_simple - y[-1]) * energy_efficiency:.4f}")
    print(f"     Last point: v = {v_simple_manual[-1]:.4f} m/s")
    
    # From function
    df = compute_track_derivatives(track_df)
    v_simple_func = df['velocity'].values
    print(f"   From function:")
    print(f"     First point: v = {v_simple_func[0]:.4f} m/s")
    print(f"     Last point: v = {v_simple_func[-1]:.4f} m/s")
    print(f"     Match: {np.allclose(v_simple_manual, v_simple_func, rtol=0.01)}")
    
    # Advanced model calculation
    print("\n2. ADVANCED MODEL CALCULATION:")
    h_initial_advanced = points[0, 2]
    h_advanced = points[:, 2]
    print(f"   h_initial = {h_initial_advanced:.2f} m")
    print(f"   h (points[:,2]) range: [{h_advanced.min():.2f}, {h_advanced.max():.2f}] m")
    print(f"   Heights match: {np.allclose(y, h_advanced)}")
    
    v0_advanced = 0.0
    
    # Manual calculation
    v_advanced_manual = np.sqrt(np.maximum(0, v0_advanced**2 + 2 * G_NORM * (h_initial_advanced - h_advanced) * energy_efficiency))
    print(f"   Manual calculation:")
    print(f"     v² = v0² + 2*G_NORM*(h_initial - h) * efficiency")
    print(f"     v² = 0² + 2*{G_NORM:.2f}*({h_initial_advanced} - h) * 0.95")
    print(f"     First point: v² = 2*{G_NORM:.2f}*({h_initial_advanced} - {h_advanced[0]}) * 0.95 = {2 * G_NORM * (h_initial_advanced - h_advanced[0]) * energy_efficiency:.4f}")
    print(f"     First point: v = {v_advanced_manual[0]:.4f} m/s")
    print(f"     Last point: v² = 2*{G_NORM:.2f}*({h_initial_advanced} - {h_advanced[-1]}) * 0.95 = {2 * G_NORM * (h_initial_advanced - h_advanced[-1]) * energy_efficiency:.4f}")
    print(f"     Last point: v = {v_advanced_manual[-1]:.4f} m/s")
    
    # From function
    acc_result = compute_acc_profile(points, dt=0.02, v0=0.0, use_energy_conservation=True)
    v_advanced_func = acc_result['v']
    print(f"   From function:")
    print(f"     First point: v = {v_advanced_func[0]:.4f} m/s")
    print(f"     Last point: v = {v_advanced_func[-1]:.4f} m/s")
    print(f"     Match: {np.allclose(v_advanced_manual, v_advanced_func, rtol=0.01)}")
    
    # Compare
    print("\n3. COMPARISON:")
    print(f"   Simple manual vs Advanced manual:")
    print(f"     First point: {v_simple_manual[0]:.4f} vs {v_advanced_manual[0]:.4f}")
    print(f"     Last point: {v_simple_manual[-1]:.4f} vs {v_advanced_manual[-1]:.4f}")
    print(f"     Match: {np.allclose(v_simple_manual, v_advanced_manual, rtol=0.01)}")
    
    print(f"   Simple func vs Advanced func:")
    print(f"     First point: {v_simple_func[0]:.4f} vs {v_advanced_func[0]:.4f}")
    print(f"     Last point: {v_simple_func[-1]:.4f} vs {v_advanced_func[-1]:.4f}")
    print(f"     Match: {np.allclose(v_simple_func, v_advanced_func, rtol=0.01)}")
    
    # Check if simple model adds minimum velocity
    print("\n4. CHECKING FOR MINIMUM VELOCITY:")
    print(f"   Simple model minimum: {v_simple_func.min():.4f} m/s")
    print(f"   Advanced model minimum: {v_advanced_func.min():.4f} m/s")
    print(f"   Simple model has minimum velocity added: {v_simple_func.min() > 0.0}")

if __name__ == "__main__":
    debug_speed_calculation()

