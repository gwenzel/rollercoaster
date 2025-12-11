"""
Debug speed calculation for app_builder
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.track import build_modular_track
from utils.acceleration import compute_acc_profile

def debug_app_speed():
    """Debug why speed might appear stuck"""
    # Create a test track similar to what app_builder might use
    track_elements = [
        {'type': 'lift_hill', 'params': {'length': 40, 'height': 80}},
        {'type': 'drop', 'params': {'length': 50, 'angle': 60}},
        {'type': 'straight', 'params': {'length': 20}},
    ]
    
    track_df = build_modular_track(track_elements)
    
    x = track_df['x'].values
    y = track_df['y'].values
    z = track_df.get('z', pd.Series(np.zeros_like(x))).values
    points = np.column_stack([x, z, y])
    
    print("="*80)
    print("APP SPEED DEBUG")
    print("="*80)
    
    print("\n1. TRACK GEOMETRY:")
    print(f"   Points: {len(points)}")
    print(f"   X (forward): [{x.min():.2f}, {x.max():.2f}] m")
    print(f"   Y (vertical): [{y.min():.2f}, {y.max():.2f}] m")
    print(f"   Z (lateral): [{z.min():.2f}, {z.max():.2f}] m")
    print(f"   Height range: {y.max() - y.min():.2f} m")
    
    # Test with app_builder settings
    print("\n2. APP BUILDER SETTINGS:")
    print("   v0 = 3.0 m/s")
    print("   use_energy_conservation = True")
    
    acc_result = compute_acc_profile(
        points,
        dt=0.02,
        mass=6000.0,
        rho=1.3,
        Cd=0.15,
        A=4.0,
        mu=0.003,
        v0=3.0,
        use_energy_conservation=True
    )
    
    velocity = acc_result['v']
    velocity_kmh = velocity * 3.6
    time = np.linspace(0, len(velocity) * 0.02, len(velocity))
    
    print("\n3. SPEED RESULTS:")
    print(f"   Speed range: [{velocity_kmh.min():.2f}, {velocity_kmh.max():.2f}] km/h")
    print(f"   Speed at start: {velocity_kmh[0]:.2f} km/h")
    print(f"   Speed at end: {velocity_kmh[-1]:.2f} km/h")
    print(f"   Speed std dev: {velocity_kmh.std():.2f} km/h")
    print(f"   Speed variation: {velocity_kmh.max() - velocity_kmh.min():.2f} km/h")
    
    # Check if speed is constant
    speed_changes = np.abs(np.diff(velocity_kmh))
    print(f"   Max speed change per step: {speed_changes.max():.4f} km/h")
    print(f"   Mean speed change per step: {speed_changes.mean():.4f} km/h")
    print(f"   Speed appears constant: {np.allclose(speed_changes, 0, atol=0.1)}")
    
    # Check first 10 and last 10 speeds
    print(f"\n4. SPEED PROFILE (first 10 points):")
    for i in range(min(10, len(velocity_kmh))):
        print(f"   t={time[i]:.2f}s: {velocity_kmh[i]:.2f} km/h (h={y[i]:.2f}m)")
    
    print(f"\n5. SPEED PROFILE (last 10 points):")
    for i in range(max(0, len(velocity_kmh)-10), len(velocity_kmh)):
        print(f"   t={time[i]:.2f}s: {velocity_kmh[i]:.2f} km/h (h={y[i]:.2f}m)")
    
    # Check energy conservation formula
    print(f"\n6. ENERGY CONSERVATION CHECK:")
    h = points[:, 2]  # Z coordinate (vertical in Z-up)
    h_initial = h[0]
    print(f"   Initial height: {h_initial:.2f} m")
    print(f"   Initial speed (v0): 3.0 m/s = {3.0*3.6:.2f} km/h")
    print(f"   Formula: v^2 = v0^2 + 2*g*(h_initial - h) * 0.95")
    
    # Manual calculation
    g = 9.81
    energy_efficiency = 0.95
    v_manual = np.sqrt(np.maximum(0, 3.0**2 + 2 * g * (h_initial - h) * energy_efficiency))
    v_manual_kmh = v_manual * 3.6
    
    print(f"   Manual calculation speed range: [{v_manual_kmh.min():.2f}, {v_manual_kmh.max():.2f}] km/h")
    print(f"   Matches function: {np.allclose(velocity_kmh, v_manual_kmh, rtol=0.01)}")
    
    # Check if height is changing
    print(f"\n7. HEIGHT PROFILE CHECK:")
    print(f"   Height changes: {np.abs(np.diff(h)).sum():.2f} m total")
    print(f"   Max height change per step: {np.abs(np.diff(h)).max():.4f} m")
    print(f"   Height appears constant: {np.allclose(np.diff(h), 0, atol=0.01)}")
    
    # Test with integration mode instead
    print(f"\n8. TESTING WITH INTEGRATION MODE (not energy conservation):")
    acc_result_int = compute_acc_profile(
        points,
        dt=0.02,
        mass=6000.0,
        rho=1.3,
        Cd=0.15,
        A=4.0,
        mu=0.003,
        v0=3.0,
        use_energy_conservation=False,
        use_velocity_verlet=True
    )
    
    velocity_int = acc_result_int['v']
    velocity_int_kmh = velocity_int * 3.6
    
    print(f"   Speed range: [{velocity_int_kmh.min():.2f}, {velocity_int_kmh.max():.2f}] km/h")
    print(f"   Speed variation: {velocity_int_kmh.max() - velocity_int_kmh.min():.2f} km/h")
    print(f"   Speed std dev: {velocity_int_kmh.std():.2f} km/h")

if __name__ == "__main__":
    debug_app_speed()

