"""
Debug time calculation issue
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.track import build_modular_track
from utils.acceleration import compute_acc_profile

def debug_time_calculation():
    """Debug why time calculation is showing 4000s instead of ~20s"""
    track_elements = [
        {'type': 'lift_hill', 'params': {'length': 40, 'height': 80}},
        {'type': 'drop', 'params': {'length': 50, 'angle': 60}},
    ]
    
    track_df = build_modular_track(track_elements)
    x = track_df['x'].values
    y = track_df['y'].values
    z = track_df.get('z', pd.Series(np.zeros_like(x))).values
    points = np.column_stack([x, z, y])
    
    acc_result = compute_acc_profile(
        points,
        dt=0.02,
        v0=3.0,
        use_energy_conservation=True
    )
    
    velocity = acc_result['v']
    velocity_kmh = velocity * 3.6
    
    # Check the time calculation
    print("="*80)
    print("TIME CALCULATION DEBUG")
    print("="*80)
    
    print(f"\n1. VELOCITY CHECK:")
    print(f"   Velocity range: [{velocity.min():.2f}, {velocity.max():.2f}] m/s")
    print(f"   Velocity range: [{velocity_kmh.min():.2f}, {velocity_kmh.max():.2f}] km/h")
    print(f"   First 5 velocities: {velocity[:5]}")
    
    # Distance calculation
    ds = np.linalg.norm(np.diff(points, axis=0, prepend=points[0:1]), axis=1)
    print(f"\n2. DISTANCE CHECK:")
    print(f"   ds range: [{ds.min():.4f}, {ds.max():.4f}] m")
    print(f"   Total distance: {ds.sum():.2f} m")
    print(f"   First 5 ds: {ds[:5]}")
    
    # Average velocity calculation
    v_avg = np.zeros_like(velocity)
    v_avg[0] = max(velocity[0], 0.1)
    for i in range(1, len(velocity)):
        v_avg[i] = 0.5 * (velocity[i-1] + velocity[i])
        v_avg[i] = max(v_avg[i], 0.1)
    
    print(f"\n3. AVERAGE VELOCITY CHECK:")
    print(f"   v_avg range: [{v_avg.min():.2f}, {v_avg.max():.2f}] m/s")
    print(f"   First 5 v_avg: {v_avg[:5]}")
    
    # Time calculation
    dt_actual = ds / v_avg
    print(f"\n4. TIME STEP CHECK:")
    print(f"   dt_actual range: [{dt_actual.min():.4f}, {dt_actual.max():.4f}] s")
    print(f"   First 5 dt_actual: {dt_actual[:5]}")
    print(f"   Total time: {dt_actual.sum():.2f} s")
    
    time = np.cumsum(dt_actual)
    print(f"\n5. CUMULATIVE TIME:")
    print(f"   Time range: [0, {time[-1]:.2f}] s")
    print(f"   First 5 times: {time[:5]}")
    print(f"   Last 5 times: {time[-5:]}")
    
    # Check for issues
    print(f"\n6. ISSUE DIAGNOSIS:")
    if time[-1] > 100:
        print(f"   [ERROR] Total time is too large: {time[-1]:.2f} s")
        print(f"   Possible causes:")
        print(f"     - ds values too large (check units)")
        print(f"     - v_avg values too small")
        print(f"     - Division error")
        
        # Check for problematic values
        large_dt = dt_actual > 10
        if large_dt.any():
            print(f"   Found {large_dt.sum()} time steps > 10s")
            print(f"   Max dt: {dt_actual.max():.4f} s")
            print(f"   At max dt: ds={ds[np.argmax(dt_actual)]:.4f} m, v_avg={v_avg[np.argmax(dt_actual)]:.4f} m/s")
    
    # Compare with expected time
    total_distance = ds.sum()
    avg_speed = velocity.mean()
    expected_time = total_distance / avg_speed if avg_speed > 0 else 0
    print(f"\n7. EXPECTED TIME:")
    print(f"   Total distance: {total_distance:.2f} m")
    print(f"   Average speed: {avg_speed:.2f} m/s")
    print(f"   Expected time: {expected_time:.2f} s")
    print(f"   Actual time: {time[-1]:.2f} s")
    print(f"   Ratio: {time[-1] / expected_time if expected_time > 0 else 0:.2f}x")

if __name__ == "__main__":
    debug_time_calculation()

